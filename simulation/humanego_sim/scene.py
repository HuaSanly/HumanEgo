# -*- coding: utf-8 -*-
"""
MJCF scene builder for the HumanEgo serve_bread rig.

The geometry is driven by the *real* calibration: each arm's base pose comes
from `T_base_in_cam` (hand-eye calibration of the deployed robot) and the
camera is placed `cam_height` above the table. The generated XML is written
next to the assets with relative paths, so the very same file can be loaded by
MuJoCo's WebAssembly build in the browser.

Geom groups (used to render the policy's "clean" view without the robot):
    0  table / rig / floor          (visible in every view)
    1  robot visual meshes           (hidden in the policy view)
    2  object visual meshes          (visible in every view)
    3  collision geometry            (never rendered)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .cameras import CameraSpec
from .robots import ASSETS, RobotSpec
from .softbody import add_soft_bread

G_STATIC, G_ROBOT, G_OBJECT, G_COLLISION = 0, 1, 2, 3


def _quat_wxyz(Rm: np.ndarray) -> List[float]:
    x, y, z, w = R.from_matrix(Rm).as_quat()
    return [float(w), float(x), float(y), float(z)]


def yaw_quat(yaw_deg: float) -> List[float]:
    return _quat_wxyz(R.from_euler("z", yaw_deg, degrees=True).as_matrix())


@dataclass
class RigGeometry:
    T_cam_in_world: np.ndarray                 # (4,4) OpenCV optical frame in world
    T_base_in_world: Dict[str, np.ndarray]     # per side
    T_base_in_cam: Dict[str, np.ndarray]


def rig_geometry(cfg: dict, sides: List[str]) -> RigGeometry:
    """World poses of the camera and the arm bases from the real calibration.

    The arm bases are level (mounted on the table frame), so the world axes are
    the base axes; the camera pose follows from inv(T_base_in_cam)."""
    rig = cfg["rig"]
    Tbc = {s: np.array(rig["T_base_in_cam"][s], dtype=np.float64) for s in ("right", "left")}
    T_cam_in_base = np.linalg.inv(Tbc["right"])            # camera in the right base frame
    # place the right base so that the camera ends up at (0, 0, cam_height)
    p_base_r = np.array([0.0, 0.0, rig["cam_height"]]) - T_cam_in_base[:3, 3]
    T_base_r = np.eye(4)
    T_base_r[:3, 3] = p_base_r
    T_cam = T_base_r @ T_cam_in_base
    T_base = {}
    for s in ("right", "left"):
        T_base[s] = T_cam @ Tbc[s]
    return RigGeometry(T_cam_in_world=T_cam, T_base_in_world={s: T_base[s] for s in sides},
                       T_base_in_cam={s: Tbc[s] for s in sides})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rel(path_from_assets: str) -> str:
    return path_from_assets.replace(os.sep, "/")


def _spec_with_rel_paths(xml_rel: str) -> mujoco.MjSpec:
    """Load an MJCF and rewrite its asset paths relative to the assets root."""
    xml_path = os.path.join(ASSETS, xml_rel)
    child = mujoco.MjSpec.from_file(xml_path)
    rel_dir = os.path.relpath(os.path.dirname(xml_path), ASSETS)
    mdir = child.meshdir or ""
    tdir = child.texturedir or ""
    for m in child.meshes:
        if m.file:
            m.file = _rel(os.path.normpath(os.path.join(rel_dir, mdir, m.file)))
    for t in child.textures:
        if t.file:
            t.file = _rel(os.path.normpath(os.path.join(rel_dir, tdir, t.file)))
    child.meshdir = ""
    child.texturedir = ""
    child.modelfiledir = ASSETS
    return child


def _load_robot_spec(robot: RobotSpec, phys: dict) -> mujoco.MjSpec:
    child = _spec_with_rel_paths(robot.xml)
    for att in robot.attach:                     # e.g. Robotiq 2F-85 on the UR5e flange
        sub = _spec_with_rel_paths(att["xml"])
        child.attach(sub, prefix=att.get("prefix", ""), site=child.site(att["site"]))
    if robot.tool_site_spec:
        ts = child.body(robot.tool_site_spec["body"]).add_site()
        ts.name = robot.tool_site
        ts.pos = list(robot.tool_site_spec["pos"])
        ts.size = [0.004, 0, 0]
        ts.group = 5
    # one visual / collision convention for every robot: visuals in group 1 (hidden in the
    # policy's clean view), collision in group 3
    for g in child.geoms:
        if g.contype == 0 and g.conaffinity == 0:
            g.group = G_ROBOT
        else:
            g.group = G_COLLISION
    # finger pads (silicone / rubber on all three hands): physics.pad_friction for every collision geom of
    # the finger bodies, at priority 0 so that a bread contact takes the larger friction of the two
    if phys.get("pad_friction") is not None:
        for fb in robot.finger_bodies:
            for g in child.body(fb).find_all(mujoco.mjtObj.mjOBJ_GEOM):
                if g.contype or g.conaffinity:
                    g.friction = [float(phys["pad_friction"]), g.friction[1], g.friction[2]]
                    g.priority = 0
    # real servos run on-board gravity compensation
    for b in child.bodies:
        if b.name != "world":
            b.gravcomp = 1.0
    # stiffer position servos (the real driver tracks within ~1 mm); keep the
    # Menagerie ratios between joints
    kp_scale = float(robot.kp_scale if robot.kp_scale is not None else phys.get("arm_kp_scale", 1.0))
    kv_ratio = float(phys.get("arm_kv_ratio", 0.0))
    arm_set = set(robot.arm_actuators)
    tune_gains = robot.kp_scale is None          # only the WidowX AI servos are re-tuned
    for a in child.actuators:
        if a.name in arm_set and tune_gains:
            kp = a.gainprm[0] * kp_scale
            a.gainprm[0] = kp
            a.biasprm[1] = -kp
            a.biasprm[2] = -kv_ratio * kp
            a.forcerange = [a.forcerange[0] * 2.0, a.forcerange[1] * 2.0]
        elif a.name == robot.gripper_actuator and (robot.gripper_force_param
                                                    or robot.name in phys.get("robot_gripper", {})):
            # physics.gripper_force / gripper_kp_scale tune the Trossen (real rig); other robots use their
            # RobotSpec values unless physics.robot_gripper.<robot> overrides them
            rg = phys.get("robot_gripper", {}).get(robot.name, {})
            trossen = robot.name == "trossen_wxai"
            f = float(phys.get("gripper_force", robot.gripper_force)) if trossen else float(rg.get("force", robot.gripper_force))
            a.forcerange = [-f, f]
            a.forcelimited = 1
            gk = float(phys.get("gripper_kp_scale", 1.0)) if trossen else float(rg.get("kp_scale", robot.gripper_kp_scale))
            if gk != 1.0:              # servo stiffness: how hard a closed-on-bread gripper squeezes
                a.gainprm[0] *= gk
                a.biasprm[1] *= gk
                a.biasprm[2] *= float(np.sqrt(gk))
    return child


def _add_box(body, name, size, pos, rgba=None, material=None, group=G_STATIC, collide=True, quat=None):
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(size)
    g.pos = list(pos)
    if quat is not None:
        g.quat = list(quat)
    if rgba is not None:
        g.rgba = list(rgba)
    if material is not None:
        g.material = material
    g.group = group
    if not collide:
        g.contype = 0
        g.conaffinity = 0
    return g


# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------
def build_scene(cfg: dict, robot: RobotSpec, cam: CameraSpec, out_name: Optional[str] = None) -> Tuple[str, dict]:
    """Build the MJCF, write it under ASSETS/scenes/, and return (xml_path, meta)."""
    task = cfg.get("task", "serve_bread")
    pick_cube = task == "pick_cube"
    mode = cfg.get("mode", "bimanual")
    sides = ["right", "left"] if mode == "bimanual" else ["right"]
    phys = cfg.get("physics", {})
    geo = rig_geometry(cfg, sides)

    spec = mujoco.MjSpec()
    spec.modelname = f"humanego_{task}_{robot.name}_{mode}"
    spec.modelfiledir = ASSETS          # relative asset paths resolve from the assets root
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    soft = not pick_cube and phys.get("bread_model", "soft") == "soft"
    spec.option.timestep = float(phys.get("timestep", 0.002))
    # flex elasticity is only integrated implicitly by the `discrete` (velocity-stepping) integrator
    spec.option.integrator = (mujoco.mjtIntegrator.mjINT_DISCRETE if soft
                              else mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # high impratio = stiff friction relative to the normal direction: no slow creep of a held bread
    spec.option.impratio = float(phys.get("impratio", 100.0 if soft else 10.0))
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1440
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.22, 0.22, 0.22]
    spec.visual.headlight.diffuse = [0.22, 0.22, 0.22]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.stat.center = [0.45, 0.0, 0.1]
    spec.stat.extent = 1.0

    # ---------------- assets ----------------
    tex = spec.add_texture()
    tex.name = "skybox"
    tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    tex.rgb1 = [0.80, 0.80, 0.78]
    tex.rgb2 = [0.55, 0.55, 0.53]
    tex.width, tex.height = 512, 3072

    t = spec.add_texture()
    t.name = "maple"
    t.type = mujoco.mjtTexture.mjTEXTURE_2D
    t.file = "objects/table/maple.png"
    m = spec.add_material()
    m.name = "maple"
    m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "maple"
    m.texrepeat = [2.0, 2.0]
    m.texuniform = True
    m.specular = 0.15
    m.shininess = 0.3

    m = spec.add_material()
    m.name = "frame_black"
    m.rgba = [0.06, 0.06, 0.065, 1.0]
    m.specular = 0.3
    m.shininess = 0.5

    m = spec.add_material()
    m.name = "floor"
    m.rgba = [0.62, 0.62, 0.60, 1.0]
    m.reflectance = 0.0

    breads = {} if pick_cube else _breads(cfg, mode, sides)
    kinds = sorted({b["type"] for b in breads.values()})
    bread_vis = {}
    for obj in kinds:
        t = spec.add_texture()
        t.name = f"{obj}_tex"
        t.type = mujoco.mjtTexture.mjTEXTURE_2D
        t.file = f"objects/{obj}/{obj}.png"
        mm = spec.add_material()
        mm.name = f"{obj}_mat"
        mm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{obj}_tex"
        mm.specular = 0.35             # glossy egg wash
        mm.shininess = 0.45
        d = os.path.join(ASSETS, "objects", obj)
        if soft:
            import trimesh
            surf = phys.get("soft", {}).get("surface", "coarse")
            fm = trimesh.load(os.path.join(d, f"{obj}_flex{'_fine' if surf == 'fine' else ''}.obj"),
                              force="mesh", process=False)
            z = np.load(os.path.join(d, f"{obj}_vis.npz"))
            rad = float(phys.get("soft", {}).get("radius", 0.001))
            # one flex mesh per distinct size of this kind (scene.bread.<side>.scale)
            for sc in sorted({_bread_scale(b) for b in breads.values() if b["type"] == obj}):
                me = spec.add_mesh()
                me.name = _scaled(f"{obj}_flex", sc)
                # the flex surface carries a contact shell of `radius`: shrink it by that much so the
                # outer contact surface stays on the visible crust (a thick shell resists tunnelling)
                Vf = np.asarray(fm.vertices) * sc
                ext_f = Vf.max(0) - Vf.min(0)
                Vf = Vf * np.maximum(ext_f - 2 * rad, 1e-3) / ext_f
                me.uservert = Vf.ravel().tolist()
                me.userface = np.asarray(fm.faces).ravel().tolist()
                bread_vis[(obj, sc)] = (z["V"] * sc, z["F"], z["UV"])
        else:
            n_col = len([f for f in os.listdir(d) if f.startswith(f"{obj}_col")])
            for sc in sorted({_bread_scale(b) for b in breads.values() if b["type"] == obj}):
                for mn, fn in [(f"{obj}_vis", f"{obj}.obj")] + [(f"{obj}_col{i}", f"{obj}_col{i}.obj") for i in range(n_col)]:
                    mc = spec.add_mesh()
                    mc.name = _scaled(mn, sc)
                    mc.file = f"objects/{obj}/{fn}"
                    if sc != 1.0:
                        mc.scale = [sc, sc, sc]

    if not pick_cube:
        with open(os.path.join(ASSETS, "objects", "plate", "plate_meta.json")) as f:
            pm = json.load(f)
        me = spec.add_mesh()
        me.name = "plate_vis"
        me.file = "objects/plate/plate.obj"
        mm = spec.add_material()
        mm.name = "plate_mat"
        mm.rgba = pm["rgba"]
        mm.specular = 0.25
        mm.shininess = 0.35

    wb = spec.worldbody

    # ---------------- lights ----------------
    l = wb.add_light()
    l.name = "key"
    l.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    l.pos = [-0.6, 0.8, 2.5]
    l.dir = list(-np.array([-0.6, 0.8, 2.5]) / np.linalg.norm([-0.6, 0.8, 2.5]))
    l.diffuse = [0.48, 0.48, 0.48]
    l.specular = [0.15, 0.15, 0.15]
    l.castshadow = True
    l2 = wb.add_light()
    l2.name = "fill"
    l2.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    l2.pos = [1.5, -1.0, 2.0]
    l2.dir = list(-np.array([1.5, -1.0, 2.0]) / np.linalg.norm([1.5, -1.0, 2.0]))
    l2.diffuse = [0.16, 0.16, 0.16]
    l2.castshadow = False

    # ---------------- floor + table ----------------
    tb = cfg["rig"]["table"]
    th = float(tb["thickness"])
    table_h = 0.76
    floor = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [4.0, 4.0, 0.05]
    floor.pos = [0.4, 0.0, -table_h]
    floor.material = "floor"
    floor.group = G_STATIC

    table = wb.add_body()
    table.name = "table"
    cx = 0.5 * (tb["x_min"] + tb["x_max"])
    sx = 0.5 * (tb["x_max"] - tb["x_min"])
    sy = float(tb["half_width"])
    gtop = _add_box(table, "table_top", [sx, sy, th / 2], [cx, 0.0, -th / 2], material="maple")
    gtop.friction = [float(phys.get("table_friction", 0.8)), 0.005, 0.0001]
    leg_h = (table_h - th) / 2
    for i, (lx, ly) in enumerate([(tb["x_min"] + 0.06, sy - 0.06), (tb["x_min"] + 0.06, -sy + 0.06),
                                  (tb["x_max"] - 0.06, sy - 0.06), (tb["x_max"] - 0.06, -sy + 0.06)]):
        _add_box(table, f"table_leg{i}", [0.025, 0.025, leg_h], [lx, ly, -th - leg_h],
                 material="frame_black", collide=False)

    # ---------------- rig frame (visual): rail behind the camera's view frustum, brackets to the
    # arm bases, and the camera mast. Nothing of it is visible from the policy camera.
    rig = wb.add_body()
    rig.name = "rig_frame"
    base_z = float(geo.T_base_in_world["right"][2, 3])
    base_x = float(geo.T_base_in_world["right"][0, 3])
    cam_p = geo.T_cam_in_world[:3, 3]
    rail_x = cam_p[0] + 0.02
    rail_y = max(max(abs(geo.T_base_in_world[s][1, 3]) for s in sides) + 0.10, 0.42)
    rail_z = base_z - 0.035
    _add_box(rig, "rail", [0.02, rail_y, 0.02], [rail_x, 0.0, rail_z], material="frame_black")
    for s in sides:
        by = float(geo.T_base_in_world[s][1, 3])
        _add_box(rig, f"bracket_{s}", [(base_x - rail_x) / 2 + 0.03, 0.045, 0.006],
                 [(base_x + rail_x) / 2, by, base_z - 0.006], material="frame_black")
    for i, yy in enumerate((rail_y - 0.02, -rail_y + 0.02)):
        _add_box(rig, f"post{i}", [0.02, 0.02, (rail_z - 0.02) / 2], [rail_x, yy, (rail_z - 0.02) / 2],
                 material="frame_black")
    mast_h = cam_p[2] + 0.05
    _add_box(rig, "mast", [0.015, 0.015, mast_h / 2], [cam_p[0] - 0.045, 0.0, mast_h / 2], material="frame_black")
    _add_box(rig, "cam_arm", [0.03, 0.012, 0.01], [cam_p[0] - 0.02, 0.0, cam_p[2] + 0.03], material="frame_black")
    # camera body (D405-sized housing) aligned with the optical axis
    R_cam = geo.T_cam_in_world[:3, :3]
    _add_box(rig, "camera_housing", [0.021, 0.0115, 0.021], cam_p - R_cam[:, 2] * 0.012,
             rgba=[0.12, 0.12, 0.13, 1], quat=_quat_wxyz(R_cam @ np.diag([1, 1, 1])), collide=False)

    # ---------------- policy camera (OpenCV -> MuJoCo: flip y and z) ----------------
    c = wb.add_camera()
    c.name = "policy_cam"
    c.pos = list(cam_p)
    c.quat = _quat_wxyz(R_cam @ np.diag([1.0, -1.0, -1.0]))
    c.fovy = cam.fovy_deg

    # a few nice viewing cameras for videos
    for name, pos, look in [("front_view", [1.45, -0.55, 0.62], [0.42, 0.0, 0.05]),
                            ("side_view", [0.55, -1.25, 0.55], [0.45, 0.0, 0.05]),
                            ("top_view", [0.50, 0.0, 1.35], [0.45, 0.0, 0.0])]:
        cc = wb.add_camera()
        cc.name = name
        pos = np.array(pos)
        fwd = np.array(look) - pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0, -1.0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        cc.pos = list(pos)
        cc.quat = _quat_wxyz(np.stack([right, up, -fwd], 1))
        cc.fovy = 40.0

    # ---------------- arms ----------------
    for s in sides:
        child = _load_robot_spec(robot, phys)
        child.option.impratio = spec.option.impratio      # the parent's options win; avoid the attach warning
        child.option.integrator = spec.option.integrator
        Tb = geo.T_base_in_world[s].copy()
        Tb[:3, 3] += np.array(robot.mount_offset)
        fr = wb.add_frame()
        fr.pos = list(Tb[:3, 3])
        fr.quat = _quat_wxyz(Tb[:3, :3])
        spec.attach(child, prefix=f"{s}/", frame=fr)
        # soft-bread proxy: real bread deforms around the fingers and gives a form-closure
        # grip even when pinched near its top. Rigid sim bread cannot, so each finger gets an
        # adhesion actuator that the driver switches on only while the gripper is closed.
        gain = 0.0 if soft or pick_cube else float(phys.get("finger_adhesion", 0.0))
        if gain > 0:
            for fb in robot.finger_bodies:
                act = spec.add_actuator()
                act.name = f"{s}/adhesion_{fb}"
                act.trntype = mujoco.mjtTrn.mjTRN_BODY
                act.target = f"{s}/{fb}"
                act.set_to_adhesion(gain)
                act.ctrlrange = [0.0, 1.0]

    # ---------------- objects ----------------
    sc = cfg["scene"]
    breads = {} if pick_cube else _breads(cfg, mode, sides)
    plate_pose = None if pick_cube else (sc["single"]["plate"] if mode == "single" else sc["plate"])

    bm = phys.get("bread_mass", {})
    bf = phys.get("bread_friction", [1.0, 0.01, 0.0005])
    contact = {"friction": list(bf), "condim": int(phys.get("bread_condim", 4)),
               "solref": list(phys.get("bread_solref", [0.004, 1.0])),
               "solimp": list(phys.get("bread_solimp", [0.9, 0.95, 0.001, 0.5, 2.0])),
               "priority": int(phys.get("bread_priority", 1)), "solmix": float(phys.get("bread_solmix", 1.0))}
    obj_meta = _add_pick_cube(wb, sc["objects"]) if pick_cube else {}
    for s, bd in breads.items():
        kind = bd["type"]
        x, y, yaw = bd["pose"]
        scl = _bread_scale(bd)                   # uniform size variation, same density
        ext = _mesh_extents(kind) * scl
        mass = float(bm.get(kind, 0.055)) * scl ** 3
        if soft:
            sp_cfg = dict(phys.get("soft", {}))
            cells = sp_cfg.get("cells", {})
            sp_cfg["cells"] = cells.get(kind, [4, 2, 2]) if isinstance(cells, dict) else cells
            # compiled undeformed with yaw 0; the episode reset places it (env.set_object_pose)
            grid = add_soft_bread(spec, f"bread_{s}", kind, [x, y, ext[2] / 2 + 0.0008], mass,
                                  bread_vis[(kind, scl)], _scaled(f"{kind}_flex", scl), f"{kind}_mat", sp_cfg, contact,
                                  flex_group=G_COLLISION, skin_group=G_OBJECT)
            obj_meta[f"bread_{s}"] = {"kind": kind, "extents": ext.tolist(), "side": s, "soft": grid}
            if scl != 1.0:
                obj_meta[f"bread_{s}"]["scale"] = scl
            continue
        b = wb.add_body()
        b.name = f"bread_{s}"
        b.pos = [x, y, ext[2] / 2 + 0.0005]
        b.quat = yaw_quat(yaw)
        fj = b.add_freejoint()
        fj.name = f"bread_{s}_free"
        g = b.add_geom()
        g.name = f"bread_{s}_vis"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = _scaled(f"{kind}_vis", scl)
        g.material = f"{kind}_mat"
        g.group = G_OBJECT
        g.contype = 0
        g.conaffinity = 0
        g.density = 0
        d = os.path.join(ASSETS, "objects", kind)
        n_col = len([f for f in os.listdir(d) if f.startswith(f"{kind}_col")])
        # MuJoCo derives mass from geom volume x density: pick the density that gives the real weight
        import trimesh
        vol = sum(trimesh.load(os.path.join(d, f"{kind}_col{i}.obj"), force="mesh").convex_hull.volume
                  for i in range(n_col)) * scl ** 3
        dens = mass / max(vol, 1e-9)
        for i in range(n_col):
            gc = b.add_geom()
            gc.name = f"bread_{s}_col{i}"
            gc.type = mujoco.mjtGeom.mjGEOM_MESH
            gc.meshname = _scaled(f"{kind}_col{i}", scl)
            gc.group = G_COLLISION
            gc.friction = contact["friction"]
            gc.solref = contact["solref"]
            gc.solimp = contact["solimp"]
            gc.priority = 1          # bread's (soft) contact parameters win against the finger pads
            gc.condim = contact["condim"]
            gc.rgba = [0.8, 0.5, 0.2, 0.0]
            gc.density = dens
        obj_meta[f"bread_{s}"] = {"kind": kind, "extents": ext.tolist(), "side": s}
        if scl != 1.0:
            obj_meta[f"bread_{s}"]["scale"] = scl

    if not pick_cube:
        # plate: visual lathe mesh + primitive collision (flat well + tilted rim ring)
        px, py, pyaw = plate_pose
        pb = wb.add_body()
        pb.name = "plate"
        pb.pos = [px, py, 0.0005]
        pb.quat = yaw_quat(pyaw)
        fj = pb.add_freejoint()
        fj.name = "plate_free"
        g = pb.add_geom()
        g.name = "plate_vis"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = "plate_vis"
        g.material = "plate_mat"
        g.group = G_OBJECT
        g.contype = 0
        g.conaffinity = 0
        g.density = 0
        Rr, H, rw, tt, fh = pm["R"], pm["H"], pm["r_well"], pm["t"], pm["foot_h"]
        gw = pb.add_geom()
        gw.name = "plate_well"
        gw.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        gw.size = [rw + 0.004, (fh + tt) / 2, 0]
        gw.pos = [0, 0, (fh + tt) / 2]
        gw.group = G_COLLISION
        gw.rgba = [0, 0, 1, 0]
        n_seg = 28
        z0, z1 = fh + tt, H
        slope = np.arctan2(z1 - z0, Rr - rw)
        L = np.hypot(Rr - rw, z1 - z0)
        r_mid = 0.5 * (Rr + rw)
        half_w = np.pi * r_mid / n_seg + 0.002
        for i in range(n_seg):
            a = 2 * np.pi * i / n_seg
            Rm = R.from_euler("zy", [a, -slope]).as_matrix()
            gr = pb.add_geom()
            gr.name = f"plate_rim{i}"
            gr.type = mujoco.mjtGeom.mjGEOM_BOX
            gr.size = [L / 2, half_w, 0.0025]
            gr.pos = [r_mid * np.cos(a), r_mid * np.sin(a), 0.5 * (z0 + z1) - 0.0025]
            gr.quat = _quat_wxyz(Rm)
            gr.group = G_COLLISION
            gr.rgba = [0, 0, 1, 0]
        plate_vol = np.pi * (rw + 0.004) ** 2 * (fh + tt) + n_seg * (L * 2 * half_w * 0.005)
        for gg in pb.geoms:
            if gg.name.startswith("plate_well") or gg.name.startswith("plate_rim"):
                gg.density = float(phys.get("plate_mass", 0.45)) / plate_vol
                if "plate_friction" in phys:             # plastic plate (MuJoCo default 1 otherwise)
                    gg.friction = [float(phys["plate_friction"]), 0.005, 0.0001]
        obj_meta["plate"] = {"kind": "plate", "R": Rr, "H": H}

    # ---------------- write ----------------
    os.makedirs(os.path.join(ASSETS, "scenes"), exist_ok=True)
    model = spec.compile()                        # validate before writing
    xml = spec.to_xml()
    if out_name is None:
        import hashlib
        h = hashlib.sha1(xml.encode()).hexdigest()[:10]
        out_name = f"{task}_{robot.name}_{mode}_{h}.xml"
    xml_path = os.path.join(ASSETS, out_name)   # sits at the assets root -> relative asset paths
    if not os.path.exists(xml_path):
        tmp = f"{xml_path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(xml)
        os.replace(tmp, xml_path)                 # atomic: parallel workers never see a partial file
    meta = {
        "sides": sides,
        "mode": mode,
        "robot": robot.name,
        "camera": cam.name,
        "T_cam_in_world": geo.T_cam_in_world.tolist(),
        "T_base_in_world": {s: geo.T_base_in_world[s].tolist() for s in sides},
        "T_base_in_cam": {s: geo.T_base_in_cam[s].tolist() for s in sides},
        "objects": obj_meta,
        "K": cam.K.tolist(),
        "image_size": [cam.height, cam.width],
        "nq": int(model.nq),
    }
    if pick_cube:
        meta["object_roles"] = {role: [n for n, obj in sc["objects"].items() if obj["role"] == role]
                                for role in ("obj1", "obj2")}
    return xml_path, meta


def _add_pick_cube(worldbody, objects: dict) -> dict:
    meta = {}
    for name, obj in objects.items():
        size = np.asarray(obj["size"], dtype=float)
        if size.shape != (3,) or np.any(size <= 0):
            raise ValueError(f"scene.objects.{name}.size must contain three positive lengths")
        half = size / 2
        x, y, yaw = obj["pose"]
        body = worldbody.add_body()
        body.name = name
        body.pos = [x, y, half[2] + 0.0008]
        body.quat = yaw_quat(yaw)
        body.add_freejoint().name = f"{name}_free"
        if obj["kind"] == "container":
            wall = float(obj["wall"])
            if not 0 < wall < min(half):
                raise ValueError(f"scene.objects.{name}.wall must be positive and smaller than half its size")
            hx, hy, hz = half
            parts = [("bottom", [hx, hy, wall / 2], [0, 0, -hz + wall / 2])]
            for sign in (-1, 1):
                parts.append((f"x{sign}", [wall / 2, hy, hz - wall / 2],
                              [sign * (hx - wall / 2), 0, wall / 2]))
                parts.append((f"y{sign}", [hx - wall, wall / 2, hz - wall / 2],
                              [0, sign * (hy - wall / 2), wall / 2]))
        elif obj["kind"] == "cube":
            parts = [("box", half, [0, 0, 0])]
        else:
            raise ValueError(f"Unsupported pick_cube prop kind: {obj['kind']}")
        volume = sum(8 * np.prod(half_size) for _, half_size, _ in parts)
        for suffix, half_size, pos in parts:
            geom = _add_box(body, f"{name}_{suffix}", half_size, pos, rgba=obj["rgba"], group=G_OBJECT)
            geom.density = float(obj["mass"]) / volume
            geom.friction = obj["friction"]
        meta[name] = {"kind": obj["kind"], "extents": size.tolist(), "role": obj["role"]}
    return meta


def _breads(cfg: dict, mode: str, sides: List[str]) -> Dict[str, dict]:
    sc = cfg["scene"]
    if mode == "single":
        return {"right": sc["single"]["bread"]}
    return {s: sc["bread"][s] for s in sides}


def _bread_scale(bd: dict) -> float:
    """scene.bread.<side>.scale: uniform scale of one bread (1 = the measured size)."""
    return float(bd.get("scale", 1.0))


def _scaled(mesh_name: str, scale: float) -> str:
    """Mesh name of a scaled bread; the unscaled name (and so the unchanged XML) at scale 1."""
    return mesh_name if scale == 1.0 else f"{mesh_name}_s{round(scale * 1e4):05d}"


def _mesh_extents(kind: str) -> np.ndarray:
    import trimesh
    m = trimesh.load(os.path.join(ASSETS, "objects", kind, f"{kind}.obj"), force="mesh", process=False)
    return np.asarray(m.extents)
