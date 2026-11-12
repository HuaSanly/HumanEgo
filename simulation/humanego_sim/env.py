# -*- coding: utf-8 -*-
"""
ServeBreadSim — the MuJoCo world: physics, the two simulated arms, and the
RGB-D camera the policy looks through.

Nothing here knows about the policy; `runner.py` wires the released HumanEgo
inference logic on top of it.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import os
from typing import Dict, List, Optional

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from .cameras import CameraSpec, get_camera
from .driver import SimArm, T_from
from .robots import RobotSpec, get_robot
from .softbody import PlasticSet, SoftBody
from .scene import G_ROBOT, build_scene, yaw_quat

SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SIM_ROOT)


def _read_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base_config", None)
    if base:
        cfg = deep_update(_read_config(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> dict:
    path = path or os.path.join(SIM_ROOT, "configs", "serve_bread.yaml")
    cfg = _read_config(path)
    # per-robot calibration (e.g. a different gripper's tool pitch) sits between the defaults
    # and explicit overrides
    robot = (overrides or {}).get("robot", cfg.get("robot"))
    per = cfg.get("per_robot", {}).get(robot)
    if per:
        cfg = deep_update(cfg, per)
    # named profile (profiles.py: production | tuned) sits between per_robot and explicit overrides
    from .profiles import profile_overrides
    profile = (overrides or {}).get("profile", cfg.get("profile"))
    cfg = deep_update(cfg, profile_overrides(profile, robot, cfg))
    if overrides:
        cfg = deep_update(cfg, overrides)
    cfg["profile"] = profile or "production"
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ServeBreadSim:
    def __init__(self, cfg: dict, seed: int = 0, render: bool = True):
        self.cfg = cfg
        self.robot: RobotSpec = get_robot(cfg["robot"])
        self.cam: CameraSpec = get_camera(cfg["camera"])
        if cfg.get("camera_intrinsics"):
            self.cam = replace(self.cam, **cfg["camera_intrinsics"])
        self.xml_path, self.meta = build_scene(cfg, self.robot, self.cam)
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.sides: List[str] = self.meta["sides"]
        self.dt = float(self.m.opt.timestep)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

        self.T_cam_in_world = np.array(self.meta["T_cam_in_world"])
        self.T_world_in_cam = np.linalg.inv(self.T_cam_in_world)
        self.K = self.cam.K.copy()
        self.H, self.W = self.cam.height, self.cam.width

        # objects: rigid ones are free bodies, soft breads are flexes (see softbody.py)
        self.obj_names: List[str] = list(self.meta["objects"])
        self.soft: Dict[str, SoftBody] = {n: SoftBody(self.m, n, o["soft"])
                                          for n, o in self.meta["objects"].items() if "soft" in o}
        pl = cfg.get("physics", {}).get("soft", {}).get("plastic", {})          # optional plastic set
        self.plastic = [PlasticSet(self.m, sb, self.meta["objects"][n]["soft"], pl)
                        for n, sb in self.soft.items()] if pl.get("enabled") else []
        self.obj_bodies = {name: self.m.body(name).id for name in self.obj_names if name not in self.soft}
        self.obj_free_qadr = {name: self.m.jnt_qposadr[self.m.body_jntadr[bid]] for name, bid in self.obj_bodies.items()}
        self.obj_free_dadr = {name: self.m.jnt_dofadr[self.m.body_jntadr[bid]] for name, bid in self.obj_bodies.items()}
        # (object type, id) -> segmentation label: geoms of rigid objects, skins of soft breads
        self.seg_label = {}
        for name, bid in self.obj_bodies.items():
            for gi in range(self.m.ngeom):
                if self.m.geom_bodyid[gi] == bid:
                    self.seg_label[(int(mujoco.mjtObj.mjOBJ_GEOM), gi)] = name
        for name in self.soft:
            self.seg_label[(int(mujoco.mjtObj.mjOBJ_SKIN), mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SKIN, f"{name}_skin"))] = name

        self._init_arms()
        # robustness hooks (humanego_sim/robustness.py): visual-only clutter in every render, and a
        # short qpos history for rendering a delayed camera frame; both off by default
        self.distractors: List[dict] = []
        self.obs_history = None
        self._d_past = None
        self.renderer = mujoco.Renderer(self.m, self.H, self.W) if render else None
        self._opt_clean = mujoco.MjvOption()
        self._opt_clean.geomgroup[G_ROBOT] = 0          # hide the robot (perfect "inpainting")

    # ------------------------------------------------------------------ arms
    def _init_arms(self):
        self.arms: Dict[str, SimArm] = {}
        for s in self.sides:
            self.arms[s] = SimArm(self.m, self.d, s, self.robot,
                                  np.array(self.meta["T_base_in_world"][s]),
                                  np.array(self.meta["T_base_in_cam"][s]),
                                  tcp_offset=float(self.cfg.get("control", {}).get("tcp_offset", 0.0)),
                                  tool_pitch_deg=float(self.cfg.get("control", {}).get("tool_pitch_deg", 0.0)))

    # ----------------------------------------------------------------- reset
    def reset(self, scene: Optional[dict] = None, settle: float = 0.4, arm_q: Optional[dict] = None):
        """Reset physics. `scene` overrides object poses:
            {"plate": [x, y, yaw], "bread_right": [x, y, yaw], "bread_left": [...]}
        """
        mujoco.mj_resetData(self.m, self.d)
        for p in self.plastic:
            p.reset(self.m)
        self.t = 0.0
        sc = self.default_scene()
        if scene:
            sc.update(scene)
        for name, pose in sc.items():
            if name not in self.obj_names:
                continue
            self.set_object_pose(name, *pose)
        for s, arm in self.arms.items():
            q = (arm_q or {}).get(s, self.initial_joints(s))
            arm.set_j_instant(q, gripper=self.robot.gripper_open)
        mujoco.mj_forward(self.m, self.d)
        # let objects settle under gravity with the arms holding position
        n = int(settle / self.dt)
        for _ in range(n):
            self.step()
        self.t = 0.0
        for arm in self.arms.values():
            arm.t = 0.0
            arm._g_t0 = 0.0

    def initial_joints(self, side: str):
        """Real-rig start pose; with rig.mirror_left_start the left arm starts at the exact mirror of
        the right arm (the real rig's left start pose is not symmetric, which costs it success)."""
        r = self.robot
        if side == "left" and self.cfg["rig"].get("mirror_left_start") and r.mirror_sign:
            return [a * b for a, b in zip(r.mirror_sign, r.initial_joints["right"])]
        return r.initial_joints[side]

    def default_scene(self) -> dict:
        sc = self.cfg["scene"]
        if self.cfg.get("task") == "pick_cube":
            return {name: list(obj["pose"]) for name, obj in sc["objects"].items()}
        if self.cfg.get("mode") == "single":
            return {"plate": list(sc["single"]["plate"]), "bread_right": list(sc["single"]["bread"]["pose"])}
        out = {"plate": list(sc["plate"])}
        for s in self.sides:
            out[f"bread_{s}"] = list(sc["bread"][s]["pose"])
        return out

    def set_object_pose(self, name: str, x: float, y: float, yaw_deg: float):
        if name == "plate":
            z = 0.0005
        else:
            z = self.meta["objects"][name]["extents"][2] / 2 + 0.0008
        self.place_object(name, [x, y, z] + list(yaw_quat(yaw_deg)))

    def object_role_masks(self, masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Map physical object instances to the dataset's obj1 / obj2 labels."""
        roles = self.meta.get("object_roles", {"obj1": [n for n in self.obj_names if n.startswith("bread")],
                                               "obj2": ["plate"]})
        return {role: np.bitwise_or.reduce([masks[n] for n in names]) for role, names in roles.items()}

    def place_object(self, name: str, pose7) -> None:
        """Rigidly place an object (pos + wxyz quat of its centre), zero velocity."""
        if name in self.soft:
            self.soft[name].place(self.d, pose7[:3], pose7[3:])
            return
        qa, da = self.obj_free_qadr[name], self.obj_free_dadr[name]
        self.d.qpos[qa:qa + 7] = pose7
        self.d.qvel[da:da + 6] = 0.0

    def object_pose7(self, name: str) -> List[float]:
        """pos + wxyz quat; for a soft bread the best rigid fit of its deformation grid."""
        if name in self.soft:
            return self.soft[name].pose7(self.d)
        bid = self.obj_bodies[name]
        return self.d.xpos[bid].tolist() + self.d.xquat[bid].tolist()

    def object_lowest_z(self, name: str) -> float:
        """Height of the object's lowest surface point (soft bread: lowest flex vertex)."""
        if name in self.soft:
            fid = self.soft[name].flex_id
            a, n = self.m.flex_vertadr[fid], self.m.flex_vertnum[fid]
            return float(self.d.flexvert_xpos[a:a + n, 2].min())
        ext = np.asarray(self.meta["objects"][name]["extents"])
        if self.cfg.get("task") == "pick_cube":
            return float(self.object_pos(name)[2] - np.abs(self.body_T_world(name)[2, :3]) @ ext / 2)
        return float(self.object_pos(name)[2] - ext[2] / 2)

    def object_pos(self, name: str) -> np.ndarray:
        return np.asarray(self.object_pose7(name)[:3])

    # ------------------------------------------------------------------ step
    def step(self, n: int = 1):
        for _ in range(n):
            for arm in self.arms.values():
                arm.update(self.t)
            mujoco.mj_step(self.m, self.d)
            for p in self.plastic:
                p.update(self.m, self.d)
            self.t += self.dt
            if self.obs_history is not None:
                self.obs_history.append((self.t, self.d.qpos.copy()))

    # ------------------------------------------------------------- geometry
    def body_T_world(self, name: str) -> np.ndarray:
        if name in self.soft:
            return self.soft[name].T_world(self.d)
        bid = self.m.body(name).id
        return T_from(self.d.xmat[bid].reshape(3, 3), self.d.xpos[bid])

    def body_T_cam(self, name: str) -> np.ndarray:
        return self.T_world_in_cam @ self.body_T_world(name)

    # ------------------------------------------------------------ rendering
    def _update_scene(self, camera: str, scene_option=None, data=None):
        r = self.renderer
        r.update_scene(self.d if data is None else data, camera=camera, scene_option=scene_option)
        if self.distractors:
            from .robustness import add_distractor_geoms
            add_distractor_geoms(r.scene, self.distractors)
        return r

    def render_rgb(self, camera: str = "policy_cam", hide_robot: bool = False,
                   qpos: Optional[np.ndarray] = None) -> np.ndarray:
        """BGR uint8 (OpenCV order, like CamRS). `qpos`: render that (past) state instead of the current one."""
        data = None
        if qpos is not None:
            if self._d_past is None:
                self._d_past = mujoco.MjData(self.m)
            data = self._d_past
            data.qpos[:] = qpos
            mujoco.mj_forward(self.m, data)
        r = self._update_scene(camera, self._opt_clean if hide_robot else None, data)
        return np.ascontiguousarray(r.render()[:, :, ::-1])

    def render_depth(self, noise: bool = True, artifacts=None) -> np.ndarray:
        """Metric depth [m], 0 = invalid. `artifacts(z_noisy, z_clean)`: optional sensor artefacts
        (robustness.DepthArtifacts) applied on top of the Gaussian noise."""
        r = self.renderer
        r.enable_depth_rendering()
        self._update_scene("policy_cam")
        z = r.render().astype(np.float32)
        r.disable_depth_rendering()
        if noise:
            z_clean = z
            sig = self.cam.depth_noise_a + self.cam.depth_noise_b * z ** 2
            z = z + self.rng.normal(0.0, 1.0, z.shape).astype(np.float32) * sig
            if artifacts is not None:
                z = artifacts(z, z_clean)
            z[(z < self.cam.depth_min) | (z > self.cam.depth_max)] = 0.0
        return z

    def render_segmentation(self) -> Dict[str, np.ndarray]:
        """Ground-truth object masks (uint8 0/255) — stands in for DINO-SAM."""
        r = self.renderer
        r.enable_segmentation_rendering()
        self._update_scene("policy_cam")
        seg = r.render()
        r.disable_segmentation_rendering()
        ids = seg[:, :, 0]
        types = seg[:, :, 1]
        masks = {name: np.zeros((self.H, self.W), np.uint8) for name in self.obj_names}
        for (ty, oid), name in self.seg_label.items():
            masks[name][(types == ty) & (ids == oid)] = 255
        return masks

    # ------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        """Compact state for the web recorder."""
        out = {"t": self.t, "qpos": {}, "objects": {}}
        for s, arm in self.arms.items():
            q = self.d.qpos[arm.qadr].tolist()
            out["qpos"][s] = q + [float(self.d.qpos[arm.g_qadr])]
        out["qfull"] = np.round(self.d.qpos, 5).tolist()
        for name in self.obj_names:
            out["objects"][name] = [round(v, 5) for v in self.object_pose7(name)]
        return out
