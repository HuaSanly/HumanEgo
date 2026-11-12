# -*- coding: utf-8 -*-
"""
Export simulator models and recorded rollouts for the three.js web viewer.

model JSON   kinematic tree (bodies, joints, visual geoms, materials, cameras)
             so the browser can run forward kinematics from recorded qpos;
mesh GLB     geometry of every visual mesh, taken from the *compiled* MuJoCo
             model so mesh frames match the simulator exactly;
rollout JSON qpos at 30 Hz + the policy log (predicted 50-step EE trajectory,
             grasp / done probabilities) + the result;
policy MP4   exactly the images the policy saw (one tile per arm).
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from .robots import ASSETS

JT = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
GT = {mujoco.mjtGeom.mjGEOM_PLANE: "plane", mujoco.mjtGeom.mjGEOM_BOX: "box",
      mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_MESH: "mesh",
      mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule"}

# web materials (PBR) keyed by MuJoCo material name suffix
WEB_MATERIALS = {
    "trossen_material": {"color": [1, 1, 1], "map": "assets/textures/trossen_BaseColor.png", "roughness": 0.42,
                         "metalness": 0.0, "lift": 2.4, "coat": 0.45, "envIntensity": 1.4},
    "maple": {"color": [1, 1, 1], "map": "assets/textures/maple.jpg", "roughness": 0.55, "metalness": 0.0, "repeat": [2, 2]},
    "frame_black": {"color": [0.05, 0.05, 0.055], "roughness": 0.35, "metalness": 0.6},
    "floor": {"color": [0.72, 0.72, 0.70], "roughness": 0.9, "metalness": 0.0},
    "plate_mat": {"color": [0.60, 0.69, 0.86], "roughness": 0.38, "metalness": 0.0, "clearcoat": 0.35},
}


def _r(x, n=5):
    return [round(float(v), n) for v in np.asarray(x).ravel()]


def glb_node_name(name: str) -> str:
    """three.js' GLTFLoader strips '/' (and '.', ':', '[]') from node names: keep names loader-safe."""
    for ch in "/.:[] ":
        name = name.replace(ch, "__")
    return name


def export_mesh_glb(m: mujoco.MjModel, mesh_ids: List[int], out_path: str, strip_prefix: str = ""):
    """Write one GLB whose nodes are the given MuJoCo meshes (geometry only)."""
    scene = trimesh.Scene()
    for mid in mesh_ids:
        name = m.mesh(mid).name
        if strip_prefix and name.startswith(strip_prefix):
            name = name[len(strip_prefix):]
        name = glb_node_name(name)
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        V = m.mesh_vert[va:va + vn]
        F = m.mesh_face[fa:fa + fn]
        # unroll faces so per-face texcoords / normals are kept (MuJoCo stores them per face corner)
        pos = V[F.reshape(-1)]
        na = m.mesh_normaladr[mid]
        nrm = m.mesh_normal[na + m.mesh_facenormal[fa:fa + fn].reshape(-1)]
        uv = None
        if m.mesh_texcoordnum[mid] > 0:
            ta = m.mesh_texcoordadr[mid]
            uv = m.mesh_texcoord[ta + m.mesh_facetexcoord[fa:fa + fn].reshape(-1)].copy()
        # weld identical (position, normal, uv) corners -> indexed mesh (3-4x smaller GLB)
        key = np.hstack([np.round(pos, 5), np.round(nrm, 3)] + ([np.round(uv, 4)] if uv is not None else []))
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        pos, nrm = uniq[:, :3], uniq[:, 3:6]
        uv = uniq[:, 6:8] if uv is not None else None
        faces = inv.reshape(-1, 3)
        vis = trimesh.visual.TextureVisuals(uv=uv) if uv is not None else None
        tm = trimesh.Trimesh(vertices=pos, faces=faces, vertex_normals=nrm, visual=vis, process=False)
        scene.add_geometry(tm, node_name=name, geom_name=name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scene.export(out_path)


def export_model(env, out_root: str, key: str) -> Dict:
    """Kinematic tree + visual geoms for the browser. `key` names the model, e.g. trossen_wxai_bimanual."""
    m = env.m
    robot = env.robot.name
    # ---- meshes: one GLB per robot (shared by both arms), objects use their own GLBs ----
    robot_mesh_ids, prefix = [], f"{env.sides[0]}/"
    for g in range(m.ngeom):
        if m.geom_group[g] == 1 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            if m.body(m.geom_bodyid[g]).name.startswith(prefix):
                robot_mesh_ids.append(int(m.geom_dataid[g]))
    glb = os.path.join(out_root, "assets", "robots", f"{robot}.glb")
    export_mesh_glb(m, sorted(set(robot_mesh_ids)), glb, strip_prefix=prefix)

    bodies = []
    for b in range(m.nbody):
        jnts = []
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            if j < 0:
                continue
            jnts.append({"type": JT[int(m.jnt_type[j])], "axis": _r(m.jnt_axis[j]), "pos": _r(m.jnt_pos[j]),
                         "qadr": int(m.jnt_qposadr[j])})
        bodies.append({"name": m.body(b).name, "parent": int(m.body_parentid[b]), "pos": _r(m.body_pos[b]),
                       "quat": _r(m.body_quat[b]), "joints": jnts})
    geoms = []
    for g in range(m.ngeom):
        grp = int(m.geom_group[g])
        if grp not in (0, 1, 2) or m.geom_rgba[g][3] == 0:
            continue
        bname = m.body(m.geom_bodyid[g]).name
        gt = GT.get(int(m.geom_type[g]))
        if gt is None:
            continue
        mat = m.mat(m.geom_matid[g]).name if m.geom_matid[g] >= 0 else None
        if mat and "/" in mat:
            mat = mat.split("/", 1)[1]
        entry = {"body": int(m.geom_bodyid[g]), "type": gt, "group": grp, "size": _r(m.geom_size[g]),
                 "pos": _r(m.geom_pos[g]), "quat": _r(m.geom_quat[g]), "material": mat,
                 "rgba": _r(m.geom_rgba[g], 3), "name": m.geom(g).name}
        if gt == "mesh":
            if bname.startswith("bread_"):
                kind = env.meta["objects"][bname]["kind"]
                entry.update({"asset": f"assets/objects/{kind}.glb", "pos": [0, 0, 0], "quat": [1, 0, 0, 0],
                              "yup": True})
            elif bname == "plate":
                entry.update({"asset": "assets/objects/plate.glb", "pos": [0, 0, 0], "quat": [1, 0, 0, 0],
                              "yup": True})
            else:
                mname = m.mesh(m.geom_dataid[g]).name
                for s in env.sides:
                    if mname.startswith(f"{s}/"):
                        mname = mname[len(s) + 1:]
                entry.update({"asset": f"assets/robots/{robot}.glb", "node": glb_node_name(mname)})
        geoms.append(entry)

    # textures used by the web materials
    tex_dir = os.path.join(out_root, "assets", "textures")
    os.makedirs(tex_dir, exist_ok=True)
    tsrc = os.path.join(ASSETS, "robots", "trossen_wxai", "assets", "materials", "1k_textures", "trossen_BaseColor.png")
    if os.path.exists(tsrc):
        shutil.copy(tsrc, os.path.join(tex_dir, "trossen_BaseColor.png"))

    tool_sites = {}
    for s in env.sides:
        sid = m.site(f"{s}/{env.robot.tool_site}").id
        tool_sites[s] = {"body": int(m.site_bodyid[sid]), "pos": _r(m.site_pos[sid]), "quat": _r(m.site_quat[sid])}
    # soft breads: textured GLB deformed in the browser by the same trilinear grid as the flex
    soft = {}
    for name, sb in env.soft.items():
        om = env.meta["objects"][name]
        soft[name] = {"asset": f"assets/objects/{om['kind']}.glb", "kind": om["kind"], "cells": om["soft"]["cells"],
                      "lo": _r(om["soft"]["lo"]), "hi": _r(om["soft"]["hi"]), "nodes": _full_grid(sb, [int(b) for b in sb.bodies], -1)}
    layout = frame_layout(env)
    model = {
        "key": key, "robot": robot, "robot_label": env.robot.label, "camera": env.cam.name,
        "task": env.cfg.get("task", "serve_bread"), "scene_only": bool(env.cfg.get("scene_only")),
        "camera_label": env.cam.label, "mode": env.cfg.get("mode"), "sides": env.sides, "nq": int(m.nq),
        "bodies": bodies, "geoms": geoms, "materials": WEB_MATERIALS, "tool_sites": tool_sites,
        "policy_camera": {"T_cam_in_world": _r(env.T_cam_in_world), "K": _r(env.K), "width": env.W, "height": env.H},
        "objects": {k: ({"soft": True} if k in env.soft else {"qadr": int(env.obj_free_qadr[k])}) for k in env.obj_names},
        "soft": soft, "frame_layout": layout,
        "finger_bodies": list(env.robot.finger_bodies),
    }
    p = os.path.join(out_root, "data", "models", f"{key}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    return model


def _full_grid(sb, vals, fill):
    """Per-corner values on the full grid (corners MuJoCo pinned to the world get `fill`)."""
    out, it = [], iter(vals)
    for v in sb.valid:
        out.append(next(it) if v else fill)
    return out


def frame_layout(env) -> Dict:
    """Columns of a rollout frame: qpos entries except the soft-bread grids, then one pose
    (pos + wxyz quat) per soft bread. The grids' deformation goes to a separate sparse track."""
    grid_q = set()
    for sb in env.soft.values():
        grid_q.update(int(i) for i in sb.qidx)
    keep = [i for i in range(env.m.nq) if i not in grid_q]
    return {"q_keep": keep, "soft": sorted(env.soft)}


def _soft_tracks(env, q_frames: List[np.ndarray], times: List[float], eps=3e-4, stride=2):
    """Per soft bread: rigid pose per frame + sparse deformation residuals (0.1 mm ints, bread frame)."""
    poses = {n: [] for n in env.soft}
    deform = {}
    for name, sb in env.soft.items():
        res_all = []
        for q in q_frames:
            P = sb.p0 + q[sb.qidx].reshape(-1, 3)
            Rm, t = sb.fit(P)
            x, y, z, w = R.from_matrix(Rm).as_quat()
            poses[name].append([round(float(v), 4) for v in t] + [round(float(v), 4) for v in (w, x, y, z)])
            res_all.append((P - t) @ Rm - sb.local)
        mags = np.array([np.abs(r).max() for r in res_all])
        on = mags > eps
        keep = set()
        for i in np.nonzero(on)[0]:
            if i % stride == 0 or not on[i - 1] or (i + 1 < len(on) and not on[i + 1]):
                keep.add(int(i))
            if i > 0 and not on[i - 1]:
                keep.add(int(i - 1))                  # zero frame bounding the run (clean interpolation)
            if i + 1 < len(on) and not on[i + 1]:
                keep.add(int(i + 1))
        ts, rs = [], []
        for i in sorted(keep):
            ts.append(round(times[i], 3))
            if on[i]:
                full = np.zeros((len(sb.valid), 3), int)
                full[sb.valid] = np.round(res_all[i] * 1e4).astype(int)
                rs.append(full.ravel().tolist())
            else:
                rs.append(0)
        deform[name] = {"t": ts, "r": rs}
    return poses, deform


def export_rollout(env, log_dir: str, out_root: str, rid: str, scene_meta: Optional[dict] = None) -> Dict:
    """Recorded episode -> data/rollouts/<rid>.json (+ policy-view mp4)."""
    with open(os.path.join(log_dir, "episode.json")) as f:
        ep = json.load(f)
    m, d = env.m, env.d
    frames = []
    if all("qfull" in fr for fr in ep["frames"]):
        layout = frame_layout(env)
        keep = np.asarray(layout["q_keep"])
        qs = [np.asarray(fr["qfull"], float) for fr in ep["frames"]]
        times = [fr["t"] for fr in ep["frames"]]
        poses, deform = _soft_tracks(env, qs, times)
        for k, q in enumerate(qs):
            row = [round(times[k], 3)] + [round(float(v), 3) for v in q[keep]]
            for name in layout["soft"]:
                row += poses[name][k]
            frames.append(row)
    for fr in ([] if frames else ep["frames"]):
        q = np.array(d.qpos)  # template
        for s, qs in fr["qpos"].items():
            arm = env.arms[s]
            q[arm.qadr] = qs[:6]
            for j in range(m.njnt):
                nm = m.joint(j).name
                if nm.startswith(f"{s}/") and "carriage" in nm:
                    q[m.jnt_qposadr[j]] = qs[6]
        for name, pq in fr["objects"].items():
            qa = env.obj_free_qadr[name]
            q[qa:qa + 7] = pq
        frames.append([round(fr["t"], 3)] + [round(float(v), 4) for v in q])
    pol = []
    for st in ep["policy"]:
        e = {"t": round(st["t"], 3), "sides": {}}
        for s, v in st["sides"].items():
            tr = v["traj_world"]
            idx = sorted(set(list(range(0, len(tr), 2)) + [len(tr) - 1])) if tr else []
            e["sides"][s] = {"traj": [[round(x, 3) for x in tr[i]] for i in idx], "traj_step": 2,
                             "grasp": [round(g, 2) for g in v["grasp"][:12]], "gmax": round(max(v["grasp"]), 2) if v["grasp"] else 0,
                             "done": round(v["done"], 3), "grasping": v["grasping"]}
        pol.append(e)
    # policy-view video: one tile per side, side-by-side, one frame per policy step
    vid_rel = None
    img_dir = os.path.join(log_dir, "policy_view")
    if os.path.isdir(img_dir) and ep["policy"]:
        vid_rel = f"data/rollouts/{rid}_policy.mp4"
        vp = os.path.join(out_root, vid_rel)
        os.makedirs(os.path.dirname(vp), exist_ok=True)
        sides = env.sides
        P = ep["policy"][1]["t"] - ep["policy"][0]["t"] if len(ep["policy"]) > 1 else 0.3
        w = imageio.get_writer(vp, fps=round(1.0 / P, 4), codec="libx264", macro_block_size=8,
                               ffmpeg_params=["-g", "8", "-crf", "28", "-preset", "slow", "-movflags", "+faststart"])
        TW, TH = 256, 192
        last = {s: np.zeros((TH, TW, 3), np.uint8) for s in sides}
        for st in ep["policy"]:
            for s in sides:
                e = st["sides"].get(s)
                if e and e.get("image"):
                    im = cv2.imread(os.path.join(img_dir, e["image"]))
                    if im is not None:
                        last[s] = cv2.resize(im, (TW, TH), interpolation=cv2.INTER_AREA)
            tile = np.hstack([last[s] for s in sides])
            w.append_data(tile[:, :, ::-1])
        w.close()
    meta = ep["meta"]
    out = {
        "id": rid, "model": None, "fps": 30, "frames": frames, "policy": pol, "policy_video": vid_rel,
        "deform": deform if env.soft and frames and "qfull" in ep["frames"][0] else {},
        "policy_period": round(P, 4) if ep["policy"] else 0.3,
        "result": meta["result"], "scene": meta["scene_init"], "extra": scene_meta or {},
    }
    p = os.path.join(out_root, "data", "rollouts", f"{rid}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return out
