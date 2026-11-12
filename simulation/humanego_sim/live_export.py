# -*- coding: utf-8 -*-
"""
Physics-only MJCF for the in-browser "Live" mode (MuJoCo WebAssembly).

The browser already renders the high-quality visual assets with three.js, so
MuJoCo there only needs collision geometry: visual geoms, textures and
materials are stripped and the remaining (collision) meshes are flattened into
one folder next to the XML. Body / joint / actuator / site names are unchanged,
so the same model JSON drives rendering from either recorded or live state.
"""
from __future__ import annotations

import json
import os
import shutil

import mujoco

from .robots import ASSETS


def _object_entry(env, n: str) -> dict:
    om = env.meta["objects"][n]
    half_h = float(om["extents"][2] / 2) if n != "plate" else 0.0
    if n in env.soft:            # soft bread: grid corners on world-aligned slide joints
        sb = env.soft[n]
        return {"soft": True, "half_height": half_h, "flex": int(sb.flex_id), "qidx": [int(i) for i in sb.qidx],
                "nodes": [int(b) for b in sb.bodies], "local": [round(float(v), 6) for v in sb.local.ravel()],
                "p0": [round(float(v), 6) for v in sb.p0.ravel()]}
    out = {"qadr": int(env.obj_free_qadr[n]), "body": int(env.obj_bodies[n]), "half_height": half_h}
    if n == "plate":
        out["R"] = om["R"]
    return out


def export_live(env, out_root: str, key: str) -> str:
    spec = mujoco.MjSpec.from_file(env.xml_path)
    spec.modelfiledir = ASSETS
    # 1) visual-only geoms
    for g in list(spec.geoms):
        if g.contype == 0 and g.conaffinity == 0:
            spec.delete(g)
    for g in spec.geoms:
        g.material = ""
    for sk in list(spec.skins):                  # soft-bread skins are visual only (three.js deforms the GLB)
        spec.delete(sk)
    for m in list(spec.materials):
        spec.delete(m)
    for t in list(spec.textures):
        spec.delete(t)
    for c in list(spec.cameras):
        if c.name != "policy_cam":
            spec.delete(c)
    # 2) keep only meshes still referenced by collision geoms, flatten their paths
    used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    out_dir = os.path.join(out_root, "assets", "live", key)
    mesh_dir = os.path.join(out_dir, "meshes")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(mesh_dir)
    files = []
    for me in list(spec.meshes):
        if me.name not in used:
            spec.delete(me)
            continue
        src = os.path.join(ASSETS, me.file)
        flat = me.name.replace("/", "__") + os.path.splitext(me.file)[1]
        shutil.copy(src, os.path.join(mesh_dir, flat))
        me.file = flat
        files.append(f"meshes/{flat}")
    spec.meshdir = "meshes"
    spec.modelfiledir = out_dir
    spec.compile()                               # validate
    xml = spec.to_xml()
    with open(os.path.join(out_dir, "scene.xml"), "w") as f:
        f.write(xml)
    with open(os.path.join(out_dir, "files.json"), "w") as f:
        json.dump({"xml": "scene.xml", "files": files}, f)
    # everything the browser runner needs besides the MJCF
    r = env.robot
    flat = lambda T: [round(float(v), 7) for v in T.ravel()]
    spec_json = {
        "key": key, "sides": env.sides, "mode": env.cfg.get("mode"),
        "task": env.cfg.get("task", "serve_bread"), "scene_only": bool(env.cfg.get("scene_only")),
        "object_roles": env.meta.get("object_roles"),
        "robot": {"name": r.name, "arm_joints": r.arm_joints, "arm_actuators": r.arm_actuators,
                  "gripper_actuator": r.gripper_actuator, "gripper_open": r.gripper_open, "gripper_closed": r.gripper_closed,
                  "tool_site": r.tool_site, "R_tool_to_ee": flat(r.R_tool_to_ee),
                  "initial_joints": {s: list(env.initial_joints(s)) for s in env.sides}, "home_joints": r.home_joints,
                  "gripper_joint": r.gripper_joint, "gripper_joint_open": r.gripper_joint_open,
                  "gripper_joint_closed": r.gripper_joint_closed, "gripper_reset_qpos": r.gripper_reset_qpos,
                  "finger_bodies": r.finger_bodies, "grip_sense": r.grip_sense, "finger_gap_closed": r.finger_gap_closed},
        "T_base_in_world": {s: flat(env.arms[s].T_base_in_world) for s in env.sides},
        "T_base_in_cam": {s: flat(env.arms[s].T_base_in_cam) for s in env.sides},
        "T_cam_in_world": flat(env.T_cam_in_world),
        "camera": {"name": env.cam.name, "K": flat(env.K), "width": env.W, "height": env.H,
                   "depth_noise": [env.cam.depth_noise_a, env.cam.depth_noise_b, env.cam.depth_min]},
        "control": env.cfg["control"], "perception": env.cfg.get("perception", {}),
        "objects": {n: _object_entry(env, n) for n in env.obj_names},
        "default_scene": env.default_scene(),
    }
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump(spec_json, f)
    size = sum(os.path.getsize(os.path.join(mesh_dir, x)) for x in os.listdir(mesh_dir))
    print(f"[live] {key}: {len(files)} collision meshes, {size / 1e6:.2f} MB")
    return out_dir
