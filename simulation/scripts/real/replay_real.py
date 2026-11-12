# -*- coding: utf-8 -*-
"""
Replay a real-robot rollout log in the simulator and compare the two, one to one.

The log comes from the AriaMimic branch `sim-logging` (inference/SimLogger.py, format
"humanego-real-log/1"; see simulation/docs/real_vs_sim.md). `sim_as_real.py` writes the same
format from a simulated episode, which is how this script is tested.

    python simulation/scripts/real/replay_real.py <log_dir> --modes scene kinematic closed policy
    python simulation/scripts/real/replay_real.py <log_dir> --modes closed --params simulation/configs/tuned/trossen_wxai.json \\
        --set 'control.grasp_assist={"enabled": true}'

Modes
  scene      rebuild the initial scene (plate + breads) from the robot's own perception at t = 0,
             fit the table plane in the logged depth (camera height / pitch vs the sim rig), run the
             sim perception on the rebuilt scene and compare centres, long axes and extents;
             writes scene_compare.jpg (real frame | sim frame of the same scene)
  kinematic  drive the sim arms through the recorded joint trajectories and gripper commands:
             forward-kinematics residual against the logged EE poses (model check), and whether
             the bread ends on the plate as it did on the real robot (contact / bread physics check)
  closed     run the released policy closed-loop in sim from the rebuilt scene (production or any
             --params/--set) and compare phase timings, grasp point, release height and EE paths
  policy     re-run the policy offline on the logged observations (policy image, EE pose, grasp
             flag, object poses) and compare with the logged plans (numerics, checkpoint, noise);
             for the steps before the first grasp also on the sim rendering of the same state, which
             measures the visual sim-to-real gap in plan space (cm)

Writes <out>/report.json (default: <log_dir>/sim_replay/).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must precede `import mujoco`
import glob
import json
import sys
from typing import Dict, List, Optional

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(SIM)
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, "scripts"))
sys.path.insert(0, HERE)

from humanego_sim.env import deep_update, load_config  # noqa: E402
from humanego_sim.perception import mirror_pose  # noqa: E402

FORMAT = "humanego-real-log/1"
PLATE_FLOOR = 0.0085
S_MIRROR = np.diag([-1.0, 1.0, 1.0])


# =============================================================================== loading
def read_depth_png(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-4


def _jsonl(p):
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass   # a rollout killed mid-write leaves a truncated last line
    return out


def load_log(d: str) -> dict:
    meta = json.load(open(os.path.join(d, "meta.json")))
    if meta.get("format") != FORMAT:
        print(f"[replay_real] warning: format {meta.get('format')} != {FORMAT}")
    inits = []
    for p in sorted(glob.glob(os.path.join(d, "init_*")), key=lambda s: int(s.rsplit("_", 1)[1])):
        o = json.load(open(os.path.join(p, "objects.json")))
        o["dir"] = p
        inits.append(o)
    st = {}
    if os.path.exists(os.path.join(d, "state.npz")):
        z = np.load(os.path.join(d, "state.npz"))
        st = {k: z[k] for k in z.files}
    outcome = json.load(open(os.path.join(d, "outcome.json"))) if os.path.exists(os.path.join(d, "outcome.json")) else {}
    return {"dir": d, "meta": meta, "inits": inits, "steps": _jsonl(os.path.join(d, "steps.jsonl")),
            "commands": _jsonl(os.path.join(d, "commands.jsonl")), "events": _jsonl(os.path.join(d, "events.jsonl")),
            "state": st, "outcome": outcome}


def sides_of(log) -> List[str]:
    s = list(log["meta"].get("arms", {}).keys())
    return [x for x in ("right", "left") if x in s] or ["right"]


# ============================================================================ geometry
def _p_to_T(p) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(np.asarray(p[3:6], dtype=np.float64)).as_matrix()
    T[:3, 3] = p[:3]
    return T


def perceived_points_world(init: dict, side: str, key: str, T_cam_in_world: np.ndarray) -> Optional[np.ndarray]:
    """Keypoints of `key` perceived for `side`, in the sim world frame (left side un-mirrored)."""
    o = init["objects"].get(side, {}).get(key)
    if not o:
        return None
    k3d = np.asarray(o.get("kpts_3d") or [], dtype=np.float64).reshape(-1, 3)
    if len(k3d) == 0:
        T, kl = np.asarray(o["T_obj_in_cam_static"]), np.asarray(o["kpts_local"]).reshape(-1, 3)
        k3d = (T[:3, :3] @ kl.T).T + T[:3, 3]
    if init["objects"][side].get("_is_flip", side == "left"):
        k3d = k3d * np.array([-1.0, 1.0, 1.0])
    return (T_cam_in_world[:3, :3] @ k3d.T).T + T_cam_in_world[:3, 3]


def footprint(pts: np.ndarray) -> dict:
    """Centre, long axis (yaw in [0, 180)), length / width / top height of a perceived object."""
    c = pts.mean(0)
    xy = pts[:, :2] - c[:2]
    ev, evec = np.linalg.eigh(xy.T @ xy)
    a = evec[:, np.argmax(ev)]
    w = np.array([-a[1], a[0]])
    yaw = float(np.degrees(np.arctan2(a[1], a[0]))) % 180.0
    return {"c": c.tolist(), "yaw_deg": yaw, "len": float(np.ptp(xy @ a)), "wid": float(np.ptp(xy @ w)),
            "top_z": float(np.sort(pts[:, 2])[-3:].mean()), "n": int(len(pts))}


def yaw_diff(a, b):
    d = (a - b) % 180.0
    return min(d, 180.0 - d)


def fit_table_plane(depth: np.ndarray, K: np.ndarray, masks: List[np.ndarray], iters: int = 300, tol: float = 0.004,
                    seed: int = 0) -> Optional[dict]:
    """RANSAC plane through the depth pixels outside the object masks (camera frame).
    Returns camera height above the plane and the angle of the optical axis below the horizon."""
    H, W = depth.shape
    ok = depth > 0.05
    for m in masks:
        if m is not None and m.shape == depth.shape:
            ok &= cv2.dilate((m > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
    ok[: H // 6] = False   # far background (wall) at the top of the image
    v, u = np.nonzero(ok)
    if len(u) < 500:
        return None
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(u), min(len(u), 40000), replace=False)
    u, v = u[sel], v[sel]
    z = depth[v, u]
    P = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    best, best_n = None, 0
    for _ in range(iters):
        i = rng.choice(len(P), 3, replace=False)
        n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        dist = np.abs((P - P[i[0]]) @ n)
        c = int((dist < tol).sum())
        if c > best_n:
            best_n, best = c, (n, P[i[0]])
    n, p0 = best
    inl = np.abs((P - p0) @ n) < tol
    Q = P[inl]
    cq = Q.mean(0)
    n = np.linalg.svd(Q - cq, full_matrices=False)[2][-1]
    if n @ cq > 0:          # normal pointing from the table towards the camera
        n = -n
    h = float(-(n @ cq))
    pitch = float(np.degrees(np.arcsin(np.clip(-n[2], -1, 1))))   # optical axis (cam +z) below the table plane
    return {"cam_height_m": round(h, 4), "optical_axis_down_deg": round(pitch, 2),
            "normal_cam": np.round(n, 4).tolist(), "inlier_frac": round(float(inl.mean()), 3), "n": int(len(P))}


# ================================================================================ scene
def reconstruct_scene(log: dict, cfg: dict, T_cam_in_world: np.ndarray, segment: int = 0) -> dict:
    init = log["inits"][segment]
    sides = sides_of(log)
    scene, info = {}, {}
    plate_pts = perceived_points_world(init, "right" if "right" in sides else sides[0], "obj2", T_cam_in_world)
    if plate_pts is not None and len(plate_pts) >= 3:
        fp = footprint(plate_pts)
        # rim keypoints: the centre is the mean, the diameter ~ the extent across
        scene["plate"] = [round(fp["c"][0], 4), round(fp["c"][1], 4), 0.0]
        info["plate"] = {"diam_cm": round(100 * 0.5 * (fp["len"] + fp["wid"]), 1), "rim_z_cm": round(100 * fp["c"][2], 2)}
    for s in sides:
        b = perceived_points_world(init, s, "obj1", T_cam_in_world)
        if b is None or len(b) < 3:
            continue
        fp = footprint(b)
        scene[f"bread_{s}"] = [round(fp["c"][0], 4), round(fp["c"][1], 4), round(fp["yaw_deg"], 1)]
        info[f"bread_{s}"] = {"len_cm": round(100 * fp["len"], 1), "wid_cm": round(100 * fp["wid"], 1),
                              "top_cm": round(100 * fp["top_z"], 1), "n_kpts": fp["n"]}
    return {"scene": scene, "perceived": info}


def bread_types(log: dict, sides) -> Dict[str, str]:
    """Sim bread type per side from the operator label (croissant / danish), else the real rig default."""
    oc = log.get("outcome", {})
    out = {}
    for s in sides:
        lab = str(oc.get(f"bread_{s}") or "").lower()
        if "croiss" in lab:
            out[s] = "croissant"
        elif lab:
            out[s] = "bun"          # sim name of the sugar danish loaf
        else:
            out[s] = "croissant" if s == "right" else "bun"
    return out


def make_cfg(log: dict, base_over: dict) -> dict:
    sides = sides_of(log)
    ov = {"mode": "bimanual" if len(sides) > 1 else "single", "robot": "trossen_wxai", "camera": "realsense_d405"}
    tb = {s: log["meta"]["arms"][s]["T_base_in_cam"] for s in sides if log["meta"]["arms"].get(s, {}).get("T_base_in_cam")}
    if tb:
        ov["rig"] = {"T_base_in_cam": tb}
    types = bread_types(log, sides)
    if len(sides) > 1:
        ov["scene"] = {"bread": {s: {"type": types[s]} for s in sides}}
    else:
        ov["scene"] = {"single": {"bread": {"type": types[sides[0]]}}}
    return load_config(None, deep_update(ov, base_over))


def sim_perceive(env, cfg, scene) -> dict:
    """Run the simulator's perception (the released modules) on `scene`; same layout as init_<k>/objects.json."""
    from humanego_sim.perception import SimPerception
    env.reset(scene, settle=0.4)
    sp = SimPerception(env, split_x_ratio=float(cfg["control"].get("split_x_ratio", 0.5)),
                       n_init_frames=int(cfg.get("perception", {}).get("n_init_frames", 10)), depth_noise=True)
    so = sp.estimate_objects(env.sides, shared_context=True)
    return {"objects": {s: {k: {"kpts_3d": np.asarray(m["kpts_3d"]).tolist(),
                                "T_obj_in_cam_static": np.asarray(m["T_obj_in_cam_static"]).tolist(),
                                "kpts_local": np.asarray(m["kpts_local"]).tolist()}
                            for k, m in so[s].objs_meta.items()} | {"_is_flip": s == "left"} for s in env.sides}}


def _object_keys(sides):
    """(scene name, perceiving side, perception key) for every object."""
    out = [("plate", "right" if "right" in sides else sides[0], "obj2")]
    return out + [(f"bread_{s}", s, "obj1") for s in sides]


def refine_scene(log, cfg, env, scene0: dict, iters: int = 2):
    """Move the rebuilt objects until the simulator's perception of the rebuilt scene matches the
    logged real perception (fixed point). Removes the bias of the visible-surface keypoints
    (their centroid sits 1-1.6 cm towards the camera) without modelling it."""
    init = log["inits"][0]
    Tcw = env.T_cam_in_world
    scene = {k: list(v) for k, v in scene0.items()}
    hist, sim_init = [], None
    for it in range(iters + 1):
        sim_init = sim_perceive(env, cfg, scene)
        worst, upd = 0.0, {}
        for nm, s, key in _object_keys(env.sides):
            pr = perceived_points_world(init, s, key, Tcw)
            ps = perceived_points_world(sim_init, s, key, Tcw)
            if nm not in scene or pr is None or ps is None or len(pr) < 3 or len(ps) < 3:
                continue
            fr, fs = footprint(pr), footprint(ps)
            d = np.array(fr["c"][:2]) - np.array(fs["c"][:2])
            dyaw = ((fr["yaw_deg"] - fs["yaw_deg"] + 90.0) % 180.0) - 90.0
            upd[nm] = (d, dyaw)
            worst = max(worst, float(np.linalg.norm(d)))
        hist.append(round(100 * worst, 2))
        if it == iters:
            break
        for nm, (d, dyaw) in upd.items():
            scene[nm][0] = round(scene[nm][0] + float(d[0]), 4)
            scene[nm][1] = round(scene[nm][1] + float(d[1]), 4)
            if nm != "plate":
                scene[nm][2] = round((scene[nm][2] + dyaw) % 180.0, 1)
    return scene, sim_init, hist


def mode_scene(log, cfg, out_dir, report, env, rec, scene, sim_init):
    Tcw = env.T_cam_in_world
    r = {"scene": scene, "first_guess": rec["scene"], "real_perceived": rec["perceived"],
         "refine_residual_cm": report.get("refine_residual_cm")}
    init = log["inits"][0]
    K_real = np.asarray(log["meta"]["camera"]["K"])
    r["K_real"] = K_real.round(2).tolist()
    r["K_sim"] = env.K.round(2).tolist()
    r["fx_ratio_real_over_sim"] = round(float(K_real[0, 0] / env.K[0, 0]), 4)
    # table plane from the logged depth
    dp = os.path.join(init["dir"], "depth_0.png")
    if os.path.exists(dp):
        depth = read_depth_png(dp)
        masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in glob.glob(os.path.join(init["dir"], "mask_*.png"))]
        tp = fit_table_plane(depth, K_real, masks)
        cam_z = float(Tcw[2, 3])
        n_sim = Tcw[:3, :3].T @ np.array([0, 0, 1.0])      # world up in the sim camera frame
        r["table_plane_real"] = tp
        r["table_plane_sim"] = {"cam_height_m": round(cam_z, 4),
                                "optical_axis_down_deg": round(float(np.degrees(np.arcsin(np.clip(-n_sim[2], -1, 1)))), 2)}
    env.reset(scene, settle=0.4)
    cmp = {}
    for s in env.sides:
        for key, nm in (("obj1", f"bread_{s}"), ("obj2", "plate")):
            pr = perceived_points_world(init, s, key, Tcw)
            ps = perceived_points_world(sim_init, s, key, Tcw)
            if pr is None or ps is None or len(pr) < 3 or len(ps) < 3:
                continue
            fr, fs = footprint(pr), footprint(ps)
            gt = env.body_T_world(nm)
            cmp[f"{s}:{nm}"] = {
                "centre_diff_cm": round(100 * float(np.linalg.norm(np.array(fr["c"][:2]) - np.array(fs["c"][:2]))), 2),
                "top_diff_cm": round(100 * (fr["top_z"] - fs["top_z"]), 2),
                "yaw_diff_deg": round(yaw_diff(fr["yaw_deg"], fs["yaw_deg"]), 1) if nm != "plate" else None,
                "len_real_cm": round(100 * fr["len"], 1), "len_sim_cm": round(100 * fs["len"], 1),
                "wid_real_cm": round(100 * fr["wid"], 1), "wid_sim_cm": round(100 * fs["wid"], 1),
                "n_kpts_real": fr["n"], "n_kpts_sim": fs["n"],
                "sim_perceived_minus_sim_true_cm": round(100 * float(np.linalg.norm(np.array(fs["c"][:2]) - gt[:2, 3])), 2),
            }
    r["perception_compare"] = cmp
    # truth available when the log came from sim_as_real.py
    gt = log["meta"].get("sim_truth", {}).get("scene_init")
    if gt:
        for tag, sc in (("first_guess", rec["scene"]), ("refined", scene)):
            err = {}
            for k, v in gt.items():
                if k in sc:
                    e = {"xy_cm": round(100 * float(np.hypot(sc[k][0] - v[0], sc[k][1] - v[1])), 2)}
                    if k != "plate":
                        e["yaw_deg"] = round(yaw_diff(sc[k][2], v[2]), 1)
                    err[k] = e
            r[f"reconstruction_error_vs_truth_{tag}"] = err
    # side-by-side image
    rgb_real = cv2.imread(os.path.join(init["dir"], "rgb.png"))
    rgb_sim = env.render_rgb()
    if rgb_real is not None:
        h = 360
        a = cv2.resize(rgb_real, (int(rgb_real.shape[1] * h / rgb_real.shape[0]), h))
        b = cv2.resize(rgb_sim, (int(rgb_sim.shape[1] * h / rgb_sim.shape[0]), h))
        both = np.concatenate([a, b], 1)
        cv2.putText(both, "real (logged)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(both, "sim (rebuilt scene)", (a.shape[1] + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out_dir, "scene_compare.jpg"), both)
        diff = cv2.absdiff(cv2.resize(rgb_real, (rgb_sim.shape[1], rgb_sim.shape[0])), rgb_sim)
        r["image_mean_abs_diff"] = round(float(diff.mean()), 2)
    report["scene"] = r
    return scene


# ============================================================================ real phases
def _speed(t, p):
    if len(t) < 3:
        return np.zeros(len(t))
    k = 5
    w = np.ones(k) / k
    ps = np.stack([np.convolve(np.pad(p[:, j], (k // 2, k // 2), mode="edge"), w, mode="valid") for j in range(3)], 1)
    return np.linalg.norm(np.gradient(ps, t, axis=0), axis=1), ps


def real_phases(log: dict, T_base_in_world: Dict[str, np.ndarray], bread_world: Dict[str, dict]) -> Dict[str, dict]:
    """The same quantities as sim_phases.py, from the logged state / commands / events."""
    st, ev, cmds = log["state"], log["events"], log["commands"]
    out = {}
    g_max = {s: float(log["meta"]["arms"][s].get("gripper_max") or 0.044) for s in sides_of(log)}
    for s in sides_of(log):
        if f"{s}_p_base" not in st:
            continue
        t = st["t"]
        m = np.isfinite(st[f"{s}_p_base"][:, 0])
        t, pb = t[m], st[f"{s}_p_base"][m]
        Tbw = T_base_in_world[s]
        pw = (Tbw[:3, :3] @ pb[:, :3].T).T + Tbw[:3, 3]
        speed, pws = _speed(t, pw)
        act = [e["t"] for e in ev if e.get("kind") == "activate" and e.get("side") == s]
        t_active = act[0] if act else 0.0
        r = {"side": s, "t_active": t_active}
        im = np.where((speed > 0.03) & (t >= t_active))[0]
        r["t_move"] = float(t[im[0]]) if len(im) else None
        cl = [c["t"] for c in cmds if c.get("side") == s and c.get("grip_closed")]
        r["t_close"] = cl[0] if cl else None
        changes = 0
        prev = False
        for c in cmds:
            if c.get("side") != s:
                continue
            if c.get("grip_closed") and not prev:
                changes += 1
            prev = bool(c.get("grip_closed"))
        r["n_close"] = changes
        g = st.get(f"{s}_g_pos")
        t_open = None
        if r["t_close"] is not None and g is not None:
            # release = the gripper reopens (> 60 % open) after it was measured closed (< 50 %);
            # a soft bread can keep the fingers above 50 % for a while after the close command
            gg = g[m] / g_max[s]
            ic = np.where((t >= r["t_close"]) & (gg < 0.5))[0]
            if len(ic):
                io = np.where((t > t[ic[0]]) & (gg > 0.6))[0]
                t_open = float(t[io[0]]) if len(io) else None
        dn = [e["t"] for e in ev if e.get("kind") == "done" and e.get("side") == s]
        r["t_done"] = dn[0] if dn else None
        r["done_reason"] = next((e.get("reason") for e in ev if e.get("kind") == "done" and e.get("side") == s), None)
        r["t_open"] = t_open if t_open is not None else r["t_done"]
        # lift proxy: the tool point 2 cm above its height when the fingers have closed
        t_lift = None
        if r["t_close"] is not None:
            k0 = int(np.argmin(np.abs(t - (r["t_close"] + 0.5))))
            il = np.where((t > t[k0]) & (pws[:, 2] > pws[k0, 2] + 0.02))[0]
            t_lift = float(t[il[0]]) if len(il) else None
        r["t_lift_ee"] = t_lift
        hm = [e["t"] for e in ev if e.get("kind") == "wrap_home" and e.get("side") == s]
        t_rest = hm[0] if hm else None
        if t_rest is None and r["t_open"] is not None:
            for i in np.where(t > r["t_open"] + 0.8)[0]:
                j = np.searchsorted(t, t[i] + 0.3)
                if j < len(t) and (speed[i:j] < 0.01).all():
                    t_rest = float(t[i])
                    break
        r["t_rest"] = t_rest
        ph = {}
        if r["t_move"] is not None and r["t_close"] is not None:
            ph["reach"] = r["t_close"] - r["t_move"]
        if t_lift is not None and r["t_close"] is not None:
            ph["grasp"] = t_lift - r["t_close"]
        if r["t_open"] is not None and t_lift is not None:
            ph["transport"] = r["t_open"] - t_lift
        if t_rest is not None and r["t_open"] is not None:
            ph["home"] = t_rest - r["t_open"]
        if t_rest is not None and r["t_move"] is not None:
            ph["total"] = t_rest - r["t_move"]
        r["phases"] = {k: round(v, 2) for k, v in ph.items()}
        bw = bread_world.get(s)
        if bw is not None and r["t_close"] is not None:
            k = int(np.argmin(np.abs(t - (r["t_close"] + 0.5))))
            c, a = np.array(bw["c"]), np.array(bw["a"])
            rel = pw[k] - c
            along = float(rel @ a)
            r["grasp_vs_perceived"] = {"along_cm": round(100 * along, 1),
                                       "along_frac": round(abs(along) / max(bw["half_len"], 1e-3), 2),
                                       "lateral_cm": round(100 * float(rel @ np.array([-a[1], a[0], 0])), 1),
                                       "above_centre_cm": round(100 * float(rel[2]), 1)}
        if r["t_open"] is not None:
            k = int(np.argmin(np.abs(t - r["t_open"])))
            r["release"] = {"tcp_above_table_cm": round(100 * float(pw[k, 2]), 1)}
        r["tcp_world"] = {"t": t, "p": pws}
        out[s] = r
    return out


def perceived_bread_world(init: dict, side: str, Tcw: np.ndarray) -> Optional[dict]:
    b = perceived_points_world(init, side, "obj1", Tcw)
    if b is None or len(b) < 3:
        return None
    c = b.mean(0)
    c[2] = 0.5 * float(np.sort(b[:, 2])[-3:].mean())
    xy = b[:, :2] - c[:2]
    ev, evec = np.linalg.eigh(xy.T @ xy)
    a2 = evec[:, np.argmax(ev)]
    return {"c": c.tolist(), "a": [a2[0], a2[1], 0.0], "half_len": float(np.abs(xy @ a2).max())}


def sim_tcp_trace(env, ep, side):
    frames = [f for f in ep["frames"] if f["t"] >= 0]
    t = np.array([f["t"] for f in frames])
    arm = env.arms[side]
    p = np.zeros((len(frames), 3))
    g = np.zeros(len(frames))
    for i, f in enumerate(frames):
        env.d.qpos[:] = f["qfull"]
        mujoco.mj_kinematics(env.m, env.d)
        p[i] = (arm.T_base_in_world @ arm.get_T_ee_in_base())[:3, 3]
        g[i] = arm.get_gripper_q()
    return t, p, g


def dtw_cm(a: np.ndarray, b: np.ndarray) -> float:
    """Mean distance (cm) along the dynamic-time-warping alignment of two 3-D paths."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    D = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    C = np.full((n + 1, m + 1), np.inf)
    C[0, 0] = 0.0
    L = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            k = np.argmin([C[i - 1, j], C[i, j - 1], C[i - 1, j - 1]])
            prev = [(i - 1, j), (i, j - 1), (i - 1, j - 1)][k]
            C[i, j] = D[i - 1, j - 1] + C[prev]
            L[i, j] = L[prev] + 1
    return float(100 * C[n, m] / L[n, m])


def _resample(t, p, hz=5.0):
    if len(t) < 2:
        return p
    tt = np.arange(t[0], t[-1], 1.0 / hz)
    return np.stack([np.interp(tt, t, p[:, j]) for j in range(p.shape[1])], 1)


def compare_phases(real: dict, sim: dict) -> dict:
    keys = ["reach", "grasp", "transport", "home", "total"]
    out = {}
    for k in keys:
        a, b = real.get("phases", {}).get(k), sim.get("phases", {}).get(k)
        out[k] = {"real": a, "sim": b, "sim_minus_real": None if a is None or b is None else round(b - a, 2)}
    return out


# ============================================================================== kinematic
class TraceInterp:
    """Stands in for the driver's joint interpolator: the recorded joint trace, linearly interpolated."""

    def __init__(self, t, q):
        self.t, self.q_tr = np.asarray(t), np.asarray(q)
        self.active = True
        self.q = self.q_tr[0].copy()

    def update(self, t):
        self.q = np.array([np.interp(t, self.t, self.q_tr[:, j]) for j in range(self.q_tr.shape[1])])
        return self.q


def served(env, side) -> dict:
    """Same rule as EpisodeRunner.evaluate."""
    Tp = env.body_T_world("plate")
    R = env.meta["objects"]["plate"]["R"]
    Tb = env.body_T_world(f"bread_{side}")
    dxy = float(np.linalg.norm(Tb[:2, 3] - Tp[:2, 3]))
    low = env.object_lowest_z(f"bread_{side}")
    on_plate = dxy < R and low > 0.003 and Tb[2, 3] < 0.12
    return {"served": bool(on_plate and not env.arms[side].gripper_cmd_closed()), "dist_to_plate_center_cm": round(100 * dxy, 1),
            "bread_z_cm": round(100 * float(Tb[2, 3]), 1)}


def mode_kinematic(log, cfg, scene, out_dir, report, record_hz=30.0):
    from humanego_sim.env import ServeBreadSim
    st = log["state"]
    if not st or "t" not in st:
        report["kinematic"] = {"error": "no state.npz in the log"}
        return
    env = ServeBreadSim(cfg, render=False)
    sides = env.sides
    t_tr = st["t"]
    traces = {}
    for s in sides:
        q = st.get(f"{s}_q")
        if q is None:
            continue
        m = np.isfinite(q[:, 0])
        traces[s] = (t_tr[m], q[m])
    t0 = max(0.0, min(tr[0][0] for tr in traces.values()))
    arm_q0 = {s: list(np.interp(t0, tr[0], tr[1][:, j]) for j in range(6)) for s, tr in traces.items()}
    env.reset(scene, settle=0.4, arm_q=arm_q0)
    env.t = t0
    for s, (tt, qq) in traces.items():
        arm = env.arms[s]
        arm.t = t0
        arm._g_t0 = t0
        arm.jinterp = TraceInterp(tt, qq[:, :6])
        arm.mode = "joint"
    # gripper command timeline: the controller's commands, then the wrap-up opening
    gcmd = {s: [] for s in sides}
    for c in log["commands"]:
        if c.get("side") in gcmd:
            gcmd[c["side"]].append((c["t"], bool(c.get("grip_closed"))))
    for e in log["events"]:
        if e.get("kind") in ("wrap_halt", "done") and e.get("side") in gcmd:
            gcmd[e["side"]].append((e["t"], False))
    for s in gcmd:
        gcmd[s].sort()
    if not any(gcmd.values()):       # no commands logged: threshold the measured opening
        for s in sides:
            g = st.get(f"{s}_g_pos")
            gm = float(log["meta"]["arms"][s].get("gripper_max") or 0.044)
            if g is not None:
                gcmd[s] = [(float(a), bool(b < 0.5 * gm)) for a, b in zip(t_tr, g) if np.isfinite(b)]
    gi = {s: 0 for s in sides}
    state = {s: False for s in sides}
    cc = cfg["control"]
    t_end = float(t_tr[-1]) + 1.0
    frames, nxt = [], t0
    fk_err = {s: [] for s in sides}
    while env.t < t_end:
        for s in sides:
            ev = gcmd[s]
            while gi[s] < len(ev) and ev[gi[s]][0] <= env.t:
                want = ev[gi[s]][1]
                if want != state[s]:
                    if want:
                        env.arms[s].close_gripper(goal_time=float(cc.get("close_gripper_goal_time", 0.5)))
                    else:
                        env.arms[s].open_gripper(goal_time=float(cc.get("open_gripper_goal_time", 0.5)))
                    state[s] = want
                gi[s] += 1
        if env.t >= nxt - 1e-9:
            snap = env.snapshot()
            snap["grip_cmd"] = {s: float(env.arms[s].g_cmd) for s in sides}
            frames.append(snap)
            nxt += 1.0 / record_hz
            for s in sides:
                pb = st.get(f"{s}_p_base")
                if pb is None:
                    continue
                k = int(np.argmin(np.abs(t_tr - env.t)))
                if np.isfinite(pb[k, 0]):
                    fk_err[s].append(float(np.linalg.norm(env.arms[s].get_T_ee_in_base()[:3, 3] - pb[k, :3])))
        env.step()
    res = {s: served(env, s) for s in sides}
    real_oc = log.get("outcome", {}).get("per_side", {})
    for s in sides:
        res[s]["real_outcome"] = real_oc.get(s)
        if real_oc.get(s):
            res[s]["agrees"] = res[s]["served"] == (real_oc[s] == "served")
        if fk_err[s]:
            res[s]["fk_residual_cm"] = {"median": round(100 * float(np.median(fk_err[s])), 2),
                                        "max": round(100 * float(np.max(fk_err[s])), 2)}
    report["kinematic"] = res
    ep = {"meta": dict(env.meta, scene_init=scene, result={"success": all(v["served"] for v in res.values()),
                                                            "per_side": {s: {"success": v["served"]} for s, v in res.items()},
                                                            "duration": env.t}),
          "frames": frames, "policy": []}
    os.makedirs(os.path.join(out_dir, "kinematic"), exist_ok=True)
    json.dump(ep, open(os.path.join(out_dir, "kinematic", "episode.json"), "w"))


# ================================================================================= closed
def mode_closed(log, cfg, scene, out_dir, report, seed=0):
    from humanego_sim.runner import EpisodeRunner
    from sim_phases import analyse_side
    runner = EpisodeRunner(cfg)
    ld = os.path.join(out_dir, "closed")
    res = runner.run(scene=scene, seed=seed, log_dir=ld, intro=False, verbose=False)
    ep = json.load(open(os.path.join(ld, "episode.json")))
    env = runner.env
    Tcw = env.T_cam_in_world
    init = log["inits"][0]
    bw = {s: perceived_bread_world(init, s, Tcw) for s in env.sides}
    Tbw = {s: env.arms[s].T_base_in_world for s in env.sides}
    real = real_phases(log, Tbw, bw)
    out = {"sim_result": {s: res.per_side[s]["success"] for s in env.sides},
           "real_outcome": log.get("outcome", {}).get("per_side", {}), "per_side": {}}
    for s in env.sides:
        sp = analyse_side(env, ep, s)
        ts, ps, _ = sim_tcp_trace(env, ep, s)
        # lift proxy on the sim side too (tool point 2 cm above its height at closing), for like-for-like
        if sp.get("t_close") is not None:
            k0 = int(np.argmin(np.abs(ts - (sp["t_close"] + 0.5))))
            il = np.where((ts > ts[k0]) & (ps[:, 2] > ps[k0, 2] + 0.02))[0]
            sp["t_lift_ee"] = float(ts[il[0]]) if len(il) else None
            if sp["t_lift_ee"] is not None and sp.get("t_open") is not None:
                sp["phases"]["grasp"] = round(sp["t_lift_ee"] - sp["t_close"], 2)
                sp["phases"]["transport"] = round(sp["t_open"] - sp["t_lift_ee"], 2)
        sim_g = None
        if bw.get(s) is not None and sp.get("t_close") is not None:
            k = int(np.argmin(np.abs(ts - (sp["t_close"] + 0.5))))
            c, a = np.array(bw[s]["c"]), np.array(bw[s]["a"])
            rel = ps[k] - c
            sim_g = {"along_cm": round(100 * float(rel @ a), 1), "lateral_cm": round(100 * float(rel @ np.array([-a[1], a[0], 0])), 1),
                     "above_centre_cm": round(100 * float(rel[2]), 1)}
        row = {"phases": compare_phases(real.get(s, {}), sp),
               "t_close": {"real": real.get(s, {}).get("t_close"), "sim": sp.get("t_close")},
               "n_close": {"real": real.get(s, {}).get("n_close"), "sim": sp.get("n_close")},
               "grasp_vs_perceived": {"real": real.get(s, {}).get("grasp_vs_perceived"), "sim": sim_g},
               "release_tcp_above_table_cm": {"real": real.get(s, {}).get("release", {}).get("tcp_above_table_cm"),
                                              "sim": sp.get("release", {}).get("tcp_above_table_cm")},
               "sim_outcome": sp.get("outcome")}
        if s in real and "tcp_world" in real[s]:
            rt, rp = real[s]["tcp_world"]["t"], real[s]["tcp_world"]["p"]
            row["ee_path_dtw_cm"] = round(dtw_cm(_resample(rt, rp), _resample(ts, ps)), 2)
            row["duration_ratio_sim_over_real"] = round(float((ts[-1] - ts[0]) / max(rt[-1] - rt[0], 1e-3)), 2)
        out["per_side"][s] = row
    report["closed"] = out


# ================================================================================= policy
def mode_policy(log, cfg, scene, out_dir, report, max_steps=40, sim_images=True):
    import torch
    from humanego_sim.runner import EpisodeRunner, T_ALIGN_ARIA
    from humanego_sim.driver import T_to_p
    from humanego_sim.perception import SideObjects, SimPerception
    from inference.data_types import ObjectState
    runner = EpisodeRunner(cfg)
    pol, env = runner.policy, runner.env
    K = np.asarray(log["meta"]["camera"]["K"], dtype=np.float64)
    W = int(log["meta"]["camera"].get("w", 640))
    Kf = K.copy()
    Kf[0, 2] = W - 1 - K[0, 2]
    hs = pol.single_hand_side
    steps = log["steps"]
    frames_dir = os.path.join(log["dir"], "frames")
    grasp_on = {e["side"]: e["t"] for e in reversed(log["events"]) if e.get("kind") == "grasp_on"}
    if sim_images:
        env.reset(scene, settle=0.4)
        sp = SimPerception(env, split_x_ratio=float(cfg["control"].get("split_x_ratio", 0.5)))
    init = log["inits"][0]
    rows = {s: [] for s in env.sides}
    idx = np.linspace(0, len(steps) - 1, min(max_steps, len(steps))).astype(int) if steps else []

    def run_pol(img, Tv, gv, objs, flip):
        x_ict, mask = pol.build_ict({hs: Tv}, {hs: gv}, {k: ObjectState(np.asarray(v)) for k, v in objs.items()}, "obj1")
        a_uv = pol.compute_anchor_uv(ObjectState(np.asarray(objs["obj1"])), Kf if flip else K, W, img.shape[0])
        traj, dp = pol.infer(pol.prepare_image(img), x_ict, mask, a_uv)
        pos, o6d, g = traj[hs]
        P = []
        for k in range(len(pos)):
            Tk = pol.decode_ee_in_cam(pos[k], o6d[k], None, T_ALIGN_ARIA)
            P.append(T_to_p(mirror_pose(Tk) if flip else Tk))
        return np.array(P), np.asarray(g).ravel(), dp

    prev_lift = {s: 0 for s in env.sides}
    for i in idx:
        stp = steps[i]
        for s, e in stp.get("sides", {}).items():
            if s not in rows or not e.get("plan_cam") or e.get("T_virtual") is None:
                continue
            if e.get("forced_lift_counter", 0) > 0 or prev_lift.get(s, 0) > 0:
                prev_lift[s] = e.get("forced_lift_counter", 0)
                continue          # the logged plan carries the forced-lift bias
            prev_lift[s] = e.get("forced_lift_counter", 0)
            p_img = os.path.join(frames_dir, f"{stp['n']:06d}_policy_{s}.jpg")
            if not os.path.exists(p_img):
                continue
            img = cv2.imread(p_img)
            flip = s == "left"
            logged = np.asarray(e["plan_cam"])[:, :3]
            P, g, dp = run_pol(img, np.asarray(e["T_virtual"]), float(e.get("gv", 0.0)), e["obj_poses"], flip)
            n = min(len(P), len(logged))
            row = {"t": stp["t"], "plan_err_cm": round(100 * float(np.linalg.norm(P[:n, :3] - logged[:n], axis=1).mean()), 2),
                   "first_err_cm": round(100 * float(np.linalg.norm(P[0, :3] - logged[0])), 2),
                   "grasp_prob_diff": round(float(np.abs(g[:n] - np.asarray(e.get("grasp_probs", g))[:n]).mean()), 3)
                   if e.get("grasp_probs") is not None else None,
                   "done_diff": round(abs(dp - float(e.get("done_prob", dp))), 3)}
            # noise floor: same inputs, another flow-matching noise draw
            n0 = pol.noise.clone()
            pol.noise = torch.randn_like(n0)
            P2, _, _ = run_pol(img, np.asarray(e["T_virtual"]), float(e.get("gv", 0.0)), e["obj_poses"], flip)
            pol.noise = n0
            row["noise_floor_cm"] = round(100 * float(np.linalg.norm(P2[:n, :3] - P[:n, :3], axis=1).mean()), 2)
            # visual sim-to-real gap: before the first grasp the bread is where perception saw it
            if sim_images and (s not in grasp_on or stp["t"] < grasp_on[s]) and e.get("state", {}).get("q"):
                arm = env.arms[s]
                q = e["state"]["q"]
                arm.set_j_instant(q[:6], gripper=q[6] if len(q) > 6 else None)
                mujoco.mj_forward(env.m, env.d)
                o = init["objects"][s]
                so = SideObjects(objs_meta={k: {"T_obj_in_cam_static": np.asarray(v["T_obj_in_cam_static"]),
                                                "kpts_local": np.asarray(v["kpts_local"]).reshape(-1, 3)}
                                            for k, v in o.items() if not k.startswith("_")},
                                 dynamic_obj_poses={k: np.asarray(v) for k, v in e["obj_poses"].items()})
                clean = sp.clean_background()
                vis = sp.make_side_image(s, so, clean, arm, float(e.get("gv", 0.0)) > 0.5)
                Ps, _, _ = run_pol(vis, np.asarray(e["T_virtual"]), float(e.get("gv", 0.0)), e["obj_poses"], flip)
                row["sim_image_plan_diff_cm"] = round(100 * float(np.linalg.norm(Ps[:n, :3] - P[:n, :3], axis=1).mean()), 2)
                row["image_mean_abs_diff"] = round(float(cv2.absdiff(cv2.resize(img, (vis.shape[1], vis.shape[0])), vis).mean()), 2)
                if len(rows[s]) == 0:
                    cv2.imwrite(os.path.join(out_dir, f"policy_input_{s}_real_vs_sim.jpg"), np.concatenate([img, vis], 1))
            rows[s].append(row)
    summ = {}
    for s, rr in rows.items():
        if not rr:
            continue
        summ[s] = {"n": len(rr)}
        for k in ("plan_err_cm", "first_err_cm", "grasp_prob_diff", "done_diff", "noise_floor_cm", "sim_image_plan_diff_cm",
                  "image_mean_abs_diff"):
            v = [r[k] for r in rr if r.get(k) is not None]
            if v:
                summ[s][k] = {"median": round(float(np.median(v)), 3), "max": round(float(np.max(v)), 3), "n": len(v)}
    report["policy"] = {"summary": summ, "rows": rows}


# =================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--modes", nargs="+", default=["scene", "kinematic", "closed", "policy"],
                    choices=["scene", "kinematic", "closed", "policy"])
    ap.add_argument("--params", default=None, help="tuned params json (configs/tuned/<robot>.json); default production")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. control.alpha=0.3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy_steps", type=int, default=40)
    ap.add_argument("--no_sim_images", action="store_true")
    ap.add_argument("--refine", type=int, default=2, help="scene refinement iterations (0 = raw perceived centroids)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from evaluate import parse_sets
    from autotune.params import params_to_overrides
    log = load_log(args.log_dir)
    over = parse_sets(args.set)
    if args.params:
        base = load_config(None, over)
        over = deep_update(params_to_overrides(json.load(open(args.params))["params"], base), over)
    cfg = make_cfg(log, over)
    out_dir = args.out or os.path.join(args.log_dir, "sim_replay")
    os.makedirs(out_dir, exist_ok=True)
    report = {"log": os.path.abspath(args.log_dir), "params": args.params, "set": args.set,
              "real_outcome": log.get("outcome", {})}
    from humanego_sim.env import ServeBreadSim
    env0 = ServeBreadSim(cfg)
    rec = reconstruct_scene(log, cfg, env0.T_cam_in_world)
    scene, sim_init, hist = refine_scene(log, cfg, env0, rec["scene"], iters=args.refine)
    report["scene_rebuilt"] = scene
    report["refine_residual_cm"] = hist
    print(f"[replay_real] scene {scene} (perception residual per iteration, cm: {hist})", flush=True)
    for m in args.modes:
        print(f"[replay_real] {m} ...", flush=True)
        if m == "scene":
            mode_scene(log, cfg, out_dir, report, env0, rec, scene, sim_init)
        elif m == "kinematic":
            mode_kinematic(log, cfg, scene, out_dir, report)
        elif m == "closed":
            mode_closed(log, cfg, scene, out_dir, report, seed=args.seed)
        elif m == "policy":
            mode_policy(log, cfg, scene, out_dir, report, max_steps=args.policy_steps, sim_images=not args.no_sim_images)
        json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1, default=lambda o: None)
    short = {k: v for k, v in report.items() if k != "policy"}
    if "policy" in report:
        short["policy"] = report["policy"]["summary"]
    print(json.dumps(short, indent=1, default=lambda o: None))


if __name__ == "__main__":
    main()
