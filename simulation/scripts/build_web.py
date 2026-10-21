# -*- coding: utf-8 -*-
"""
Build the web viewer's data: models, rollouts, manifest — then pack them for the web.

    python simulation/scripts/build_web.py --configs trossen_wxai:realsense_d405:bimanual --n 24 --workers 6
    python simulation/scripts/build_web.py --profile tuned --configs franka_panda:realsense_d405:bimanual:10
    python simulation/scripts/build_web.py --configs ur5e:realsense_d405:bimanual:10:tuned   # per-config profile
    python simulation/scripts/build_web.py --manifest-only            # re-generate the manifest + pack

Config spec: robot:camera:mode[:n[:profile]]. Rollouts come from the released policy on
randomized initial conditions (paper protocol, fixed seeds) with the physical drop-in intro
recorded. The manifest's numbers and notes are generated (humanego_sim/web_manifest.py); the
build is packed in place afterwards (humanego_sim/web_pack.py, skip with --no-pack).
For the whole site use scripts/rebuild_site.py.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)
os.environ.setdefault("MUJOCO_GL", "egl")

from humanego_sim.cameras import CAMERAS  # noqa: E402
from humanego_sim.env import ServeBreadSim, deep_update, load_config  # noqa: E402
from humanego_sim.evaluate import evaluate  # noqa: E402
from humanego_sim.live_export import export_live  # noqa: E402
from humanego_sim.profiles import PROFILES, add_profile_arg, load_tuned, profile_info  # noqa: E402
from humanego_sim.robots import ROBOTS  # noqa: E402
from humanego_sim.sampling import sample_scene  # noqa: E402
from humanego_sim import web_manifest  # noqa: E402
from humanego_sim.web_export import export_model, export_rollout  # noqa: E402

WEB = os.path.join(SIM, "web")
LOG_ROOT = os.path.join(SIM, "runs", "web")
DEFAULT = {"robot": "trossen_wxai", "camera": "realsense_d405", "mode": "bimanual"}


def mirror_pose(T):
    S = np.diag([-1.0, 1.0, 1.0, 1.0])
    return S @ T @ S


def T_free(q):
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, 3] = q[:3]
    T[:3, :3] = R.from_quat([q[4], q[5], q[6], q[3]]).as_matrix()
    return T


def rollout_extras(env, log_dir, cfg):
    """Keypoints / PCA frames in object body frames, appear times, look-ahead."""
    ep = json.load(open(os.path.join(log_dir, "episode.json")))
    meta, frames = ep["meta"], ep["frames"]
    f0 = next(f for f in frames if f["t"] >= 0)
    T_cw = np.array(meta["T_cam_in_world"])
    kpts = {}
    colors = {"obj1": "#22d3ee", "obj2": "#3ddc97"}
    for side, objs in meta["perception"].items():
        for k, v in objs.items():
            roles = env.meta.get("object_roles")
            body = roles[k][0] if roles else (f"bread_{side}" if k == "obj1" else "plate")
            if body in kpts:
                continue
            Ts = np.array(v["T_obj_in_cam_static"])
            kl = np.array(v["kpts_local"]).reshape(-1, 3)
            if side == "left":
                Ts = mirror_pose(Ts)
                kl = kl * np.array([-1.0, 1.0, 1.0])   # local coords mirror with the frame
            pc = (Ts[:3, :3] @ kl.T).T + Ts[:3, 3]
            pw = (T_cw[:3, :3] @ pc.T).T + T_cw[:3, 3]
            Tb = T_free(np.array(f0["objects"][body]))
            Tbi = np.linalg.inv(Tb)
            pb = (Tbi[:3, :3] @ pw.T).T + Tbi[:3, 3]
            Tf = Tbi @ T_cw @ Ts
            kpts[body] = {"pts": np.round(pb, 4).tolist(), "color": colors[k], "frame": np.round(Tf, 4).ravel().tolist()}
    appear = {}
    for f in frames:
        hid = set(f.get("hidden", []))
        for name in f["objects"]:
            if name not in hid and name not in appear:
                appear[name] = f["t"]
    return {"kpts": kpts, "appear": appear, "look_ahead": int(cfg["control"]["look_ahead_offset"])}


def build_config(robot, camera, mode, n, workers, seed, overrides, profile="production"):
    key = f"{robot}_{camera}_{mode}"
    ov = deep_update(overrides, {"robot": robot, "camera": camera, "mode": mode, "profile": profile})
    cfg = load_config(None, ov)
    env = ServeBreadSim(cfg)
    model_key = key
    export_model(env, WEB, model_key)
    export_live(env, WEB, model_key)          # physics-only MJCF (+ this profile's controller) for Live mode
    rng = np.random.default_rng(seed)
    scenes = [sample_scene(rng, mode) for _ in range(n)]
    log_root = os.path.join(LOG_ROOT, key)
    shutil.rmtree(log_root, ignore_errors=True)
    st = evaluate(ov, scenes=scenes, workers=workers, seed=seed, log_root=log_root, intro=True)
    rolls = []
    for r in st["results"]:
        if "error" in r:
            print("  episode error:", r["error"])
            continue
        ld = os.path.join(log_root, f"ep{r['idx']:03d}")
        rid = f"{key}_{r['idx']:03d}"
        ro = export_rollout(env, ld, WEB, rid)
        ro.update(rollout_extras(env, ld, cfg))
        ro["model"] = model_key
        with open(os.path.join(WEB, "data", "rollouts", f"{rid}.json"), "w") as f:
            json.dump(ro, f, separators=(",", ":"))
        rolls.append({"id": rid, "scene": ro["scene"], "per_side": {s: {"success": v["success"]}
                      for s, v in ro["result"]["per_side"].items()}, "success": ro["result"]["success"]})
    info = profile_info(profile, robot)
    info["eval"] = {"n": st["n"], "per_arm": st["per_side_success_rate"], "sides": st["side_success"],
                    "both": st["success_rate"], "failure_modes": st["failure_modes"], "seed": seed}
    return {"robot": robot, "camera": camera, "mode": mode, "model": model_key, "profile": info, "rollouts": rolls}


def _git_rev():
    try:
        import subprocess
        return subprocess.run(["git", "-C", SIM, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def build_task_config(cfg, seed=0):
    from humanego_sim.scene_preview import record_scene
    key = f"{cfg['task']}_{cfg['robot']}_{cfg['camera']}_{cfg['mode']}"
    preview = bool(cfg.get("scene_only"))
    env = ServeBreadSim(cfg, seed=seed, render=not preview)
    log_dir = os.path.join(LOG_ROOT, key, "ep000")
    if preview:
        record_scene(env, log_dir)
    else:
        from humanego_sim.runner import EpisodeRunner
        EpisodeRunner(cfg, env=env).run(seed=seed, log_dir=log_dir)
    export_model(env, WEB, key)
    if preview:
        export_live(env, WEB, key)
    rid = f"{key}_000"
    ro = export_rollout(env, log_dir, WEB, rid)
    ro.update(model=key, kpts={}, appear={}, look_ahead=int(cfg["control"]["look_ahead_offset"]))
    if not preview:
        ro.update(rollout_extras(env, log_dir, cfg))
    with open(os.path.join(WEB, "data", "rollouts", f"{rid}.json"), "w") as f:
        json.dump(ro, f, separators=(",", ":"))
    if env.renderer is not None:
        env.renderer.close()
    return {"robot": cfg["robot"], "camera": cfg["camera"], "mode": cfg["mode"], "task": cfg["task"],
            "model": key, "scene_only": preview, "objects": cfg["scene"]["objects"],
            "policy": cfg["policy"],
            "rollouts": [{"id": rid, "scene": ro["scene"], "per_side": ro["result"]["per_side"],
                          "success": ro["result"]["success"]}]}


def write_manifest(configs, tuning_robot="trossen_wxai", media=None):
    man_p = os.path.join(WEB, "data", "manifest.json")
    old = json.load(open(man_p)) if os.path.exists(man_p) else {"configs": []}
    keep = {(c.get("task", "serve_bread"), c["robot"], c["camera"], c["mode"]): c for c in old.get("configs", [])}
    for c in configs:
        keep[(c.get("task", "serve_bread"), c["robot"], c["camera"], c["mode"])] = c
    cfgs = sorted(keep.values(), key=lambda c: (list(ROBOTS).index(c["robot"]) if c["robot"] in ROBOTS else 9,
                                                list(CAMERAS).index(c["camera"]) if c["camera"] in CAMERAS else 9,
                                                c["mode"] != "bimanual"))
    for c in cfgs:
        c["live"] = os.path.exists(os.path.join(WEB, "assets", "live", c["model"], "spec.json")) and \
            (c.get("scene_only") or (c.get("task", "serve_bread") == "serve_bread" and
                                    os.path.exists(os.path.join(WEB, "assets", "policy", "serve_bread_meta.json"))))
        # refresh the quoted numbers from configs/tuned (same parameters: same autotune run), e.g. after
        # the reserved test set was run on a kept build (rebuild_site.py --skip-build)
        p = c.get("profile")
        if isinstance(p, dict) and p.get("profile") == "tuned":
            fresh = profile_info("tuned", c["robot"])
            if fresh.get("autotune_run") == p.get("autotune_run"):
                c["profile"] = fresh
    # the tuning plot shows the run behind the profile the default robot's rollouts use
    prof = next((c.get("profile") for c in cfgs if c["robot"] == tuning_robot and c.get("profile")), None)
    tuning = None
    if prof and prof.get("profile") == "tuned":
        tuning = web_manifest.tuning_section(load_tuned(tuning_robot))
    extra = {"built": {"date": time.strftime("%Y-%m-%d"), "commit": _git_rev()}}
    if media or old.get("media"):
        extra["media"] = media or old.get("media")
    if old.get("packed"):
        extra["packed"] = old["packed"]
    default = dict(DEFAULT, task="serve_bread")
    if not any(c.get("task", "serve_bread") == "serve_bread" and
               all(c[k] == DEFAULT[k] for k in DEFAULT) for c in cfgs):
        first = next((c for c in cfgs if not c.get("scene_only")), cfgs[0])
        default = {k: first[k] for k in ("robot", "camera", "mode")}
        default["task"] = first.get("task", "serve_bread")
    man = web_manifest.assemble(cfgs, {k: v.label for k, v in ROBOTS.items()},
                                {k: v.label for k, v in CAMERAS.items()}, default, tuning, extra)
    with open(man_p, "w") as f:
        json.dump(man, f, separators=(",", ":"))
    print(f"[web] manifest: {len(cfgs)} configs, {sum(len(c['rollouts']) for c in cfgs)} rollouts")
    return man


def parse_spec(spec, n_default, profile_default):
    parts = spec.split(":")
    robot, camera, mode = parts[:3]
    n = int(parts[3]) if len(parts) > 3 and parts[3] else n_default
    profile = parts[4] if len(parts) > 4 else profile_default
    if profile not in PROFILES:
        raise SystemExit(f"{spec}: unknown profile {profile} (choose from {PROFILES})")
    return robot, camera, mode, n, profile


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", default="trossen_wxai:realsense_d405:bimanual",
                    help="comma-separated robot:camera:mode[:n[:profile]]")
    ap.add_argument("--config", help="pick_cube YAML: run its trained policy, or preview with scene_only=true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--params", default=None, help="extra tuned parameters JSON (best.json) on top of the profile")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="extra override, e.g. control.alpha=0.3")
    ap.add_argument("--no-pack", action="store_true", help="leave the JSON rollouts / GLBs / videos unpacked")
    add_profile_arg(ap)
    args = ap.parse_args(argv)
    overrides = {}
    if args.params:
        from autotune.params import params_to_overrides
        best = json.load(open(args.params))
        overrides = params_to_overrides(best["params"], load_config())
    if args.set:
        from evaluate import parse_sets
        overrides = deep_update(overrides, parse_sets(args.set))
    built = []
    if args.config and not args.manifest_only:
        cfg = load_config(args.config, deep_update(overrides, {"profile": args.profile}))
        if cfg.get("task") != "pick_cube":
            ap.error("--config expects pick_cube.yaml; use --configs for serve_bread rollouts")
        built.append(build_task_config(cfg, args.seed))
    elif not args.manifest_only:
        for spec in args.configs.split(","):
            robot, camera, mode, n, profile = parse_spec(spec, args.n, args.profile)
            print(f"[web] building {robot} / {camera} / {mode} ({n} rollouts, profile {profile})", flush=True)
            built.append(build_config(robot, camera, mode, n, args.workers, args.seed, overrides, profile))
            write_manifest(built)         # usable after every config
    man = write_manifest(built)
    if not args.no_pack:
        from humanego_sim.web_pack import pack_web
        pack_web(WEB, log_root=LOG_ROOT)
        man = json.load(open(os.path.join(WEB, "data", "manifest.json")))
    errs = web_manifest.validate(man, WEB)
    if errs:
        raise SystemExit("[web] manifest invalid:\n  " + "\n  ".join(errs))
    return man


if __name__ == "__main__":
    main()
