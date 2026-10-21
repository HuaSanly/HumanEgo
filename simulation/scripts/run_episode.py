# -*- coding: utf-8 -*-
"""
Run the released HumanEgo serve_bread policy closed-loop in simulation.

    python simulation/scripts/run_episode.py                       # bimanual, Trossen WidowX AI + D405
    python simulation/scripts/run_episode.py --mode single --video
    python simulation/scripts/run_episode.py --robot franka_panda --camera zed2i
    python simulation/scripts/run_episode.py --profile tuned           # shipped tuned controller + grasp skill

Outputs land in simulation/runs/<name>/ (episode.json, policy_view/*.jpg, episode.mp4).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from humanego_sim.env import ServeBreadSim, load_config  # noqa: E402
from humanego_sim.profiles import add_profile_arg  # noqa: E402
from humanego_sim.replay import render_episode_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", default=None, choices=[None, "single", "bimanual"])
    ap.add_argument("--robot", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default=None, help='JSON, e.g. \'{"bread_right":[0.45,-0.25,30]}\'')
    ap.add_argument("--set", action="append", default=[], help="override, e.g. control.alpha=0.3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--camera_view", default="front_view")
    ap.add_argument("--duration", type=float, default=5.0, help="physics preview duration (scene-only configs)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    add_profile_arg(ap)
    args = ap.parse_args()

    ov = {"profile": args.profile}
    for k in ("mode", "robot", "camera"):
        if getattr(args, k):
            ov[k] = getattr(args, k)
    for s in args.set:
        key, val = s.split("=", 1)
        cur = ov
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = json.loads(val)
    cfg = load_config(args.config, ov)
    prefix = f"{cfg['task']}_" if cfg.get("task") != "serve_bread" else ""
    name = args.out or f"{prefix}{cfg['robot']}_{cfg['camera']}_{cfg['mode']}_s{args.seed}"
    log_dir = os.path.join(os.path.dirname(HERE), "runs", name)
    scene = json.loads(args.scene) if args.scene else None
    if cfg.get("scene_only"):
        from humanego_sim.scene_preview import record_scene
        env = ServeBreadSim(cfg, seed=args.seed, render=False)
        result = record_scene(env, log_dir, scene=scene, duration=args.duration)
        print(json.dumps(dict(result, scene_only=True, log_dir=log_dir), indent=1))
        if args.video:
            print("video:", render_episode_video(env, log_dir, camera=args.camera_view))
        return
    from humanego_sim.runner import EpisodeRunner
    runner = EpisodeRunner(cfg, device=args.device)
    res = runner.run(scene=scene, seed=args.seed, log_dir=log_dir)
    print(json.dumps({"success": res.success, "per_side": res.per_side, "duration": res.duration,
                      "wall": res.wall_time}, indent=1))
    if args.video:
        p = render_episode_video(runner.env, log_dir, camera=args.camera_view)
        print("video:", p)


if __name__ == "__main__":
    main()
