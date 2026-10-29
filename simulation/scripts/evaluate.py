# -*- coding: utf-8 -*-
"""
Batch success rate of the released policy in simulation.

    python simulation/scripts/evaluate.py --n 32 --workers 8
    python simulation/scripts/evaluate.py --mode single --n 40 --set control.alpha=0.3
    python simulation/scripts/evaluate.py --profile tuned --robot ur5e --n 24 --seed 7777

Prints per-arm / per-side / both-served success with Wilson 95 % intervals. Scene sets: --seed
7777 / 8888 for development, 9001-9099 only with --final (README, "Evaluation protocol").
"""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from humanego_sim.evaluate import evaluate  # noqa: E402
from humanego_sim.profiles import add_profile_arg  # noqa: E402
from humanego_sim.sampling import sample_scenes  # noqa: E402
from humanego_sim.stats import check_seed  # noqa: E402


def parse_sets(sets):
    ov = {}
    for s in sets:
        key, val = s.split("=", 1)
        cur = ov
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = json.loads(val)
    return ov


def evaluate_pick_cube(args, overrides):
    import yaml
    from humanego_sim.env import load_config
    from humanego_sim.pick_cube_evaluation import (
        SAMPLING_CONFIG, checkpoint_identity, load_study, prepare_study, read_results,
        render_videos, run_study, summarize,
    )
    from humanego_sim.pick_cube_sampling import make_trials

    if not args.out or args.log:
        raise ValueError("pick_cube evaluation requires --out NAME (or an absolute path); use --out instead of --log")
    if args.variation:
        raise ValueError("pick_cube uses --sampling, not serve_bread --variation")
    if args.retry_errors and not args.resume:
        raise ValueError("--retry-errors requires --resume")
    root = Path(args.out).expanduser()
    if not root.is_absolute():
        root = Path(HERE).parent / "runs" / root
    root = root.resolve()
    if args.resume or args.report_only:
        cfg, sampling, manifest, protocol = load_study(root, verify_runtime=not args.report_only)
        if args.set or args.sampling or args.trials or args.mode or args.robot or args.camera:
            raise ValueError("resume/report uses frozen config and trials; overrides require a new --out")
        if args.config and not args.report_only:
            requested = load_config(args.config, overrides)
            requested["policy"]["ckpt"] = checkpoint_identity(requested)["path"]
            if requested != cfg:
                raise ValueError("live task config differs from the frozen study; omit --config when resuming")
        if args.n is not None and args.n != manifest["n_scenes"]:
            raise ValueError("--n differs from frozen scene count")
        if args.policy_seeds is not None and args.policy_seeds != manifest["policy_seeds"]:
            raise ValueError("--policy-seeds differs from frozen trials")
        if args.scene_seed is not None and args.scene_seed != manifest["scene_seed"]:
            raise ValueError("--scene-seed differs from frozen trials")
    else:
        cfg = load_config(args.config, overrides)
        sampling = yaml.safe_load(Path(args.sampling or SAMPLING_CONFIG).read_text())
        if args.trials:
            manifest = json.loads(Path(args.trials).read_text())
            if args.n is not None and args.n != manifest["n_scenes"]:
                raise ValueError("--n differs from imported trials")
            if args.policy_seeds is not None and args.policy_seeds != manifest["policy_seeds"]:
                raise ValueError("--policy-seeds differs from imported trials")
            if args.scene_seed is not None and args.scene_seed != manifest["scene_seed"]:
                raise ValueError("--scene-seed differs from imported trials")
        else:
            manifest = make_trials(cfg, sampling, args.n if args.n is not None else 50,
                                   args.scene_seed if args.scene_seed is not None else args.seed,
                                   args.policy_seeds if args.policy_seeds is not None else [0, 1, 2])
        check_seed([manifest["scene_seed"]], args.final)
        protocol = prepare_study(root, cfg, sampling, manifest, args.device)
        cfg, _, _, _ = load_study(root)
    check_seed([manifest["scene_seed"]], args.final)
    if any(index < 0 or index >= len(manifest["trials"]) for index in args.video_trials):
        raise ValueError("--video-trials indices are outside the trial list")
    if args.prepare_only or args.report_only:
        summary = summarize(root, manifest, protocol)
    else:
        summary = run_study(root, args.workers if args.workers is not None else 1, args.device, args.retry_errors)
    rows = list(read_results(root, manifest, protocol).values())
    render_videos(root, cfg, rows, args.videos, args.video_trials)
    print(json.dumps({**{key: summary[key] for key in
                         ("planned", "completed", "pending", "successes", "errors", "success_rate", "scene_bootstrap95")},
                      "output": str(root)}, indent=2))
    return 1 if summary["errors"] else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="task YAML; omit for the legacy serve_bread evaluator")
    ap.add_argument("--n", type=int, default=None, help="scene count; pick_cube trials = n × number of policy seeds")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--robot", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--log", default=None, help="save every episode under simulation/runs/<log>/")
    ap.add_argument("--variation", default=None,
                    help="scene variation JSON (sampling.sample_scenes), e.g. '{\"light\": {\"intensity\": 0.35}}'")
    ap.add_argument("--final", action="store_true", help="allow the reserved seeds 9001-9099 (final reports only)")
    ap.add_argument("--scene-seed", type=int, default=None, help="pick_cube layout seed; defaults to --seed")
    ap.add_argument("--policy-seeds", type=int, nargs="+", default=None, help="explicit per-episode policy noise seeds")
    ap.add_argument("--sampling", help="pick_cube evaluation sampling YAML")
    ap.add_argument("--trials", help="reuse an existing trials.json in a new evaluation")
    ap.add_argument("--out", help="pick_cube evaluation name under simulation/runs, or an absolute directory")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--resume", action="store_true", help="continue the frozen study, skipping completed trials")
    ap.add_argument("--retry-errors", action="store_true", help="explicitly retry errored trials on resume")
    ap.add_argument("--prepare-only", action="store_true", help="freeze config, seeds and scenes without loading the model")
    ap.add_argument("--report-only", action="store_true", help="rebuild reports from saved results without inference")
    ap.add_argument("--videos", choices=["none", "failures", "all"], default="none")
    ap.add_argument("--video-trials", type=int, nargs="*", default=[], help="trial indices to render from recorded physics")
    add_profile_arg(ap)
    args = ap.parse_args()
    ov = parse_sets(args.set)
    ov.setdefault("profile", args.profile)
    for k in ("mode", "robot", "camera"):
        if getattr(args, k):
            ov[k] = getattr(args, k)
    from humanego_sim.env import load_config
    if args.resume or args.report_only or load_config(args.config, ov).get("task") == "pick_cube":
        try:
            return evaluate_pick_cube(args, ov)
        except (ValueError, FileNotFoundError) as error:
            ap.error(str(error))
    if (args.config or args.out or args.policy_seeds or args.scene_seed is not None or args.prepare_only
            or args.sampling or args.trials or args.retry_errors or args.video_trials or args.videos != "none"):
        ap.error("new evaluation options require --config simulation/configs/pick_cube.yaml")
    check_seed([args.seed], args.final)
    n, workers = args.n if args.n is not None else 16, args.workers if args.workers is not None else 6
    log_root = os.path.join(os.path.dirname(HERE), "runs", args.log) if args.log else None
    scenes = None
    if args.variation:
        from humanego_sim.env import load_config
        scenes = sample_scenes(n, args.seed, load_config(None, ov).get("mode", "bimanual"), args.difficulty,
                               variation=json.loads(args.variation))
    st = evaluate(ov, n_episodes=n, workers=workers, seed=args.seed, difficulty=args.difficulty,
                  log_root=log_root, scenes=scenes, device=args.device)
    st.pop("results", None)
    print(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
