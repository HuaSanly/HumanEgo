"""Frozen, resumable pick_cube scene × policy-noise evaluation."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
import yaml

from .env import REPO_ROOT, SIM_ROOT
from .pick_cube_sampling import validate_sampling, validate_trials
from .stats import scene_bootstrap_ci, wilson

STATE = {}
SAMPLING_CONFIG = Path(SIM_ROOT) / "configs/eval_pick_cube.yaml"


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def code_digest() -> str:
    result = hashlib.sha256()
    root = Path(REPO_ROOT)
    for directory in ("simulation/humanego_sim", "simulation/scripts", "inference", "training", "utils"):
        for path in sorted((root / directory).rglob("*.py")):
            result.update(str(path.relative_to(root)).encode())
            result.update(path.read_bytes())
    return result.hexdigest()


def source_patch() -> str:
    """Include new Python modules as well as edits to already tracked files."""
    patch = subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=REPO_ROOT, text=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--",
                                        "simulation/humanego_sim", "simulation/scripts", "inference", "training", "utils"],
                                       cwd=REPO_ROOT, text=True).splitlines()
    for path in untracked:
        if path.endswith(".py"):
            added = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", path],
                                   cwd=REPO_ROOT, text=True, capture_output=True)
            if added.returncode not in (0, 1):
                raise RuntimeError(added.stderr)
            patch += added.stdout
    return patch


def runtime_versions() -> dict:
    values = {"python": platform.python_version()}
    for package in ("mujoco", "torch", "torchvision", "numpy", "scipy", "scikit-learn",
                    "opencv-python", "opencv-python-headless", "Pillow", "PyYAML"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    return values


def checkpoint_identity(cfg: dict) -> dict:
    path = Path(cfg["policy"]["ckpt"]).expanduser()
    if not path.is_absolute():
        path = Path(REPO_ROOT) / path
    path = path.resolve()
    return {"path": str(path), "sha256": file_digest(path),
            "config_sha256": file_digest(path.parent / "config.json"),
            "stats_sha256": file_digest(path.parent / "dataset_stats.json")}


def prepare_study(root: Path, cfg: dict, sampling: dict, manifest: dict, device: str) -> dict:
    if cfg.get("task") != "pick_cube" or cfg.get("mode") != "single" or cfg.get("scene_only"):
        raise ValueError("pick_cube evaluation requires the single-arm policy task, not a scene preview")
    validate_sampling(sampling)
    validate_trials(manifest, cfg, sampling)
    if float(cfg["control"]["max_episode_time"]) <= 0:
        raise ValueError("max_episode_time must be positive")
    if root.exists() and any(root.iterdir()):
        raise ValueError("output directory is not empty; use --resume or choose a new --out")
    checkpoint = checkpoint_identity(cfg)
    cfg = {**cfg, "policy": {**cfg["policy"], "ckpt": checkpoint["path"]}}
    git = lambda *args: subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)
    protocol = {"schema_version": 1, "config_sha256": digest(cfg), "sampling_sha256": digest(sampling),
                "trials_sha256": digest(manifest), "checkpoint": checkpoint, "code_sha256": code_digest(),
                "runtime": runtime_versions(), "device": device,
                "evaluation": {"intro": False, "record_hz": 10, "lift_height": 0.025,
                               "lift_duration": 0.2, "secure_lift_duration": 0.1, "pad_force": 0.1,
                               "gripper_open_min": 0.9},
                "assists": {"grasp": bool(cfg["control"].get("grasp_assist", {}).get("enabled")),
                            "forced_lift_height": cfg["control"].get("forced_lift_height", 0.05),
                            "forced_lift_frames": cfg["control"].get("forced_lift_frames", 15),
                            "release": bool(cfg["control"].get("release_assist", {}).get("enabled"))},
                "git": {"commit": git("rev-parse", "HEAD").strip(), "status": git("status", "--short")}}
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (root / "sampling.yaml").write_text(yaml.safe_dump(sampling, sort_keys=False))
    (root / "source.patch").write_text(source_patch())
    write_json(root / "trials.json", manifest)
    write_json(root / "protocol.json", protocol)
    return protocol


def load_study(root: Path, verify_runtime: bool = True) -> tuple[dict, dict, dict, dict]:
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    sampling = yaml.safe_load((root / "sampling.yaml").read_text())
    manifest = json.loads((root / "trials.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    for key, value in (("config_sha256", cfg), ("sampling_sha256", sampling), ("trials_sha256", manifest)):
        if digest(value) != protocol[key]:
            raise ValueError(f"frozen study changed: {key}; use a new output directory")
    if verify_runtime:
        if code_digest() != protocol["code_sha256"] or runtime_versions() != protocol["runtime"]:
            raise ValueError("evaluation code or dependencies changed; use a new output directory")
        if checkpoint_identity(cfg) != protocol["checkpoint"]:
            raise ValueError("checkpoint or training metadata changed; use a new output directory")
    return cfg, sampling, manifest, protocol


def read_results(root: Path, manifest: dict, protocol: dict) -> dict:
    results = {}
    for trial in manifest["trials"]:
        path = root / "episodes" / trial["trial_id"] / "result.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        if (row.get("trial") != trial or row.get("protocol_sha256") != digest(protocol)
                or row.get("status") not in ("success", "failure", "error")
                or row.get("success") is not (row.get("status") == "success")):
            raise ValueError(f"result does not match the frozen trial: {path}")
        results[trial["trial_id"]] = row
    return results


def failure_reason(row: dict) -> str:
    if row["success"]:
        return "success"
    if not row["physical"]["both_pads"]:
        return "not_grasped"
    if not row["physical"]["lifted"]:
        return "not_lifted"
    if not row["physical"]["secure_grasp"]:
        return "unstable_grasp"
    side = row["per_side"]["right"]
    if not side["released"]:
        return "not_released" if row["holding_at_end"] else "dropped_in_transit"
    if not side["inside_container"]:
        return "outside_container"
    if not side["settled"]:
        return "not_settled"
    return "placement_failed"


def _init_worker(root: str, device: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    STATE.clear()
    STATE.update(root=Path(root), device=device)
    try:
        from .pick_cube_metrics import MeasuredSim
        from .runner import EpisodeRunner
        cfg, _, _, protocol = load_study(Path(root))
        env = MeasuredSim(cfg)
        STATE.update(cfg=cfg, protocol=protocol, env=env,
                     runner=EpisodeRunner(cfg, device=device, env=env))
        STATE["policy"] = STATE["runner"].policy
        if checkpoint_identity(cfg) != protocol["checkpoint"]:
            raise ValueError("checkpoint changed while the worker was loading it")
    except Exception:
        STATE["init_error"] = traceback.format_exc()


def _run_trial(trial: dict) -> dict:
    output = STATE["root"] / "episodes" / trial["trial_id"]
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((STATE["root"] / "protocol.json").read_text())
    row = {"trial": trial, "protocol_sha256": digest(protocol), "success": False}
    started = time.time()
    try:
        if "init_error" in STATE:
            raise RuntimeError(STATE["init_error"])
        if STATE["runner"] is None:
            from .pick_cube_metrics import MeasuredSim
            from .runner import EpisodeRunner
            STATE["env"] = MeasuredSim(STATE["cfg"])
            STATE["runner"] = EpisodeRunner(STATE["cfg"], policy=STATE["policy"],
                                             device=STATE["device"], env=STATE["env"])
        env, runner = STATE["env"], STATE["runner"]
        value = runner.run(scene=trial["scene"], seed=trial["env_seed"], policy_seed=trial["policy_seed"],
                           log_dir=str(output), save_policy_images=False, record_hz=10, intro=False, verbose=False)
        measured = dict(env.measurements)
        physical = {"both_pads": measured["both_pads"], "lifted": measured["max_lift_duration"] >= 0.2,
                    "secure_grasp": measured["max_secure_duration"] >= 0.1,
                    "transported": measured["max_horizontal"] > 0.05 and measured["max_lift_duration"] >= 0.2}
        opening = env.arms["right"].get_gripper_q()
        success = bool(value.success and physical["secure_grasp"] and opening >= 0.9)
        row.update(status="success" if success else "failure", success=success, runner_success=value.success,
                   per_side=value.per_side, physical=physical, measurements=measured,
                   gripper_opening=opening, holding_at_end=bool(np.all(np.array(env.telemetry[-1]["forces"]) > 0.1)),
                   duration=value.duration, wall=value.wall_time, metrics=value.metrics)
        row["per_side"]["right"]["released"] = bool(row["per_side"]["right"]["released"] and opening >= 0.9)
        row["failure_reason"] = failure_reason(row)
        # Keep replay metadata and the batch's stricter physical score consistent.
        path = output / "episode.json"
        episode = json.loads(path.read_text())
        episode["meta"]["result"].update(runner_success=value.success, success=success)
        episode["meta"]["result"]["per_side"]["right"]["success"] = success
        episode["meta"]["result"]["per_side"]["right"]["released"] = row["per_side"]["right"]["released"]
        episode["meta"]["evaluation"] = {"physical": physical, "trial": trial}
        write_json(path, episode)
    except Exception as error:
        row.update(status="error", success=False, error=repr(error), trace=traceback.format_exc(),
                   failure_reason="error", wall=time.time() - started)
        # The next job starts from a new environment after an interrupted simulation.
        if "env" in STATE:
            STATE["env"].renderer.close()
        STATE["runner"] = None
    write_json(output / "result.json", row)
    return row


def summarize(root: Path, manifest: dict, protocol: dict) -> dict:
    results = read_results(root, manifest, protocol)
    rows = list(results.values())
    successes = sum(row["success"] for row in rows)
    counts = Counter(row["status"] for row in rows)
    planned = len(manifest["trials"])
    complete = len(rows) == planned
    per_noise, per_scene, outcomes = {}, {}, {}
    for seed in manifest["policy_seeds"]:
        selected = [r for r in rows if r["trial"]["policy_seed"] == seed]
        k = sum(r["success"] for r in selected)
        per_noise[str(seed)] = {"completed": len(selected), "successes": k,
                                "success_rate": k / len(selected) if selected else None,
                                "wilson95": list(wilson(k, len(selected))) if selected else None}
    for scene_id in range(manifest["n_scenes"]):
        selected = [r for r in rows if r["trial"]["scene_id"] == scene_id]
        per_scene[str(scene_id)] = {"completed": len(selected), "successes": sum(r["success"] for r in selected),
                                    "errors": sum(r["status"] == "error" for r in selected)}
        outcomes[scene_id] = {str(r["trial"]["policy_seed"]): float(r["success"]) for r in selected}
    valid = [r for r in rows if r["status"] != "error"]
    intervals = list(scene_bootstrap_ci(outcomes, seed=manifest["scene_seed"])) if complete else None
    rates = [v["success_rate"] for v in per_noise.values() if v["success_rate"] is not None]
    summary = {"planned": planned, "completed": len(rows), "pending": planned - len(rows),
               "successes": successes, "failures": counts["failure"], "errors": counts["error"],
               "success_rate": successes / planned if complete else None,
               "completed_success_rate": successes / len(rows) if rows else None,
               "scene_bootstrap95": intervals, "per_policy_seed": per_noise, "per_scene": per_scene,
               "noise_rate_std": float(np.std(rates, ddof=1)) if complete and len(rates) > 1 else None,
               "failure_modes": dict(Counter(r["failure_reason"] for r in rows)),
               "physical_rates": {key: sum(r["physical"][key] for r in valid) / len(rows) if rows else None
                                  for key in ("both_pads", "lifted", "secure_grasp", "transported")},
               "release_assist_triggered": sum(r["per_side"]["right"].get("release_assist", {}).get("triggered", False)
                                               for r in valid),
               "mean_duration": float(np.mean([r["duration"] for r in valid])) if valid else None,
               "assists": protocol["assists"]}
    triggered = [r for r in valid if r["per_side"]["right"].get("release_assist", {}).get("triggered")]
    summary["release_assist_success_rate"] = sum(r["success"] for r in triggered) / len(triggered) if triggered else None
    summary["inside_container_rate"] = sum(r["per_side"]["right"]["inside_container"] for r in valid) / len(rows) if rows else None
    summary["ik_failures"] = sum(r["metrics"].get("ik_fail", 0) for r in valid)
    write_json(root / "summary.json", summary)
    flat = []
    for trial in manifest["trials"]:
        row = results.get(trial["trial_id"], {})
        item = {k: trial[k] for k in ("trial_id", "scene_id", "env_seed", "policy_seed")}
        for name in ("cube", "container"):
            item.update({f"{name}_{axis}": value for axis, value in zip(("x", "y", "yaw_deg"), trial["scene"][name])})
        item.update(status=row.get("status", "pending"), success=row.get("success"),
                    duration=row.get("duration"), wall=row.get("wall"), error=row.get("error"),
                    failure_reason=row.get("failure_reason"))
        item.update({key: row.get("physical", {}).get(key) for key in ("both_pads", "lifted", "secure_grasp", "transported")})
        side = row.get("per_side", {}).get("right", {})
        item.update({key: side.get(key) for key in ("inside_container", "released", "settled")})
        item["release_assist_triggered"] = side.get("release_assist", {}).get("triggered")
        item["ik_failures"] = row.get("metrics", {}).get("ik_fail")
        flat.append(item)
    with (root / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    rate = f"{100 * summary['success_rate']:.1f}%" if complete else "待全部轮次完成后统计"
    lines = ["# pick_cube 多轮评测", "", f"计划 {planned} 轮；完成 {len(rows)}；异常 {counts['error']}；待运行 {planned-len(rows)}。",
             f"最终成功率：{rate}（异常计失败）。", f"按场景 bootstrap 的 95% 区间：{intervals}。", "",
             f"辅助设置：`{json.dumps(protocol['assists'], ensure_ascii=False)}`。",
             "真实抓取要求方块抬高超过 2.5 cm，双指各有 >0.1 N 接触，连续保持至少 0.1 s；最终还须释放、入盒、落稳。",
             "这里比较同一权重的推理噪声种子，不代表多个训练种子的模型。", "",
             "| 推理噪声种子 | 完成轮数 | 成功数 | 成功率 |", "|---|---:|---:|---:|"]
    for seed, values in per_noise.items():
        display = f"{values['success_rate']:.1%}" if values["success_rate"] is not None else "—"
        lines.append(f"| {seed} | {values['completed']} | {values['successes']} | {display} |")
    lines += ["", "## 结果分类", "", *[f"- {key}: {count}" for key, count in summary["failure_modes"].items()],
              "", "## 各场景", "", "| 场景 | 完成 | 成功 | 异常 |", "|---|---:|---:|---:|"]
    lines += [f"| {key} | {v['completed']} | {v['successes']} | {v['errors']} |" for key, v in per_scene.items()]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def render_videos(root: Path, cfg: dict, rows: list[dict], mode: str, indices: list[int]) -> None:
    from .env import ServeBreadSim
    from .replay import render_episode_video
    selected = [r for r in rows if r["status"] != "error" and
                (r["trial"]["idx"] in indices or mode == "all" or (mode == "failures" and not r["success"]))]
    if not selected:
        return
    env = ServeBreadSim(cfg, render=False)
    try:
        for row in selected:
            directory = root / "episodes" / row["trial"]["trial_id"]
            if not (directory / "episode.mp4").exists():
                render_episode_video(env, str(directory), fps=10, show_policy=False)
    finally:
        if env.renderer is not None:
            env.renderer.close()


def run_study(root: Path, workers: int, device: str, retry_errors: bool = False) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    _, _, manifest, protocol = load_study(root)
    if device != protocol["device"]:
        raise ValueError("device changed; use a new output directory")
    results = read_results(root, manifest, protocol)
    jobs = [t for t in manifest["trials"] if t["trial_id"] not in results or
            (retry_errors and results[t["trial_id"]]["status"] == "error")]
    if jobs:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = "1"
        print(f"[pick_cube eval] planned={len(manifest['trials'])} remaining={len(jobs)} workers={workers}", flush=True)
        started = time.time()
        try:
            with mp.get_context("spawn").Pool(min(workers, len(jobs)), initializer=_init_worker,
                                               initargs=(str(root), device)) as pool:
                for row in pool.imap_unordered(_run_trial, jobs):
                    print(f"[pick_cube eval] {row['trial']['trial_id']} {row['status']} "
                          f"reason={row['failure_reason']} wall={time.time()-started:.1f}s", flush=True)
        finally:
            summarize(root, manifest, protocol)
    return summarize(root, manifest, protocol)
