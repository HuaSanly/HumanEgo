# -*- coding: utf-8 -*-
"""
Parallel batch evaluation: N episodes (randomized or given initial conditions)
spread over worker processes, each holding its own simulator + policy.

    from humanego_sim.evaluate import evaluate
    stats = evaluate(cfg_overrides, n_episodes=32, workers=8, seed=0)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from typing import Dict, List, Optional

import numpy as np

from .stats import fmt_rate

_W = {}


def _worker_init(cfg_overrides: dict, device: str, intro: bool = False, viewer: bool = False):
    os.environ.setdefault("MUJOCO_GL", "egl")
    # one thread per worker: OpenCV / torch / BLAS thread pools oversubscribe the CPU otherwise
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    import warnings
    warnings.filterwarnings("ignore")
    from .env import load_config
    from .runner import EpisodeRunner
    cfg = load_config(None, cfg_overrides)
    viewer = viewer or cfg.get("execution_mode", "sync") == "realtime"
    _W["runner"] = EpisodeRunner(cfg, device=device)
    _W["cfg"] = cfg
    _W["cfg_overrides"] = cfg_overrides
    _W["device"] = device
    _W["var_key"] = ""
    _W["intro"] = intro
    _W["viewer"] = viewer


def variant_overrides(scene: Optional[dict], base_overrides: dict):
    """Config overrides a scene variation needs a rebuilt model for (bread size, table height);
    returns (key, overrides). key == "" means the base model."""
    from .env import deep_update, load_config
    var = (scene or {}).get("_var") or {}
    ov = dict(var.get("cfg") or {})
    dz = float(var.get("table_dz", 0.0))
    if dz:
        base = load_config(None, base_overrides)
        ov = deep_update(ov, {"rig": {"cam_height": round(float(base["rig"]["cam_height"]) - dz, 4)}})
    return (json.dumps(ov, sort_keys=True) if ov else ""), ov


def apply_runtime_variation(env, scene: Optional[dict]) -> None:
    """The scene-variation settings that need no rebuild: lights, distractors, and the robot's
    belief about the table height (it keeps the calibrated, nominal camera pose)."""
    from .robustness import set_lights
    var = (scene or {}).get("_var") or {}
    dz = float(var.get("table_dz", 0.0))
    if dz:
        # table dz higher relative to the rig than believed: the true rig is dz lower (cam_height - dz),
        # the robot computes in the world frame of the nominal rig
        T = np.array(env.meta["T_cam_in_world"], float)
        T[2, 3] += dz
        env.T_cam_in_world, env.T_world_in_cam = T, np.linalg.inv(T)
    set_lights(env, var.get("light"))
    env.distractors = list(var.get("distractors") or [])


def apply_scene_variation(runner, scene: Optional[dict], base_overrides: dict, cur_key: str, device: str = "cuda"):
    """Set up the simulator for a scene with sampling.apply_variation settings (scene["_var"]).
    Size / table-height variation needs a rebuilt model (~0.5 s); the policy is kept. Returns
    (runner, key of the built variant)."""
    from .env import ServeBreadSim, deep_update, load_config
    from .runner import EpisodeRunner
    key, ov = variant_overrides(scene, base_overrides)
    if key != cur_key:
        cfg = load_config(None, deep_update(base_overrides, ov) if ov else base_overrides)
        if runner.env.renderer is not None:
            runner.env.renderer.close()
        runner = EpisodeRunner(cfg, policy=runner.policy, device=device, env=ServeBreadSim(cfg))
    apply_runtime_variation(runner.env, scene)
    return runner, key


def _worker_run(job):
    idx, scene, seed, log_dir, log_images = job
    try:
        if (scene and "_var" in scene) or _W.get("var_used"):
            _W["runner"], _W["var_key"] = apply_scene_variation(_W["runner"], scene, _W["cfg_overrides"],
                                                                _W.get("var_key", ""), _W["device"])
            _W["var_used"] = True
        from .viewer import show_simulation
        with show_simulation(_W["runner"].env, _W.get("viewer", False)):
            r = _W["runner"].run(scene=scene, seed=seed, log_dir=log_dir, save_policy_images=log_dir is not None and log_images,
                                 verbose=False, intro=_W.get("intro", False))
        return {"idx": idx, "seed": seed, "scene": scene, "success": r.success, "per_side": r.per_side,
                "duration": r.duration, "wall": r.wall_time, "metrics": r.metrics}
    except Exception as e:  # keep the batch alive, report the failure
        return {"idx": idx, "seed": seed, "scene": scene, "success": False, "error": repr(e),
                "trace": traceback.format_exc()[-2000:]}


def evaluate(cfg_overrides: Optional[dict] = None, n_episodes: int = 16, workers: int = 6, seed: int = 0,
             scenes: Optional[List[dict]] = None, difficulty: float = 1.0, device: str = "cuda",
             log_root: Optional[str] = None, verbose: bool = True, intro: bool = False,
             log_images: bool = True, viewer: bool = False) -> Dict:
    from .env import load_config
    from .sampling import sample_scene
    cfg_overrides = cfg_overrides or {}
    cfg = load_config(None, cfg_overrides)
    viewer = viewer or cfg.get("execution_mode", "sync") == "realtime"
    rng = np.random.default_rng(seed)
    if scenes is None:
        scenes = [sample_scene(rng, cfg.get("mode", "bimanual"), difficulty) for _ in range(n_episodes)]
    jobs = []
    for i, sc in enumerate(scenes):
        ld = os.path.join(log_root, f"ep{i:03d}") if log_root else None
        jobs.append((i, sc, seed * 1000 + i, ld, log_images))
    t0 = time.time()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"           # inherited by the spawned workers
    if viewer:
        if workers != 1:
            raise ValueError("--viewer requires --workers 1")
        _worker_init(cfg_overrides, device, intro, viewer=True)
        try:
            results = [_worker_run(job) for job in jobs]
        finally:
            if _W["runner"].env.renderer is not None:
                _W["runner"].env.renderer.close()
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(jobs)), initializer=_worker_init,
                      initargs=(cfg_overrides, device, intro)) as pool:
            results = list(pool.imap_unordered(_worker_run, jobs))
    results.sort(key=lambda r: r["idx"])
    stats = summarize(results)
    stats["wall"] = time.time() - t0
    if verbose:
        ci = stats["ci"]
        print(f"[eval] per-arm {fmt_rate(ci['per_arm']['k'], ci['per_arm']['n'])} "
              f"both {fmt_rate(ci['both']['k'], ci['both']['n'])} (Wilson 95 %) "
              f"side={stats['side_success']} fails={stats['failure_modes']} wall={stats['wall']:.0f}s", flush=True)
    if log_root:
        os.makedirs(log_root, exist_ok=True)
        with open(os.path.join(log_root, "results.json"), "w") as f:
            json.dump({"stats": stats, "results": results, "overrides": cfg_overrides}, f, indent=1)
    stats["results"] = results
    return stats


def classify_failure(side_res: dict) -> str:
    if side_res.get("success"):
        return "success"
    if not side_res.get("grasped"):
        return "never_closed"
    lifted = side_res.get("bread_max_z", 1.0) - side_res.get("table_surface_z", 0.0) > 0.07
    d = side_res.get("dist_to_plate_center", 1.0)
    if not lifted:
        return "grasp_missed"            # gripper closed but the bread never left the table
    if d > 0.16:
        return "dropped_in_transit"
    if side_res.get("done_reason") == "timeout":
        return "timeout"
    return "rim_or_edge"


def summarize(results: List[dict]) -> Dict:
    from .stats import summarize_ci
    n = len(results)
    ns = sum(1 for r in results if r.get("success"))
    side_succ, modes = {}, {}
    dists = []
    for r in results:
        if "error" in r:
            modes["error"] = modes.get("error", 0) + 1
            continue
        for s, v in r["per_side"].items():
            side_succ.setdefault(s, []).append(1.0 if v["success"] else 0.0)
            m = classify_failure(v)
            modes[m] = modes.get(m, 0) + 1
            dists.append(v.get("dist_to_plate_center", 1.0))
    return {
        "n": n, "n_success": ns, "success_rate": ns / max(n, 1),
        "side_success": {s: round(float(np.mean(v)), 3) for s, v in side_succ.items()},
        "per_side_success_rate": float(np.mean([x for v in side_succ.values() for x in v])) if side_succ else 0.0,
        "failure_modes": modes,
        "mean_dist_to_plate": float(np.mean(dists)) if dists else 1.0,
        "ci": summarize_ci(results),
    }
