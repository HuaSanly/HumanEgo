"""Independent, prefix-stable pick_cube layouts and policy-noise trials."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from .cameras import get_camera
from .robots import get_robot
from .scene import rig_geometry


def _corners(obj: dict, pose: list) -> np.ndarray:
    half = np.asarray(obj["size"], float) / 2
    if obj["kind"] == "container":
        half[:2] += float(obj.get("rim", 0.0))
        half[1] += 0.016  # include the handles in table and view checks
    points = np.array([[x, y, z] for x in (-half[0], half[0])
                       for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    yaw = np.deg2rad(pose[2])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    points[:, :2] = points[:, :2] @ rotation.T + pose[:2]
    points[:, 2] += float(obj["size"][2]) / 2 + 0.0008
    return points


def layout_reason(scene: dict, cfg: dict, sampling: dict) -> str:
    """Geometric screening only; never inspect the policy outcome to accept a scene."""
    cam = replace(get_camera(cfg["camera"]), **cfg.get("camera_intrinsics", {}))
    world_to_cam = np.linalg.inv(rig_geometry(cfg, ["right"]).T_cam_in_world)
    table, workspace = cfg["rig"]["table"], sampling["workspace"]
    margin = sampling["table_margin"]
    points = {}
    for name in ("cube", "container"):
        corners = _corners(cfg["scene"]["objects"][name], scene[name])
        points[name] = corners
        if (corners[:, 0].min() < table["x_min"] + margin
                or corners[:, 0].max() > table["x_max"] - margin
                or np.abs(corners[:, 1]).max() > table["half_width"] - margin):
            return "off_table"
        if any(corners[:, axis].min() < workspace[key][0] or corners[:, axis].max() > workspace[key][1]
               for axis, key in enumerate(("x", "y"))):
            return "outside_workspace"
        camera_points = corners @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
        if np.any(camera_points[:, 2] < cam.depth_min) or np.any(camera_points[:, 2] > cam.depth_max):
            return "outside_depth_range"
        uv = camera_points @ cam.K.T
        uv = uv[:, :2] / uv[:, 2:3]
        pad = sampling["camera_margin_px"]
        if np.any(uv < pad) or np.any(uv > np.array([cam.width, cam.height]) - 1 - pad):
            return "outside_camera"
    # Separating-axis test of the full rectangular footprints, with clearance.
    gap = sampling["min_object_gap"]
    for name in ("cube", "container"):
        angle = np.deg2rad(scene[name][2])
        for axis in (np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])):
            a, b = points["cube"][:, :2] @ axis, points["container"][:, :2] @ axis
            if a.max() + gap <= b.min() or b.max() + gap <= a.min():
                return ""
    return "objects_overlap"


def validate_sampling(sampling: dict) -> None:
    for name in ("cube", "container"):
        jitter = np.asarray(sampling[f"{name}_xy_jitter"], float)
        if jitter.shape != (2,) or not np.isfinite(jitter).all() or np.any(jitter < 0):
            raise ValueError(f"{name}_xy_jitter must contain two finite nonnegative widths")
        angle = sampling[f"{name}_yaw_jitter_deg"]
        if not np.isfinite(angle) or angle < 0:
            raise ValueError(f"{name}_yaw_jitter_deg must be finite and nonnegative")
    for key in ("table_margin", "min_object_gap", "camera_margin_px"):
        if not np.isfinite(sampling[key]) or sampling[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for bounds in sampling["workspace"].values():
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("workspace bounds must be finite increasing pairs")
    if not isinstance(sampling["max_attempts_per_scene"], int) or sampling["max_attempts_per_scene"] < 1:
        raise ValueError("max_attempts_per_scene must be a positive integer")


def make_trials(cfg: dict, sampling: dict, n: int, scene_seed: int, policy_seeds: list[int]) -> dict:
    validate_sampling(sampling)
    if n < 1 or not policy_seeds or len(set(policy_seeds)) != len(policy_seeds):
        raise ValueError("n must be positive and policy seeds must be nonempty and unique")
    if any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in [scene_seed, *policy_seeds]):
        raise ValueError("seeds must be integers in [0, 2**32)")
    rejected, trials = Counter(), []
    for scene_id in range(n):
        rng = np.random.default_rng([scene_seed, scene_id, 11])
        for _ in range(sampling["max_attempts_per_scene"]):
            scene = {}
            for name in ("cube", "container"):
                pose = np.asarray(cfg["scene"]["objects"][name]["pose"], float).copy()
                widths = np.r_[sampling[f"{name}_xy_jitter"], sampling[f"{name}_yaw_jitter_deg"]]
                scene[name] = (pose + rng.uniform(-widths, widths)).tolist()
            reason = layout_reason(scene, cfg, sampling)
            if not reason:
                break
            rejected[reason] += 1
        else:
            raise ValueError(f"cannot sample scene {scene_id}; rejected={dict(rejected)}; check sampling bounds")
        env_seed = int(np.random.SeedSequence([scene_seed, scene_id, 29]).generate_state(1)[0])
        for policy_seed in policy_seeds:
            idx = len(trials)
            trials.append({"trial_id": f"trial_{idx:06d}", "idx": idx, "scene_id": scene_id,
                           "scene": scene, "env_seed": env_seed, "policy_seed": policy_seed})
    return {"schema_version": 1, "n_scenes": n, "scene_seed": scene_seed,
            "policy_seeds": policy_seeds, "rejections": dict(rejected), "trials": trials}


def validate_trials(manifest: dict, cfg: dict, sampling: dict) -> None:
    """Reject edited or incompatible manifests rather than silently regenerating them."""
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported trials schema")
    n, seeds = manifest["n_scenes"], manifest["policy_seeds"]
    if n < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("invalid scene count or policy seeds")
    if len(manifest["trials"]) != n * len(seeds):
        raise ValueError("trial count does not match scenes × policy seeds")
    scenes = {}
    for idx, trial in enumerate(manifest["trials"]):
        scene_id = idx // len(seeds)
        if (trial["trial_id"] != f"trial_{idx:06d}" or trial["idx"] != idx or trial["scene_id"] != scene_id
                or trial["policy_seed"] != seeds[idx % len(seeds)]):
            raise ValueError(f"invalid trial identity at index {idx}")
        if set(trial["scene"]) != {"cube", "container"}:
            raise ValueError("a pick_cube trial needs exactly cube and container")
        for pose in trial["scene"].values():
            if np.asarray(pose).shape != (3,) or not np.isfinite(pose).all():
                raise ValueError("object poses must be finite [x, y, yaw_deg] triples")
        for seed in (trial["env_seed"], trial["policy_seed"], manifest["scene_seed"]):
            if not isinstance(seed, int) or not 0 <= seed < 2**32:
                raise ValueError("trial seeds must be integers in [0, 2**32)")
        identity = (trial["scene"], trial["env_seed"])
        if scene_id in scenes and scenes[scene_id] != identity:
            raise ValueError("repeated scenes must keep identical poses and environment seed")
        if scene_id not in scenes:
            reason = layout_reason(trial["scene"], cfg, sampling)
            if reason:
                raise ValueError(f"scene {scene_id} is invalid: {reason}")
            scenes[scene_id] = identity
