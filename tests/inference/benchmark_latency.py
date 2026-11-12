"""Synchronized stage timings and precision comparison using one real deployment."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import time
from unittest.mock import patch

import draccus
import numpy as np
import torch
from lerobot.inference import ExecutionMode, ObservationSnapshot
from lerobot.remote_inference import ServerConfig, load_deployment
from lerobot.utils import register_third_party_plugins


def summary(values):
    return {
        "mean_ms": float(np.mean(values) * 1000),
        "p95_ms": float(np.percentile(values, 95) * 1000),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--object-update-mode", choices=["continuous", "events"], default="continuous"
    )
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["float32/float32", "float16/float16", "bfloat16/bfloat16"],
    )
    args = parser.parse_args()
    register_third_party_plugins()
    with (args.bundle / "server.yaml").open() as handle:
        cfg = draccus.load(ServerConfig, handle)
    cfg.model.repo_or_path = str(args.bundle.resolve())
    start = time.perf_counter()
    runner, _ = load_deployment(cfg)
    torch.cuda.synchronize()
    report = {
        "initialization_s": time.perf_counter() - start,
        "gpu": torch.cuda.get_device_name(),
        "variants": {},
    }
    processor = next(
        s.processor for s in runner.preprocessor.steps if hasattr(s, "processor")
    )
    processor.config.perception["object_update_mode"] = args.object_update_mode
    report["object_update_mode"] = args.object_update_mode
    perception = processor.perception
    engine = perception.segmentation
    policy = runner.policy.policy
    from lerobot_policy_humanego import processor_humanego

    data = np.load(args.observation)
    features = {
        "observation.images.head_left": data["left"],
        "observation.images.head_right": data["right"],
        "observation.state": data["state"],
    }
    stages = defaultdict(float)

    def timed(fn, name):
        def call(*a, **kw):
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = fn(*a, **kw)
            torch.cuda.synchronize()
            stages[name] += time.perf_counter() - start
            return value

        return call

    targets = [
        (engine.sam, "set_image", "sam_encode"),
        (engine.model, "forward", "dino"),
        (engine.sam, "predict", "sam_masks"),
        (perception.stereo, "estimate", "stereo"),
        (perception.stereo, "points_from_mask", "keypoints"),
        (perception.inpainter, "inpaint", "lama"),
        (processor_humanego, "render_scene", "render"),
        (policy, "prepare_image", "policy_image"),
        (policy, "build_ict", "ict"),
        (policy, "compute_anchor_uv", "anchor_uv"),
        (policy, "infer", "policy_ode"),
        (policy, "decode_ee_in_cam", "action_decode"),
    ]
    reference = None
    args.output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for obj, attr, name in targets:
            stack.enter_context(
                patch.object(obj, attr, timed(getattr(obj, attr), name))
            )
        # __call__ is looked up on the class, so wrap the instance at its owner instead.
        stack.enter_context(
            patch.object(
                engine, "processor", timed(engine.processor, "dino_preprocess")
            )
        )
        for variant in args.variants:
            engine.dino_precision, engine.sam_precision = variant.split("/")
            samples = defaultdict(list)
            runner.reset(full=True)
            try:
                for index in range(args.requests + 2):
                    stages.clear()
                    snapshot = ObservationSnapshot(
                        features, time.monotonic(), "pick_cube"
                    )
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = runner.predict(snapshot, mode=ExecutionMode.CHUNK)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if index == 0:
                        initialization_ms = elapsed * 1000
                    if index >= 2:
                        samples["total"].append(elapsed)
                        samples["other"].append(elapsed - sum(stages.values()))
                        for name, duration in stages.items():
                            samples[name].append(duration)
                actions = result.canonical_actions.cpu().numpy()
                masks = {k: v.copy() for k, v in perception.masks.items()}
                if reference is None:
                    reference = actions.copy(), masks
                position = np.linalg.norm(actions[:, :3] - reference[0][:, :3], axis=1)
                comparisons = {
                    "mask_iou": {
                        k: float(
                            np.logical_and(v, reference[1][k]).sum()
                            / max(1, np.logical_or(v, reference[1][k]).sum())
                        )
                        for k, v in masks.items()
                    },
                    "position_max_m": float(position.max()),
                    "position_mean_m": float(position.mean()),
                    "rotation6d_max_abs": float(
                        np.abs(actions[:, 3:9] - reference[0][:, 3:9]).max()
                    ),
                    "grasp_max_abs": float(
                        np.abs(actions[:, 9] - reference[0][:, 9]).max()
                    ),
                }
                report["variants"][variant] = {
                    "initial_scene_ms": initialization_ms,
                    "timings": {k: summary(v) for k, v in samples.items()},
                    "comparison_to_first_variant": comparisons,
                    "valid_points": perception.diagnostics["valid_points"],
                }
                np.savez(
                    args.output / (variant.replace("/", "_") + ".npz"),
                    actions=actions,
                    **masks,
                )
            except (RuntimeError, ValueError) as exc:
                report["variants"][variant] = {"error": str(exc)}
            print(variant, json.dumps(report["variants"][variant]), flush=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
