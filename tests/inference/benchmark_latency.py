"""Synchronized stage timings and precision comparison using one real deployment."""

import argparse
from copy import deepcopy
from functools import wraps
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import time
from unittest.mock import patch

import draccus
import cv2
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
        "--settings", type=Path, help="JSON list of perception overrides"
    )
    parser.add_argument(
        "--diagnostics", choices=["off", "canvas", "full"], default="off"
    )
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
    if args.diagnostics == "off":
        processor.config.deployment["diagnostic_dir"] = None
    else:
        processor.config.deployment["diagnostic_dir"] = str(
            (args.output / "diagnostics").resolve()
        )
        processor.config.deployment["diagnostic_mode"] = args.diagnostics
    report["object_update_mode"] = args.object_update_mode
    perception = processor.perception
    engine = perception.segmentation
    policy = runner.policy.policy
    from lerobot_policy_humanego import processor_humanego

    if args.observation.is_dir():
        data = {
            eye: cv2.cvtColor(
                cv2.imread(str(args.observation / f"{eye}_rgb.png")), cv2.COLOR_BGR2RGB
            )
            for eye in ("left", "right")
        }
        data["state"] = np.load(args.observation / "state.npy")
    else:
        data = np.load(args.observation)
    features = {
        "observation.images.head_left": data["left"],
        "observation.images.head_right": data["right"],
        "observation.state": data["state"],
    }
    stages = defaultdict(float)

    def timed(fn, name):
        @wraps(fn)
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
        (type(perception.inpainter), "inpaint", "lama"),
        (processor_humanego, "render_scene", "render"),
        (policy, "prepare_image", "policy_image"),
        (policy, "build_ict", "ict"),
        (policy, "compute_anchor_uv", "anchor_uv"),
        (policy, "infer", "policy_ode"),
        (policy, "decode_ee_in_cam", "action_decode"),
    ]
    reference = None
    settings = json.loads(args.settings.read_text()) if args.settings else [{}]
    variants = [(v, s) for s in settings for v in args.variants]
    original_lama = deepcopy(perception.inpainter.opts)
    image_processor = engine.processor.image_processor
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
        names = set()
        for variant_index, (precision, settings) in enumerate(variants):
            variant = precision
            if args.settings:
                variant = settings.get(
                    "name", f"setting_{variant_index // len(args.variants)}"
                )
                if len(args.variants) > 1:
                    variant += "_" + precision.replace("/", "_")
            if variant in names:
                raise ValueError(f"Duplicate benchmark variant name: {variant}")
            names.add(variant)
            setup_start = time.perf_counter()
            engine.dino_precision, engine.sam_precision = precision.split("/")
            if args.settings:
                from lerobot_policy_humanego.perception import Inpainter

                processor.config.perception["lama"] = {
                    **original_lama,
                    **settings.get("lama", {}),
                }
                # Release the previous CUDA session before creating another.
                if perception.inpainter.executor is not None:
                    perception.inpainter.executor.shutdown()
                perception.inpainter = None
                perception.inpainter = Inpainter(processor.config)
                image_processor.size["shortest_edge"] = settings.get(
                    "dino_shortest_edge", 800
                )
                engine.joint_object_detection = settings.get(
                    "joint_object_detection", False
                )
                engine.set_dino_kernel(settings.get("dino_cuda_kernel_revision"))
                # Replacing the session also discards its startup warmup/capture.
                zeros = np.zeros((480, 640, 3), np.uint8)
                perception.inpainter.inpaint(
                    zeros, np.ones((480, 640), bool), np.zeros((480, 640), bool)
                )
                engine.segment(
                    zeros,
                    processor.config.perception["object_prompts"],
                    processor.config.perception["erase_prompt"],
                )
                torch.cuda.synchronize()
            setup_s = time.perf_counter() - setup_start
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
                        valid_points = deepcopy(perception.diagnostics["valid_points"])
                        initial_masks = {
                            k: v.copy() for k, v in perception.masks.items()
                        }
                        initial_poses = {
                            k: v.T_in_cam.copy()
                            for k, v in processor.scene.objects.items()
                        }
                    if index >= 2:
                        samples["total"].append(elapsed)
                        samples["other"].append(elapsed - sum(stages.values()))
                        for name, duration in stages.items():
                            samples[name].append(duration)
                actions = result.canonical_actions.cpu().numpy()
                masks = {k: v.copy() for k, v in perception.masks.items()}
                clean = perception.erase_arms(
                    processor_humanego.observation_from_batch(
                        runner._batch(snapshot)
                    ).frame
                )
                if reference is None:
                    reference = (
                        actions.copy(),
                        masks,
                        clean.copy(),
                        initial_masks,
                        initial_poses,
                    )
                position = np.linalg.norm(actions[:, :3] - reference[0][:, :3], axis=1)
                comparisons = {
                    "initial_mask_iou": {
                        k: float(
                            np.logical_and(v, reference[3][k]).sum()
                            / max(1, np.logical_or(v, reference[3][k]).sum())
                        )
                        for k, v in initial_masks.items()
                    },
                    "object_translation_delta_m": {
                        k: float(np.linalg.norm(v[:3, 3] - reference[4][k][:3, 3]))
                        for k, v in initial_poses.items()
                    },
                    "clean_mae": float(
                        np.abs(clean.astype(float) - reference[2]).mean()
                    ),
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
                    "variant_setup_s": setup_s,
                    "timings": {k: summary(v) for k, v in samples.items()},
                    "comparison_to_first_variant": comparisons,
                    "valid_points": valid_points,
                    "settings": settings,
                    "precision": precision,
                }
                np.savez(
                    args.output / (variant.replace("/", "_") + ".npz"),
                    actions=actions,
                    **masks,
                )
                cv2.imwrite(
                    str(args.output / f"{variant.replace('/', '_')}_clean.png"), clean
                )
            except (RuntimeError, ValueError) as exc:
                report["variants"][variant] = {"error": str(exc)}
            print(variant, json.dumps(report["variants"][variant]), flush=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
