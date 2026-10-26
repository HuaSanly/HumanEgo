"""Real CUDA perception + native runner + original-policy numerical comparison."""

import argparse
import json
import logging
from pathlib import Path
import time

import cv2
import draccus
import numpy as np
import torch
from lerobot.inference import ObservationSnapshot, ExecutionMode
from lerobot.remote_inference import load_deployment, ServerConfig
from lerobot.utils import register_third_party_plugins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    register_third_party_plugins()
    with (args.bundle / "server.yaml").open() as f:
        cfg = draccus.load(ServerConfig, f)
    cfg.model.repo_or_path = str(args.bundle.resolve())
    start = time.perf_counter()
    runner, identity = load_deployment(cfg)
    torch.cuda.synchronize()
    initialization = time.perf_counter() - start
    data = np.load(args.observation)
    features = {
        "observation.images.head_left": data["left"],
        "observation.images.head_right": data["right"],
        "observation.state": data["state"],
    }
    durations = []
    for index in range(args.requests):
        snapshot = ObservationSnapshot(
            features, time.monotonic(), "pick_cube", observation_id=str(index)
        )
        start = time.perf_counter()
        result = runner.predict(snapshot, mode=ExecutionMode.CHUNK)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        assert result.canonical_actions.shape == (50, 10)
        assert torch.isfinite(result.canonical_actions).all()
        print("request", index, "seconds", durations[-1], flush=True)
    adapter = runner.policy
    step = next(
        step for step in runner.preprocessor.steps if hasattr(step, "processor")
    )
    processor = step.processor
    assert processor.scene is not None
    assert processor.perception.diagnostics["inpainting"] == "inpainted"
    assert all(
        n >= 8 for n in processor.perception.diagnostics["valid_points"].values()
    )
    assert processor.perception.masks["arms"].any()
    # Same prepared observation and saved noise; direct calls use the single owned original policy.
    from lerobot_policy_humanego.processor_humanego import observation_from_batch

    prepared = processor(observation_from_batch(runner._batch(snapshot)))
    wrapped = adapter.predict_action_chunk(prepared)[0].cpu().numpy()
    p = adapter.policy
    scene = prepared.scene
    x = p.prepare_image(prepared.clean_bgr)
    ict, mask = p.build_ict(
        {"right": scene.hand_in_cam},
        {"right": scene.gripper},
        scene.objects,
        reference_anchor=scene.reference_anchor,
    )
    h, w = prepared.clean_bgr.shape[:2]
    uv = p.compute_anchor_uv(scene.objects["obj1"], prepared.K, w, h)
    trajectory, done = p.infer(x, ict, mask, uv)
    pos, rot, grasp = trajectory["right"]
    poses = np.stack(
        [
            p.decode_ee_in_cam(a, b, scene.reference_anchor, adapter.T_align)
            for a, b in zip(pos, rot)
        ]
    )
    expected = np.concatenate(
        [poses[:, :3, 3], poses[:, :3, :2].reshape(50, 6), grasp], axis=-1
    )
    np.testing.assert_array_equal(wrapped, expected.astype(np.float32))
    np.testing.assert_array_equal(wrapped[:, -1], grasp[:, 0])
    assert np.any((wrapped[:, -1] > 0) & (wrapped[:, -1] < 1))
    assert adapter.model is p.model
    assert p.noise is adapter.fixed_noise
    protected = processor.perception.masks["obj1"] | processor.perception.masks["obj2"]
    cleaned = processor.perception.erase_arms(prepared.frame)
    original = cv2.cvtColor(prepared.frame.left_rgb, cv2.COLOR_RGB2BGR)
    np.testing.assert_array_equal(cleaned[protected], original[protected])
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / "clean.png"), cleaned)
    cv2.imwrite(str(args.output / "canvas.png"), prepared.clean_bgr)
    cv2.imwrite(
        str(args.output / "model.png"), cv2.resize(prepared.clean_bgr, (320, 240))
    )
    cv2.imwrite(
        str(args.output / "arms.png"),
        processor.perception.masks["arms"].astype(np.uint8) * 255,
    )
    np.save(args.output / "actions.npy", wrapped)
    # After startup reset, wire-level warmup text/zero tensors receive normal validation.
    zeros = {f.name: np.zeros(f.shape, dtype=f.dtype) for f in cfg.features}
    try:
        runner.predict(
            ObservationSnapshot(zeros, 0, "warmup"), mode=ExecutionMode.CHUNK
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Startup probe remained reachable after reset")
    from unittest.mock import patch

    dual = data["left"].copy()
    dual[300:, :220] = data["left"][300:, -220:][:, ::-1]
    engine = processor.perception.segmentation
    with patch.object(engine.sam, "set_image", wraps=engine.sam.set_image) as encode:
        dual_masks = engine.segment(
            dual,
            processor.config.perception["object_prompts"],
            processor.config.perception["erase_prompt"],
        )
        assert encode.call_count == 1
    assert dual_masks["arms"][300:, :220].sum() > 1000
    assert dual_masks["arms"][300:, -220:].sum() > 1000
    cv2.imwrite(
        str(args.output / "dual_arm_mask.png"),
        dual_masks["arms"].astype(np.uint8) * 255,
    )
    noise = p.noise.clone()
    runner.reset(full=True)
    assert processor.scene is None
    assert processor.perception.masks is None
    torch.testing.assert_close(p.noise, noise, rtol=0, atol=0)
    report = dict(
        identity=identity,
        initialization_s=initialization,
        request_seconds=durations,
        mean_s=float(np.mean(durations)),
        p95_s=float(np.percentile(durations, 95)),
        original_policy_max_error=float(np.max(np.abs(wrapped - expected))),
        done=done,
        grasp_range=[float(grasp.min()), float(grasp.max())],
        gpu=torch.cuda.get_device_name(),
        checks=[
            "real DINO/SAM2/LaMa CUDA",
            "metric object points",
            "nonempty arm mask",
            "protected object pixels",
            "dual-arm union on mirrored test image",
            "one SAM encoding",
            "original policy equality",
            "continuous grasp",
            "startup probe closes",
            "reset keeps noise",
        ],
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
