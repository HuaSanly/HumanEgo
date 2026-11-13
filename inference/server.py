"""Serve HumanEgo directly from a project inference configuration."""

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import signal
import time
from uuid import uuid4

import numpy as np
from lerobot.inference import ExecutionMode, FeatureSpec, ObservationSnapshot, PolicyRunner
from lerobot.policies import RTCConfig
from lerobot.remote_inference import (
    ExecutionConfig,
    ModelConfig,
    PolicyServer,
    ServerConfig,
)
from lerobot.transport import ZenohConfig
from lerobot.utils import init_logging

from .config import InferenceConfig, load_config, policy_files, validate_runtime_config
from .configuration_humanego import HumanEgoConfig
from .modeling_humanego import HumanEgoPolicy
from .processor_humanego import make_humanego_pre_post_processors
from .runtime_logging import LoggedSessionWorker, LoggedZenohTransport

logger = logging.getLogger(__name__)
SEMANTICS = "humanego.right-ee.left-rectified-optical.m.row-major-rotation6d.continuous-grasp.v1"
COMPONENTS = ("x", "y", "z", "R00", "R01", "R10", "R11", "R20", "R21", "grasp")


def server_config(config: InferenceConfig) -> ServerConfig:
    settings = config.deployment
    features = [
        FeatureSpec(
            name=f"observation.images.head_{side}",
            shape=(*config.camera["image_size_hw"], 3),
            dtype="uint8",
            kind="rgb",
            semantics=f"{side} rectified optical RGB; synchronized stereo640x480",
        )
        for side in ("left", "right")
    ]
    features.append(
        FeatureSpec(
            name="observation.state",
            shape=(10,),
            dtype="float32",
            names=COMPONENTS,
            semantics="measured right EE; left rectified optical; metres; R[:, :2].reshape(-1); gripper 0=open 1=closed",
        )
    )
    return ServerConfig(
        deployment=settings["name"],
        model=ModelConfig(
            repo_or_path=config.policy["ckpt"], device=config.policy["device"]
        ),
        execution=ExecutionConfig(
            supported_modes=[settings["mode"]],
            action_fps=settings["action_fps"],
            rtc=RTCConfig(enabled=False),
            action_deadline_s=settings.get("action_deadline_s", 30.0),
            idle_timeout_s=settings.get("idle_timeout_s", 60.0),
            warmup_calls=settings.get("warmup_calls", 2),
        ),
        zenoh=ZenohConfig(listen_endpoints=[settings["listen_endpoint"]]),
        semantics=SEMANTICS,
        robot_type="humanego_brx",
        features=features,
        action_feature=FeatureSpec(
            name="action",
            shape=(10,),
            dtype="float32",
            names=COMPONENTS,
            semantics="absolute right EE target; left rectified optical; metres; R[:, :2].reshape(-1); raw policy sigmoid grasp",
        ),
        log_level=settings.get("log_level", "INFO"),
    )


def load_runner(
    config: InferenceConfig, cfg: ServerConfig, *, run_dir: Path | None = None
) -> tuple[PolicyRunner, str]:
    validate_runtime_config(config)
    files = policy_files(config)
    file_versions = {role: path.stat() for role, path in files.items()}
    hashes = {}
    for role, path in files.items():
        with path.open("rb") as stream:
            hashes[role] = hashlib.file_digest(stream, "sha256").hexdigest()

    native = HumanEgoConfig(device=cfg.model.device, push_to_hub=False)
    if native.device != cfg.model.device:
        raise ValueError(f"Configured policy device is unavailable: {cfg.model.device}")
    policy = HumanEgoPolicy(native, config)
    # A training job may replace latest.pt while we hash or load it.
    for role, path in files.items():
        before, after = file_versions[role], path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise RuntimeError(f"Policy file changed during startup: {path}")

    noise_hash = hashlib.sha256(policy.fixed_noise.cpu().numpy().tobytes()).hexdigest()
    fingerprint = {
        "files": hashes,
        "fixed_noise_sha256": noise_hash,
        "intrinsics": config.intrinsics,
        "extrinsics": config.extrinsics,
        "perception": config.perception,
        "num_inference_steps": config.policy["num_inference_steps"],
        "mode": config.deployment["mode"],
        "action_fps": cfg.execution.action_fps,
        "robot_type": cfg.robot_type,
        "semantics": cfg.semantics,
        "features": [asdict(feature) for feature in cfg.features],
        "action_feature": asdict(cfg.action_feature),
    }
    identity = "sha256:" + hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    if run_dir is not None:
        snapshot = asdict(config)
        snapshot["source"] = str(config.source)
        snapshot["server"] = asdict(cfg)
        snapshot["artifact_identity"] = identity
        snapshot["fixed_noise_sha256"] = noise_hash
        snapshot["policy_files"] = {
            role: {"path": str(path), "sha256": hashes[role]}
            for role, path in files.items()
        }
        path = run_dir / "config.json"
        path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        logger.info("config_snapshot path=%s artifact=%s", path, identity)

    pre, post = make_humanego_pre_post_processors(config)
    runner = PolicyRunner(
        policy,
        pre,
        post,
        action_interval=1 / cfg.execution.action_fps,
        features=tuple(cfg.features),
        action_feature=cfg.action_feature,
        modes=(ExecutionMode.CHUNK,),
        robot_type=cfg.robot_type,
        language_enabled=cfg.language.enabled,
        max_text_input=cfg.language.max_input_chars,
        max_text_output=cfg.language.max_output_chars,
    )
    warmup = ObservationSnapshot(
        {
            feature.name: np.zeros(feature.shape, dtype=feature.dtype)
            for feature in cfg.features
        },
        0.0,
        "warmup",
        observation_id="warmup",
    )
    try:
        for _ in range(cfg.execution.warmup_calls):
            runner.predict(warmup, mode=ExecutionMode.CHUNK)
    finally:
        runner.reset(full=True)
    return runner, identity


def serve(config: InferenceConfig) -> None:
    cfg = server_config(config)
    run_name = f"run_{time.time_ns()}_{uuid4().hex}"
    run_dir = Path(config.deployment["log_dir"]) / run_name
    run_dir.mkdir(parents=True)
    init_logging(
        log_file=run_dir / "server.log",
        console_level=cfg.log_level,
        file_level=cfg.log_level,
    )
    logger.info(
        "server_loading deployment=%s config=%s checkpoint=%s device=%s",
        cfg.deployment,
        config.source,
        cfg.model.repo_or_path,
        cfg.model.device,
    )
    try:
        runner, identity = load_runner(config, cfg, run_dir=run_dir)
    except Exception:
        logger.exception("server_initialization_failed config=%s", config.source)
        raise

    trace_root = None
    diagnostic_dir = config.deployment.get("diagnostic_dir")
    if diagnostic_dir and config.deployment.get("diagnostic_mode") == "full":
        trace_root = Path(diagnostic_dir) / run_name
        logger.info("full_diagnostics root=%s", trace_root)
    transport = LoggedZenohTransport(cfg.zenoh, trace_root=trace_root)
    worker = LoggedSessionWorker(
        runner,
        trace_root=trace_root,
        deployment=cfg.deployment,
        artifact_identity=identity,
        semantics=cfg.semantics,
        action_deadline_s=cfg.execution.action_deadline_s,
        idle_timeout_s=cfg.execution.idle_timeout_s,
        max_input_chars=cfg.language.max_input_chars,
        max_output_chars=cfg.language.max_output_chars,
    )
    server = PolicyServer(worker, transport)
    logger.info(
        "server_warmed deployment=%s execution_steps=%d action_hz=%.1f",
        cfg.deployment,
        runner.capabilities.execution_steps,
        1 / runner.capabilities.action_interval,
    )
    previous_handlers = {
        sig: signal.signal(
            sig, lambda signum, _: server.stop(reason=signal.Signals(signum).name)
        )
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        server.serve()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path, required=True)
    args = parser.parse_args()
    serve(load_config(args.config_path))


if __name__ == "__main__":
    main()
