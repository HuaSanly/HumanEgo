"""Export a self-contained native deployment, retaining original checkpoint sidecars."""

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from .config import InferenceConfig, validate_runtime_config, load_config

LEROBOT_REVISION = "5517a326eeb9b457cfe07d59b9bfdf291799c8d9"
SEMANTICS = "humanego.right-ee.left-rectified-optical.m.row-major-rotation6d.continuous-grasp.v1"
COMPONENTS = ["x", "y", "z", "R00", "R01", "R10", "R11", "R20", "R21", "grasp"]


def server_config(output, config):
    features = [
        dict(
            name=f"observation.images.head_{side}",
            shape=[480, 640, 3],
            dtype="uint8",
            kind="rgb",
            semantics=f"{side} rectified optical RGB; synchronized stereo640x480",
        )
        for side in ("left", "right")
    ]
    features.append(
        dict(
            name="observation.state",
            shape=[10],
            dtype="float32",
            names=COMPONENTS,
            semantics="measured right EE; left rectified optical; metres; R[:, :2].reshape(-1); gripper 0=open 1=closed",
        )
    )
    action = dict(
        name="action",
        shape=[10],
        dtype="float32",
        names=COMPONENTS,
        semantics="absolute right EE target; left rectified optical; metres; R[:, :2].reshape(-1); raw policy sigmoid grasp",
    )
    return dict(
        deployment=config.deployment["name"],
        model=dict(repo_or_path=str(output.resolve()), device=config.policy["device"]),
        execution=dict(
            supported_modes=["chunk"],
            action_fps=30.0,
            action_deadline_s=30.0,
            idle_timeout_s=60.0,
            warmup_calls=2,
        ),
        language=dict(enabled=False),
        zenoh=dict(listen_endpoints=[config.deployment["listen_endpoint"]]),
        semantics=SEMANTICS,
        robot_type="humanego_brx",
        features=features,
        action_feature=action,
        log_level="INFO",
    )


def export_policy(config: InferenceConfig, output: Path) -> Path:
    validate_runtime_config(config)
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty export directory: {output}"
        )
    from .policy import ICTPolicy
    from .configuration_humanego import HumanEgoConfig
    from .processor_humanego import make_humanego_pre_post_processors

    policy = ICTPolicy(config.policy, device=config.policy["device"])
    if not (
        policy.single_hand
        and policy.single_hand_side == "right"
        and policy.action_mode == "absolute"
        and policy.frame_mode == "anchor_frame"
        and policy.pred_horizon == 50
        and policy.num_inference_steps == 20
        and not policy.use_pcd_features
    ):
        raise ValueError(
            "Only the approved HumanEgo_anchor single-right absolute baseline can be exported"
        )
    (output / "policy").mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config.policy["ckpt"])
    for src, name in (
        (checkpoint, "latest.pt"),
        (checkpoint.parent / "config.json", "config.json"),
        (checkpoint.parent / "dataset_stats.json", "dataset_stats.json"),
    ):
        shutil.copyfile(src, output / "policy" / name)
    assets = output / "assets"
    assets.mkdir()
    params = dict(
        policy=deepcopy(config.policy),
        camera=deepcopy(config.camera),
        perception=deepcopy(config.perception),
        deployment=deepcopy(config.deployment),
    )
    params["policy"]["ckpt"] = "../policy/latest.pt"
    params["camera"].update(
        intrinsics_path="intrinsics.yaml", extrinsics_path="extrinsics.yaml"
    )
    # Diagnostics are an explicit operator setting, not a path back into training.
    params["deployment"]["diagnostic_dir"] = None
    for name, values in (
        ("params.yaml", params),
        ("intrinsics.yaml", config.intrinsics),
        ("extrinsics.yaml", config.extrinsics),
    ):
        (assets / name).write_text(yaml.safe_dump(values, sort_keys=False))
    native = HumanEgoConfig(
        device=config.policy["device"], pretrained_path=output, push_to_hub=False
    )
    native.save_pretrained(output)
    np.save(
        output / "noise.npy", policy.noise.detach().cpu().numpy(), allow_pickle=False
    )
    pre, post = make_humanego_pre_post_processors(native)
    # Persist relative paths; the native load hook resolves them inside the bundle.
    pre.steps[1].params_path = native.params_file
    pre.save_pretrained(output)
    post.save_pretrained(output)
    (output / "server.yaml").write_text(
        yaml.safe_dump(server_config(output, config), sort_keys=False)
    )
    versions = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    manifest = dict(
        lerobot_revision=LEROBOT_REVISION,
        versions=versions,
        semantics=SEMANTICS,
        components=COMPONENTS,
        rotation6d="R[:, :2].reshape(-1) (row-major)",
        state_gripper="measured normalized state: 0=open, 1=closed",
        action_grasp="original ICTPolicy continuous sigmoid output, unchanged",
        rendering=config.perception["rendering"],
        stationary_episode=["head", "torso", "base"],
        files={},
    )
    for path in sorted(output.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(output))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_policy(load_config(args.config_path), args.output))


if __name__ == "__main__":
    main()
