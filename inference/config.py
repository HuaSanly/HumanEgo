"""Read deployment parameters without importing policy or perception runtimes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class InferenceConfig:
    """Resolved paths plus the parameter groups owned by each subsystem."""

    source: Path
    policy: dict[str, Any]
    camera: dict[str, Any]
    perception: dict[str, Any]
    deployment: dict[str, Any]
    intrinsics: dict[str, Any]
    extrinsics: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return values


def _resolve_path(value: str, source: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: expected a nonempty file path, got {value!r}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else source.parent / path).resolve()


def load_config(path: str | Path) -> InferenceConfig:
    """Load params and calibration templates; paths are relative to params.yaml.

    Null calibration values are allowed for inspection. Call
    validate_runtime_config before constructing any inference models.
    """
    source = Path(path).expanduser().resolve()
    values = _read_yaml(source)
    groups = ("policy", "camera", "perception", "deployment")
    for name in groups:
        if not isinstance(values.get(name), dict):
            raise ValueError(f"{source}: {name} must be a mapping")

    policy = dict(values["policy"])
    camera = dict(values["camera"])
    policy["ckpt"] = str(_resolve_path(policy.get("ckpt"), source))
    steps = policy.get("num_inference_steps")
    if type(steps) is not int or steps <= 0:
        raise ValueError(f"{source}: policy.num_inference_steps must be a positive integer")
    if camera.get("image_size_hw") != [480, 640]:
        raise ValueError(f"{source}: camera.image_size_hw must be [480, 640]")

    for name in ("intrinsics_path", "extrinsics_path"):
        camera[name] = str(_resolve_path(camera.get(name), source))
    intrinsics = _read_yaml(Path(camera["intrinsics_path"]))
    if intrinsics.get("image_size_hw") != camera["image_size_hw"]:
        raise ValueError("intrinsics.image_size_hw differs from the transmitted image size")
    if intrinsics.get("rectified") is not True:
        raise ValueError("intrinsics.rectified must be true for the serving contract")
    for side in ("left", "right"):
        if not isinstance(intrinsics.get(side), dict):
            raise ValueError(f"intrinsics.{side} must be a mapping with K")

    return InferenceConfig(
        source=source, policy=policy, camera=camera,
        perception=dict(values["perception"]), deployment=dict(values["deployment"]),
        intrinsics=intrinsics, extrinsics=_read_yaml(Path(camera["extrinsics_path"])),
    )


def validate_runtime_config(config: InferenceConfig) -> None:
    """Reject absent/invalid metric calibration and missing checkpoint sidecars.

    This checks configuration readiness only; it does not load or validate a
    policy, initialize perception, or certify a physical calibration.
    """
    matrices = {
        "intrinsics.left.K": config.intrinsics.get("left", {}).get("K"),
        "intrinsics.right.K": config.intrinsics.get("right", {}).get("K"),
        "extrinsics.T_left_from_right": config.extrinsics.get("T_left_from_right"),
        "extrinsics.T_EE_from_hand": config.extrinsics.get("T_EE_from_hand"),
    }
    missing = [name for name, value in matrices.items() if value is None]
    if missing:
        raise ValueError("Missing calibration: " + ", ".join(missing))
    if config.extrinsics.get("translation_unit") != "m":
        raise ValueError("extrinsics.translation_unit must be 'm'")
    for name, value in matrices.items():
        try:
            matrix = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric matrix values") from exc
        shape = (3, 3) if name.endswith(".K") else (4, 4)
        if matrix.shape != shape or not np.isfinite(matrix).all():
            raise ValueError(f"{name} must be a finite {shape} matrix")
        if shape == (3, 3):
            if (
                matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0, 0, 1])
            ):
                raise ValueError(f"{name} must contain positive focal lengths and row [0, 0, 1]")
        elif (
            not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)
        ):
            raise ValueError(f"{name} must be a rigid transform")
    if np.linalg.norm(np.asarray(matrices["extrinsics.T_left_from_right"])[:3, 3]) == 0:
        raise ValueError("extrinsics.T_left_from_right must have a nonzero stereo baseline")

    checkpoint = Path(config.policy["ckpt"])
    required_files = (checkpoint, checkpoint.parent / "config.json", checkpoint.parent / "dataset_stats.json")
    for file in required_files:
        if not file.is_file():
            raise FileNotFoundError(f"Required policy file is missing: {file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True, type=Path)
    parser.add_argument("--validate-runtime", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config_path)
    if args.validate_runtime:
        validate_runtime_config(config)
    print(f"Configuration loaded: {config.source}")
    print(f"Checkpoint: {config.policy['ckpt']}")
    print("Stereo images: RGB uint8, each 640x480")


if __name__ == "__main__":
    main()
