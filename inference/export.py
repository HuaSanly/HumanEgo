"""Deployment artifact preparation interface; no export is performed yet."""

from pathlib import Path

from .config import InferenceConfig


def export_policy(config: InferenceConfig, output: Path) -> Path:
    """Prepare native model/processor artifacts without replacing ICTPolicy."""
    raise NotImplementedError("Native LeRobot artifact export is not implemented")
