"""Input validation and episode lifecycle; perception orchestration is pending."""

from __future__ import annotations

import numpy as np

from .config import InferenceConfig
from .data_types import Observation, PreparedObservation, SceneState
from .perception import Perception


def validate_observation(observation: Observation) -> None:
    """Check decoded right-arm inputs without changing buffers or units."""
    for name, image in (("left", observation.frame.left_rgb), ("right", observation.frame.right_rgb)):
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"{name} image must be RGB uint8 with shape (480, 640, 3)")
    pose = observation.robot.T_ee_in_cam
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Right EE pose must be a finite (4, 4) matrix")
    if not 0 <= observation.robot.gripper <= 1:
        raise ValueError("Right gripper must be in [0, 1] (0 open, 1 closed)")


class HumanEgoProcessor:
    """Own session scene state, never a second ICTPolicy or action queue."""

    def __init__(self, config: InferenceConfig, perception: Perception):
        self.config = config
        self.perception = perception
        self.scene: SceneState | None = None

    def __call__(self, observation: Observation) -> PreparedObservation:
        validate_observation(observation)
        raise NotImplementedError("Perception and clean-image orchestration is not implemented")

    def reset(self) -> None:
        """Clear the episode reference/objects while retaining engine ownership."""
        self.scene = None
