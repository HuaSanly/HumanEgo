"""Scene and image processing interfaces; no model is loaded at import time."""

from __future__ import annotations

import numpy as np

from .config import InferenceConfig
from .data_types import Observation, SceneState, StereoFrame


class Perception:
    """Own future segmentation/inpainting engines, with scene state held outside."""

    def __init__(self, config: InferenceConfig):
        self.config = config

    def initialize_scene(self, observation: Observation) -> SceneState:
        """Locate both objects and copy the initial obj1 pose as the reference."""
        raise NotImplementedError("Object initialization and hand alignment are not implemented")

    def update_scene(self, observation: Observation, scene: SceneState) -> SceneState:
        """Update visible/latched objects without changing the fixed reference."""
        raise NotImplementedError("Object tracking, grasp latching and release are not implemented")

    def erase_arms(self, frame: StereoFrame) -> np.ndarray:
        """Return a BGR uint8 left image with both real arms/grippers removed.

        Object geometry must be measured from the original pair, before LaMa.
        """
        raise NotImplementedError("Arm segmentation and LaMa inpainting are not implemented")
