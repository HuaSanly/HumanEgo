"""Training-compatible visual input rendering interface."""

from __future__ import annotations

import numpy as np

from .data_types import SceneState


def render_scene(background_bgr: np.ndarray, scene: SceneState, K: np.ndarray) -> np.ndarray:
    """Draw the model right hand and current object keypoints on a 640x480 image.

    Returns BGR uint8 for ICTPolicy.prepare_image. Rendering must reproduce
    the actual training convention rather than assume the old helper matches it.
    """
    raise NotImplementedError("Training-compatible gripper and object rendering is not implemented")
