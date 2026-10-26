"""Metric stereo processing interfaces; algorithms are not implemented yet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import InferenceConfig
from .data_types import StereoFrame


@dataclass(frozen=True)
class StereoDepthResult:
    """Left-image arrays (480, 640); depth is metres, valid_mask is boolean."""

    disparity_px: np.ndarray
    depth_m: np.ndarray
    valid_mask: np.ndarray


class StereoDepthEstimator:
    def __init__(self, config: InferenceConfig):
        self.config = config

    def estimate(self, frame: StereoFrame) -> StereoDepthResult:
        """Estimate depth from the original rectified RGB pair."""
        raise NotImplementedError("640x480 stereo depth estimation is not implemented")

    def points_from_mask(self, depth: StereoDepthResult, mask: np.ndarray) -> np.ndarray:
        """Return valid masked (N, 3) points in the left optical frame, in metres."""
        raise NotImplementedError("Metric stereo backprojection is not implemented")
