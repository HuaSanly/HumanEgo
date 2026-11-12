"""Rectified metric stereo; measurements always use the original RGB pair."""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import InferenceConfig
from .data_types import StereoFrame


@dataclass(frozen=True)
class StereoDepthResult:
    disparity_px: np.ndarray
    depth_m: np.ndarray
    valid_mask: np.ndarray


class StereoDepthEstimator:
    def __init__(self, config: InferenceConfig):
        self.K = np.asarray(config.intrinsics["left"]["K"], np.float64)
        right_K = np.asarray(config.intrinsics["right"]["K"], np.float64)
        transform = np.asarray(config.extrinsics["T_left_from_right"], np.float64)
        if not np.allclose(transform[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError("Rectified stereo requires parallel optical axes")
        if transform[0, 3] <= 0 or not np.allclose(transform[1:3, 3], 0, atol=1e-6):
            raise ValueError(
                "Right optical origin must lie on positive left X (metres)"
            )
        if not np.allclose(self.K, right_K, atol=1e-5):
            raise ValueError(
                "SGBM contract requires matching rectified left/right intrinsics"
            )
        self.P_left = self.K @ np.eye(4)[:3]
        self.P_right = right_K @ np.linalg.inv(transform)[:3]
        self.focal_baseline = -self.P_right[0, 3]
        opts = config.perception.get("stereo", {})
        self.min_depth, self.max_depth = opts.get("depth_range_m", [0.15, 2.5])
        self.lr_threshold = float(opts.get("lr_threshold_px", 1.0))
        self.left = self._matcher(0)
        self.right = self._matcher(-128)

    @staticmethod
    def _matcher(min_disparity):
        return cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=128,
            blockSize=5,
            P1=8 * 25,
            P2=32 * 25,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def estimate(self, frame: StereoFrame) -> StereoDepthResult:
        left = cv2.cvtColor(frame.left_rgb, cv2.COLOR_RGB2GRAY)
        right = cv2.cvtColor(frame.right_rgb, cv2.COLOR_RGB2GRAY)
        disparity = self.left.compute(left, right).astype(np.float32) / 16
        reverse = self.right.compute(right, left).astype(np.float32) / 16
        yy, xx = np.indices(disparity.shape)
        xr = np.rint(xx - disparity).astype(int)
        in_bounds = (xr >= 0) & (xr < left.shape[1])
        reverse_sample = reverse[yy, np.clip(xr, 0, left.shape[1] - 1)]
        valid = in_bounds & (disparity > 0) & (reverse_sample > -129)
        valid &= np.abs(disparity + reverse_sample) <= self.lr_threshold
        depth = self.focal_baseline / np.maximum(disparity, 1e-6)
        valid &= (depth >= self.min_depth) & (depth <= self.max_depth)
        depth[~valid] = np.nan
        return StereoDepthResult(disparity, depth.astype(np.float32), valid)

    def points_from_mask(
        self, depth: StereoDepthResult, mask: np.ndarray
    ) -> np.ndarray:
        """Sample the training selector's ordered inner contour, retaining valid depths."""
        from box import Box
        from preprocess.KptsSelector import KeypointsSelectorEngine

        # Reuse the training algorithm without a per-request configuration file.
        selector = KeypointsSelectorEngine.__new__(KeypointsSelectorEngine)
        selector.cfg = Box(
            dict(
                kpts_patch_close=True,
                kpts_close_kernel=7,
                kpts_patch_erode=True,
                kpts_erode_kernel=5,
                kpts_erode_iters=1,
                kpts_use_inner_edge=True,
                kpts_edge_erode_kernel=5,
                kpts_min_edge_pixels=50,
                kpts_n_bands=10,
            )
        )
        pixels = selector.select_points((mask > 0).astype(np.uint8) * 255)
        points = []
        for x, y in pixels:
            ys, xs = slice(max(0, y - 2), y + 3), slice(max(0, x - 2), x + 3)
            good = depth.valid_mask[ys, xs] & mask[ys, xs].astype(bool)
            values = depth.depth_m[ys, xs][good]
            if len(values) < 3:
                continue
            z = float(np.median(values))
            points.append(
                [
                    (x - self.K[0, 2]) * z / self.K[0, 0],
                    (y - self.K[1, 2]) * z / self.K[1, 1],
                    z,
                ]
            )
        points = np.asarray(points, np.float64).reshape(-1, 3)
        if len(points):
            z = points[:, 2]
            median = np.median(z)
            mad = np.median(np.abs(z - median))
            points = points[np.abs(z - median) <= max(3 * 1.4826 * mad, 0.01)]
        return points
