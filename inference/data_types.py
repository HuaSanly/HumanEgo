"""Serving data contracts. Poses use metres in the left rectified optical frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObjectState:
    """Object pose and local keypoints; preserves the original policy container."""

    T_in_cam: np.ndarray  # (4, 4), object -> camera
    kpts_local: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), np.float32)
    )


@dataclass(frozen=True)
class StereoFrame:
    """One synchronized rectified pair, each RGB uint8 (480, 640, 3).

    capture_time_ns is the original source timestamp, not a server deadline.
    """

    left_rgb: np.ndarray
    right_rgb: np.ndarray
    capture_time_ns: int = 0


@dataclass(frozen=True)
class RobotState:
    """Measured right EE pose (4, 4) and gripper: 0 open, 1 closed."""

    T_ee_in_cam: np.ndarray
    gripper: float


@dataclass(frozen=True)
class Observation:
    """Stereo and measured robot state from the same capture."""

    frame: StereoFrame
    robot: RobotState


@dataclass
class SceneState:
    """Current entities and an independent, fixed episode reference.

    reference_anchor must own a copy of the initial anchor pose; moving an
    object must never update this reference. hand_in_cam uses the MODEL hand
    convention, after applying T_EE_from_hand to the measured EE pose.
    """

    objects: dict[str, ObjectState]
    reference_anchor: ObjectState
    hand_in_cam: np.ndarray
    gripper: float
    latched_object: str | None = None
    T_object_in_hand: np.ndarray | None = None


@dataclass(frozen=True)
class PreparedObservation:
    """Perception result consumed by the original ICTPolicy methods.

    clean_bgr is uint8 (480, 640, 3); K is the matching left rectified (3, 3)
    intrinsics. ICTPolicy.prepare_image performs the final RGB/320x240 conversion.
    """

    frame: StereoFrame
    scene: SceneState
    clean_bgr: np.ndarray
    K: np.ndarray
