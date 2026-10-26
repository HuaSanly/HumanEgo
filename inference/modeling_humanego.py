"""Thin LeRobot chunk-adapter interface around the original ICTPolicy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .data_types import PreparedObservation

if TYPE_CHECKING:
    import torch

    from .policy import ICTPolicy


class HumanEgoPolicy:
    """Hold one existing policy; native LeRobot integration is still pending.

    The implementation will call prepare_image, build_ict, compute_anchor_uv,
    infer and decode_ee_in_cam on that instance, using the same fixed reference
    for encoding and decoding. It must not rebuild a model or implement an ODE.
    """

    def __init__(self, policy: ICTPolicy):
        self.policy = policy

    def predict_action_chunk(self, observation: PreparedObservation) -> torch.Tensor:
        """Return canonical float32 [1, 50, 10] right-EE camera-frame targets."""
        raise NotImplementedError("LeRobot chunk adaptation of ICTPolicy is not implemented")
