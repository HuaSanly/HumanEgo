"""Intended LeRobot policy capabilities; native registration is not implemented."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HumanEgoConfig:
    """Public contract for the single-right-hand HumanEgo_anchor baseline.

    Architecture and normalization still come from ICTPolicy's checkpoint
    sidecars. This declaration is not a LeRobot PreTrainedConfig yet.
    """

    policy_type: str = "humanego"
    modes: tuple[str, ...] = ("chunk",)
    image_shape: tuple[int, int, int] = (480, 640, 3)
    state_dim: int = 10
    action_dim: int = 10
    prediction_steps: int = 50
    current_observation_only: bool = True
    retains_session_state: bool = True
