"""Native LeRobot configuration for the fixed right-hand anchor baseline."""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature


@PreTrainedConfig.register_subclass("humanego")
@dataclass
class HumanEgoConfig(PreTrainedConfig):
    chunk_size: int = 50
    n_action_steps: int = 50
    params_file: str = "assets/params.yaml"
    noise_file: str = "noise.npy"

    def __post_init__(self):
        super().__post_init__()
        if (
            self.n_obs_steps != 1
            or self.chunk_size != 50
            or not 0 < self.n_action_steps <= 50
        ):
            raise ValueError(
                "HumanEgo requires one observation and a 50-step prediction horizon"
            )
        if self.use_amp or self.dtype is not None:
            raise ValueError(
                "HumanEgo preserves the original float32 policy; AMP/dtype overrides are unsupported"
            )
        if not self.input_features:
            self.input_features = {
                f"observation.images.head_{side}": PolicyFeature(
                    FeatureType.VISUAL, (3, 480, 640)
                )
                for side in ("left", "right")
            }
            self.input_features["observation.state"] = PolicyFeature(
                FeatureType.STATE, (10,)
            )
        if not self.output_features:
            self.output_features = {"action": PolicyFeature(FeatureType.ACTION, (10,))}
        self.validate_features()

    def validate_features(self):
        expected = {
            f"observation.images.head_{side}": PolicyFeature(
                FeatureType.VISUAL, (3, 480, 640)
            )
            for side in ("left", "right")
        }
        expected["observation.state"] = PolicyFeature(FeatureType.STATE, (10,))
        if self.input_features != expected or self.output_features != {
            "action": PolicyFeature(FeatureType.ACTION, (10,))
        }:
            raise ValueError(
                "HumanEgo features must match the stereo640/right-EE10 contract"
            )

    @property
    def observation_delta_indices(self):
        return [0]

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError(
            "This plugin serves the existing ICTPolicy; use the original trainer"
        )

    def get_scheduler_preset(self):
        return None
