"""Native serving adapter; all neural inference remains in the original ICTPolicy."""

import logging
from pathlib import Path
import time

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.inference import ChunkPolicySpec, ExecutionMode

from .config import load_config, validate_runtime_config
from .configuration_humanego import HumanEgoConfig
from .data_types import PreparedObservation
from .policy import ICTPolicy

logger = logging.getLogger(__name__)


class HumanEgoPolicy(PreTrainedPolicy):
    config_class = HumanEgoConfig
    name = "humanego"

    def __init__(self, config: HumanEgoConfig, *, policy: ICTPolicy | None = None):
        super().__init__(config)
        root = Path(config.pretrained_path or ".")
        self.serving_config = load_config(root / config.params_file)
        validate_runtime_config(self.serving_config)
        self.policy = (
            policy
            if policy is not None
            else ICTPolicy(self.serving_config.policy, device=config.device)
        )
        original = self.policy
        if (
            not original.single_hand
            or original.single_hand_side != "right"
            or original.action_mode != "absolute"
            or original.frame_mode != "anchor_frame"
            or original.pred_horizon != 50
            or original.use_pcd_features
            or original.num_inference_steps != 20
        ):
            raise ValueError(
                "Checkpoint does not implement the HumanEgo_anchor serving baseline"
            )
        # Register the EXISTING network and noise, without constructing any second model.
        self.model = original.model
        self.register_buffer("fixed_noise", original.noise, persistent=False)
        self.T_align = np.asarray(
            self.serving_config.extrinsics["T_EE_from_hand"], np.float32
        )
        self.eval()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        self.policy.noise = self.fixed_noise
        self.policy.device = str(self.fixed_noise.device)
        return result

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs):
        root = Path(pretrained_name_or_path).resolve()
        config = config or HumanEgoConfig.from_pretrained(root)
        config.pretrained_path = root
        instance = cls(config)
        noise = np.load(root / config.noise_file, allow_pickle=False)
        if (
            noise.shape != tuple(instance.fixed_noise.shape)
            or not np.isfinite(noise).all()
        ):
            raise ValueError("Deployment fixed noise has an invalid shape or values")
        instance.fixed_noise.copy_(
            torch.as_tensor(noise, device=instance.fixed_noise.device)
        )
        instance.policy.noise = instance.fixed_noise
        return instance.eval()

    def chunk_inference_spec(self):
        return ChunkPolicySpec(
            prediction_steps=50,
            execution_steps=self.config.n_action_steps,
            modes=(ExecutionMode.CHUNK,),
            current_observation_only=True,
            retains_session_state=True,
        )

    @torch.inference_mode()
    def predict_action_chunk(self, batch, **kwargs):
        start = time.perf_counter()
        if isinstance(batch, PreparedObservation):
            prepared = batch
        elif batch.get("observation.humanego.startup_probe") is True:
            # Native startup warmup is discarded by load_deployment before admission.
            image = np.zeros((240, 320, 3), np.uint8)
            x_rgb = self.policy.prepare_image(image)
            ict, mask = self.policy.build_ict({"right": np.eye(4)}, {"right": 0.0}, {})
            uv = (
                torch.full((1, 2), 0.5, device=self.policy.device)
                if self.policy.use_region_attn
                else None
            )
            prediction, done = self.policy.infer(x_rgb, ict, mask, uv)
            if not np.isfinite(done) or not all(
                np.isfinite(value).all() for value in prediction["right"]
            ):
                raise ValueError("ICTPolicy produced non-finite startup predictions")
            return torch.zeros(
                (1, 50, 10), device=self.policy.device, dtype=torch.float32
            )
        else:
            prepared = batch["observation.humanego.prepared"]
        scene = prepared.scene
        x_rgb = self.policy.prepare_image(prepared.clean_bgr)
        ict, mask = self.policy.build_ict(
            {"right": scene.hand_in_cam},
            {"right": scene.gripper},
            scene.objects,
            reference_anchor=scene.reference_anchor,
        )
        h, w = prepared.clean_bgr.shape[:2]
        uv = self.policy.compute_anchor_uv(scene.objects["obj1"], prepared.K, w, h)
        prediction, done = self.policy.infer(x_rgb, ict, mask, uv)
        pos, rot, grasp = prediction["right"]
        poses = np.stack(
            [
                self.policy.decode_ee_in_cam(p, r, scene.reference_anchor, self.T_align)
                for p, r in zip(pos, rot, strict=True)
            ]
        )
        actions = np.concatenate(
            [
                poses[:, :3, 3],
                poses[:, :3, :2].reshape(50, 6),
                np.asarray(grasp).reshape(50, 1),
            ],
            axis=-1,
        )
        if (
            actions.shape != (50, 10)
            or not np.isfinite(actions).all()
            or not np.isfinite(done)
        ):
            raise ValueError(
                "ICTPolicy produced a non-finite or malformed action chunk"
            )
        logger.info(
            "Policy done=%s inference_s=%.4f grasp_range=[%.5f,%.5f]",
            done,
            time.perf_counter() - start,
            float(actions[:, -1].min()),
            float(actions[:, -1].max()),
        )
        return torch.as_tensor(
            actions, dtype=torch.float32, device=self.policy.device
        ).unsqueeze(0)

    def select_action(self, batch, **kwargs):
        raise NotImplementedError(
            "Use the official chunk runtime; this adapter has no action queue"
        )

    def forward(self, batch):
        raise NotImplementedError("Use the existing HumanEgo trainer")

    def get_optim_params(self):
        return self.model.parameters()

    def reset(self):
        # Session state belongs to the processor; the model and fixed noise survive.
        pass
