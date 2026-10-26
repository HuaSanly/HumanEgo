"""Episode perception and native LeRobot processor serialization."""

from dataclasses import dataclass
import logging
from pathlib import Path
import time

import cv2
import numpy as np

from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.factory import (
    make_policy_processor_pipelines,
    load_pretrained_policy_processors,
)
from lerobot.processor import DeviceProcessorStep, RenameObservationsProcessorStep


from .config import InferenceConfig, load_config, validate_runtime_config
from .data_types import Observation, PreparedObservation, RobotState, StereoFrame
from .perception import Perception
from .rendering import canvas_intrinsics, render_scene

logger = logging.getLogger(__name__)


def validate_observation(observation: Observation) -> None:
    for name, image in (
        ("left", observation.frame.left_rgb),
        ("right", observation.frame.right_rgb),
    ):
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"{name} image must be RGB uint8 with shape (480, 640, 3)")
    T = observation.robot.T_ee_in_cam
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Right EE pose must be finite (4,4)")
    if (
        not np.allclose(T[3], [0, 0, 0, 1])
        or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-4)
    ):
        raise ValueError("Right EE pose must be SE(3)")
    if (
        not np.isfinite(observation.robot.gripper)
        or not 0 <= observation.robot.gripper <= 1
    ):
        raise ValueError("Measured gripper must be in [0,1], 0 open and 1 closed")


def observation_from_batch(batch):
    """Recover lossless RGB from the official runner's canonical float32 CHW batch."""
    state = batch["observation.state"].detach().cpu().numpy()
    if state.shape != (1, 10) or not np.isfinite(state).all():
        raise ValueError("observation.state must be finite [1,10]")
    state = state[0]
    columns = state[3:9].reshape(3, 2)
    rotation = np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])
    T = np.eye(4, dtype=np.float32)
    T[:3, :3], T[:3, 3] = rotation, state[:3]
    images = []
    for side in ("left", "right"):
        image = batch[f"observation.images.head_{side}"].detach().cpu().numpy()
        if (
            image.shape != (1, 3, 480, 640)
            or not np.isfinite(image).all()
            or image.min() < 0
            or image.max() > 1
        ):
            raise ValueError(f"Invalid canonical RGB image head_{side}")
        scaled = image[0].transpose(1, 2, 0) * 255
        if not np.allclose(scaled, np.rint(scaled), atol=1e-4):
            raise ValueError("Canonical image must originate from uint8 RGB")
        images.append(np.rint(scaled).astype(np.uint8))
    obs = Observation(StereoFrame(*images), RobotState(T, float(state[9])))
    validate_observation(obs)
    return obs


class HumanEgoProcessor:
    def __init__(self, config: InferenceConfig, perception: Perception):
        self.config, self.perception = config, perception
        self.scene = None

    def __call__(self, observation: Observation) -> PreparedObservation:
        validate_observation(observation)
        start = time.perf_counter()
        scene = (
            self.perception.initialize_scene(observation)
            if self.scene is None
            else self.perception.update_scene(observation, self.scene)
        )
        clean = self.perception.erase_arms(observation.frame)
        K = np.asarray(self.config.intrinsics["left"]["K"], np.float64)
        settings = self.config.perception["rendering"]
        image = render_scene(clean, scene, K, settings)
        self.scene = scene
        self.perception.diagnostics["processor_s"] = time.perf_counter() - start
        logger.info("Perception %s", self.perception.diagnostics)
        folder = self.config.deployment.get("diagnostic_dir")
        if folder:
            path = Path(folder)
            path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path / f"{time.time_ns()}_model_canvas.png"), image)
        return PreparedObservation(
            observation.frame,
            scene,
            image,
            canvas_intrinsics(K, canvas_hw=settings["canvas_hw"]),
        )

    def reset(self):
        self.scene = None
        self.perception.reset()


@ProcessorStepRegistry.register("humanego_perception")
@dataclass
class HumanEgoPerceptionStep(ProcessorStep):
    params_path: str
    device: str = "cuda"

    def __post_init__(self):
        self.processor = None
        self.startup_probe_allowed = True

    def _processor(self):
        if self.processor is None:
            cfg = load_config(self.params_path)
            cfg.policy["device"] = self.device
            validate_runtime_config(cfg)
            self.processor = HumanEgoProcessor(cfg, Perception(cfg))
        return self.processor

    def __call__(self, transition):
        batch = transition["observation"]
        task = (transition.get("complementary_data") or {}).get("task")
        probe = (
            self.startup_probe_allowed
            and task == ["warmup"]
            and all(not value.any().item() for value in batch.values())
        )
        processor = self._processor()
        if probe:
            # Startup-only probe creates no reference, objects or actionable scene.
            zeros = np.zeros((480, 640, 3), np.uint8)
            processor.perception.segmentation.segment(
                zeros,
                processor.config.perception["object_prompts"],
                processor.config.perception["erase_prompt"],
            )
            processor.perception.inpainter.inpaint(
                zeros, np.ones((480, 640), bool), np.zeros((480, 640), bool)
            )
            transition["observation"] = {"observation.humanego.startup_probe": True}
        else:
            observation = observation_from_batch(batch)
            transition["observation"] = {
                "observation.humanego.prepared": processor(observation)
            }
        return transition

    def reset(self):
        self.startup_probe_allowed = False
        if self.processor is not None:
            self.processor.reset()

    def get_config(self):
        return {"params_path": self.params_path, "device": self.device}

    def transform_features(self, features):
        return features


def make_humanego_pre_post_processors(config, dataset_stats=None):
    root = Path(config.pretrained_path or ".")
    return make_policy_processor_pipelines(
        [
            RenameObservationsProcessorStep({}),
            HumanEgoPerceptionStep(str(root / config.params_file), config.device),
            DeviceProcessorStep(config.device),
        ],
        [DeviceProcessorStep("cpu")],
    )


def make_humanego_pre_post_processors_from_pretrained(
    config,
    pretrained_path,
    preprocessor_overrides=None,
    postprocessor_overrides=None,
    **kwargs,
):
    overrides = dict(preprocessor_overrides or {})
    overrides["humanego_perception"] = {
        "params_path": str(Path(pretrained_path) / config.params_file),
        "device": config.device,
    }
    return load_pretrained_policy_processors(
        pretrained_path,
        preprocessor_overrides=overrides,
        postprocessor_overrides=postprocessor_overrides,
    )
