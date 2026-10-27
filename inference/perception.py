"""Persistent GPU vision, metric object updates and episode grasp association."""

from copy import deepcopy
from contextlib import nullcontext
import logging
import time

import cv2
import numpy as np

from .config import InferenceConfig
from .data_types import ObjectState, Observation, SceneState, StereoFrame
from .stereo import StereoDepthEstimator

logger = logging.getLogger(__name__)


class Segmentation:
    """DINO uses PIL RGB; SAM encodes each RGB image once for all prompts."""

    def __init__(self, config):
        from huggingface_hub import hf_hub_download
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = config.policy["device"]
        opts = config.perception["segmentation"]
        self.threshold = opts["box_threshold"]
        self.dino_precision = opts.get("dino_precision", "float32")
        self.sam_precision = opts.get("sam_precision", "float32")
        for precision in (self.dino_precision, self.sam_precision):
            if precision not in ("float32", "float16", "bfloat16"):
                raise ValueError(f"Unsupported segmentation precision: {precision}")
        self.processor = AutoProcessor.from_pretrained(
            opts["dino_model_id"], revision=opts["dino_revision"], backend="pil"
        )
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                opts["dino_model_id"], revision=opts["dino_revision"]
            )
            .to(self.device)
            .eval()
        )
        checkpoint = hf_hub_download(
            opts["sam2_repo_id"], "sam2_hiera_tiny.pt", revision=opts["sam2_revision"]
        )
        self.sam = SAM2ImagePredictor(
            build_sam2(
                "configs/sam2/sam2_hiera_t.yaml",
                checkpoint,
                device=self.device,
                mode="eval",
            )
        )

    def _autocast(self, precision):
        import torch

        if precision == "float32":
            return nullcontext()
        return torch.autocast(
            device_type=torch.device(self.device).type, dtype=getattr(torch, precision)
        )

    def segment(self, rgb, object_prompts, erase_prompt, object_boxes=None):
        import torch
        from PIL import Image

        with torch.inference_mode():
            with self._autocast(self.sam_precision):
                self.sam.set_image(rgb)
            masks = {}
            prompts = {**object_prompts, "arms": erase_prompt}
            for key, prompt in prompts.items():
                batch = self.processor(
                    images=Image.fromarray(rgb), text=prompt, return_tensors="pt"
                ).to(self.device)
                with self._autocast(self.dino_precision):
                    output = self.model(**batch)
                scores = output.logits.sigmoid().amax(-1)[0]
                keep = scores > self.threshold
                boxes = output.pred_boxes[0, keep]
                if not len(boxes):
                    masks[key] = np.zeros(rgb.shape[:2], bool)
                    continue
                if key != "arms":
                    boxes = boxes[scores[keep].argmax()].unsqueeze(0)
                cx, cy, w, h = boxes.unbind(-1)
                xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
                xyxy *= xyxy.new_tensor(
                    [rgb.shape[1], rgb.shape[0], rgb.shape[1], rgb.shape[0]]
                )
                with self._autocast(self.sam_precision):
                    result, _, _ = self.sam.predict(
                        box=xyxy.float().cpu().numpy(), multimask_output=False
                    )
                masks[key] = np.any(result.reshape(-1, *rgb.shape[:2]), axis=0)
            # Existing metric poses provide prompts without another DINO forward.
            for key, box in (object_boxes or {}).items():
                if box is None:
                    masks[key] = np.zeros(rgb.shape[:2], bool)
                    continue
                with self._autocast(self.sam_precision):
                    result, _, _ = self.sam.predict(box=box, multimask_output=False)
                masks[key] = np.any(result.reshape(-1, *rgb.shape[:2]), axis=0)
        return masks


class Inpainter:
    """Training-side BGR LaMa normalization, dilation and soft blending."""

    def __init__(self, config):
        import torch  # Load CUDA/cuDNN libraries before ONNX Runtime.
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        self.opts = config.perception["lama"]
        path = hf_hub_download(
            self.opts["repo_id"], self.opts["filename"], revision=self.opts["revision"]
        )
        ort.preload_dlls()
        device_id = torch.device(config.policy["device"]).index or 0
        self.session = ort.InferenceSession(
            path,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device_id,
                        "cudnn_conv_algo_search": "HEURISTIC",
                    },
                ),
                "CPUExecutionProvider",
            ],
        )
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(
                "LaMa requires CUDAExecutionProvider; GPU initialization failed"
            )

    def inpaint(self, bgr, mask, protected):
        mask = cv2.dilate(
            mask.astype(np.uint8) * 255,
            np.ones((self.opts["mask_dilation"],) * 2, np.uint8),
            iterations=2,
        )
        mask[protected] = 0
        if not np.any(mask):
            return bgr.copy()
        size = tuple(self.opts["input_size_hw"][::-1])
        image = cv2.resize(bgr, size).astype(np.float32).transpose(2, 0, 1)[None] / 255
        small = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        inputs = {
            self.session.get_inputs()[0].name: image,
            self.session.get_inputs()[1].name: (small > 127).astype(np.float32)[
                None, None
            ],
        }
        output = self.session.run(None, inputs)[0][0].transpose(1, 2, 0)
        output = np.clip(
            output * 255 if output.max() <= 1.1 else output, 0, 255
        ).astype(np.uint8)
        output = cv2.resize(output, bgr.shape[1::-1])
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (5, 5), 0)[..., None]
        result = (output * alpha + bgr * (1 - alpha)).astype(np.uint8)
        result[protected] = bgr[protected]
        return result


class Perception:
    def __init__(
        self,
        config: InferenceConfig,
        *,
        segmentation=None,
        inpainter=None,
        clock=time.monotonic,
    ):
        self.config = config
        self.stereo = StereoDepthEstimator(config)
        self.segmentation = (
            segmentation if segmentation is not None else Segmentation(config)
        )
        self.inpainter = inpainter if inpainter is not None else Inpainter(config)
        self.clock = clock
        self.masks = None
        self.diagnostics = {}

    def reset(self):
        self.masks = None
        self.diagnostics = {}

    def observe(self, observation, object_keys=("obj1", "obj2"), scene=None):
        """The original stereo and segmentation share one capture, before inpainting."""
        start = time.perf_counter()
        prompts = {
            key: self.config.perception["object_prompts"][key] for key in object_keys
        }
        boxes = (
            None
            if scene is None
            else {
                key: self._object_box(obj)
                for key, obj in scene.objects.items()
                if key not in object_keys
            }
        )
        masks = self.segmentation.segment(
            observation.frame.left_rgb,
            prompts,
            self.config.perception["erase_prompt"],
            **({"object_boxes": boxes} if boxes is not None else {}),
        )
        self.masks = masks
        depth = self.stereo.estimate(observation.frame) if object_keys else None
        points = {
            key: self.stereo.points_from_mask(depth, masks[key]) for key in object_keys
        }
        self.diagnostics = {
            "localized_objects": list(object_keys),
            "valid_points": {k: len(v) for k, v in points.items()},
            "mask_pixels": {k: int(v.sum()) for k, v in masks.items()},
            "geometry_segmentation_s": time.perf_counter() - start,
        }
        return points

    def _object_box(self, obj):
        points = obj.kpts_local @ obj.T_in_cam[:3, :3].T + obj.T_in_cam[:3, 3]
        points = points[points[:, 2] > 0]
        if len(points) < 3:
            return None
        uv = points @ self.stereo.K.T
        uv = uv[:, :2] / uv[:, 2:]
        low = np.maximum(uv.min(axis=0) - 4, [0, 0])
        high = np.minimum(uv.max(axis=0) + 4, [639, 479])
        return (
            np.concatenate([low, high]).astype(np.float32)
            if np.all(high > low)
            else None
        )

    def _locate(self, points, existing=None):
        from preprocess.OrientAnything import estimate_frame_pca1

        objects = {}
        for key in ("obj1", "obj2"):
            pts = points.get(key, [])
            if len(pts) < 8:
                continue
            anchor = objects.get("obj1") or (existing or {}).get("obj1")
            if key == "obj2" and anchor is None:
                continue
            T, _ = estimate_frame_pca1(
                pts,
                is_anchor=key == "obj1",
                anchor_center_cam=None if anchor is None else anchor.T_in_cam[:3, 3],
            )
            if not np.isfinite(T).all() or not np.allclose(
                T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-4
            ):
                continue
            local = (pts - T[:3, 3]) @ T[:3, :3]
            objects[key] = ObjectState(T.astype(np.float32), local.astype(np.float32))
        return objects

    def initialize_scene(self, observation: Observation) -> SceneState:
        objects = self._locate(self.observe(observation))
        missing = {"obj1", "obj2"} - objects.keys()
        if missing:
            raise ValueError(
                f"Cannot initialize scene: missing valid metric objects {sorted(missing)}; {self.diagnostics}"
            )
        now = self.clock()
        hand = observation.robot.T_ee_in_cam @ np.asarray(
            self.config.extrinsics["T_EE_from_hand"]
        )
        scene = SceneState(
            objects,
            deepcopy(objects["obj1"]),
            hand,
            observation.robot.gripper,
            last_seen={key: now for key in objects},
        )
        self._associate(scene, now)
        logger.info("Scene initialized: fixed reference copied; %s", self.diagnostics)
        return scene

    def update_scene(self, observation: Observation, scene: SceneState) -> SceneState:
        # Commit a complete update only after all necessary object states are valid.
        scene = deepcopy(scene)
        now = self.clock()
        scene.hand_in_cam = observation.robot.T_ee_in_cam @ np.asarray(
            self.config.extrinsics["T_EE_from_hand"]
        )
        scene.gripper = observation.robot.gripper
        events = (
            self.config.perception.get("object_update_mode", "continuous") == "events"
        )
        held = scene.latched_object
        if held:
            scene.objects[held].T_in_cam = scene.hand_in_cam @ scene.T_object_in_hand
            scene.last_seen[held] = now
            if scene.gripper <= 0.5:
                scene.latched_object, scene.T_object_in_hand = None, None
                scene.relocalization_started[held] = now
        if events:
            points = self.observe(
                observation, tuple(scene.relocalization_started), scene
            )
        else:
            points = self.observe(observation)
        visible = self._locate(points, scene.objects)
        for key, obj in visible.items():
            if key != scene.latched_object:
                scene.objects[key] = obj
                scene.last_seen[key] = now
                scene.relocalization_started.pop(key, None)
        self._associate(scene, now, require_fresh=not events)
        ttl = self.config.perception.get("object_ttl_s", 1.0)
        deadlines = scene.relocalization_started if events else scene.last_seen
        for key, last_valid in deadlines.items():
            if key != scene.latched_object and now - last_valid > ttl:
                raise ValueError(f"{key}: visual localization expired after {ttl}s")
        return scene

    @staticmethod
    def _associate(scene, now, require_fresh=True):
        if scene.latched_object is not None or scene.gripper <= 0.5:
            return
        # Event mode assumes stationary unheld objects; pending relocation is untrusted.
        if (
            "obj2" not in scene.relocalization_started
            and (not require_fresh or scene.last_seen["obj2"] == now)
            and np.linalg.norm(
                scene.objects["obj2"].T_in_cam[:3, 3] - scene.hand_in_cam[:3, 3]
            )
            < 0.20
        ):
            scene.latched_object = "obj2"
            scene.T_object_in_hand = (
                np.linalg.inv(scene.hand_in_cam) @ scene.objects["obj2"].T_in_cam
            )

    def erase_arms(self, frame: StereoFrame) -> np.ndarray:
        if self.masks is None:
            raise RuntimeError(
                "Scene measurement must precede inpainting for this capture"
            )
        bgr = cv2.cvtColor(frame.left_rgb, cv2.COLOR_RGB2BGR)
        protected = self.masks["obj1"] | self.masks["obj2"]
        mask = self.masks["arms"] & ~protected
        if not mask.any():
            self.diagnostics["inpainting"] = "empty_mask"
            return bgr
        start = time.perf_counter()
        result = self.inpainter.inpaint(bgr, mask, protected)
        self.diagnostics.update(
            inpainting="inpainted", inpaint_s=time.perf_counter() - start
        )
        return result
