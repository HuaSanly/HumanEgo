# -*- coding: utf-8 -*-
"""
SimPerception — the real HumanEgo perception pipeline, fed by the simulator.

At episode start (`estimate_objects_bimanual`) it runs exactly what
`AriaMimic inference/Inference.py: do_object_centric_bimanual` runs on the
robot, using the *released* modules of this repo:

    masks        ground-truth segmentation       (replaces DINO-SAM; same output)
    split        connected components at split_x (shared_context: only obj1 is split)
    keypoints    preprocess/KptsSelector          (20 equidistant contour points)
    3D lift      retained CamRS.lift3d_for_multi_frames over N noisy depth frames
    filtering    postprocess_kpts_3d              (MAD gate + DBSCAN, ported)
    pose         preprocess/OrientAnything.estimate_frame_pca1
    mirroring    utils/utils_math.mirror_pose     (left arm sees a flipped world)

Robustness options (all off by default, README "Robustness"; models in robustness.py):
perception.mask_noise / depth_artifacts / calib_error / latency perturb what this module sees,
perception.mask_frames (multi-frame mask fusion) and perception.lift_reject (keypoint outlier
rejection) are the perception-side mitigations.

Every step (`make_side_image`) it composes the policy's clean image the same way
as `_run_side`: the robot-free render (= perfect LaMa inpainting), object
keypoints projected from the latched object poses, and the virtual gripper
wireframe from `preprocess/VisualKpts.process_single_gripper`.
"""
from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from collections import deque

from . import robustness as rb
from .env import REPO_ROOT

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# CamRS imports pyrealsense2 at module import time; the lifting helpers we use do not need it.
if "pyrealsense2" not in sys.modules:
    sys.modules["pyrealsense2"] = types.ModuleType("pyrealsense2")

from preprocess.KptsSelector import KeypointsSelectorEngine  # noqa: E402
from preprocess.OrientAnything import estimate_frame_pca1  # noqa: E402
from preprocess.VisualKpts import VisualKptsEngine  # noqa: E402
from utils.utils_math import mirror_intrinsics, mirror_pose  # noqa: E402
from .legacy_inference.CamRS import CamRS  # noqa: E402
from utils.utils_io import load_cfg  # noqa: E402

CFG_BASE = os.path.join(REPO_ROOT, "cfg", "preprocess", "base")
CAM_CFG = os.path.join(os.path.dirname(__file__), "legacy_inference", "cfg", "CamRS.yaml")


def _make_lifter() -> CamRS:
    """A CamRS instance without opening a device: only its lifting helpers are used."""
    cam = CamRS.__new__(CamRS)
    cam.cfg = load_cfg(CAM_CFG)
    return cam


def postprocess_kpts_3d(kpts_3d, point_robust_mad_k=4.0, cluster_eps_m=0.06, cluster_min_points=3) -> np.ndarray:
    """Port of AriaMimic inference/InferenceOps.postprocess_kpts_3d (MAD gate + DBSCAN)."""
    pts = [p for p in kpts_3d if p is not None and np.isfinite(p).all()]
    if len(pts) < 3:
        return np.array(pts).reshape(-1, 3)
    P = np.stack(pts, 0)
    med = np.median(P, 0)
    dists = np.linalg.norm(P - med[None], axis=1)
    md = np.median(dists)
    mad = np.median(np.abs(dists - md))
    keep = dists <= md + point_robust_mad_k * mad + 1e-6
    Pr = P[keep] if keep.sum() >= 3 else P
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=cluster_eps_m, min_samples=cluster_min_points).fit(Pr).labels_
    if labels.size == 0 or labels.max() < 0:
        return Pr
    ul, cnt = np.unique(labels[labels >= 0], return_counts=True)
    return Pr[labels == ul[np.argmax(cnt)]]


def split_mask_by_x(mask: np.ndarray, split_x: int) -> Dict[str, np.ndarray]:
    """Port of Inference.split_mask_by_x."""
    if mask is None or mask.max() == 0:
        z = np.zeros_like(mask)
        return {"right": z, "left": z.copy()}
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    right = np.zeros_like(mask)
    left = np.zeros_like(mask)
    comps = [(i, cents[i][0]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 50]
    if not comps:
        return {"right": right, "left": left}
    if len(comps) == 1:
        cm = (labels == comps[0][0]).astype(np.uint8) * 255
        right, left = cm.copy(), cm.copy()
        right[:, :split_x] = 0
        left[:, split_x:] = 0
    else:
        for i, cx in comps:
            cm = (labels == i).astype(np.uint8) * 255
            if cx >= split_x:
                right = cv2.bitwise_or(right, cm)
            else:
                left = cv2.bitwise_or(left, cm)
    return {"right": right, "left": left}


def fuse_masks(masks: List[np.ndarray], size_tol: float = 0.35) -> np.ndarray:
    """Multi-frame mask fusion: ignore empty frames and frames whose area is off the median by more
    than size_tol (a missed instance, a merged blob), then keep pixels set in most remaining frames."""
    areas = np.array([np.count_nonzero(m) for m in masks], float)
    ok = areas > 0
    if not ok.any():
        return np.zeros_like(masks[0])
    med = float(np.median(areas[ok]))
    ok &= np.abs(areas - med) <= size_tol * med
    keep = [m for m, k in zip(masks, ok) if k]
    votes = np.sum([(m > 0) for m in keep], axis=0)
    return (votes * 2 > len(keep)).astype(np.uint8) * 255


@dataclass
class SideObjects:
    """Per-side object-centric state (SideContext.objs_meta + dynamic poses)."""
    objs_meta: Dict[str, dict] = field(default_factory=dict)
    T_anchor_in_cam_static: Optional[np.ndarray] = None
    dynamic_obj_poses: Dict[str, np.ndarray] = field(default_factory=dict)


class SimPerception:
    def __init__(self, env, obj_keys=("obj1", "obj2"), anchor_key="obj1", split_x_ratio=0.5,
                 n_init_frames=10, depth_noise=True):
        self.env = env
        self.obj_keys = list(obj_keys)
        self.anchor_key = anchor_key
        self.split_x_ratio = split_x_ratio
        self.n_init_frames = n_init_frames
        self.depth_noise = depth_noise
        self.kptsselector = KeypointsSelectorEngine(os.path.join(CFG_BASE, "KptsSelector.yaml"))
        self.visualkpts = VisualKptsEngine(os.path.join(CFG_BASE, "VisualKpts.yaml"))
        self.lifter = _make_lifter()
        self.K = env.K.copy()
        self.W = env.W
        self.K_flip = mirror_intrinsics(self.K, self.W)
        self.debug: Dict[str, dict] = {}
        self._setup_robustness(env)

    # ------------------------------------------------------------------ robustness
    def _setup_robustness(self, env):
        """Per-episode perturbations; each has its own random stream derived from the episode seed."""
        cfg = env.cfg
        pc = cfg.get("perception", {}) or {}
        self.mask_frames = int(pc.get("mask_frames", 1))
        # "gt": rendered ground truth (+ mask_noise); "dinosam": the real DINO-SAM on the rendered RGB
        self.mask_source = str(pc.get("mask_source", "gt"))
        self._rgb_rng = rb.child_rng(env.rng, 5)
        self.lift_reject = dict(pc.get("lift_reject", {}) or {})
        cal = rb.params(cfg, "calib_error", rb.CALIB_ERROR)
        self.calib_dT = rb.camera_error(rb.child_rng(env.rng, 1), cal) if cal["enabled"] else None
        rb.set_policy_camera(env, self.calib_dT)
        mn = rb.params(cfg, "mask_noise", rb.MASK_NOISE)
        self.mask_noise = rb.MaskNoise(mn, rb.child_rng(env.rng, 2), list(env.obj_names)) if mn["enabled"] else None
        dc = rb.params(cfg, "distractor_confusion", rb.DISTRACTOR_CONFUSION)
        r6 = rb.child_rng(env.rng, 6)
        self.distractor_bread = [bool(dc["enabled"] and r6.random() < float(dc["p"].get(d["type"], 0.0)))
                                 for d in env.distractors]
        da = rb.params(cfg, "depth_artifacts", rb.DEPTH_ARTIFACTS)
        self.depth_artifacts = rb.DepthArtifacts(da, rb.child_rng(env.rng, 3), (env.H, env.W)) if da["enabled"] else None
        lat = rb.params(cfg, "latency", rb.LATENCY)
        self.latency = lat if lat["enabled"] else None
        self._lat_rng = rb.child_rng(env.rng, 4)
        self._last_clean = None
        # what was drawn this episode, saved with the episode log (runner._save copies env.meta)
        env.meta.pop("robustness", None)
        info = {}
        if self.calib_dT is not None:
            info["calib_dT"] = np.round(self.calib_dT, 6).tolist()
        if self.mask_noise is not None:
            info["mask_bias_px"] = {k: round(v, 2) for k, v in self.mask_noise.bias.items()}
        if self.depth_artifacts is not None:
            info["depth_scale"] = round(self.depth_artifacts.scale, 5)
        if any(self.distractor_bread):
            info["distractor_as_bread"] = [d["type"] for d, b in zip(env.distractors, self.distractor_bread) if b]
        if info:
            env.meta["robustness"] = info
        env.obs_history = None
        if self.latency:
            age_max = (float(lat["image_ms"]) + float(lat["jitter_ms"])) / 1000.0
            env.obs_history = deque(maxlen=int(age_max / env.dt) + 4)

    # ------------------------------------------------------------------
    def object_masks(self, gt: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """One detector frame. obj1 = every bread on the table ("a piece of bread ."), obj2 = plate
        ("a plate ."). Ground truth, or with perception.mask_noise DINO-SAM-like errors on top of it."""
        gt = self.env.render_segmentation() if gt is None else gt
        mn = self.mask_noise
        if mn is not None:
            gt = {name: mn.instance(name, mk) for name, mk in gt.items()}
        if self.env.cfg.get("task") == "pick_cube":
            return self.env.object_role_masks(gt)
        obj1 = np.zeros((self.env.H, self.env.W), np.uint8)
        for name, mk in gt.items():
            if name.startswith("bread"):
                obj1 = cv2.bitwise_or(obj1, mk)
        for dm, as_bread in zip(getattr(self, "_distractor_masks", []), self.distractor_bread):
            if as_bread:                     # a bread-like distractor segmented as bread (whole object)
                obj1 = cv2.bitwise_or(obj1, dm)
        if mn is not None:
            b = mn.blob(obj1.shape, table_rows=int(0.3 * self.env.H), anchors=list(gt.values()))
            if b is not None:
                obj1 = cv2.bitwise_or(obj1, b)
        return {"obj1": obj1, "obj2": gt["plate"]}

    def detect_masks(self) -> Dict[str, np.ndarray]:
        """The masks perception works with: one detector frame (the released pipeline), or the fusion
        of perception.mask_frames frames (mitigation: drop empty / off-size frames, pixel majority vote)."""
        if self.mask_source == "dinosam":
            return self._dinosam_masks()
        gt = self.env.render_segmentation()
        if any(self.distractor_bread) and not hasattr(self, "_distractor_masks"):
            self._distractor_masks = rb.distractor_masks(self.env, self.env.distractors)
        if self.mask_frames <= 1:
            return self.object_masks(gt)
        frames = [self.object_masks(gt) for _ in range(self.mask_frames)]
        return {k: fuse_masks([f[k] for f in frames]) for k in self.obj_keys}

    def _dinosam_masks(self) -> Dict[str, np.ndarray]:
        """The released DINO-SAM (Grounding-DINO + SAM2) on the rendered camera frame, robot included,
        as on the real rig. With mask_frames > 1 each frame gets fresh sensor noise (2 grey levels)."""
        from .dinosam_client import client
        rgb = self.env.render_rgb()
        frames = []
        for _ in range(max(1, self.mask_frames)):
            img = rgb
            if self.mask_frames > 1:
                img = np.clip(rgb + self._rgb_rng.normal(0, 2.0, rgb.shape), 0, 255).astype(np.uint8)
            frames.append(client().segment(img))
        if len(frames) == 1:
            return frames[0]
        return {k: fuse_masks([f[k] for f in frames]) for k in self.obj_keys}

    def _process_mask_to_3d(self, mask, depth_list, K):
        if mask is None or mask.max() == 0:
            return np.zeros((0, 3)), []
        k2d = self.kptsselector.select_points(mask)
        lr = self.lift_reject
        lift_mask = mask
        if lr.get("erode_px"):
            # lift from pixels at least erode_px inside the mask: a dilated mask or flying pixels at the
            # silhouette then cannot pull a contour keypoint onto the table behind it
            e = int(lr["erode_px"])
            em = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
            if np.count_nonzero(em) > 200:
                lift_mask = em
        k3d = []
        for uv in k2d:
            pt, _ = self.lifter.lift3d_for_multi_frames(uv, depth_list, lift_mask, K)
            k3d.append(pt)
        if lr.get("table_margin") is not None:
            k3d = self._reject_off_object(k3d, float(lr["table_margin"]))
        return k3d, k2d

    def _reject_off_object(self, k3d, margin: float):
        """Drop lifted keypoints that sit on (or below) the table plane known from the calibration: a
        bread keypoint lifted that low took its depth from the table, not the bread."""
        Tcw = self.env.T_cam_in_world        # calibration (the robot's belief), not the true camera
        out = []
        for p in k3d:
            if p is None or not np.isfinite(p).all():
                out.append(p)
                continue
            zw = float(Tcw[2, :3] @ p + Tcw[2, 3])
            out.append(p if zw > margin else None)
        return out

    def _build_side_objs_meta(self, ok3d, ok2d) -> Tuple[dict, np.ndarray]:
        objs_meta, T_anchor, anchor_center = {}, np.eye(4), None
        if self.anchor_key in ok3d:
            k3d = ok3d[self.anchor_key]
            T = estimate_frame_pca1(k3d, is_anchor=True)[0] if len(k3d) >= 3 else np.eye(4)
            Ti = np.linalg.inv(T)
            kl = (Ti[:3, :3] @ k3d.T + Ti[:3, 3:4]).T if len(k3d) else np.zeros((0, 3))
            objs_meta[self.anchor_key] = {"T_obj_in_cam_static": T, "kpts_2d": ok2d[self.anchor_key],
                                          "kpts_3d": k3d, "kpts_local": kl}
            anchor_center = T[:3, 3]
            T_anchor = T.copy()
        for k in sorted(ok3d):
            if k == self.anchor_key:
                continue
            k3d = ok3d[k]
            T = (estimate_frame_pca1(k3d, is_anchor=False, anchor_center_cam=anchor_center)[0]
                 if len(k3d) >= 3 else np.eye(4))
            Ti = np.linalg.inv(T)
            kl = (Ti[:3, :3] @ k3d.T + Ti[:3, 3:4]).T if len(k3d) else np.zeros((0, 3))
            objs_meta[k] = {"T_obj_in_cam_static": T, "kpts_2d": ok2d[k], "kpts_3d": k3d, "kpts_local": kl}
        return objs_meta, T_anchor

    def estimate_objects(self, sides: List[str], shared_context: bool = True) -> Dict[str, SideObjects]:
        """do_object_centric_bimanual (or the single-arm variant when sides == ['right'])."""
        masks_full = self.detect_masks()
        depth_list = [self.env.render_depth(noise=self.depth_noise, artifacts=self.depth_artifacts)
                      for _ in range(self.n_init_frames)]
        split_x = int(self.W * self.split_x_ratio)
        out = {}
        bimanual = len(sides) > 1
        for side in sides:
            is_flip = side == "left"
            ok3d, ok2d = {}, {}
            for k in self.obj_keys:
                full = masks_full[k]
                need_split = bimanual and ((not shared_context) or k == self.anchor_key)
                mask = split_mask_by_x(full, split_x)[side] if need_split else full
                k3d, k2d = self._process_mask_to_3d(mask, depth_list, self.K)
                k3d = postprocess_kpts_3d(k3d)
                if is_flip and len(k3d) > 0:
                    k3d = k3d.copy()
                    k3d[:, 0] = -k3d[:, 0]
                    k2d = [(self.W - 1 - u, v) for u, v in k2d]
                ok3d[k], ok2d[k] = k3d, k2d
            meta, T_anchor = self._build_side_objs_meta(ok3d, ok2d)
            so = SideObjects(objs_meta=meta, T_anchor_in_cam_static=T_anchor,
                             dynamic_obj_poses={k: v["T_obj_in_cam_static"].copy() for k, v in meta.items()})
            out[side] = so
            self.debug[side] = {"masks": masks_full}
        return out

    # ------------------------------------------------------------------
    def clean_background(self) -> np.ndarray:
        """Robot-free render == the LaMa-inpainted frame of the real loop (BGR). With
        perception.latency the frame is image_ms + U(0, jitter_ms) old, and with p_drop the camera
        did not deliver a new frame (the previous image is reused)."""
        lat, env = self.latency, self.env
        if lat is None:
            return env.render_rgb(hide_robot=True)
        rng = self._lat_rng
        if self._last_clean is not None and rng.random() < float(lat["p_drop"]):
            return self._last_clean
        age = (float(lat["image_ms"]) + rng.uniform(0.0, float(lat["jitter_ms"]))) / 1000.0
        qpos = None
        hist = env.obs_history
        if hist:
            t_want = env.t - age
            qpos = hist[0][1]
            for t_h, q_h in hist:
                if t_h <= t_want + 1e-9:
                    qpos = q_h
                else:
                    break
        img = env.render_rgb(hide_robot=True, qpos=qpos) if qpos is not None else env.render_rgb(hide_robot=True)
        self._last_clean = img
        return img

    def make_side_image(self, side: str, so: SideObjects, clean_bg: np.ndarray, arm, is_grasp: bool,
                        draw_obj_kpts: bool = True, draw_arm_kpts: bool = True) -> np.ndarray:
        """Policy input image for one side (in that side's domain: flipped for the left arm)."""
        is_flip = side == "left"
        vis = cv2.flip(clean_bg, 1) if is_flip else clean_bg.copy()
        draw_K = self.K.copy()
        active_K = (self.K_flip if is_flip else self.K).copy()
        resolution = self.env.cfg.get("appearance", {}).get("keypoint_resolution")
        if resolution:
            height, width = resolution
            scale = np.diag([width / vis.shape[1], height / vis.shape[0], 1])
            draw_K, active_K = scale @ draw_K, scale @ active_K
            vis = cv2.resize(vis, (width, height), interpolation=cv2.INTER_LINEAR)
        self.visualkpts.reset_obj_counter()
        if draw_obj_kpts:
            for ok in so.objs_meta:
                kl = so.objs_meta[ok]["kpts_local"]
                if len(kl) == 0:
                    continue
                Td = so.dynamic_obj_poses.get(ok, so.objs_meta[ok]["T_obj_in_cam_static"])
                p3 = (Td[:3, :3] @ kl.T).T + Td[:3, 3]
                Z = p3[:, 2] + 1e-6
                u = active_K[0, 0] * p3[:, 0] / Z + active_K[0, 2]
                v = active_K[1, 1] * p3[:, 1] / Z + active_K[1, 2]
                vis = self.visualkpts.process_single_obj(vis, np.stack([u, v], 1))
        if draw_arm_kpts:
            if is_flip:
                tmp = self.visualkpts.process_single_gripper(cv2.flip(vis, 1), arm.get_p_in_cam(), is_grasp, draw_K)
                vis = cv2.flip(tmp, 1)
            else:
                vis = self.visualkpts.process_single_gripper(vis, arm.get_p_in_cam(), is_grasp, draw_K)
        if resolution:
            vis = cv2.resize(vis, (clean_bg.shape[1], clean_bg.shape[0]), interpolation=cv2.INTER_AREA)
        return vis
