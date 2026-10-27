from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from lerobot_policy_humanego.config import load_config, validate_runtime_config
from lerobot_policy_humanego.data_types import (
    ObjectState,
    Observation,
    RobotState,
    StereoFrame,
)
from lerobot_policy_humanego.perception import Perception
from lerobot_policy_humanego.rendering import gripper_points, render_scene
from lerobot_policy_humanego.stereo import StereoDepthEstimator

ROOT = Path(__file__).resolve().parents[2]


def config():
    cfg = load_config(ROOT / "cfg/inference/params.yaml")
    cfg.perception["object_update_mode"] = "continuous"
    K = [[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]]
    cfg.intrinsics.update(left={"K": K}, right={"K": K})
    T = np.eye(4)
    T[0, 3] = 0.064
    cfg.extrinsics.update(
        T_left_from_right=T.tolist(), T_EE_from_hand=np.eye(4).tolist()
    )
    return cfg


def pose(x=0, y=0, z=0.8):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [x, y, z]
    return T


def observation(x=0, gripper=0):
    image = np.zeros((480, 640, 3), np.uint8)
    return Observation(StereoFrame(image, image), RobotState(pose(x), gripper))


def test_metric_stereo_and_reverse_consistency():
    estimator = StereoDepthEstimator(config())
    rng = np.random.default_rng(91)
    left = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-24] = left[:, 24:]
    result = estimator.estimate(StereoFrame(left, right))
    center = result.depth_m[30:-30, 150:490]
    assert np.isfinite(center).mean() > 0.9
    assert np.nanmedian(center) == pytest.approx(400 * 0.064 / 24, rel=0.01)
    assert not result.valid_mask[:, :24].any()
    broken = estimator.estimate(
        StereoFrame(left, rng.integers(0, 256, left.shape, dtype=np.uint8))
    )
    assert broken.valid_mask.mean() < 0.15
    mask = np.zeros((480, 640), bool)
    mask[150:330, 220:430] = True
    pts = estimator.points_from_mask(result, mask)
    assert len(pts) >= 8
    assert np.max(np.abs(pts[:, 2] - 400 * 0.064 / 24)) < 0.01


@pytest.mark.parametrize("field", ["direction", "rotation", "units", "intrinsics"])
def test_rejects_wrong_rectification(field):
    cfg = config()
    if field == "direction":
        cfg.extrinsics["T_left_from_right"][0][3] *= -1
    if field == "rotation":
        cfg.extrinsics["T_left_from_right"][0][0] = 0
    if field == "units":
        cfg.extrinsics["translation_unit"] = "mm"
    if field == "intrinsics":
        cfg.intrinsics["right"]["K"] = np.eye(3).tolist()
    with pytest.raises(ValueError):
        validate_runtime_config(cfg)


class EmptyEngine:
    def inpaint(self, *args):
        raise AssertionError("Empty mask must skip LaMa")


def scene_perception():
    now = [0.0]
    perception = Perception(
        config(), segmentation=object(), inpainter=EmptyEngine(), clock=lambda: now[0]
    )
    objects = {
        "obj1": ObjectState(pose(-0.3), np.zeros((8, 3), np.float32)),
        "obj2": ObjectState(pose(0), np.zeros((8, 3), np.float32)),
    }
    perception.observe = lambda obs: None
    perception._locate = lambda pts, existing=None: deepcopy(objects)
    return perception, objects, now


def test_scene_initialization_requires_both_objects():
    p, objects, _ = scene_perception()
    del objects["obj2"]
    with pytest.raises(ValueError, match="obj2"):
        p.initialize_scene(observation())


def test_moving_objects_fixed_reference_grasp_occlusion_release_expiry():
    p, objects, now = scene_perception()
    scene = p.initialize_scene(observation())
    fixed = scene.reference_anchor.T_in_cam.copy()
    now[0] = 0.1
    objects["obj1"].T_in_cam = pose(-0.2)
    scene = p.update_scene(observation(), scene)
    assert scene.objects["obj1"].T_in_cam[0, 3] == pytest.approx(-0.2)
    np.testing.assert_array_equal(fixed, scene.reference_anchor.T_in_cam)
    now[0] = 0.2
    scene = p.update_scene(observation(gripper=1), scene)
    assert scene.latched_object == "obj2"
    del objects["obj2"]
    now[0] = 0.4
    scene = p.update_scene(observation(x=0.12, gripper=1), scene)
    assert scene.objects["obj2"].T_in_cam[0, 3] == pytest.approx(0.12)
    now[0] = 0.6
    scene = p.update_scene(observation(x=0.18, gripper=0), scene)
    assert scene.latched_object is None
    assert scene.objects["obj2"].T_in_cam[0, 3] == pytest.approx(0.18)
    now[0] = 1.7
    with pytest.raises(ValueError, match="obj2.*expired"):
        p.update_scene(observation(), scene)
    # A failed update cannot mutate the last valid scene/reference.
    np.testing.assert_array_equal(fixed, scene.reference_anchor.T_in_cam)
    objects["obj2"] = ObjectState(pose(0.25))
    now[0] = 1.8
    scene = p.update_scene(observation(), scene)
    assert scene.objects["obj2"].T_in_cam[0, 3] == pytest.approx(0.25)


def test_non_grasped_anchor_can_never_latch():
    p, objects, _ = scene_perception()
    scene = p.initialize_scene(observation(x=-0.3, gripper=1))
    assert scene.latched_object is None


def test_empty_arm_mask_and_render_axis():
    p, _, _ = scene_perception()
    scene = p.initialize_scene(observation())
    p.masks = {key: np.zeros((480, 640), bool) for key in ["obj1", "obj2", "arms"]}
    image = p.erase_arms(observation().frame)
    assert p.diagnostics["inpainting"] == "empty_mask"
    render = render_scene(image, scene, p.stereo.K)
    assert render.shape == (1232, 1624, 3)
    points = np.asarray([pt for pt, _ in gripper_points(0)])
    assert points[:, 1].min() == pytest.approx(-0.16)
    assert points[:, 1].max() == 0
    assert np.all(points[:, 2] == 0)


def test_render_matches_training_segments_and_point_order():
    from preprocess.VisualKpts import VisualKptsEngine, interpolate_bgr
    from lerobot_policy_humanego.rendering import canvas_intrinsics

    engine = VisualKptsEngine(str(ROOT / "cfg/preprocess/base/VisualKpts.yaml"))
    p, _, _ = scene_perception()
    scene = p.initialize_scene(observation())
    K = canvas_intrinsics(p.stereo.K)
    for grasp in (0, 1):
        scene.gripper = grasp
        scene.objects = {}
        actual = render_scene(np.zeros((480, 640, 3), np.uint8), scene, p.stereo.K)
        expected = np.zeros_like(actual)
        segments, closed = engine._get_segments_3d_oriented_binary(
            {
                "midpoint_pose_opt_world": scene.hand_in_cam.tolist(),
                "grasp_state": grasp,
            }
        )
        palette = [
            getattr(engine.cfg, f"color_hand_closed_{i}")
            if closed
            else getattr(engine.cfg, f"color_hand_{i}_line")
            for i in (1, 2, 3, 4)
        ]
        c1, c2, c3, c4 = palette
        for index, (a, b) in enumerate(segments):
            start, end = (c4, c3) if index < 2 else (c2, c1)
            points = np.linspace(a, b, 6)
            projected = points @ K.T
            for j, pt in enumerate(projected):
                uv = pt[:2] / pt[2]
                engine._draw_kpt_with_core(
                    expected, *uv, interpolate_bgr(start, end, j / 5), 4
                )
        np.testing.assert_array_equal(actual, expected)


def test_event_updates_skip_static_objects_and_relocate_after_release():
    p, objects, now = scene_perception()
    p.config.perception["object_update_mode"] = "events"
    requested = []

    def observe(obs, object_keys=("obj1", "obj2"), scene=None):
        requested.append(tuple(object_keys))
        return {key: obj for key, obj in objects.items() if key in object_keys}

    p.observe = observe
    p._locate = lambda pts, existing=None: deepcopy(pts)
    scene = p.initialize_scene(observation())
    fixed = scene.reference_anchor.T_in_cam.copy()
    now[0] = 10
    scene = p.update_scene(observation(), scene)
    assert requested[-1] == ()
    assert scene.last_seen == {"obj1": 0, "obj2": 0}
    now[0] = 11
    scene = p.update_scene(observation(gripper=1), scene)
    assert scene.latched_object == "obj2"
    del objects["obj2"]
    now[0] = 12
    scene = p.update_scene(observation(x=0.1, gripper=1), scene)
    assert requested[-1] == ()
    assert scene.objects["obj2"].T_in_cam[0, 3] == pytest.approx(0.1)
    now[0] = 13
    scene = p.update_scene(observation(x=0.2), scene)
    assert requested[-1] == ("obj2",)
    assert scene.relocalization_started == {"obj2": 13}
    now[0] = 13.5
    scene = p.update_scene(observation(x=0.2, gripper=1), scene)
    assert scene.latched_object is None  # Pending relocation cannot be grasped again.
    now[0] = 14.1
    with pytest.raises(ValueError, match="obj2.*expired"):
        p.update_scene(observation(), scene)
    objects["obj2"] = ObjectState(pose(0.25))
    scene = p.update_scene(observation(), scene)
    assert not scene.relocalization_started
    assert scene.objects["obj2"].T_in_cam[0, 3] == pytest.approx(0.25)
    now[0] = 30
    scene = p.update_scene(observation(), scene)
    assert requested[-1] == ()
    np.testing.assert_array_equal(fixed, scene.reference_anchor.T_in_cam)


def test_event_masks_use_current_projection_without_depth_or_object_detection():
    from unittest.mock import Mock

    p, objects, _ = scene_perception()
    scene = p.initialize_scene(observation())
    points = np.array([[-0.03, -0.03, 0], [0.03, -0.03, 0], [0.03, 0.03, 0]])
    scene.objects["obj2"].kpts_local = points
    before = p._object_box(scene.objects["obj2"])
    scene.objects["obj2"].T_in_cam[0, 3] += 0.1
    after = p._object_box(scene.objects["obj2"])
    np.testing.assert_allclose(after - before, [50, 0, 50, 0], atol=1e-4)
    p.segmentation = Mock()
    p.segmentation.segment.return_value = {
        k: np.zeros((480, 640), bool) for k in ("obj1", "obj2", "arms")
    }
    p.stereo.estimate = Mock(side_effect=AssertionError("No depth needed"))
    points = Perception.observe(p, observation(), (), scene)
    assert points == {}
    assert p.segmentation.segment.call_args.args[1] == {}
    assert set(p.segmentation.segment.call_args.kwargs["object_boxes"]) == {
        "obj1",
        "obj2",
    }
    p.stereo.estimate.assert_not_called()
