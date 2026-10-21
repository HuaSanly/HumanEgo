"""Static training references, dynamic object tokens and camera-frame compatibility."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanego_sim.driver import T_to_p
from humanego_sim.perception import SideObjects, mirror_pose
from humanego_sim.runner import EpisodeRunner, ICTPolicy, ObjectState, SideCtx, T_ALIGN_ARIA


def pose(xyz, angles=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    T[:3, 3] = xyz
    return T


def policy(frame_mode):
    pol = ICTPolicy.__new__(ICTPolicy)
    pol.frame_mode, pol.action_mode = frame_mode, "absolute"
    pol.device, pol.sides = "cpu", ["right"]
    pol.single_hand, pol.single_hand_side = True, "right"
    pol.max_ict, pol.ict_dim, pol.pred_horizon = 8, 20, 1
    pol.pos_mean = np.array([0.03, -0.02, 0.1], np.float32)
    pol.pos_std = np.array([0.2, 0.3, 0.4], np.float32)
    pol.use_object_tokens, pol.use_region_attn = True, False
    pol.img_size = (8, 8)
    return pol


def objects(anchor):
    points = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]) + anchor[:3, 3]
    return SideObjects(
        objs_meta={"obj1": {"kpts_3d": points}},
        T_anchor_in_cam_static=anchor.copy(),
        dynamic_obj_poses={"obj1": anchor.copy(), "obj2": pose([0.2, 0.1, 0.5])},
    )


def test_anchor_ict_and_decode_match_translated_rotated_reference():
    ref = pose([0.2, -0.1, 0.6], (15, -10, 35))
    hand_in_ref = pose([0.04, 0.08, 0.12], (20, 5, -15))
    hand = ref @ hand_in_ref
    # The anchor object has moved since the reference was established.
    states = {"obj1": ObjectState(ref @ pose([0.1, 0, 0], (0, 0, 20))),
              "obj2": ObjectState(ref @ pose([-0.04, 0.02, 0.01]))}
    pol = policy("anchor_frame")
    ict, mask = pol.build_ict({"right": hand}, {"right": 0}, states,
                              reference_anchor=ObjectState(ref))
    np.testing.assert_allclose(ict[0, 0, 1:10], pol._encode(hand_in_ref), atol=1e-6)
    np.testing.assert_allclose(ict[0, 1, 1:10], pol._encode(np.linalg.inv(ref) @ states["obj1"].T_in_cam), atol=1e-6)
    camera_ict, camera_mask = policy("camera_frame").build_ict({"right": hand}, {"right": 0}, states)
    np.testing.assert_array_equal(mask, camera_mask)
    np.testing.assert_allclose(ict[0, :3, 10:19], camera_ict[0, :3, 10:19], atol=1e-6)
    encoded = pol._encode(hand_in_ref)
    decoded = pol.decode_ee_in_cam(encoded[:3], encoded[3:], ObjectState(ref), T_ALIGN_ARIA)
    np.testing.assert_allclose(decoded, hand @ np.linalg.inv(T_ALIGN_ARIA), atol=1e-6)


@pytest.mark.parametrize("frame_mode", ["camera_frame", "anchor_frame"])
def test_optional_reference_preserves_legacy_ict(frame_mode):
    pol = policy(frame_mode)
    ref = ObjectState(pose([0.2, 0.1, 0.6], (10, 20, 30)))
    states = {"obj1": ref}
    hands = {"right": pose([0.1, -0.1, 0.7], (30, -20, 10))}
    original = pol.build_ict(hands, {"right": 0}, states)
    explicit_default = pol.build_ict(hands, {"right": 0}, states, reference_anchor=None)
    for actual, expected in zip(explicit_default, original):
        np.testing.assert_array_equal(actual, expected)
    if frame_mode == "camera_frame":
        ignored_reference = pol.build_ict(hands, {"right": 0}, states,
                                          reference_anchor=ObjectState(pose([1, 2, 3])))
        for actual, expected in zip(ignored_reference, original):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("frame_mode", ["camera_frame", "anchor_frame"])
def test_runner_uses_one_reference_for_ict_and_decode(side, frame_mode):
    pol = policy(frame_mode)
    anchor = pose([0.2, 0.1, 0.6], (10, -20, 30))
    target_hand = anchor @ pose([0.04, -0.02, 0.08], (20, 10, -15))
    side_anchor = mirror_pose(anchor) if side == "left" else anchor
    side_hand = mirror_pose(target_hand) if side == "left" else target_hand
    so = objects(side_anchor)
    arm = SimpleNamespace(get_T_ee_in_cam=lambda: target_hand @ np.linalg.inv(T_ALIGN_ARIA),
                          get_gripper_q=lambda: 1.0, gripper_settled_closed=lambda: False)
    env = SimpleNamespace(arms={side: arm}, T_cam_in_world=pose([0.3, 0, 0.5], (20, 0, 0)), W=8, H=8)
    cfg = {"task": "pick_cube", "mode": "single", "control": {}}
    runner = EpisodeRunner(cfg, policy=pol, env=env)
    runner.perception = SimpleNamespace(K=np.eye(3), K_flip=np.eye(3),
                                        make_side_image=Mock(return_value=np.zeros((8, 8, 3), np.uint8)))
    ctx = SideCtx(side, side == "left", so)
    if frame_mode == "anchor_frame":
        ctx.reference_anchor = runner._static_reference_anchor(side, so)
        # Updating perception or a latched object's pose must not move the reference.
        so.T_anchor_in_cam_static[:] = pose([1, 2, 3])
        so.dynamic_obj_poses["obj1"][:] = pose([0.4, 0.2, 0.5], (0, 0, 60))
        encoded = pol._encode(np.linalg.inv(side_anchor) @ side_hand)
    else:
        encoded = pol._encode(side_hand)
    pol.infer = Mock(return_value=({"right": (encoded[None, :3], encoded[None, 3:], np.zeros((1, 1)))}, 0.0))
    pol.build_ict = Mock(wraps=pol.build_ict)
    pol.decode_ee_in_cam = Mock(wraps=pol.decode_ee_in_cam)
    raw, _, _, world, _ = runner._run_side(ctx, np.zeros((8, 8, 3), np.uint8), 0)
    target_ee = target_hand @ np.linalg.inv(T_ALIGN_ARIA)
    np.testing.assert_allclose(raw[0], T_to_p(target_ee), atol=1e-6)
    np.testing.assert_allclose(world[0], (env.T_cam_in_world @ target_ee)[:3, 3], atol=1e-6)
    if frame_mode == "anchor_frame":
        np.testing.assert_array_equal(ctx.reference_anchor.T_in_cam, side_anchor)
        assert pol.build_ict.call_args.kwargs["reference_anchor"] is ctx.reference_anchor
        assert pol.decode_ee_in_cam.call_args.args[2] is ctx.reference_anchor
        ict = pol.infer.call_args.args[1]
        np.testing.assert_allclose(ict[0, 0, 1:10], encoded, atol=1e-6)
    else:
        assert pol.build_ict.call_args.kwargs == {}
        assert pol.decode_ee_in_cam.call_args.args[2] is None


@pytest.mark.parametrize("invalid", ["missing_object", "missing_pose", "few_points", "nan_points",
                                     "no_static_pose", "wrong_shape", "nan_pose", "singular",
                                     "reflection", "bad_bottom_row"])
def test_anchor_requires_valid_initial_perception(invalid):
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.anchor_key = "obj1"
    so = objects(pose([0.2, 0.1, 0.6]))
    if invalid == "missing_object":
        so.objs_meta.clear()
    elif invalid == "missing_pose":
        so.dynamic_obj_poses.clear()
    elif invalid == "few_points":
        so.objs_meta["obj1"]["kpts_3d"] = np.zeros((2, 3))
    elif invalid == "nan_points":
        so.objs_meta["obj1"]["kpts_3d"][0] = np.nan
    elif invalid == "no_static_pose":
        so.T_anchor_in_cam_static = None
    elif invalid == "wrong_shape":
        so.T_anchor_in_cam_static = np.eye(3)
    elif invalid == "nan_pose":
        so.T_anchor_in_cam_static[0, 3] = np.nan
    elif invalid == "singular":
        so.T_anchor_in_cam_static[:3, :3] = 0
    elif invalid == "reflection":
        so.T_anchor_in_cam_static[0, 0] = -1
    else:
        so.T_anchor_in_cam_static[3, 0] = 1
    with pytest.raises(ValueError, match="anchor_frame: right anchor 'obj1'"):
        runner._static_reference_anchor("right", so)


@pytest.mark.parametrize("frame_mode,action_mode", [("world_frame", "absolute"),
                                                   ("camera_frame", "delta"), ("anchor_frame", "delta")])
def test_runner_rejects_unsupported_modes(frame_mode, action_mode):
    with pytest.raises(NotImplementedError, match="absolute checkpoints"):
        EpisodeRunner({"control": {}}, policy=SimpleNamespace(frame_mode=frame_mode, action_mode=action_mode),
                      env=SimpleNamespace())


def test_anchor_runner_never_falls_back_without_initialized_reference():
    runner = EpisodeRunner({"control": {}}, policy=policy("anchor_frame"),
                           env=SimpleNamespace(arms={"right": SimpleNamespace()}))
    with pytest.raises(ValueError, match="static reference anchor was not initialized"):
        runner._run_side(SideCtx("right", False, objects(np.eye(4))), np.zeros((8, 8, 3), np.uint8), 0)
