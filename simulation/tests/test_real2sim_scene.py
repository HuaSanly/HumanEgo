"""Physical and camera invariants needed for the appearance/geometry ablation."""
import os
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from humanego_sim.container import contains_points
from humanego_sim.env import SIM_ROOT, ServeBreadSim, deep_update, load_config
from humanego_sim.scene import rig_geometry


def config():
    return load_config(os.path.join(SIM_ROOT, "configs", "pick_cube.yaml"))


def test_camera_move_preserves_bases_and_updates_all_extrinsics():
    original = config()
    modified = deep_update(original, {"rig": {"camera_pose": {"position": [0, 0, 0.53], "pitch_deg": 49}}})
    a, b = rig_geometry(original, ["right", "left"]), rig_geometry(modified, ["right", "left"])
    for side in a.T_base_in_world:
        np.testing.assert_allclose(a.T_base_in_world[side], b.T_base_in_world[side])
        np.testing.assert_allclose(b.T_cam_in_world @ b.T_base_in_cam[side], b.T_base_in_world[side], atol=1e-12)
    np.testing.assert_allclose(b.T_cam_in_world[:3, :3].T @ b.T_cam_in_world[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(b.T_cam_in_world[:3, :3]) == pytest.approx(1)


def rounded_config():
    return deep_update(config(), {"scene": {"objects": {"container": {"shape": "rounded", "size": [0.15, 0.23, 0.096],
                                                                      "radius": 0.018, "taper": 0.84},
                                                       "cube": {"size": [0.0335] * 3}}}})


def test_rounded_container_is_open_supports_cube_and_has_correct_mass():
    cfg = rounded_config()
    env = ServeBreadSim(cfg, render=False)
    env.reset()
    center = env.object_pos("container")
    env.place_object("cube", [*center[:2], 0.20, 1, 0, 0, 0])
    mujoco.mj_forward(env.m, env.d)
    env.step(1000)
    expected = cfg["scene"]["objects"]["container"]["wall"] + cfg["scene"]["objects"]["cube"]["size"][2] / 2
    assert env.object_pos("cube")[2] == pytest.approx(expected, abs=0.001)
    for name in env.obj_names:
        assert env.m.body_mass[env.obj_bodies[name]] == pytest.approx(cfg["scene"]["objects"][name]["mass"], abs=1e-6)
    assert np.isfinite(env.d.qpos).all()


def test_rounded_cavity_rejects_outer_corners_and_tapered_bottom():
    obj = rounded_config()["scene"]["objects"]["container"]
    assert contains_points(np.array([[0, 0, 0], [0.01, 0.02, -0.02]]), obj)
    assert not contains_points(np.array([[0.070, 0.110, 0.04]]), obj)
    assert not contains_points(np.array([[0.068, 0, -0.04]]), obj)
    assert not contains_points(np.array([[0, 0, 0.07]]), obj)


def test_appearance_preserves_dynamic_geometry_and_contacts():
    a = ServeBreadSim(config(), render=False)
    b = ServeBreadSim(deep_update(config(), {"appearance": {"table": {"rgba": [0.7, 0.7, 0.7, 1]},
                                                           "markers": [[0.1, 0.2]],
                                                           "wall": {"x": 0.95, "rgba": [0.6, 0.6, 0.6, 1]}}}), render=False)
    for name in a.obj_names:
        aid, bid = a.obj_bodies[name], b.obj_bodies[name]
        assert a.m.body_mass[aid] == pytest.approx(b.m.body_mass[bid])
        np.testing.assert_allclose(a.m.body_inertia[aid], b.m.body_inertia[bid])
        ga, gb = np.where(a.m.geom_bodyid == aid)[0], np.where(b.m.geom_bodyid == bid)[0]
        np.testing.assert_allclose(a.m.geom_size[ga], b.m.geom_size[gb])
        np.testing.assert_allclose(a.m.geom_friction[ga], b.m.geom_friction[gb])


def test_camera_intrinsics_match_render_projection():
    intrinsics = {"fx": 407.0, "fy": 409.0, "cx": 319.5, "cy": 239.5}
    env = ServeBreadSim(deep_update(config(), {"camera_intrinsics": intrinsics}), render=False)
    cid = env.m.camera("policy_cam").id
    sensor, focal = env.m.cam_sensorsize[cid], env.m.cam_intrinsic[cid, :2]
    rendered_focal = focal / sensor * env.m.cam_resolution[cid]
    np.testing.assert_allclose(rendered_focal, [407, 409], rtol=1e-6)
    assert env.K[0, 0] == 407 and env.K[1, 1] == 409


def test_overlay_matches_training_relative_dot_size():
    from humanego_sim.perception import SimPerception
    background = np.zeros((480, 640, 3), np.uint8)
    objects = SimpleNamespace(objs_meta={"obj1": {"kpts_local": np.array([[0, 0, 1]]),
                                                 "T_obj_in_cam_static": np.eye(4)}}, dynamic_obj_poses={})
    counts = []
    for appearance in ({}, {"keypoint_resolution": [1232, 1624]}):
        cfg = config()
        cfg["appearance"] = appearance
        env = ServeBreadSim(cfg, render=False)
        perception = SimPerception(env)
        image = perception.make_side_image("right", objects, background, None, False, draw_arm_kpts=False)
        y, x = np.where(image.any(axis=2))
        counts.append(len(x))
        assert abs(x.mean() - env.K[0, 2]) < 1 and abs(y.mean() - env.K[1, 2]) < 1
        assert image.shape == background.shape and image.dtype == np.uint8
    assert counts[1] < counts[0] * 0.4


def test_ablation_pairing_and_bounded_randomization_do_not_mutate_configs():
    from real2sim_ablation import trial_config
    original = rounded_config()
    original["rig"]["camera_pose"] = {"position": [0, 0, 0.53], "pitch_deg": 49}
    varied = deep_update(original, {"ablation_randomization": True})
    first, base = trial_config(original, 4, 20261023)
    second, variation = trial_config(varied, 4, 20261023)
    assert base["scene"] == variation["scene"]
    assert first["scene"]["objects"] == second["scene"]["objects"]
    assert first["control"] == second["control"]
    assert abs(second["rig"]["camera_pose"]["position"][2] - 0.53) <= 0.02
    assert abs(second["rig"]["camera_pose"]["pitch_deg"] - 49) <= 3
    assert 0.92 <= variation["light"]["intensity"] <= 1.08
    assert varied["rig"]["camera_pose"]["position"][2] == 0.53


def test_report_uses_requested_prefix_instead_of_extra_completed_trials(tmp_path):
    from real2sim_ablation import report
    visual = {label: {"geometry_mae": 0.01, "object_rgb_mae": 1} for label in "ABCDE"}
    (tmp_path / "visual_metrics.json").write_text(json.dumps({"metrics": visual}))
    for label in "ABCDE":
        for index in range(11):
            path = tmp_path / "episodes" / label / f"ep{index:03d}"
            path.mkdir(parents=True)
            value = {"idx": index, "success": index == 10, "physical_lift": False, "physical_transport": False,
                     "per_side": {"right": {"success": index == 10, "grasped": False, "inside_container": False,
                                             "dist_to_container_center": 0.2}}}
            (path / "result.json").write_text(json.dumps(value))
    report(tmp_path, 10)
    result = json.loads((tmp_path / "summary.json").read_text())
    for group in result["groups"].values():
        assert group["completed"] == group["valid"] == 10
        assert group["successes"] == 0
