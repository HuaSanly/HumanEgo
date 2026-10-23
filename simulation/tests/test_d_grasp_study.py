"""The D study observer must detect real contacts without perturbing physics."""
import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from d_grasp_study import MeasuredSim
from humanego_sim.env import SIM_ROOT, ServeBreadSim, load_config
from humanego_sim.runner import EpisodeRunner
from types import SimpleNamespace


def test_contact_observer_preserves_physics():
    cfg = load_config(os.path.join(SIM_ROOT, "configs", "pick_cube.yaml"))
    measured, ordinary = MeasuredSim(cfg, render=False), ServeBreadSim(cfg, render=False)
    measured.reset()
    ordinary.reset()
    measured.step(100)
    ordinary.step(100)
    np.testing.assert_array_equal(measured.d.qpos, ordinary.d.qpos)
    np.testing.assert_array_equal(measured.d.qvel, ordinary.d.qvel)
    assert not measured.measurements["one_pad"]
    assert measured.measurements["max_lift"] < 0.001


def test_rigid_cube_contacts_register_both_fingers():
    cfg = load_config(os.path.join(SIM_ROOT, "configs", "pick_cube.yaml"))
    env = MeasuredSim(cfg, render=False)
    env.reset()
    arm = env.arms["right"]
    T = arm.T_base_in_world @ arm.get_T_ee_in_base()
    point = T[:3, 3] - 0.005 * T[:3, 2]
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    env.place_object("cube", [*point, w, x, y, z])
    # Fixture only: isolate squeeze sensing from gravity while the arm holds still.
    env.m.body_gravcomp[env.obj_bodies["cube"]] = 1
    mujoco.mj_forward(env.m, env.d)
    env.initial_cube = env.object_pos("cube").copy()
    env.reset_metrics()
    arm.close_gripper(goal_time=0.5)
    env.step(500)
    assert env.measurements["one_pad"] and env.measurements["both_pads"]
    assert env.measurements["max_secure_duration"] == 0


def test_cube_grasp_refinement_uses_perceived_obj2_and_table_height():
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.task, runner.anchor_key = "pick_cube", "obj1"
    runner.cfg = {}
    runner.cc = {"grasp_assist": {"enabled": True, "refine": True}}
    runner.env = SimpleNamespace(T_cam_in_world=np.eye(4))
    cube = np.array([[0.4 + x, -0.2 + y, z] for x in (-0.02, 0.02)
                     for y in (-0.02, 0.02) for z in (0, 0.04)])
    other = cube + [3, 3, 3]
    objects = SimpleNamespace(objs_meta={role: {"T_obj_in_cam_static": np.eye(4), "kpts_local": points}
                                        for role, points in (("obj1", other), ("obj2", cube))})
    geometry = runner._perceived_geometry("right", objects)
    np.testing.assert_allclose(geometry["bread"]["c"], [0.4, -0.2, 0.02])
    assert "plate" not in geometry and "place_z" not in geometry
    runner.cc["grasp_assist"]["refine"] = False
    assert runner._perceived_geometry("right", objects) == {}
