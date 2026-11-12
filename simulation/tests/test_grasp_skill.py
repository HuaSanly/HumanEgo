# -*- coding: utf-8 -*-
"""The grasp skill of the low-level controller (control.grasp_assist) on synthetic inputs:
a fake arm that goes wherever it is told, a constant plan, and a known bread pose."""
import numpy as np
import pytest

from humanego_sim.controller import SimController

UP = np.array([0.0, 0.0, 1.0])
C = np.array([0.50, -0.25, 0.03])          # bread centre (world == camera frame here)
A = np.array([0.0, 1.0, 0.0])              # long axis
W = np.cross(UP, A)                        # across the bread


class FakeArm:
    """Teleports to every command; the grip sensor is whatever the test says."""

    def __init__(self, p0):
        self.p = np.asarray(p0, float).copy()
        self.closed = False
        self.empty = False
        self.cmds = []

    def get_p_in_cam(self):
        return self.p.copy()

    def move_p_in_cam(self, cmd, duration=0.0, blocking=False, safe_z_min=-0.2, v_ff_in_cam=None):
        self.cmds.append(np.array(cmd))
        self.p = np.array(cmd, float)
        return True

    def close_gripper(self, goal_time=1.0, blocking=False):
        self.closed = True

    def open_gripper(self, goal_time=1.0, blocking=False):
        self.closed = False

    def holding_nothing(self, q_empty, gap_empty=0.015):
        return self.empty


GA = dict(enabled=True, lateral_gain=0.5, along_max=0.02, height_max=0.004, height_min=-0.008, arrive_tol=0.01,
          close_hold=0.6, regrasp_max=2, q_empty=0.1, retry_lower=0.005, retry_inward=0.01, regrasp_cooldown=1.2)


def make(ga=GA, start=None):
    cfg = dict(control_freq=10.0, look_ahead_offset=45, step_stride=2, alpha=0.5, alpha_rot=0.0,
               grasp_threshold=0.6, should_grasp_method="any", max_pos_step=0.5, max_rot_step=3.0,
               grasp_assist=dict(ga))
    # pinch point the "policy" asks for: 3 cm along the bread, 2 cm off its axis, 1 cm above centre
    plan_pt = C + 0.03 * A + 0.02 * W + 0.01 * UP
    arm = FakeArm(np.r_[plan_pt + np.array([0.0, 0.0, 0.05]), 0.0, 0.0, 0.0] if start is None else start)
    ctl = SimController({"right": arm}, cfg)
    ctl.set_geometry("right", {"c": C.tolist(), "a": A.tolist(), "half_len": 0.06, "half_wid": 0.03}, np.eye(4))
    chunk = [np.r_[plan_pt, 0.0, 0.0, 0.0] for _ in range(50)]
    return ctl, arm, chunk


def feed(ctl, chunk, grasp=0.9):
    ctl.update_target({"right": chunk}, {"right": np.full(len(chunk), grasp)})


def step(ctl, chunk, grasp=0.9):
    """One control tick after a fresh plan (the plan alone runs out after (50 - 45) / 2 ticks)."""
    feed(ctl, chunk, grasp)
    ctl.tick()


def refined(n_tries, ga=GA):
    """Expected pinch point after `n_tries` re-grasps (controller._refine, by hand)."""
    s_a = float(np.clip(0.03, -(ga["along_max"] - n_tries * ga["retry_inward"]),
                        ga["along_max"] - n_tries * ga["retry_inward"]))
    s_w = 0.02 * (1 - ga["lateral_gain"]) ** (1 + n_tries)
    up_max = ga["height_max"] - n_tries * ga["retry_lower"]
    s_u = float(np.clip(0.01, min(ga["height_min"], up_max), up_max))
    return C + s_a * A + s_w * W + s_u * UP


def test_refine_projects_onto_the_bread():
    ctl, arm, chunk = make()
    step(ctl, chunk)
    gs = ctl.gs["right"]
    assert gs["state"] == "approach"
    np.testing.assert_allclose(gs["target"][:3], refined(0), atol=1e-12)
    # the arm is sent straight to the pinch point (no EMA) with the gripper still open
    np.testing.assert_allclose(arm.cmds[-1][:3], refined(0), atol=1e-12)
    assert not arm.closed


def test_refine_can_hold_a_reachable_world_grasp_orientation():
    from scipy.spatial.transform import Rotation as R
    ctl, arm, chunk = make(ga={**GA, "orientation_world_rpy_deg": [180, 0, 180]})
    ctl.T_cam_in_world = np.eye(4)
    ctl.T_cam_in_world[:3, :3] = R.from_euler("y", 40, degrees=True).as_matrix()
    refined_target = ctl._refine("right", np.asarray(chunk[0]))
    actual_world = ctl.T_cam_in_world[:3, :3] @ R.from_rotvec(refined_target[3:]).as_matrix()
    expected = R.from_euler("xyz", [180, 0, 180], degrees=True).as_matrix()
    np.testing.assert_allclose(actual_world, expected, atol=1e-12)


def test_close_on_arrival_hold_then_holding():
    ctl, arm, chunk = make()
    step(ctl, chunk)                              # approach: arm teleports onto the pinch point
    step(ctl, chunk)                              # arrived -> close
    gs = ctl.gs["right"]
    assert gs["state"] == "closing" and arm.closed
    t_close = gs["t0"]
    while gs["state"] == "closing":
        step(ctl, chunk)
        assert arm.closed
        np.testing.assert_allclose(arm.p[:3], refined(0), atol=1e-12)   # held still while closing
    assert ctl.t - t_close == pytest.approx(GA["close_hold"], abs=0.1 + 1e-9)
    assert gs["state"] == "holding"
    step(ctl, chunk)
    assert arm.closed and gs["state"] == "holding"


def test_approach_times_out_into_closing():
    ctl, arm, chunk = make(ga={**GA, "arrive_tol": 1e-9, "approach_timeout": 0.5})
    step(ctl, chunk)
    arm.move_p_in_cam = lambda cmd, **kw: True    # the arm stops responding
    arm.p = arm.p + np.r_[0.0, 0.0, 0.05, 0, 0, 0]
    for _ in range(4):
        step(ctl, chunk)
        assert ctl.gs["right"]["state"] == "approach"
    for _ in range(3):
        step(ctl, chunk)
    assert ctl.gs["right"]["state"] == "closing"


def test_empty_grip_regrasps_lower_and_more_central():
    ctl, arm, chunk = make()
    for _ in range(10):
        step(ctl, chunk)
    gs = ctl.gs["right"]
    assert gs["state"] == "holding"
    arm.empty = True                              # the fingers closed on air
    for _ in range(3):
        step(ctl, chunk)
        assert gs["tries"] == 0                   # needs > 0.3 s of "empty"
    step(ctl, chunk)
    step(ctl, chunk)
    assert gs["tries"] == 1 and gs["state"] == "free" and not arm.closed
    t_regrasp = [e[0] for e in ctl.events["right"] if e[1] == "regrasp:empty"]
    assert len(t_regrasp) == 1
    arm.empty = False
    # cooldown: the gripper stays open even though the policy still wants to grasp
    assert gs["cool_until"] == pytest.approx(t_regrasp[0] + GA["regrasp_cooldown"], abs=0.006)
    while ctl.t + 0.1 < gs["cool_until"] - 1e-6:      # every tick that lands inside the cooldown
        step(ctl, chunk)
        assert gs["state"] == "free" and not arm.closed
    while gs["state"] == "free":
        step(ctl, chunk)
    assert gs["state"] == "approach"
    np.testing.assert_allclose(gs["target"][:3], refined(1), atol=1e-12)
    assert gs["target"][2] < refined(0)[2]                               # lower ...
    assert abs((gs["target"][:3] - C) @ A) < abs((refined(0) - C) @ A)   # ... and more central


def test_regrasp_attempts_are_capped():
    ctl, arm, chunk = make(ga={**GA, "regrasp_max": 1, "regrasp_cooldown": 0.0})
    gs = ctl.gs["right"]
    arm.empty = True
    for _ in range(80):
        step(ctl, chunk)
    assert gs["tries"] == 1                       # one re-grasp, then it keeps the (empty) grip
    assert gs["state"] == "holding" and arm.closed


def test_policy_letting_go_aborts_the_approach():
    ctl, arm, chunk = make(ga={**GA, "arrive_tol": 1e-9, "approach_timeout": 99.0})
    arm.move_p_in_cam = lambda cmd, **kw: True    # never arrives
    step(ctl, chunk)
    assert ctl.gs["right"]["state"] == "approach"
    step(ctl, chunk, grasp=0.1)
    assert ctl.gs["right"]["state"] == "free" and not arm.closed


def test_skill_off_closes_on_the_policy_flag():
    ctl, arm, chunk = make(ga={"enabled": False})
    step(ctl, chunk)
    assert arm.closed                             # production controller: close as soon as the flag is up
    assert ctl.gs["right"]["state"] == "free"     # state machine untouched
    plan_pt = chunk[0][:3]
    np.testing.assert_allclose(arm.cmds[-1][:3], plan_pt, atol=1e-12)   # first command = plan target
