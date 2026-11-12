"""Real BRX contacts during clearance, release, and the swept return path."""
import json
from pathlib import Path
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from humanego_sim.env import ServeBreadSim, load_config, SIM_ROOT
from humanego_sim.robots import ASSETS, ROBOTS
from humanego_sim.withdrawal import SafeWithdrawal


@pytest.fixture(scope="module")
def world():
    if not (Path(ASSETS) / ROBOTS["brx20260825"].xml).is_file():
        pytest.skip("BRX assets not imported; run simulation/scripts/import_brx.py")
    return ServeBreadSim(load_config(str(Path(SIM_ROOT) / "configs/pick_cube_brx.yaml")), render=False)


@pytest.fixture
def held(world):
    saved = json.loads((Path(__file__).parent / "data/brx_release_pose.json").read_text())
    world.reset(settle=0, arm_q={"right": saved["arm"]})
    arm = world.arms["right"]
    for name, q in zip(world.robot.gripper_reset_qpos, saved["jaws"]):
        world.d.qpos[world.m.jnt_qposadr[world.m.joint("right/" + name).id]] = q
    for name in ("cube", "container"):
        world.place_object(name, saved[name])
    arm.g_cmd = arm._g_from = arm._g_to = world.robot.gripper_closed
    world.d.ctrl[arm.gid] = arm.g_cmd
    arm._g_t0, arm._g_T = 0, 0.001
    mujoco.mj_forward(world.m, world.d)
    return world


def test_planning_uses_scratch_state_and_rejects_the_direct_home_sweep(held):
    guard = SafeWithdrawal(held, "right", {})
    arm = held.arms["right"]
    before = held.d.qpos.copy()
    assert not guard.clear_of_rim(arm.get_j())
    assert not guard.edge_free(arm.get_j(), np.asarray(arm.robot.home_joints))
    assert guard.prepare_release(0)
    np.testing.assert_array_equal(held.d.qpos, before)
    assert arm.gripper_cmd_closed()


def test_real_clearance_opening_and_return_leave_the_container_untouched(held):
    guard = SafeWithdrawal(held, "right", held.cfg["control"]["withdrawal"])
    arm = held.arms["right"]
    initial_box = held.object_pos("container").copy()
    assert guard.prepare_release(0)
    maximum_force = 0.0
    maximum_shift = 0.0
    wrench = np.zeros(6)

    def step():
        nonlocal maximum_force, maximum_shift
        held.step()
        maximum_shift = max(maximum_shift, float(np.linalg.norm(held.object_pos("container") - initial_box)))
        for index, contact in enumerate(held.d.contact[:held.d.ncon]):
            bodies = held.m.geom_bodyid[contact.geom]
            if held.obj_bodies["container"] in bodies and any(int(b) in guard.moving for b in bodies):
                mujoco.mj_contactForce(held.m, held.d, index, wrench)
                maximum_force = max(maximum_force, abs(float(wrench[0])))

    for _ in range(round(4 / held.dt)):
        guard.preparation_update(held.t)
        step()
        if guard.outcome != "preparing":
            break
    assert guard.outcome == "ready_to_open", guard.report()
    assert guard.clear_of_rim(arm.get_j())
    arm.open_gripper(goal_time=0.5)
    for _ in range(round(1 / held.dt)):
        step()
    assert guard.start(held.t), guard.report()
    for _ in range(round(6 / held.dt)):
        guard.update(held.t)
        step()
        if guard.outcome != "moving":
            break
    assert guard.outcome == "finished", guard.report()
    assert maximum_force < 0.05
    assert maximum_shift < 0.003
    np.testing.assert_allclose(arm.get_j(), arm.robot.home_joints, atol=0.012)
    from humanego_sim.container import contains_points
    half = np.asarray(held.cfg["scene"]["objects"]["cube"]["size"]) / 2
    corners = np.array([[x, y, z] for x in (-half[0], half[0])
                        for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    relative = np.linalg.inv(held.body_T_world("container")) @ held.body_T_world("cube")
    assert contains_points(corners @ relative[:3, :3].T + relative[:3, 3],
                           held.cfg["scene"]["objects"]["container"])


def test_no_ik_solution_holds_position_and_never_commands_home(held, monkeypatch):
    guard = SafeWithdrawal(held, "right", {})
    arm = held.arms["right"]
    monkeypatch.setattr(guard, "_ik", lambda *args: None)
    home = Mock()
    monkeypatch.setattr(arm, "go_home", home)
    assert not guard.prepare_release(0)
    assert guard.outcome == "blocked" and arm.gripper_cmd_closed()
    np.testing.assert_allclose(arm.jinterp.q, arm.get_j())
    assert not arm.jinterp.active
    home.assert_not_called()


def test_unsafe_command_origin_stops_at_measured_joints_without_teleporting(held):
    guard = SafeWithdrawal(held, "right", {})
    arm = held.arms["right"]
    measured = arm.get_j().copy()
    # A small tracking error places the old target inside the container wall.
    arm.q_cmd = measured.copy()
    arm.q_cmd[1] += 0.02
    held.d.qvel[arm.dadr] = 0.01
    positions, velocities = held.d.qpos.copy(), held.d.qvel.copy()
    assert not guard.prepare_release(0)
    assert guard.outcome == "blocked" and guard.reason == "command_path_blocked"
    np.testing.assert_array_equal(arm.q_cmd, measured)
    np.testing.assert_array_equal(arm.jinterp.q, measured)
    assert not arm.jinterp.active and not arm.cinterp.active
    assert arm.gripper_cmd_closed()
    arm.update(held.t + held.dt)
    np.testing.assert_array_equal(held.d.ctrl[arm.aids], measured)
    np.testing.assert_array_equal(held.d.qpos, positions)
    np.testing.assert_array_equal(held.d.qvel, velocities)


def test_closed_gripper_cannot_start_the_return(held):
    guard = SafeWithdrawal(held, "right", {})
    assert not guard.start(0)
    assert guard.outcome == "blocked" and guard.reason == "gripper_not_open"


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_clearance_is_rejected(held, value):
    with pytest.raises(ValueError, match="finite and positive"):
        SafeWithdrawal(held, "right", {"rim_clearance": value})
