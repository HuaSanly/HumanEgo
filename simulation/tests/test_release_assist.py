"""Release geometry, actual rigid contacts, timing and controller takeover."""
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanego_sim.controller import SimController
from humanego_sim.perception import SideObjects
from humanego_sim.release_assist import ReleaseAssist, release_geometry
from humanego_sim.runner import EpisodeRunner, SideCtx


PROPS = {"cube": {"size": [0.034] * 3},
         "container": {"size": [0.15, 0.23, 0.06], "wall": 0.004,
                       "shape": "rounded", "radius": 0.018, "taper": 0.84}}


def pose(xyz=(0, 0, 0), angles=(0, 0, 0)):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


@pytest.mark.parametrize("xyz,angles,expected", [
    ((0, 0, 0.08), (0, 0, 0), ""),
    ((0, 0, 0), (0, 0, 0), ""),  # already lowered into the cavity
    ((0, 0, 0.08), (20, 10, 45), ""),
    ((0.06, 0, 0.08), (0, 0, 0), "outside_drop_region"),
    ((0.045, 0, 0.08), (0, 0, 0), "outside_drop_region"),  # mouth fits, bottom does not
    ((0.035, 0, 0.08), (0, 0, 45), "outside_drop_region"),  # rotated corner crosses margin
    ((0.039, 0.075, 0.08), (0, 0, 0), "outside_drop_region"),  # rounded corner
    ((0, 0, 0.15), (0, 0, 0), "too_high"),
    ((0, 0, -0.02), (0, 0, 0), "below_floor"),
])
@pytest.mark.parametrize("yaw", [0, 35, 90])
def test_drop_geometry_uses_full_oriented_block_and_container(xyz, angles, expected, yaw):
    container = pose((0.3, 0.1, 0.03), (0, 0, yaw))
    cube = container @ pose(xyz, angles)
    assert release_geometry(cube, container, PROPS, 0.005, 0.08) == expected


def test_container_tilt_and_box_cavity():
    container = pose(angles=(10, 0, 0))
    assert release_geometry(container @ pose((0, 0, 0.08)), container, PROPS, 0.005, 0.08) == "container_tilted"
    props = {**PROPS, "container": {**PROPS["container"], "shape": "box"}}
    assert release_geometry(pose((0.045, 0, 0.08)), np.eye(4), props, 0.005, 0.08) == ""


@pytest.fixture
def physics():
    # Both static pads slightly compress a real free block. A separate static body
    # represents the container frame; no renderer, policy checkpoint or GPU needed.
    model = mujoco.MjModel.from_xml_string('''
      <mujoco><option gravity="0 0 0"/><worldbody>
        <body name="container" pos="0 0 0.03">
          <geom type="box" size="0.071 0.111 0.002" pos="0 0 -0.028"/>
          <geom type="box" size="0.002 0.115 0.03" pos="-0.073 0 0"/>
          <geom type="box" size="0.002 0.115 0.03" pos="0.073 0 0"/>
          <geom type="box" size="0.071 0.002 0.03" pos="0 -0.113 0"/>
          <geom type="box" size="0.071 0.002 0.03" pos="0 0.113 0"/>
        </body>
        <body name="cube" pos="0 0 0.11"><freejoint/>
          <geom type="box" size="0.017 0.017 0.017" mass="0.05"/></body>
        <body name="right/left_pad" pos="-0.0265 0 0.11">
          <geom type="box" size="0.01 0.025 0.025"/></body>
        <body name="right/right_pad" pos="0.0265 0 0.11">
          <geom type="box" size="0.01 0.025 0.025"/><site name="hand"/></body>
        <body name="decoy" pos="0.2 0 0.11"><freejoint/>
          <geom type="box" size="0.017 0.017 0.017" mass="0.05"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm = SimpleNamespace(prefix="right/", robot=SimpleNamespace(finger_bodies=["left_pad", "right_pad"]),
                          sid=model.site("hand").id, gripper_cmd_closed=Mock(return_value=True))
    env = SimpleNamespace(m=model, d=data, arms={"right": arm}, cfg={"scene": {"objects": PROPS}},
                          obj_bodies={name: model.body(name).id for name in ("cube", "container")})
    env.body_T_world = lambda name: pose(data.xpos[model.body(name).id])
    env.object_pos = lambda name: data.xpos[model.body(name).id].copy()
    env.obj_free_dadr = {"cube": 0}
    return env


def test_actual_rigid_contacts_and_continuous_sim_time(physics):
    assist = ReleaseAssist(physics, "right", {"enabled": True})
    assert assist.blocking_reason() == ""
    assert not assist.update(10.0)
    assert not assist.update(10.9)
    assert assist.update(11.0)
    assert not assist.update(12.0)  # one trigger per episode
    assert assist.report()["waited_time"] == pytest.approx(1.0)


def test_lost_grip_and_policy_opening_reset_wait(physics):
    assist = ReleaseAssist(physics, "right", {"enabled": True})
    assert not assist.update(0.0)
    physics.arms["right"].gripper_cmd_closed.return_value = False
    assert not assist.update(0.9)
    assert assist.last_block_reason == "not_holding_cube"
    physics.arms["right"].gripper_cmd_closed.return_value = True
    assert not assist.update(1.0)
    assert not assist.update(1.9, wants_close=False)
    assert assist.last_block_reason == "policy_opening"
    assert not assist.update(2.0)
    assert not assist.update(2.9)
    assert assist.update(3.0)


def test_one_pad_or_contacts_with_other_objects_do_not_count(physics):
    assist = ReleaseAssist(physics, "right", {"enabled": True})
    assert assist._holding_cube()
    physics.d.qpos[0] = -0.002
    mujoco.mj_forward(physics.m, physics.d)
    assert assist.blocking_reason() == "not_holding_cube"
    physics.d.qpos[0] = 0.2
    physics.d.qpos[7] = 0.0  # the pads now squeeze another body, not the task's cube
    mujoco.mj_forward(physics.m, physics.d)
    assert not assist._holding_cube()
    assert physics.d.ncon > 0


@pytest.mark.parametrize("dof,reason", [(0, "moving_too_fast"), (3, "rotating_too_fast")])
def test_motion_invalidates_dwell(physics, dof, reason):
    assist = ReleaseAssist(physics, "right", {"enabled": True})
    assert not assist.update(0)
    physics.d.qvel[dof] = 0.5
    mujoco.mj_forward(physics.m, physics.d)
    assert not assist.update(1)
    assert assist.last_block_reason == reason
    physics.d.qvel[:] = 0
    mujoco.mj_forward(physics.m, physics.d)
    assert not assist.update(2)
    assert assist.update(3)


def test_disabled_assist_never_triggers(physics):
    assist = ReleaseAssist(physics, "right", {})
    assert not assist.update(0) and not assist.update(100)
    assert not assist.report()["triggered"]


def release_runner(physics):
    arm = physics.arms["right"]
    arm.get_p = lambda: np.zeros(6)
    arm.move_p = Mock()
    arm.open_gripper = Mock(side_effect=lambda **kw: setattr(arm, "closed", False))
    arm.close_gripper = Mock(side_effect=lambda **kw: setattr(arm, "closed", True))
    arm.get_gripper_q = Mock(return_value=1.0)
    arm.go_home = Mock()
    arm.closed = True
    arm.gripper_cmd_closed = Mock(side_effect=lambda: arm.closed)
    controller = SimController(physics.arms, {"open_gripper_goal_time": 0.5})
    controller.reperceive["right"] = 99
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.env = physics
    runner.evaluate = Mock(return_value={"right": {"success": True}})
    ctx = SideCtx("right", False, SideObjects())
    ctx.hand.update(has_ever_grasped=True, is_grasping=True, latched_obj="obj2")
    assist = ReleaseAssist(physics, "right", {"enabled": True})
    assist.triggered_at = 1.0
    runner._start_assisted_release(ctx, controller, 1.0)
    return runner, controller, ctx, assist


def test_takeover_ignores_new_close_commands_and_waits_for_stable_placement(physics):
    runner, controller, ctx, assist = release_runner(physics)
    arm = physics.arms["right"]
    assert not arm.closed and controller.muted["right"]
    assert controller.reperceive["right"] is None and ctx.hand["latched_obj"] is None
    # Even a late inference delivered after takeover cannot command the arm.
    controller.update_target({"right": [np.zeros(6)] * 50}, {"right": np.ones(50)})
    controller.tick()
    arm.close_gripper.assert_not_called()
    runner._advance_assisted_release(ctx, assist, controller, 1.49)
    assert ctx.wrap_phase == "assist_open"
    runner._advance_assisted_release(ctx, assist, controller, 1.5)
    assert ctx.wrap_phase == "assist_settle"
    runner._advance_assisted_release(ctx, assist, controller, 1.6)
    runner._advance_assisted_release(ctx, assist, controller, 2.0)
    arm.go_home.assert_not_called()
    runner._advance_assisted_release(ctx, assist, controller, 2.1)
    assert assist.outcome == "placed" and ctx.wrap_phase == "home"
    arm.go_home.assert_called_once()


@pytest.mark.parametrize("opens,outcome", [(True, "settle_timeout"), (False, "open_timeout")])
def test_failed_open_or_drop_remains_a_failed_attempt(physics, opens, outcome):
    runner, controller, ctx, assist = release_runner(physics)
    runner.evaluate.return_value = {"right": {"success": False}}
    if opens:
        runner._advance_assisted_release(ctx, assist, controller, 1.5)
        runner._advance_assisted_release(ctx, assist, controller, 3.6)
    else:
        physics.arms["right"].get_gripper_q.return_value = 0.3
        runner._advance_assisted_release(ctx, assist, controller, 3.6)
    assert assist.outcome == outcome and ctx.wrap_phase == "home"
    assert not runner.evaluate({"right": ctx})["right"]["success"]


def test_triggered_release_really_drops_block_into_container(physics):
    assist = ReleaseAssist(physics, "right", {"enabled": True})
    assert not assist.update(0.0)
    assert assist.update(1.0)
    runner, controller, ctx, _ = release_runner(physics)
    runner.cfg, runner.task = physics.cfg, "pick_cube"
    runner.evaluate = EpisodeRunner.evaluate.__get__(runner)
    # Open the physical pads; let MuJoCo, rather than a mocked success value,
    # determine whether the released block actually lands and rests in the box.
    for name in ("right/left_pad", "right/right_pad"):
        physics.m.body_pos[physics.m.body(name).id, 0] *= 3
    physics.m.opt.gravity[:] = [0, 0, -9.81]
    mujoco.mj_forward(physics.m, physics.d)
    for _ in range(250):
        mujoco.mj_step(physics.m, physics.d)
    runner._advance_assisted_release(ctx, assist, controller, 1.5)
    assert ctx.wrap_phase == "assist_settle"
    for t in np.arange(1.6, 2.3, 0.1):
        for _ in range(50):
            mujoco.mj_step(physics.m, physics.d)
        runner._advance_assisted_release(ctx, assist, controller, float(t))
    result = runner.evaluate({"right": ctx})["right"]
    assert result["success"] and result["inside_container"] and result["settled"]
    assert assist.outcome == "placed" and ctx.wrap_phase == "home"
