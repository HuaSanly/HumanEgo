"""Simulation-only release guard for a rigid block held above its container."""
from __future__ import annotations

import mujoco
import numpy as np

from .container import contains_xy


def release_geometry(cube_pose: np.ndarray, container_pose: np.ndarray, props: dict,
                     margin: float, max_height: float) -> str:
    """Return a blocking reason, or '' for a clear, short gravity drop into the cavity."""
    half = np.asarray(props["cube"]["size"], float) / 2
    corners = np.array([[x, y, z] for x in (-half[0], half[0])
                        for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    relative = np.linalg.inv(container_pose) @ cube_pose
    points = corners @ relative[:3, :3].T + relative[:3, 3]
    container = props["container"]
    container_half = np.asarray(container["size"], float) / 2
    # An overturned container has no reliable vertical drop path into its cavity.
    down = container_pose[:3, :3].T @ np.array([0.0, 0.0, -1.0])
    if -down[2] < np.cos(np.deg2rad(5.0)):
        return "container_tilted"
    floor = -container_half[2] + float(container["wall"])
    lowest = float(points[:, 2].min())
    if lowest < floor - 0.003:
        return "below_floor"
    if lowest - container_half[2] > max_height:
        return "too_high"
    landing = points + down * (max(0.0, lowest - floor) / -down[2])
    # Checking both endpoints of the drop also handles small container tilts. The
    # landing must fit the narrower bottom, not just pass through the wider mouth.
    if not contains_xy(points, container, -margin) or not contains_xy(landing, container, -margin):
        return "outside_drop_region"
    return ""


class ReleaseAssist:
    """Observe true physics at control ticks; the runner owns all actuator commands."""

    def __init__(self, env, side: str, config: dict):
        self.env, self.side, self.config = env, side, dict(config)
        self.enabled = bool(config.get("enabled", False))
        self.wait_time = float(config.get("wait_time", 1.0))
        self.margin = float(config.get("interior_margin", 0.005))
        self.max_height = float(config.get("max_height_above_rim", 0.08))
        self.max_speed = float(config.get("max_linear_speed", 0.02))
        self.max_angular_speed = float(config.get("max_angular_speed", 0.3))
        self.settle_timeout = float(config.get("settle_timeout", 2.0))
        if (self.wait_time <= 0 or self.margin < 0 or self.max_height < 0
                or self.max_speed <= 0 or self.max_angular_speed <= 0 or self.settle_timeout < 0.5):
            raise ValueError("release_assist needs positive timing/speed limits, nonnegative margins, "
                             "and settle_timeout >= 0.5")
        self.ready_since = None
        self.triggered_at = None
        self.waited_time = 0.0
        self.settled_since = None
        self.last_block_reason = "not_checked"
        self.outcome = "not_triggered"
        self._pad_geoms = None

    def _holding_cube(self) -> bool:
        env, arm = self.env, self.env.arms[self.side]
        if not arm.gripper_cmd_closed():
            return False
        if self._pad_geoms is None:
            col = (env.m.geom_contype > 0) | (env.m.geom_conaffinity > 0)
            fingers = [env.m.body(arm.prefix + name).id for name in arm.robot.finger_bodies[:2]]
            self._pad_geoms = [np.flatnonzero((env.m.geom_bodyid == body) & col) for body in fingers]
            self._cube_geoms = np.flatnonzero(env.m.geom_bodyid == env.obj_bodies["cube"])
        forces = np.zeros(2)
        force = np.zeros(6)
        for contact_id in range(env.d.ncon):
            contact = env.d.contact[contact_id]
            for pad, geoms in enumerate(self._pad_geoms):
                if ((contact.geom1 in geoms and contact.geom2 in self._cube_geoms)
                        or (contact.geom2 in geoms and contact.geom1 in self._cube_geoms)):
                    mujoco.mj_contactForce(env.m, env.d, contact_id, force)
                    forces[pad] += abs(float(force[0]))
        return bool(np.all(forces > 0.05))

    def _velocity(self, kind, object_id: int) -> np.ndarray:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.env.m, self.env.d, kind, object_id, velocity, 0)
        return velocity

    def blocking_reason(self) -> str:
        env, arm = self.env, self.env.arms[self.side]
        if not self._holding_cube():
            return "not_holding_cube"
        reason = release_geometry(env.body_T_world("cube"), env.body_T_world("container"),
                                  env.cfg["scene"]["objects"], self.margin, self.max_height)
        if reason:
            return reason
        container_velocity = self._velocity(mujoco.mjtObj.mjOBJ_BODY, env.obj_bodies["container"])
        cube_velocity = self._velocity(mujoco.mjtObj.mjOBJ_BODY, env.obj_bodies["cube"])
        hand_velocity = self._velocity(mujoco.mjtObj.mjOBJ_SITE, arm.sid)
        container_pos = env.object_pos("container")
        for velocity, position in ((cube_velocity, env.object_pos("cube")),
                                   (hand_velocity, env.d.site_xpos[arm.sid])):
            relative_linear = velocity[3:] - container_velocity[3:] - np.cross(
                container_velocity[:3], position - container_pos)
            if np.linalg.norm(relative_linear) > self.max_speed:
                return "moving_too_fast"
            if np.linalg.norm(velocity[:3] - container_velocity[:3]) > self.max_angular_speed:
                return "rotating_too_fast"
        return ""

    def update(self, t: float, wants_close: bool = True) -> bool:
        if not self.enabled or self.triggered_at is not None:
            return False
        self.last_block_reason = self.blocking_reason() if wants_close else "policy_opening"
        if self.last_block_reason:
            self.ready_since = None
            self.waited_time = 0.0
            return False
        if self.ready_since is None:
            self.ready_since = t
        self.waited_time = t - self.ready_since
        if self.waited_time < self.wait_time - 1e-9:
            return False
        self.triggered_at = t
        self.outcome = "releasing"
        return True

    def report(self) -> dict:
        return {"enabled": self.enabled, "triggered": self.triggered_at is not None,
                "trigger_time": self.triggered_at, "waited_time": round(self.waited_time, 4),
                "last_block_reason": self.last_block_reason, "outcome": self.outcome,
                "config": self.config}
