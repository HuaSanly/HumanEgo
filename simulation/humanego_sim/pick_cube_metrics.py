"""Read-only contact and lift telemetry shared by pick_cube evaluations."""
from __future__ import annotations

import mujoco
import numpy as np

from .env import ServeBreadSim


class MeasuredSim(ServeBreadSim):
    """Observe actual rigid-cube contacts without feeding privileged state to the controller."""

    def __init__(self, cfg: dict, render: bool = True):
        super().__init__(cfg, render=render)
        self.cube_body = self.obj_bodies["cube"]
        self.finger_bodies = [self.m.body("right/" + name).id for name in self.robot.finger_bodies[:2]]
        self.force = np.zeros(6)
        self.initial_cube = self.object_pos("cube").copy()
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.measurements = {"one_pad": False, "both_pads": False, "max_lift": 0.0,
                             "max_lift_duration": 0.0, "max_secure_duration": 0.0,
                             "max_horizontal": 0.0, "min_pinch_distance": float("inf"),
                             "first_close_distance": None, "first_close_delta": None}
        self.lift_duration = self.secure_duration = 0.0
        self.telemetry = []
        self.next_record = 0.0

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.initial_cube = self.object_pos("cube").copy()
        self.reset_metrics()

    def step(self, n: int = 1):
        for _ in range(n):
            super().step()
            self.measure()

    def measure(self) -> None:
        if not hasattr(self, "measurements"):
            return
        forces = np.zeros(2)
        for index in range(self.d.ncon):
            geoms = self.d.contact[index].geom
            bodies = self.m.geom_bodyid[geoms]
            if self.cube_body not in bodies:
                continue
            other = bodies[1] if bodies[0] == self.cube_body else bodies[0]
            if other in self.finger_bodies:
                mujoco.mj_contactForce(self.m, self.d, index, self.force)
                forces[self.finger_bodies.index(other)] += abs(float(self.force[0]))
        cube = self.object_pos("cube")
        arm = self.arms["right"]
        pinch = self.d.site_xpos[arm.sid]
        delta = pinch - cube
        distance = float(np.linalg.norm(delta))
        m = self.measurements
        touching = forces > 0.1
        m["one_pad"] |= bool(touching.any())
        m["both_pads"] |= bool(touching.all())
        lift = float(cube[2] - self.initial_cube[2])
        self.lift_duration = self.lift_duration + self.dt if lift > 0.025 else 0.0
        self.secure_duration = self.secure_duration + self.dt if lift > 0.025 and touching.all() else 0.0
        m["max_lift"] = max(m["max_lift"], lift)
        m["max_lift_duration"] = max(m["max_lift_duration"], self.lift_duration)
        m["max_secure_duration"] = max(m["max_secure_duration"], self.secure_duration)
        if lift > 0.025:
            m["max_horizontal"] = max(m["max_horizontal"], float(np.linalg.norm(cube[:2] - self.initial_cube[:2])))
        m["min_pinch_distance"] = min(m["min_pinch_distance"], distance)
        if arm.gripper_cmd_closed() and m["first_close_distance"] is None:
            m["first_close_distance"] = distance
            m["first_close_delta"] = delta.tolist()
        if self.t >= self.next_record:
            self.telemetry.append({"t": round(self.t, 4), "cube": cube.tolist(), "pinch": pinch.tolist(),
                                   "forces": forces.tolist(), "gripper": arm.get_gripper_q()})
            self.next_record += 0.1
