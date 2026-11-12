"""Simulation-only clearance before release and collision-checked return paths.

Planning uses a private MjData; it never moves or welds the physical cube.
The actual robot executes checked joint paths through its position servos.
"""
from __future__ import annotations

import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .release_assist import release_geometry


class SafeWithdrawal:
    def __init__(self, env, side, config):
        self.env, self.arm, self.config = env, env.arms[side], config
        self.m = env.m
        self.d = mujoco.MjData(self.m)
        self.side = side
        self.clearance = float(config.get("rim_clearance", 0.02))
        self.margin = float(config.get("collision_margin", 0.003))
        self.joint_step = float(config.get("joint_step", 0.025))
        self.home_time = float(config.get("home_time", 2.2))
        self.speed = float(config.get("speed", 0.10))
        self.timeout = float(config.get("timeout", 5.0))
        values = [self.clearance, self.margin, self.joint_step, self.home_time, self.speed, self.timeout]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("withdrawal clearance, margins, sampling, speed and timing must be finite and positive")
        self.outcome = "not_started"
        self.waypoints = []
        self.index = 0
        self.started = self.commanded = 0.0
        self.plan_wall = 0.0
        self.reason = ""
        self.open_geometry = True
        self.holding = False
        self.prepared_q = None
        self.held_cube_in_tcp = None
        self.next_check = 0.0
        self.preparation_time = None
        self.finished_time = None
        root = int(self.m.jnt_bodyid[self.arm.jids[0]])
        wrist = int(self.m.jnt_bodyid[self.arm.jids[-1]])
        moving, distal = {root}, {wrist}
        for b in range(root + 1, self.m.nbody):
            if self.m.body_parentid[b] in moving:
                moving.add(b)
            if self.m.body_parentid[b] in distal:
                distal.add(b)
        active = (self.m.geom_contype != 0) | (self.m.geom_conaffinity != 0)
        self.moving = moving
        self.arm_geoms = np.array([g for g in range(self.m.ngeom) if active[g] and self.m.geom_bodyid[g] in moving])
        self.distal_geoms = np.array([g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in distal])
        self.obstacles = np.array([g for g in range(self.m.ngeom) if active[g]
            and not self.m.body(int(self.m.geom_bodyid[g])).name.startswith(self.arm.prefix)])
        handles = [g for g in range(self.m.ngeom) if (self.m.geom(g).name or "").startswith("container_handle_")]
        self.obstacles = np.unique(np.r_[self.obstacles, handles]).astype(int)
        self.container = env.obj_bodies["container"]
        self.container_geoms = np.flatnonzero((self.m.geom_bodyid == self.container) & active)
        self.box = np.asarray(env.cfg["scene"]["objects"]["container"]["size"]) / 2

    def _pose(self, q):
        self.d.qpos[:] = self.env.d.qpos
        self.d.qpos[self.arm.qadr] = q
        if self.open_geometry:
            for name, value in self.arm.robot.gripper_reset_qpos.items():
                joint = self.m.joint(self.arm.prefix + name).id
                self.d.qpos[self.m.jnt_qposadr[joint]] = value
        self.d.qvel[:] = 0
        if self.holding and self.held_cube_in_tcp is not None:
            mujoco.mj_kinematics(self.m, self.d)
            hand = np.eye(4)
            hand[:3, :3] = self.d.site_xmat[self.arm.sid].reshape(3, 3)
            hand[:3, 3] = self.d.site_xpos[self.arm.sid]
            cube = hand @ self.arm.C @ self.held_cube_in_tcp
            address = self.env.obj_free_qadr["cube"]
            self.d.qpos[address:address + 3] = cube[:3, 3]
            self.d.qpos[address + 3:address + 7] = Rotation.from_matrix(cube[:3, :3]).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_fwdPosition(self.m, self.d)

    def collision_free(self, q):
        self._pose(q)
        margin = min(self.margin, 0.001) if self.holding and not self.open_geometry else self.margin
        for c in self.d.contact[:self.d.ncon]:
            bodies = self.m.geom_bodyid[c.geom]
            held_cube = self.holding and self.env.obj_bodies["cube"] in bodies
            if not held_cube and any(int(b) in self.moving for b in bodies) and c.dist < -0.0001:
                return False
        for g in self.arm_geoms:
            distances = np.linalg.norm(self.d.geom_xpos[self.obstacles] - self.d.geom_xpos[g], axis=1)
            nearby = self.obstacles[distances < self.m.geom_rbound[self.obstacles] + self.m.geom_rbound[g] + margin]
            for other in nearby:
                if self.holding and self.m.geom_bodyid[other] == self.env.obj_bodies["cube"]:
                    continue
                distance = mujoco.mj_geomDistance(self.m, self.d, int(g), int(other), margin, None)
                if distance < margin:
                    return False
        if self.holding:
            cube_geoms = np.flatnonzero(self.m.geom_bodyid == self.env.obj_bodies["cube"])
            for g in cube_geoms:
                for other in self.container_geoms:
                    if mujoco.mj_geomDistance(self.m, self.d, int(g), int(other), margin, None) < margin:
                        return False
        return True

    def edge_free(self, first, last):
        travel = np.sum(np.abs(last - first)) * (self.arm.robot.reach + 0.2)
        steps = max(1, int(np.ceil(np.max(np.abs(last - first)) / self.joint_step)),
                    int(np.ceil(travel / 0.002)))
        return all(self.collision_free(first + u * (last - first)) for u in np.linspace(0, 1, steps + 1))

    def clear_of_rim(self, q):
        previous = self.open_geometry
        bounds = []
        # Jaw motion is linear. The union of endpoint boxes contains the whole
        # opening sweep, including tilted fingers which move down as they open.
        for opened in (False, True):
            self.open_geometry = opened
            self._pose(q)
            boxes = []
            for g in self.distal_geoms:
                local = self.m.geom_aabb[g]
                R = self.d.geom_xmat[g].reshape(3, 3)
                centre = self.d.geom_xpos[g] + R @ local[:3]
                radius = abs(R) @ local[3:]
                boxes.append([centre - radius, centre + radius])
            bounds.append(boxes)
        self.open_geometry = previous
        centre = self.d.xpos[self.container]
        rotation = self.d.xmat[self.container].reshape(3, 3)
        # Conservative horizontal container bounds include the rim and handles.
        extent = abs(rotation) @ (self.box + [0.005, 0.021, 0.0])
        lower, upper = centre - extent, centre + extent
        bounds = np.asarray(bounds)
        for index in range(len(self.distal_geoms)):
            lo, hi = bounds[:, index, 0].min(axis=0), bounds[:, index, 1].max(axis=0)
            overlaps = np.all(hi[:2] > lower[:2] - self.margin) and np.all(lo[:2] < upper[:2] + self.margin)
            if overlaps and lo[2] < upper[2] + self.clearance:
                return False
        return True

    def prepare_release(self, t):
        wall = time.perf_counter()
        self.preparation_time = t
        self.started = t
        self.holding = True
        start = self.arm.get_j()
        T = self.arm.T_base_in_world @ self.arm.get_T_ee_in_base()
        cube_in_hand = np.linalg.inv(T) @ self.env.body_T_world("cube")
        self.held_cube_in_tcp = cube_in_hand
        release = self.env.cfg["control"].get("release_assist", {})
        xaxis = T[:3, 0].copy()
        xaxis[2] = 0
        if np.linalg.norm(xaxis) < 1e-6:
            xaxis = np.array([1.0, 0, 0])
        xaxis /= np.linalg.norm(xaxis)
        vertical = np.column_stack([xaxis, np.cross([0, 0, -1], xaxis), [0, 0, -1]])
        turn = Slerp([0, 1], Rotation.from_matrix([T[:3, :3], vertical]))
        candidates = [(turn(fraction).as_matrix(), dz) for fraction in (0, 0.5, 1)
                      for dz in (0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10)]
        for rotation, dz in candidates:
            q = self._ik(T[:3, 3] + [0, 0, dz], rotation, start)
            if q is None or not self.clear_of_rim(q):
                continue
            predicted = self.arm.T_base_in_world @ self.arm.fk_raw(q) @ cube_in_hand
            if release_geometry(predicted, self.env.body_T_world("container"),
                    self.env.cfg["scene"]["objects"], float(release.get("interior_margin", 0.005)),
                    float(release.get("max_height_above_rim", 0.08))):
                continue
            self.open_geometry = False
            if not self.edge_free(start, q):
                continue
            self.open_geometry = True
            if not self.collision_free(q):
                continue
            self.prepared_q = q
            self.waypoints = [(q, max(0.25, dz / self.speed), "clearance")]
            self.index = 0
            self.outcome = "preparing"
            self.open_geometry = False
            commanded = self._command(t)
            self.plan_wall += time.perf_counter() - wall
            return commanded
        self.open_geometry = True
        self.outcome, self.reason = "blocked", "no_safe_release_pose"
        self.plan_wall += time.perf_counter() - wall
        self.hold(t)
        return False

    def preparation_update(self, t):
        if not self._live_clear(t):
            return
        q, duration, _ = self.waypoints[0]
        arrived = np.max(abs(self.arm.get_j() - q)) < 0.008 and np.max(abs(self.env.d.qvel[self.arm.dadr])) < 0.05
        if arrived and t - self.commanded >= duration and self.clear_of_rim(self.arm.get_j()):
            release = self.env.cfg["control"].get("release_assist", {})
            reason = release_geometry(self.env.body_T_world("cube"), self.env.body_T_world("container"),
                self.env.cfg["scene"]["objects"], float(release.get("interior_margin", 0.005)),
                float(release.get("max_height_above_rim", 0.08)))
            if reason:
                self.reason = "release_geometry:" + reason
                self.hold(t)
                return
            self.outcome = "ready_to_open"
        elif t - self.started > self.timeout:
            self.reason = "clearance_timeout"
            self.hold(t)

    def _live_clear(self, t):
        if t >= self.next_check:
            self.next_check = t + 0.05
            if not self.collision_free(self.arm.get_j()):
                self.reason = "clearance_lost"
                self.hold(t)
                return False
        return True

    def hold(self, t):
        self.arm.hold_position()
        self.outcome, self.finished_time = "blocked", t

    def _ik(self, xyz, rotation, seed):
        target = np.eye(4)
        target[:3, :3], target[:3, 3] = rotation, xyz
        q, error = self.arm.ik(self.arm.T_world_in_base @ target, seed, iters=250)
        actual = self.arm.T_base_in_world @ self.arm.fk_raw(q)
        angle = Rotation.from_matrix(rotation @ actual[:3, :3].T).magnitude()
        return q if error < 0.002 and angle < np.deg2rad(2) else None

    def plan(self):
        started = time.perf_counter()
        start = self.arm.get_j()
        self.waypoints = []
        self.index = 0
        self.holding = False
        self.open_geometry = True
        home = np.asarray(self.arm.robot.home_joints, float)
        T = self.arm.T_base_in_world @ self.arm.get_T_ee_in_base()
        self.reason = "no_collision_free_exit"
        # Keep the wrist fixed while clearing the rim, then test the entire
        # joint-space return sweep; no motion is issued while searching.
        for dz in (0.0, 0.01, 0.02, 0.03, 0.04, 0.06):
            xyz = T[:3, 3] + [0, 0, dz]
            q = self._ik(xyz, T[:3, :3], start)
            if q is None or not self.clear_of_rim(q) or not self.edge_free(start, q):
                continue
            if self.edge_free(q, home):
                self.waypoints = ([(q, max(0.25, dz / self.speed), "lift")] if dz > 0 else [])
                self.waypoints.append((home, self.home_time, "home"))
                self.reason = ""
                break
            # An outward waypoint keeps the hand outside the container before
            # changing wrist orientation toward its ready pose.
            for dy in (-0.08, -0.14, -0.20, 0.08, 0.14):
                outer = xyz + [0, dy, 0]
                mid = self._ik(outer, T[:3, :3], q)
                if mid is not None and self.clear_of_rim(mid) and self.edge_free(q, mid) and self.edge_free(mid, home):
                    self.waypoints = [(q, max(0.25, dz / self.speed), "lift"),
                                      (mid, max(0.3, abs(dy) / self.speed), "outside"),
                                      (home, self.home_time, "home")]
                    self.reason = ""
                    break
            if self.waypoints:
                break
        self.plan_wall += time.perf_counter() - started
        return bool(self.waypoints)

    def start(self, t):
        self.started = t
        if self.arm.get_gripper_q() < 0.9:
            self.outcome, self.reason = "blocked", "gripper_not_open"
            self.hold(t)
            return False
        if not self.plan():
            self.hold(t)
            return False
        self.outcome = "moving"
        self.next_check = t
        return self._command(t)

    def _command(self, t):
        q, duration, stage = self.waypoints[self.index]
        # Joint interpolation starts at the driver's command, which can differ
        # slightly from the measured joints because of servo tracking error.
        if not self.edge_free(self.arm.q_cmd, q):
            self.reason = "command_path_blocked"
            self.hold(t)
            return False
        self.arm.move_j(q, duration=duration)
        self.commanded = t
        return True

    def update(self, t):
        if self.outcome != "moving":
            return
        if not self._live_clear(t):
            return
        q, duration, stage = self.waypoints[self.index]
        # Advance on measured arrival and velocity, not only elapsed duration.
        arrived = np.max(abs(self.arm.get_j() - q)) < 0.012 and np.max(abs(self.env.d.qvel[self.arm.dadr])) < 0.05
        if arrived and t - self.commanded >= duration:
            self.index += 1
            if self.index == len(self.waypoints):
                self.outcome = "finished"
                self.finished_time = t
            else:
                self._command(t)
        elif t - self.started > self.timeout:
            self.reason = "withdrawal_timeout"
            self.hold(t)

    def report(self):
        return {"outcome": self.outcome, "reason": self.reason, "plan_wall": self.plan_wall,
                "duration": (self.finished_time - self.preparation_time
                             if self.finished_time is not None and self.preparation_time is not None else None),
                "waypoints": [stage for _, _, stage in self.waypoints]}
