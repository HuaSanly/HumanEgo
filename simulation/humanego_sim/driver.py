# -*- coding: utf-8 -*-
"""
SimArm — a MuJoCo stand-in for the real arm driver.

It reproduces the *behaviour* the HumanEgo inference stack relies on from
`inference/RobotArmTrossen.py` (trossen_arm driver):

  * `set_cartesian_positions(goal, cartesian, goal_time, blocking=False)`:
    the driver re-plans a quintic trajectory from its *current interpolator
    state* (position, velocity, acceleration) to the goal every time a new
    command arrives. Re-targeting at 10 Hz with goal_time = 1.2 s is what makes
    the real arm glide instead of stop-and-go. Orientation is interpolated the
    same way in the goal's tangent space.
  * joint-space moves (`go_home`, `go_initial`) with quintic profiles,
  * timed gripper position commands,
  * an on-board IK (damped least squares on the MuJoCo Jacobian) that turns the
    Cartesian set-point into joint targets for the position servos every
    physics step, and rejects goals it cannot reach (like the real
    "inverse kinematics" error),
  * the frame conventions of `RobotArmTrossen`: `get_p` / `move_p` work in the
    *corrected* EE frame (Z approach, X finger spread), `*_in_cam` use the
    hand-eye extrinsic `T_base_in_cam`.

Everything is driven by simulation time (`SimArm.update(t)` once per physics
step), so rollouts are deterministic and independent of wall-clock speed.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .robots import RobotSpec


# ---------------------------------------------------------------------------
# small SE(3) helpers
# ---------------------------------------------------------------------------
def T_from(Rm: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = p
    return T


def p_to_T(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return T_from(R.from_rotvec(p[3:6]).as_matrix(), p[:3])


def T_to_p(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_rotvec()])


def _quintic_coeffs(x0, v0, a0, x1, T, v1=None):
    """Quintic Hermite from (x0, v0, a0) to (x1, v1, 0) in time T (vectorised); v1 = 0 unless a
    goal feed-forward velocity is given (trossen_arm goal_feedforward_velocities)."""
    x0, v0, a0, x1 = map(np.asarray, (x0, v0, a0, x1))
    d = x1 - x0
    c0, c1, c2 = x0, v0, a0 / 2.0
    if v1 is None:
        c3 = (20 * d - 12 * v0 * T - 3 * a0 * T ** 2) / (2 * T ** 3)
        c4 = (-30 * d + 16 * v0 * T + 3 * a0 * T ** 2) / (2 * T ** 4)
        c5 = (12 * d - 6 * v0 * T - a0 * T ** 2) / (2 * T ** 5)
    else:
        v1 = np.asarray(v1)
        c3 = (20 * d - (8 * v1 + 12 * v0) * T - 3 * a0 * T ** 2) / (2 * T ** 3)
        c4 = (-30 * d + (14 * v1 + 16 * v0) * T + 3 * a0 * T ** 2) / (2 * T ** 4)
        c5 = (12 * d - 6 * (v1 + v0) * T - a0 * T ** 2) / (2 * T ** 5)
    return np.stack([c0, c1, c2, c3, c4, c5])


def _quintic_eval(C, tau):
    p = C[0] + tau * (C[1] + tau * (C[2] + tau * (C[3] + tau * (C[4] + tau * C[5]))))
    v = C[1] + tau * (2 * C[2] + tau * (3 * C[3] + tau * (4 * C[4] + tau * 5 * C[5])))
    a = 2 * C[2] + tau * (6 * C[3] + tau * (12 * C[4] + tau * 20 * C[5]))
    return p, v, a


class CartesianInterpolator:
    """Task-space quintic re-targeting with velocity/acceleration continuity."""

    def __init__(self, T0: np.ndarray):
        self.reset(T0)

    def reset(self, T0: np.ndarray):
        self.p = T0[:3, 3].copy()
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.Rm = T0[:3, :3].copy()
        self.w = np.zeros(3)          # angular velocity (base frame)
        self.al = np.zeros(3)         # angular acceleration
        self.active = False
        self._Cp = self._Cr = None
        self._R_goal = self.Rm.copy()
        self.p_goal = self.p.copy()
        self._t0 = 0.0
        self._T = 1.0

    def set_goal(self, T_goal: np.ndarray, duration: float, t_now: float, v_goal=None):
        duration = max(float(duration), 1e-3)
        self._R_goal = T_goal[:3, :3].copy()
        self.p_goal = T_goal[:3, 3].copy()
        # rotation expressed as r = log(R_goal^T R) in the goal frame
        r0 = R.from_matrix(self._R_goal.T @ self.Rm).as_rotvec()
        dr0 = self._R_goal.T @ self.w
        ddr0 = self._R_goal.T @ self.al
        self._Cp = _quintic_coeffs(self.p, self.v, self.a, T_goal[:3, 3], duration, v_goal)
        self._Cr = _quintic_coeffs(r0, dr0, ddr0, np.zeros(3), duration)
        self._t0, self._T = t_now, duration
        self.active = True

    def update(self, t: float) -> np.ndarray:
        if self.active:
            tau = min(max(t - self._t0, 0.0), self._T)
            self.p, self.v, self.a = _quintic_eval(self._Cp, tau)
            r, dr, ddr = _quintic_eval(self._Cr, tau)
            self.Rm = self._R_goal @ R.from_rotvec(r).as_matrix()
            self.w = self._R_goal @ dr
            self.al = self._R_goal @ ddr
            if t - self._t0 >= self._T:
                self.active = False
                self.v[:] = 0.0
                self.a[:] = 0.0
                self.w[:] = 0.0
                self.al[:] = 0.0
        return T_from(self.Rm, self.p)


class JointInterpolator:
    def __init__(self, q0: np.ndarray):
        self.q = np.array(q0, dtype=np.float64)
        self.active = False

    def set_goal(self, q_goal, duration, t_now):
        duration = max(float(duration), 1e-3)
        z = np.zeros_like(self.q)
        self._C = _quintic_coeffs(self.q, z, z, np.asarray(q_goal, dtype=np.float64), duration)
        self._t0, self._T = t_now, duration
        self.active = True

    def update(self, t):
        if self.active:
            tau = min(max(t - self._t0, 0.0), self._T)
            self.q, _, _ = _quintic_eval(self._C, tau)
            if t - self._t0 >= self._T:
                self.active = False
        return self.q


# ---------------------------------------------------------------------------
class SimArm:
    """One simulated arm + gripper with the RobotArmTrossen API surface."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, side: str, robot: RobotSpec,
                 T_base_in_world: np.ndarray, T_base_in_cam: np.ndarray,
                 ik_damping: float = 0.02, ik_iters: int = 3, tcp_offset: float = 0.0, tool_pitch_deg: float = 0.0):
        self.m, self.d = model, data
        self.side = side
        self.robot = robot
        self.prefix = f"{side}/"
        self.T_base_in_world = np.asarray(T_base_in_world, dtype=np.float64)
        self.T_world_in_base = np.linalg.inv(self.T_base_in_world)
        self._T_base_in_cam = np.asarray(T_base_in_cam, dtype=np.float64)
        self.C = np.eye(4)
        self.C[:3, :3] = robot.R_tool_to_ee      # raw tool frame -> corrected EE frame
        # TCP calibration: shift the EE point along the approach axis (negative = towards the
        # palm, so the fingers reach further past the commanded pinch point). Real rig: the
        # tool site itself (t_flange_tool = 0.156 m), i.e. 0.
        self.C[:3, 3] = robot.R_tool_to_ee @ np.array([0.0, 0.0, float(tcp_offset)])
        # tool pitch calibration: the gripper is held rotated by `tool_pitch_deg` about the EE's
        # spread axis (through the TCP) relative to the commanded EE pose; > 0 levels a downward
        # approach. 0 for the real rig; a per-gripper choice when moving the policy to a new hand.
        if tool_pitch_deg:
            Rx = np.eye(4)
            Rx[:3, :3] = R.from_euler("x", float(tool_pitch_deg), degrees=True).as_matrix()
            self.C = self.C @ Rx
        self.C_inv = np.linalg.inv(self.C)
        self.ik_damping = ik_damping
        self.ik_iters = ik_iters

        self.jids = [model.joint(self.prefix + j).id for j in robot.arm_joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.qlo = model.jnt_range[self.jids, 0].copy()
        self.qhi = model.jnt_range[self.jids, 1].copy()
        self.aids = [model.actuator(self.prefix + a).id for a in robot.arm_actuators]
        self.gid = model.actuator(self.prefix + robot.gripper_actuator).id
        self.sid = model.site(self.prefix + robot.tool_site).id
        gj = model.joint(self.prefix + robot.gripper_joint).id
        self.g_qadr = model.jnt_qposadr[gj]
        self.g_dadr = model.jnt_dofadr[gj]
        self.adh_ids = [model.actuator(f"{self.prefix}adhesion_{fb}").id for fb in robot.finger_bodies
                        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.prefix}adhesion_{fb}") >= 0]

        # scratch data for IK / FK so the physics state is never touched
        self._dk = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        self.mode = "joint"            # "joint" | "cartesian" | "idle"
        q = self.get_j()
        self.q_cmd = q.copy()
        self.jinterp = JointInterpolator(q)
        self.cinterp = CartesianInterpolator(self.fk_raw(q))
        self.g_cmd = robot.gripper_open
        self._g_from = self.g_cmd
        self._g_to = self.g_cmd
        self._g_t0, self._g_T = 0.0, 1e-3
        self.t = 0.0
        self.ik_failures = 0
        self._fix_rotation = False

    # ------------------------------------------------------------------ state
    def get_j(self) -> np.ndarray:
        return self.d.qpos[self.qadr].copy()

    def set_j_instant(self, q, gripper: Optional[float] = None):
        """Teleport (used for resets only)."""
        q = np.asarray(q, dtype=np.float64)
        self.d.qpos[self.qadr] = q
        self.d.qvel[self.dadr] = 0.0
        self.d.ctrl[self.aids] = q
        self.q_cmd = q.copy()
        self.jinterp = JointInterpolator(q)
        self.cinterp.reset(self.fk_raw(q))
        self.mode = "joint"
        if gripper is not None:
            self.g_cmd = self._g_from = self._g_to = gripper
            self.d.ctrl[self.gid] = gripper
            for jn, v in self.robot.gripper_reset_qpos.items():
                self.d.qpos[self.m.jnt_qposadr[self.m.joint(self.prefix + jn).id]] = v

    def fk_raw(self, q: np.ndarray) -> np.ndarray:
        """Raw tool-site pose in the base frame for joint vector q."""
        dk = self._dk
        dk.qpos[:] = self.d.qpos
        dk.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, dk)
        T_w = T_from(dk.site_xmat[self.sid].reshape(3, 3), dk.site_xpos[self.sid])
        return self.T_world_in_base @ T_w

    def _T_raw_in_base(self) -> np.ndarray:
        T_w = T_from(self.d.site_xmat[self.sid].reshape(3, 3), self.d.site_xpos[self.sid])
        return self.T_world_in_base @ T_w

    # --------------------------------------------------- RobotArmTrossen API
    @property
    def T_base_in_cam(self) -> np.ndarray:
        return self._T_base_in_cam

    def get_p_orig(self) -> np.ndarray:
        return T_to_p(self._T_raw_in_base())

    def get_p(self) -> np.ndarray:
        return T_to_p(self._T_raw_in_base() @ self.C)

    def get_T_ee_in_base(self) -> np.ndarray:
        return self._T_raw_in_base() @ self.C

    def get_T_ee_in_cam(self) -> np.ndarray:
        return self._T_base_in_cam @ self.get_T_ee_in_base()

    def get_p_in_cam(self) -> np.ndarray:
        return T_to_p(self.get_T_ee_in_cam())

    def p_in_cam_to_p_in_base(self, p_in_cam):
        p_in_cam = np.asarray(p_in_cam, dtype=np.float64)
        T_cam_in_base = np.linalg.inv(self._T_base_in_cam)
        if len(p_in_cam) == 3:
            return (T_cam_in_base @ np.append(p_in_cam, 1.0))[:3]
        return T_to_p(T_cam_in_base @ p_to_T(p_in_cam))

    def p_in_base_to_p_in_cam(self, p_in_base):
        p_in_base = np.asarray(p_in_base, dtype=np.float64)
        if len(p_in_base) == 3:
            return (self._T_base_in_cam @ np.append(p_in_base, 1.0))[:3]
        return T_to_p(self._T_base_in_cam @ p_to_T(p_in_base))

    @staticmethod
    def T_to_p(T):
        return T_to_p(T)

    @staticmethod
    def p_to_T(p):
        return p_to_T(p)

    def move_p(self, pose, duration: float = 3.0, blocking: bool = False, use_joint_interp: bool = False,
               safe_z_min: float = -0.120, v_ff=None) -> bool:
        T_target = p_to_T(pose)
        T_raw = T_target @ self.C_inv
        if T_raw[2, 3] < safe_z_min:
            T_raw[2, 3] = safe_z_min
        if self._fix_rotation:
            T_raw[:3, :3] = self._T_raw_in_base()[:3, :3]
        # reachability check, like the driver's IK error
        q_goal, err = self.ik(T_raw, self.q_cmd, iters=30)
        if err > 0.01:
            self.ik_failures += 1
            return False
        if self.mode != "cartesian":
            # hand over from joint mode: start the interpolator at the current set-point
            self.cinterp.reset(self.fk_raw(self.q_cmd))
            self.mode = "cartesian"
        self.cinterp.set_goal(T_raw, duration, self.t, v_ff)
        return True

    def move_p_in_cam(self, p_in_cam, duration: float = 3.0, blocking: bool = False,
                      use_joint_interp: bool = False, safe_z_min: float = -0.120, v_ff_in_cam=None) -> bool:
        v_ff = None if v_ff_in_cam is None else self.T_base_in_cam[:3, :3].T @ np.asarray(v_ff_in_cam)
        return self.move_p(self.p_in_cam_to_p_in_base(p_in_cam), duration, blocking, use_joint_interp, safe_z_min, v_ff)

    def move_j(self, joints, duration: float = 3.0, blocking: bool = False) -> bool:
        joints = np.asarray(joints, dtype=np.float64)
        q = joints[: len(self.qadr)]
        self.jinterp = JointInterpolator(self.q_cmd)
        self.jinterp.set_goal(np.clip(q, self.qlo, self.qhi), duration, self.t)
        self.mode = "joint"
        if len(joints) > len(self.qadr):
            self.set_gripper_position(float(joints[len(self.qadr)]), goal_time=duration)
        return True

    def hold_position(self):
        """Cancel arm motion at measured joints without changing physical state."""
        self.q_cmd = self.get_j().copy()
        self.jinterp = JointInterpolator(self.q_cmd)
        self.cinterp.reset(self.fk_raw(self.q_cmd))
        self.mode = "joint"
        self.d.ctrl[self.aids] = self.q_cmd

    def go_home(self, duration: float = 3.0, blocking: bool = False):
        self.move_j(self.robot.home_joints or np.zeros(len(self.qadr)), duration)

    def go_initial(self, duration: float = 2.5, blocking: bool = False):
        self.move_j(self.robot.initial_joints[self.side], duration)

    def motion_done(self) -> bool:
        return not (self.jinterp.active or self.cinterp.active)

    # gripper: actuator control units, robot.gripper_closed .. robot.gripper_open
    def set_gripper_position(self, pos: float, goal_time: float = 1.0):
        lo, hi = sorted((self.robot.gripper_closed, self.robot.gripper_open))
        pos = float(np.clip(pos, lo, hi))
        if abs(pos - self._g_to) < 1e-6:
            return
        self._g_from = self.g_cmd
        self._g_to = pos
        self._g_t0 = self.t
        self._g_T = max(goal_time, 1e-3)

    def close_gripper(self, blocking: bool = False, goal_time: float = 1.0):
        self.set_gripper_position(self.robot.gripper_closed, goal_time)

    def open_gripper(self, blocking: bool = False, goal_time: float = 1.0):
        self.set_gripper_position(self.robot.gripper_open, goal_time)

    def get_gripper_q(self) -> float:
        """1 = open, 0 = closed (measured from the gripper joint)."""
        r = self.robot
        g = float(self.d.qpos[self.g_qadr])
        return float(np.clip((g - r.gripper_joint_closed) / (r.gripper_joint_open - r.gripper_joint_closed), 0.0, 1.0))

    def finger_opening(self) -> float:
        """Gap between the finger pads (m), from the finger bodies (0 = closed on air)."""
        if not hasattr(self, "_finger_ids"):
            self._finger_ids = [self.m.body(self.prefix + b).id for b in self.robot.finger_bodies]
        a, b = self._finger_ids[:2]
        return float(np.linalg.norm(self.d.xpos[a] - self.d.xpos[b])) - self.robot.finger_gap_closed

    def holding_nothing(self, q_empty: float, gap_empty: float = 0.015) -> bool:
        """Grip check from proprioception only: has the gripper closed on air?"""
        if self.robot.grip_sense == "gap":
            return self.finger_opening() < gap_empty
        return self.get_gripper_q() < q_empty

    # ------------------------------------------------------------ grip sensing
    def grip_effort(self) -> float:
        """Gripper motor effort (N): the force of the gripper servo on its joint / tendon. What the real grippers
        report from their motor: trossen_arm `get_gripper_external_effort` (N), Robotiq gCU (current, ~10 mA
        per count, proportional to force). The Franka Hand reports none (width and is_grasped only)."""
        return abs(float(self.d.actuator_force[self.gid]))

    def grip_effort_max(self) -> float:
        lo, hi = self.m.actuator_forcerange[self.gid]
        return float(max(abs(lo), abs(hi))) if self.m.actuator_forcelimited[self.gid] else float("inf")

    def gripper_speed(self) -> float:
        """Gripper joint speed as a fraction of its full stroke per second."""
        r = self.robot
        return abs(float(self.d.qvel[self.g_dadr])) / max(abs(r.gripper_joint_open - r.gripper_joint_closed), 1e-6)

    def pad_forces(self) -> Tuple[float, float]:
        """Normal contact force (N) between each finger's collision geoms and the breads. A tactile pad: NOT on
        the real grippers (they report finger position and motor effort only)."""
        m, d = self.m, self.d
        if not hasattr(self, "_pad_geoms"):
            fb = [m.body(self.prefix + b).id for b in self.robot.finger_bodies[:2]]
            col = (m.geom_contype > 0) | (m.geom_conaffinity > 0)
            self._pad_geoms = [np.nonzero((m.geom_bodyid == b) & col)[0] for b in fb]
            self._f6 = np.zeros(6)
        n = d.ncon
        out = [0.0, 0.0]
        if n == 0:
            return 0.0, 0.0
        g, fl = d.contact.geom[:n], d.contact.flex[:n]
        for k, pg in enumerate(self._pad_geoms):
            hit = (np.isin(g[:, 0], pg) & (fl[:, 1] >= 0)) | (np.isin(g[:, 1], pg) & (fl[:, 0] >= 0))
            for i in np.nonzero(hit)[0]:
                mujoco.mj_contactForce(m, d, int(i), self._f6)
                out[k] += abs(float(self._f6[0]))
        return out[0], out[1]

    def gripper_cmd_closed(self) -> bool:
        r = self.robot
        return abs(self._g_to - r.gripper_closed) < abs(self._g_to - r.gripper_open)

    def gripper_settled_closed(self, vel_tol: float = 0.004) -> bool:
        """Commanded closed, closing profile finished, and the fingers have stopped
        (either fully closed or stalled on an object)."""
        if not self.gripper_cmd_closed() or self.t - self._g_t0 < self._g_T:
            return False
        r = self.robot
        gv = abs(float(self.d.qvel[self.g_dadr])) / max(abs(r.gripper_joint_open - r.gripper_joint_closed), 1e-6)
        return gv < vel_tol / 0.044

    def set_draggable(self, enable: bool):
        pass

    def set_mode(self, mode: str):
        if mode == "idle":
            self.mode = "idle"

    def close(self):
        pass

    # ------------------------------------------------------------------ IK
    def ik(self, T_raw_target: np.ndarray, q0: np.ndarray, iters: Optional[int] = None,
           tol: float = 2e-4) -> Tuple[np.ndarray, float]:
        """Damped least squares on the tool site; returns (q, position error)."""
        dk = self._dk
        q = np.array(q0, dtype=np.float64)
        T_w = self.T_base_in_world @ T_raw_target
        p_t, R_t = T_w[:3, 3], T_w[:3, :3]
        lam2 = self.ik_damping ** 2
        err_p = 1e9
        for _ in range(iters or self.ik_iters):
            dk.qpos[:] = self.d.qpos
            dk.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, dk)
            mujoco.mj_comPos(self.m, dk)
            p = dk.site_xpos[self.sid]
            Rm = dk.site_xmat[self.sid].reshape(3, 3)
            e_p = p_t - p
            e_r = R.from_matrix(R_t @ Rm.T).as_rotvec()
            err_p = float(np.linalg.norm(e_p))
            if err_p < tol and np.linalg.norm(e_r) < 2e-3:
                break
            mujoco.mj_jacSite(self.m, dk, self._jacp, self._jacr, self.sid)
            J = np.vstack([self._jacp[:, self.dadr], self._jacr[:, self.dadr]])
            e = np.concatenate([e_p, 0.5 * e_r])
            W = np.diag([1, 1, 1, 0.5, 0.5, 0.5])
            JW = W @ J
            dq = JW.T @ np.linalg.solve(JW @ JW.T + lam2 * np.eye(6), W @ e)
            nrm = np.linalg.norm(dq)
            if nrm > 0.3:
                dq *= 0.3 / nrm
            q = np.clip(q + dq, self.qlo, self.qhi)
        # final error
        dk.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, dk)
        err_p = float(np.linalg.norm(p_t - dk.site_xpos[self.sid]))
        return q, err_p

    # ---------------------------------------------------------------- servo
    def update(self, t: float):
        """Called once per physics step: advance interpolators, run IK, set servos."""
        self.t = t
        if self.mode == "joint":
            self.q_cmd = self.jinterp.update(t)
        elif self.mode == "cartesian":
            T_raw = self.cinterp.update(t)
            self.q_cmd, _ = self.ik(T_raw, self.q_cmd)
        if self.mode != "idle":
            self.d.ctrl[self.aids] = self.q_cmd
        # gripper (linear position profile, like set_gripper_position(goal_time))
        a = min(max((t - self._g_t0) / self._g_T, 0.0), 1.0)
        self.g_cmd = self._g_from + (self._g_to - self._g_from) * a
        self.d.ctrl[self.gid] = self.g_cmd
        if self.adh_ids:
            on = 1.0 if (self.gripper_cmd_closed() and a > 0.6) else 0.0
            self.d.ctrl[self.adh_ids] = on
