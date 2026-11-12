# -*- coding: utf-8 -*-
"""
SimController — the production `AsyncRobotController` (AriaMimic
inference/InferenceController.py) on a simulation clock.

The real controller runs in its own thread at `control_freq`; here `tick()` is
called by the runner every 1/control_freq seconds of *sim* time, so the same
behaviour is reproduced deterministically:

  update_target():  should-grasp vote over the chunk, temporal ensembling of
                    the last `max_history_len` chunks (exp(-0.25*offset) weights,
                    Markley quaternion averaging), optional grasp latch.
  tick():           pick the target at `look_ahead_offset` (advancing by
                    `step_stride` each tick), safety cage around the *measured*
                    pose, EMA + Slerp against the previous command, then a
                    non-blocking `move_p_in_cam(duration = dt * duration_smooth_param)`
                    and the gripper open/close command.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


class SimController:
    def __init__(self, arms: Dict[str, object], cfg: dict):
        self.arms = arms
        c = cfg
        self.control_freq = float(c.get("control_freq", 10.0))
        self.max_history_len = int(c.get("max_history_len", 1))
        self.step_stride = int(c.get("step_stride", 1))
        self.look_ahead_offset = int(c.get("look_ahead_offset", 30))
        self.alpha_pos = float(c.get("alpha", 0.8))
        self.alpha_rot = float(c.get("alpha_rot", self.alpha_pos))
        self.duration_smooth_param = float(c.get("duration_smooth_param", 1.0))
        self.grasp_threshold = float(c.get("grasp_threshold", 0.5))
        self.should_grasp_method = c.get("should_grasp_method", "first")
        self.grasp_k = {"first_k": int(c.get("grasp_k_first", 1)), "last_k": int(c.get("grasp_k_last", 1)),
                        "sum_k": int(c.get("grasp_k_sum", 3)), "any_k": int(c.get("grasp_k_any", 50))}
        self.use_safety_cage = bool(c.get("use_safety_cage", True))
        self.max_pos_step = float(c.get("max_pos_step", 0.018))
        self.max_rot_step = float(c.get("max_rot_step", 0.10))
        self.safe_z_min = float(c.get("safe_z_min", -0.20))
        self.close_goal_time = float(c.get("close_gripper_goal_time", 1.0))
        self.open_goal_time = float(c.get("open_gripper_goal_time", 1.0))
        self.grasp_latch = bool(c.get("grasp_latch", False))
        self.fix_rotation = bool(c.get("fix_rotation", False))
        # not in the real controller: pass the plan's velocity as the driver's goal feed-forward
        # velocity (trossen_arm set_cartesian_positions(goal_feedforward_velocities=...)); 0 = off
        self.feedforward_gain = float(c.get("feedforward_gain", 0.0))
        # ---- grasp skill (not in the real controller; off by default) ----
        # refine the plan's pinch point with the perceived bread pose, close only once the arm
        # has arrived there (and hold still while closing), check the grip from the gripper
        # reading and re-grasp if it closed on air or the bread slipped out.
        ga = c.get("grasp_assist", {}) or {}
        self.ga_on = bool(ga.get("enabled", False))
        self.ga_refine = bool(ga.get("refine", True))
        self.ga_lat_gain = float(ga.get("lateral_gain", 0.7))      # pull towards the bread's long axis
        self.ga_along_max = float(ga.get("along_max", 0.045))      # m from the bread centre along its axis
        self.ga_up_max = float(ga.get("height_max", 0.004))        # m above the perceived bread centre
        self.ga_up_min = float(ga.get("height_min", -0.008))
        self.ga_tol = float(ga.get("arrive_tol", 0.01))            # m: close once this close to the pinch point
        self.ga_timeout = float(ga.get("approach_timeout", 1.5))   # s
        # > 0: the approach timeout only counts once the arm is this close (m) to the pinch point (a re-grasp
        # starts from high above and timed out in mid-air), with approach_max (s) as the hard limit
        self.ga_near = float(ga.get("timeout_near", 0.0))
        self.ga_approach_max = float(ga.get("approach_max", 4.0))
        self.ga_hold = float(ga.get("close_hold", 0.6))            # s holding still while the fingers close
        self.ga_max_tries = int(ga.get("regrasp_max", 2))
        self.ga_q_empty = float(ga.get("q_empty", 0.10))           # gripper opening (0..1) that means nothing held
        self.ga_gap_empty = float(ga.get("gap_empty", 0.015))      # m between the pads ("gap" sensing grippers)
        self.ga_cooldown = float(ga.get("regrasp_cooldown", 1.2))  # s with the gripper open before a new attempt
        # grip check while holding: "q" (gripper opening, or the pad gap for "gap" grippers), "touch" (contact force
        # on each finger pad: a tactile sensor the real grippers do NOT have) or "effort" (the gripper motor's effort,
        # what trossen_arm / the 2F-85 current report). Empty: below the threshold for empty_time; "touch" also calls
        # a pinch with only one pad on the bread empty (one_pad_empty) and, with slip_drop > 0, a fall of the weaker
        # pad's force below slip_drop x its peak since closing a slip (re-grasp after slip_time)
        self.ga_check = str(ga.get("grip_check", "q"))
        self.ga_f_min = float(ga.get("touch_min", 0.3))            # N on a pad that counts as contact
        self.ga_one_pad = bool(ga.get("one_pad_empty", False))
        self.ga_eff_empty = float(ga.get("effort_empty", 3.0))     # N of gripper effort below which nothing is held
        self.ga_slip_drop = float(ga.get("slip_drop", 0.0))
        self.ga_slip_time = float(ga.get("slip_time", 0.1))
        self.ga_empty_time = float(ga.get("empty_time", 0.3))
        # wrist yaw at the pinch point: "off", "align" (turn the hand about the vertical so its fingers close across
        # the bread's short axis) or "fit" (only as far as needed for the bread's width across the fingers, an
        # ellipse of the perceived half length / width, to fit the hand's opening minus yaw_margin), at most
        # yaw_max_deg; closing then also waits for the orientation (arrive_rot_deg)
        self.ga_yaw = str(ga.get("yaw", "off"))
        self.ga_orientation_world = ga.get("orientation_world_rpy_deg")
        self.ga_yaw_max = np.radians(float(ga.get("yaw_max_deg", 60.0)))
        self.ga_yaw_margin = float(ga.get("yaw_margin", 0.01))
        robot = getattr(next(iter(arms.values()), None), "robot", None)
        self.ga_open = float(ga.get("max_opening", getattr(robot, "max_opening", 0.08)))   # m, the hand's opening
        self.ga_rot_tol = np.radians(float(ga.get("arrive_rot_deg", 6.0)))
        # slow final approach: within slow_dist (m) of the pinch point the arm moves at most slow_speed (m/s)
        self.ga_slow_dist = float(ga.get("slow_dist", 0.0))
        self.ga_slow_speed = float(ga.get("slow_speed", 0.05))
        self.ga_retry_down = float(ga.get("retry_lower", 0.005))   # each re-grasp aims this much lower ...
        self.ga_retry_in = float(ga.get("retry_inward", 0.01))     # ... and this much closer to the bread's middle
        # re-perceive the bread before a re-grasp (it may have moved, or dropped elsewhere): "off", "skill" (update
        # the pinch-point geometry) or "policy" (also the policy's object state); the runner runs the perception
        # once the bread is visible again, the next approach waits for it up to reperceive_timeout
        rp = ga.get("reperceive", "off")
        self.ga_reperceive = {True: "policy", False: "off", None: "off"}.get(rp, rp) if isinstance(rp, (bool, type(None))) else str(rp)
        self.ga_rp_timeout = float(ga.get("reperceive_timeout", 1.5))
        self.t = 0.0
        self.geom = {}                                              # per side: perceived bread geometry (world)
        self.T_cam_in_world = None

        sides = list(arms)
        self.shared_traj = {s: [] for s in sides}
        self.shared_grip = {s: 0.0 for s in sides}
        self.history = {s: [] for s in sides}
        self.active_traj = {s: [] for s in sides}
        self.traj_idx = {s: self.look_ahead_offset for s in sides}
        self.last_cmd = {s: None for s in sides}
        self.muted = {s: False for s in sides}
        self._latch_ever_open = {s: False for s in sides}
        self._latch_locked = {s: False for s in sides}
        self.fixed_rot = {}
        self.n_ik_fail = {s: 0 for s in sides}
        self.shared_gprob = {s: None for s in sides}
        self.active_gprob = {s: None for s in sides}
        self.gs = {s: {"state": "free", "target": None, "t0": 0.0, "tries": 0, "empty_t": None, "cool_until": 0.0}
                   for s in sides}
        self.events = {s: [] for s in sides}
        self.reperceive = {s: None for s in sides}                 # pending re-perception: its deadline (s)

    # ------------------------------------------------------------------
    @staticmethod
    def _avg_quat(quats, weights):
        Q = np.zeros((4, 4))
        ws = float(np.sum(weights))
        for q, w in zip(quats, weights):
            q = np.asarray(q)
            Q += (w / ws) * np.outer(q, q)
        evals, evecs = np.linalg.eigh(Q)
        return evecs[:, np.argmax(evals)]

    def _should_grasp(self, g):
        g = np.asarray(g).flatten()
        th = self.grasp_threshold
        m = self.should_grasp_method
        if m == "any":
            return bool(np.any(g > th))
        if m == "any_first_k":     # grasp intent anywhere in the next k predicted steps (k = horizon == "any")
            return bool(np.any(g[: max(1, self.grasp_k["any_k"])] > th))
        if m == "average":
            return bool(np.mean(g) > th)
        if m == "last":
            return bool(g[-1] > th)
        if m == "first_k":
            return bool(np.mean(g[: min(self.grasp_k["first_k"], len(g))]) > th)
        if m == "last_k":
            return bool(np.mean(g[-min(self.grasp_k["last_k"], len(g)):]) > th)
        if m == "sum_k":
            return bool(np.sum(g > th) >= self.grasp_k["sum_k"])
        return bool(g[0] > th)

    def update_target(self, traj_dict: Dict[str, List[np.ndarray]], grasp_dict: Dict[str, np.ndarray]):
        for side, chunk in traj_dict.items():
            if side not in self.arms or len(chunk) == 0:
                continue
            should_grasp = self._should_grasp(grasp_dict.get(side, np.zeros(len(chunk))))
            hb = self.history[side]
            hb.append(chunk)
            if len(hb) > self.max_history_len:
                hb.pop(0)
            ens = []
            for k in range(len(chunk)):
                sp, quats, ws = np.zeros(3), [], []
                for hi, ht in enumerate(hb):
                    off = len(hb) - 1 - hi
                    kk = k + off
                    if kk < len(ht):
                        w = np.exp(-0.25 * off)
                        pt = ht[kk]
                        sp += pt[:3] * w
                        quats.append(R.from_rotvec(pt[3:]).as_quat())
                        ws.append(w)
                if not ws:
                    break
                q = self._avg_quat(quats, ws)
                ens.append(np.concatenate([sp / np.sum(ws), R.from_quat(q).as_rotvec()]))
            if self.grasp_latch:
                if not should_grasp:
                    self._latch_ever_open[side] = True
                elif self._latch_ever_open[side]:
                    self._latch_locked[side] = True
                if self._latch_locked[side]:
                    should_grasp = True
            if not self.muted[side]:
                self.shared_traj[side] = ens
                self.shared_grip[side] = 1.0 if should_grasp else 0.0
                self.shared_gprob[side] = np.asarray(grasp_dict.get(side, np.zeros(len(chunk)))).ravel()[:len(ens)]

    # ------------------------------------------------------------------ grasp skill
    def set_geometry(self, side: str, geom: dict, T_cam_in_world: np.ndarray):
        """geom: bread centre `c` (world), unit long axis `a` (horizontal, world), half length/width."""
        self.geom[side] = geom
        self.T_cam_in_world = np.asarray(T_cam_in_world, float)

    def reperceived(self, side: str, geom: dict):
        """The runner re-ran the perception before a re-grasp: new bread geometry, the next approach may start."""
        old = self.geom.get(side)
        self.geom[side] = geom
        self.reperceive[side] = None
        moved = float(np.linalg.norm(np.subtract(geom["c"][:2], old["c"][:2]))) if old else 0.0
        self.events[side].append((round(self.t, 2), "reperceive", round(moved, 4)))

    def _refine(self, side: str, p_cam: np.ndarray) -> np.ndarray:
        g = self.geom.get(side)
        if not self.ga_refine or g is None:
            return p_cam
        Tcw = self.T_cam_in_world
        pw = Tcw[:3, :3] @ p_cam[:3] + Tcw[:3, 3]
        c, a = np.asarray(g["c"]), np.asarray(g["a"])
        up = np.array([0.0, 0.0, 1.0])
        w = np.cross(up, a)
        d = pw - c
        s_a, s_w, s_u = float(d @ a), float(d @ w), float(d[2])
        n = self.gs[side]["tries"]                  # adaptive re-grasp: lower and more central each time
        along_max = max(0.0, self.ga_along_max - n * self.ga_retry_in)
        s_a = float(np.clip(s_a, -along_max, along_max))
        s_w *= (1.0 - self.ga_lat_gain) ** (1 + n)
        up_max = self.ga_up_max - n * self.ga_retry_down
        s_u = float(np.clip(s_u, min(self.ga_up_min, up_max), up_max))
        pw2 = c + s_a * a + s_w * w + s_u * up
        out = p_cam.copy()
        out[:3] = Tcw[:3, :3].T @ (pw2 - Tcw[:3, 3])
        if self.ga_orientation_world is not None:
            rotation = R.from_euler("xyz", self.ga_orientation_world, degrees=True).as_matrix()
            out[3:] = R.from_matrix(Tcw[:3, :3].T @ rotation).as_rotvec()
        elif self.ga_yaw != "off":
            out[3:] = self._yaw_fix(g, out[3:])
        return out

    def _yaw_fix(self, g: dict, rv_cam: np.ndarray) -> np.ndarray:
        """Turn the commanded hand about the world vertical (through the pinch point) so the finger spread axis
        crosses the bread's long axis at 90 deg ("align") or just enough for the bread to fit ("fit")."""
        Rc = self.T_cam_in_world[:3, :3]
        Rw = Rc @ R.from_rotvec(rv_cam).as_matrix()
        sp = Rw[:2, 0]
        if np.linalg.norm(sp) < 1e-6:
            return rv_cam
        a = np.asarray(g["a"][:2], float)
        phi = float(np.arctan2(sp[1], sp[0]) - np.arctan2(a[1], a[0]))      # spread vs long axis
        # candidate yaw corrections that put the spread axis at +-90 deg to the long axis; take the smallest
        cands = [np.pi / 2 - phi, -np.pi / 2 - phi]
        cands = [(c + np.pi) % (2 * np.pi) - np.pi for c in cands]
        d = min(cands, key=abs)
        if self.ga_yaw == "fit":
            L, W = float(g.get("half_len", 0.07)), float(g.get("half_wid", 0.035))
            room = 0.5 * (self.ga_open - self.ga_yaw_margin)
            width = lambda ang: np.sqrt((L * np.cos(ang)) ** 2 + (W * np.sin(ang)) ** 2)   # half extent across the fingers
            if width(phi) <= room:
                return rv_cam
            if W < room:                                              # smallest turn that makes it fit
                ang_fit = np.arccos(np.sqrt(max(0.0, (room ** 2 - W ** 2) / max(L ** 2 - W ** 2, 1e-9))))
                d = float(np.sign(d) * max(0.0, abs(d) - (np.pi / 2 - ang_fit)))
        d = float(np.clip(d, -self.ga_yaw_max, self.ga_yaw_max))
        Rz = R.from_euler("z", d).as_matrix()
        return R.from_matrix(Rc.T @ Rz @ Rw).as_rotvec()

    def _grasp_point(self, side: str, traj, idx: int) -> np.ndarray:
        gp = self.active_gprob[side]
        k = idx
        if gp is not None and len(gp):
            hits = np.nonzero(gp > self.grasp_threshold)[0]
            if len(hits):
                k = int(hits[0])
        k = min(max(k, 0), len(traj) - 1)
        return self._refine(side, np.array(traj[k], dtype=np.float64))

    def _regrasp(self, side: str, arm, why: str):
        gs = self.gs[side]
        gs["tries"] += 1
        gs["state"], gs["target"], gs["empty_t"] = "free", None, None
        gs["cool_until"] = self.t + self.ga_cooldown
        self._latch_locked[side] = False
        self._latch_ever_open[side] = False
        self.shared_grip[side] = 0.0
        arm.open_gripper(goal_time=self.open_goal_time)
        self.events[side].append((round(self.t, 2), "regrasp:" + why))
        if self.ga_reperceive != "off":
            self.reperceive[side] = self.t + self.ga_rp_timeout

    def _skill(self, side: str, arm, want_close: bool, traj, idx: int, actual: np.ndarray):
        """Grasp skill state machine -> (target override or None, close?)."""
        gs = self.gs[side]
        rp = self.reperceive[side]
        if rp is not None and self.t >= rp:                     # bread never visible enough: keep the old geometry
            self.reperceive[side] = rp = None
            self.events[side].append((round(self.t, 2), "reperceive:timeout"))
        if gs["state"] == "free":
            if want_close and self.t >= gs["cool_until"] and gs["tries"] <= self.ga_max_tries and rp is None:
                gs.update(state="approach", target=self._grasp_point(side, traj, idx), t0=self.t, near_t=None)
                Tcw = self.T_cam_in_world
                tw = (Tcw[:3, :3] @ gs["target"][:3] + Tcw[:3, 3]) if Tcw is not None else gs["target"][:3]
                self.events[side].append((round(self.t, 2), "approach", [round(float(v), 4) for v in tw]))
            else:
                return None, False
        if gs["state"] == "approach":
            if not want_close:                      # the policy no longer wants to grasp
                gs.update(state="free", target=None)
                return None, False
            err = float(np.linalg.norm(actual[:3] - gs["target"][:3]))
            if self.ga_near > 0:
                if gs.get("near_t") is None and err < self.ga_near:
                    gs["near_t"] = self.t
                timed_out = (gs.get("near_t") is not None and self.t - gs["near_t"] > self.ga_timeout) \
                    or self.t - gs["t0"] > self.ga_approach_max
            else:
                timed_out = self.t - gs["t0"] > self.ga_timeout
            arrived = err < self.ga_tol
            if arrived and (self.ga_yaw != "off" or self.ga_orientation_world is not None):
                arrived = (R.from_rotvec(gs["target"][3:]) * R.from_rotvec(actual[3:]).inv()).magnitude() < self.ga_rot_tol
            if arrived or timed_out:
                gs.update(state="closing", t0=self.t)
                self.events[side].append((round(self.t, 2), "close", round(err, 4)))
            else:
                tgt = gs["target"]
                if self.ga_slow_dist > 0 and err < self.ga_slow_dist:   # slow final approach
                    step = self.ga_slow_speed / self.control_freq
                    if err > step:
                        tgt = tgt.copy()
                        tgt[:3] = actual[:3] + (gs["target"][:3] - actual[:3]) * (step / err)
                return tgt, False
        if gs["state"] == "closing":
            if self.t - gs["t0"] >= self.ga_hold:
                gs.update(state="holding", t0=self.t, empty_t=None, f_peak=None)
            return gs["target"], True
        if gs["state"] == "holding":
            if not want_close:
                gs.update(state="free", target=None)
                return None, False
            why = self._grip_status(arm, gs)
            if why:                                                     # nothing (or one pad only) between the fingers
                gs["empty_t"] = gs["empty_t"] if gs["empty_t"] is not None else self.t
                wait = self.ga_slip_time if why == "slip" else self.ga_empty_time
                if self.t - gs["empty_t"] > wait and gs["tries"] < self.ga_max_tries:
                    self._regrasp(side, arm, why)
                    return None, False
            else:
                gs["empty_t"] = None
            return None, True
        return None, want_close

    def _grip_status(self, arm, gs) -> str:
        """'' while the grip holds, else why not: 'empty', 'one_pad' or 'slip'."""
        if self.ga_check == "touch":
            f = sorted(arm.pad_forces())
            if f[1] < self.ga_f_min:
                return "empty"
            if self.ga_one_pad and f[0] < self.ga_f_min:
                return "one_pad"
            gs["f_peak"] = max(gs.get("f_peak") or 0.0, f[0])
            if self.ga_slip_drop > 0 and gs["f_peak"] >= 2 * self.ga_f_min and f[0] < self.ga_slip_drop * gs["f_peak"]:
                return "slip"
            return ""
        if self.ga_check == "effort":
            e = arm.grip_effort()
            if e < self.ga_eff_empty:
                return "empty"
            gs["f_peak"] = max(gs.get("f_peak") or 0.0, e)
            if self.ga_slip_drop > 0 and e < self.ga_slip_drop * gs["f_peak"]:
                return "slip"
            return ""
        return "empty" if arm.holding_nothing(self.ga_q_empty, self.ga_gap_empty) else ""

    def clear(self, side: str):
        self.shared_traj[side] = []
        self.history[side] = []
        self.active_traj[side] = []
        self.last_cmd[side] = None

    # ------------------------------------------------------------------
    def tick(self):
        """One control step (call every 1/control_freq s of sim time)."""
        dt = 1.0 / self.control_freq
        self.t += dt
        for side, arm in self.arms.items():
            if len(self.shared_traj[side]) > 0:
                self.active_traj[side] = self.shared_traj[side]
                self.active_gprob[side] = self.shared_gprob[side]
                self.traj_idx[side] = min(self.look_ahead_offset, len(self.active_traj[side]) - 1)
                self.shared_traj[side] = []
            if self.muted[side]:
                self.active_traj[side] = []
                self.last_cmd[side] = None
                continue
            traj = self.active_traj[side]
            idx = self.traj_idx[side]
            if not (len(traj) > 0 and idx < len(traj)):
                continue
            target = np.array(traj[idx], dtype=np.float64)
            actual = arm.get_p_in_cam()
            want_close = self.shared_grip[side] > self.grasp_threshold
            override = None
            if self.ga_on:
                override, want_close = self._skill(side, arm, want_close, traj, idx, actual)
                if override is not None:
                    target = override.copy()
            if self.use_safety_cage:
                dp = target[:3] - actual[:3]
                dist = np.linalg.norm(dp)
                if dist > self.max_pos_step:
                    target[:3] = actual[:3] + dp / dist * self.max_pos_step
                ra, rt = R.from_rotvec(actual[3:]), R.from_rotvec(target[3:])
                ang = (rt * ra.inv()).magnitude()
                if ang > self.max_rot_step:
                    sl = Slerp([0, 1], R.from_quat([ra.as_quat(), rt.as_quat()]))
                    target[3:] = sl(self.max_rot_step / ang).as_rotvec()
            last = self.last_cmd[side]
            if last is None or override is not None:     # the grasp skill servoes straight to its point
                cmd = target.copy()
            else:
                pos = self.alpha_pos * last[:3] + (1 - self.alpha_pos) * target[:3]
                sl = Slerp([0, 1], R.from_quat([R.from_rotvec(last[3:]).as_quat(),
                                                R.from_rotvec(target[3:]).as_quat()]))
                cmd = np.concatenate([pos, sl(1.0 - self.alpha_rot).as_rotvec()])
            if self.fix_rotation:
                if side not in self.fixed_rot:
                    self.fixed_rot[side] = actual[3:].copy()
                cmd[3:] = self.fixed_rot[side]
            self.last_cmd[side] = cmd.copy()
            v_ff = None
            if self.feedforward_gain > 0 and len(traj) > 1:
                i1 = min(idx + self.step_stride, len(traj) - 1)
                i0 = max(i1 - self.step_stride, 0)
                if i1 > i0:     # the servo walks the plan at step_stride indices per tick
                    v_ff = self.feedforward_gain * (np.asarray(traj[i1][:3]) - np.asarray(traj[i0][:3])) \
                        * (self.step_stride / (i1 - i0)) / dt
            ok = arm.move_p_in_cam(cmd, duration=dt * self.duration_smooth_param, blocking=False,
                                   safe_z_min=self.safe_z_min, v_ff_in_cam=v_ff)
            if not ok:
                self.n_ik_fail[side] += 1
                self.shared_traj[side] = []
                self.history[side] = []
                self.last_cmd[side] = actual.copy()
            if want_close:
                arm.close_gripper(goal_time=self.close_goal_time)
            else:
                arm.open_gripper(goal_time=self.open_goal_time)
            self.traj_idx[side] += self.step_stride
