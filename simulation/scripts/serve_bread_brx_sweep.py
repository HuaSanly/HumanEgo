"""Test nine bread positions with fixed policy noise and measured soft-bread grasps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanego_sim.env import SIM_ROOT, ServeBreadSim, load_config
from humanego_sim.runner import EpisodeRunner, load_policy


class MeasuredBreadSim(ServeBreadSim):
    """Contact telemetry observes physics without feeding the controller."""

    def reset(self, *args, **kwargs):
        self.measuring = False
        super().reset(*args, **kwargs)
        self.initial_bread = self.object_pos("bread_right").copy()
        self.measurements = {"one_pad": False, "both_pads": False, "max_lift": 0.0,
                             "max_lift_duration": 0.0, "max_secure_duration": 0.0,
                             "min_pinch_distance": float("inf"), "first_close_distance": None,
                             "first_close_delta": None}
        self.lift_duration = self.secure_duration = 0.0
        self.next_measure, self.last_measure = 0.0, 0.0
        self.telemetry = []
        self.measuring = True

    def step(self, n=1):
        for _ in range(n):
            super().step()
            if getattr(self, "measuring", False) and self.t >= self.next_measure:
                self.measure()
                self.next_measure = self.t + 0.01

    def measure(self):
        arm, metrics = self.arms["right"], self.measurements
        bread = self.object_pos("bread_right")
        pinch = self.d.site_xpos[arm.sid].copy()
        forces = np.asarray(arm.pad_forces())
        touching = forces > 0.1
        lift = float(bread[2] - self.initial_bread[2])
        dt = self.t - self.last_measure
        self.last_measure = self.t
        self.lift_duration = self.lift_duration + dt if lift > 0.025 else 0.0
        self.secure_duration = self.secure_duration + dt if lift > 0.025 and touching.all() else 0.0
        metrics["one_pad"] |= bool(touching.any())
        metrics["both_pads"] |= bool(touching.all())
        metrics["max_lift"] = max(metrics["max_lift"], lift)
        metrics["max_lift_duration"] = max(metrics["max_lift_duration"], self.lift_duration)
        metrics["max_secure_duration"] = max(metrics["max_secure_duration"], self.secure_duration)
        distance = float(np.linalg.norm(pinch - bread))
        metrics["min_pinch_distance"] = min(metrics["min_pinch_distance"], distance)
        if arm.gripper_cmd_closed() and metrics["first_close_distance"] is None:
            metrics["first_close_distance"] = distance
            metrics["first_close_delta"] = (pinch - bread).tolist()
        self.telemetry.append({"t": round(self.t, 4), "bread": bread.tolist(), "pinch": pinch.tolist(),
                               "forces": forces.tolist(), "gripper": arm.get_gripper_q()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(SIM_ROOT) / "configs" / "serve_bread_brx.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=7)
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(9)))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    cfg = load_config(args.config, {"control": {"max_episode_time": args.duration}})
    if cfg["task"] != "serve_bread" or cfg["mode"] != "single" or cfg["physics"]["bread_model"] != "soft":
        parser.error("the contact probe requires single-arm serve_bread with soft bread")
    if any(index not in range(9) for index in args.indices) or len(set(args.indices)) != len(args.indices):
        parser.error("indices must be unique values from 0 through 8")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    bread = cfg["scene"]["single"]["bread"]["pose"]
    positions = [[round(bread[0] + dx, 4), round(bread[1] + dy, 4), bread[2]]
                 for dy in (-0.05, 0.0, 0.05) for dx in (-0.05, 0.0, 0.05)]
    checkpoint = Path(cfg["policy"]["ckpt"]).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path(SIM_ROOT).parent / checkpoint
    protocol = {"seed": args.seed, "policy_seed": args.policy_seed, "duration_limit": args.duration,
                "indices": args.indices, "positions": positions,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "grasp_criteria": {"pad_force_N": 0.1, "lift_m": 0.025,
                                   "lift_duration_s": 0.2, "secure_lift_duration_s": 0.1}}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    policy = load_policy(cfg, args.device)
    env = MeasuredBreadSim(cfg)
    rows = []
    try:
        runner = EpisodeRunner(cfg, policy=policy, device=args.device, env=env)
        for index in args.indices:
            scene = {"bread_right": positions[index]}
            folder = args.out / f"ep{index:02d}"
            result = runner.run(scene=scene, seed=args.seed, policy_seed=args.policy_seed,
                                log_dir=str(folder), record_hz=10, save_policy_images=True,
                                verbose=False, intro=False)
            row = {"index": index, "scene": scene, "success": result.success,
                   "duration": result.duration, "ik_fail": result.metrics["ik_fail"],
                   "per_side": result.per_side, **env.measurements}
            row["lifted"] = row["max_lift_duration"] >= 0.2
            row["secure_lift"] = row["max_secure_duration"] >= 0.1
            rows.append(row)
            (folder / "contacts.json").write_text(json.dumps(env.telemetry))
            (folder / "result.json").write_text(json.dumps(row, indent=2))
            summary = {"seed": args.seed, "policy_seed": args.policy_seed, "duration_limit": args.duration,
                       "results": rows,
                       "counts": {key: sum(bool(r[key]) for r in rows)
                                  for key in ("one_pad", "both_pads", "lifted", "secure_lift", "success")}}
            (args.out / "results.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(row), flush=True)
    finally:
        env.renderer.close()


if __name__ == "__main__":
    main()
