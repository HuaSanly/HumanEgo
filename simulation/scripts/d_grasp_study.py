"""Paired parameter studies on the fixed D scene, measuring real finger contacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import mujoco
import numpy as np
import yaml

from humanego_sim.env import ServeBreadSim, deep_update, load_config
from humanego_sim.runner import EpisodeRunner, load_policy
from humanego_sim.stats import paired, wilson
from real2sim_ablation import trial_config, write_json

STATE = {}


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


def init_worker(root: str, device: str) -> None:
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    output = Path(root)
    STATE.update(root=output, device=device, runner=None, label=None)
    STATE["configs"] = {path.stem: load_config(str(path)) for path in output.glob("*.yaml")}
    STATE["policy"] = load_policy(STATE["configs"]["baseline"], device)


def run_trial(job: tuple[str, int, int]) -> dict:
    label, index, seed = job
    directory = STATE["root"] / "episodes" / label / f"ep{index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    cfg, variation = trial_config(STATE["configs"][label], index, seed)
    result = {"idx": index, "variant": label, "seed": seed * 1000 + index, "scene": variation["scene"]}
    try:
        if STATE["label"] != label:
            if STATE["runner"] is not None:
                STATE["runner"].env.renderer.close()
            env = MeasuredSim(cfg)
            STATE["runner"] = EpisodeRunner(cfg, policy=STATE["policy"], device=STATE["device"], env=env)
            STATE["label"] = label
        runner = STATE["runner"]
        value = runner.run(scene=result["scene"], seed=result["seed"], log_dir=str(directory),
                           save_policy_images=index < 2, record_hz=10, intro=False, verbose=False)
        result.update(success=value.success, per_side=value.per_side, duration=value.duration,
                      wall=value.wall_time, ik_fail=value.metrics["ik_fail"], **runner.env.measurements)
        result["lifted"] = result["max_lift_duration"] >= 0.2
        result["secure_lift"] = result["max_secure_duration"] >= 0.1
        result["transported"] = result["lifted"] and result["max_horizontal"] > 0.05
        write_json(directory / "contact_telemetry.json", runner.env.telemetry)
    except Exception as error:
        result.update(error=repr(error), trace=traceback.format_exc())
    write_json(directory / "result.json", result)
    return result


def summarize(root: Path, labels: list[str], n: int) -> dict:
    groups, results = {}, {}
    for label in labels:
        rows = [json.loads(path.read_text()) for path in sorted((root / "episodes" / label).glob("*/result.json"))
                if int(path.parent.name[2:]) < n]
        valid = [row for row in rows if "error" not in row]
        results[label] = valid
        groups[label] = {"completed": len(rows), "errors": len(rows) - len(valid), "n": len(valid),
                         **{key: sum(row[key] for row in valid) for key in ("one_pad", "both_pads", "lifted", "secure_lift", "transported", "success")},
                         "median_min_pinch_distance": float(np.median([r["min_pinch_distance"] for r in valid])) if valid else None,
                         "mean_ik_fail": float(np.mean([r["ik_fail"] for r in valid])) if valid else None,
                         "secure_lift_wilson95": wilson(sum(r["secure_lift"] for r in valid), len(valid))}
    comparisons = {}
    for label in labels:
        if label == "baseline":
            continue
        comparisons[label] = {}
        for metric in ("lifted", "secure_lift", "success"):
            mapped = lambda rows: [{"idx": r["idx"], "per_side": {"right": {"success": r[metric]}}} for r in rows]
            comparisons[label][metric] = paired(mapped(results[label]), mapped(results["baseline"]))
    summary = {"n_per_group": n, "groups": groups, "paired_vs_baseline": comparisons}
    write_json(root / "summary.json", summary)
    lines = ["# D 场景抓取参数实验", "", "D 的材质、相机、尺寸、质量和摩擦固定；每组使用相同的布局扰动和推理噪声。具体改动见 variants.json。", "",
             "| 组别 | 完成 | 错误 | 单指接触 | 双指接触 | 持续抬起 | 双指持稳抬起 | 搬运 | 入盒 | 最近夹点距离 cm | 平均 IK 拒绝 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, value in groups.items():
        if value["n"]:
            lines.append(f"| {label} | {value['completed']}/{n} | {value['errors']} | {value['one_pad']} | {value['both_pads']} | {value['lifted']} | {value['secure_lift']} | {value['transported']} | {value['success']} | {100*value['median_min_pinch_distance']:.2f} | {value['mean_ik_fail']:.1f} |")
    lines += ["", "持续抬起：方块比初始位置高 2.5 cm，持续至少 0.2 s；双指持稳抬起还要求两指对方块均有 >0.1 N 接触力，持续至少 0.1 s。",
              "接触力只用于评估，不回馈给策略或改变物理。夹点为物理工具 site，与人为偏移后的 TCP 分开计量。",
              "每组只有 10 次，本轮属于开发调参；按开发结果挑出的组合需要另用新的场景/噪声种子验证。", ""]
    (root / "report.md").write_text("\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="simulation/configs/pick_cube.yaml")
    parser.add_argument("--variants", type=Path, help="parameter groups as JSON; saved in the output directory")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output / "variants.json").exists():
        if args.variants is None:
            parser.error("a new study requires --variants; existing studies keep their frozen variants.json")
        variants = json.loads(args.variants.read_text())
        write_json(args.output / "variants.json", variants)
        base = load_config(args.config)
        for label, overrides in variants.items():
            (args.output / f"{label}.yaml").write_text(yaml.safe_dump(deep_update(base, overrides), sort_keys=False))
    labels = list(json.loads((args.output / "variants.json").read_text()))
    protocol = {"seed": args.seed, "n": args.n,
                "config_sha256": {label: hashlib.sha256((args.output / f"{label}.yaml").read_bytes()).hexdigest() for label in labels}}
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("frozen protocol/config changed; use a new output directory")
    write_json(path, protocol)
    jobs = [(label, index, args.seed) for label in labels for index in range(args.n)
            if not (args.output / "episodes" / label / f"ep{index:03d}/result.json").exists()]
    if args.report_only or not jobs:
        summarize(args.output, labels, args.n)
        return
    started = time.time()
    print(f"[D study] groups={labels} remaining={len(jobs)}", flush=True)
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker, initargs=(str(args.output), args.device)) as pool:
        for done, result in enumerate(pool.imap_unordered(run_trial, jobs), 1):
            if done % 5 == 0 or "error" in result:
                print(f"[D study] {done}/{len(jobs)} {result['variant']}:{result['idx']} lift={result.get('lifted')} secure={result.get('secure_lift')} success={result.get('success')} error={result.get('error')} wall={time.time()-started:.1f}s", flush=True)
    summarize(args.output, labels, args.n)


if __name__ == "__main__":
    main()
