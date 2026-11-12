"""Reproducible paired pick_cube A–E evaluation with incremental results and reports."""
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

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from humanego_sim.env import ServeBreadSim, deep_update, load_config
from humanego_sim.stats import paired, wilson

WORKER = {}


def write_json(path: Path, data: dict | list) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def trial_config(config: dict, index: int, seed: int) -> tuple[dict, dict]:
    """Same normalized layout perturbation for every group, at its own nominal geometry."""
    import copy
    cfg = copy.deepcopy(config)
    rng = np.random.default_rng([seed, index, 11])
    scene = {}
    for name in ("container", "cube"):
        pose = np.asarray(cfg["scene"]["objects"][name]["pose"], float)
        pose += rng.uniform([-0.015, -0.015, -8], [0.015, 0.015, 8])
        scene[name] = pose.tolist()
    variation = {}
    if cfg.get("ablation_randomization"):
        rng = np.random.default_rng([seed, index, 29])
        view = cfg["rig"]["camera_pose"]
        view["position"][2] += float(rng.uniform(-0.02, 0.02))
        view["pitch_deg"] += float(rng.uniform(-3, 3))
        variation = {"intensity": float(rng.uniform(0.92, 1.08)), "tint": float(rng.uniform(-0.025, 0.025))}
    return cfg, {"scene": scene, "light": variation}


def init_worker(root: str, device: str) -> None:
    import torch
    from humanego_sim.runner import load_policy
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    WORKER.update(root=Path(root), device=device, runner=None, key=None)
    WORKER["configs"] = {label: load_config(str(Path(root) / f"{label}.yaml")) for label in "ABCDE"}
    WORKER["policy"] = load_policy(WORKER["configs"]["A"], device=device)


def run_trial(job: tuple[str, int, int]) -> dict:
    from humanego_sim.runner import EpisodeRunner
    from humanego_sim.robustness import set_lights
    label, index, seed = job
    output = WORKER["root"] / "episodes" / label / f"ep{index:03d}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cfg, variation = trial_config(WORKER["configs"][label], index, seed)
    result = {"variant": label, "idx": index, "seed": seed * 1000 + index, **variation}
    try:
        key = json.dumps(cfg, sort_keys=True)
        if key != WORKER["key"]:
            if WORKER["runner"] is not None:
                WORKER["runner"].env.renderer.close()
            WORKER["runner"] = EpisodeRunner(cfg, policy=WORKER["policy"], device=WORKER["device"])
            WORKER["key"] = key
        runner = WORKER["runner"]
        set_lights(runner.env, variation["light"] or None)
        if index < 2:
            runner.env.reset(variation["scene"])
            cv2.imwrite(str(output / "initial_scene.png"), runner.env.render_rgb(hide_robot=True))
        value = runner.run(scene=variation["scene"], seed=result["seed"], log_dir=str(output),
                           save_policy_images=index < 2, record_hz=10, intro=False, verbose=False)
        result.update(success=value.success, per_side=value.per_side, duration=value.duration,
                      wall=value.wall_time, metrics=value.metrics)
        # Log lifted/transported milestones from actual body positions, independent of grasp flags.
        episode = json.loads((output / "episode.json").read_text())
        initial_cube = np.asarray(episode["frames"][0]["objects"]["cube"][:3])
        cube = np.asarray([frame["objects"]["cube"][:3] for frame in episode["frames"]])
        result["physical_lift"] = bool(cube[:, 2].max() - initial_cube[2] > 0.025)
        result["physical_transport"] = bool(result["physical_lift"] and np.linalg.norm(cube[:, :2] - initial_cube[:2], axis=1).max() > 0.05)
        result["final_cube"] = cube[-1].tolist()
        result["effective_config"] = cfg if label == "E" else None
    except Exception as error:
        result.update(error=repr(error), trace=traceback.format_exc(), wall=time.time() - started)
    write_json(output / "result.json", result)
    return result


def render_comparison(root: Path) -> dict:
    from humanego_sim.perception import SimPerception
    from real2sim_calibrate import region_stats
    stats = json.loads((root / "training_statistics.json").read_text())
    target = stats["image_targets"]
    panels, metrics = [], {}
    for label in "ABCDE":
        cfg, variation = trial_config(load_config(str(root / f"{label}.yaml")), 0, 20261023)
        env = ServeBreadSim(cfg)
        env.reset()
        from humanego_sim.robustness import set_lights
        set_lights(env, variation["light"] or None)
        raw = env.render_rgb(hide_robot=True)
        masks = env.render_segmentation()
        measured = {role: region_stats(raw, masks[name]) for role, name in (("obj1", "container"), ("obj2", "cube"))}
        measured["geometry_mae"] = float(np.mean([abs(measured[r][k] - target[r][k]) for r in ("obj1", "obj2") for k in ("width", "height", "cx", "cy")]))
        measured["object_rgb_mae"] = float(np.mean([np.abs(np.asarray(measured[r]["rgb"]) - target[r]["rgb"]).mean() for r in ("obj1", "obj2")]))
        perception = SimPerception(env)
        objects = perception.estimate_objects(env.sides, shared_context=True)
        policy = perception.make_side_image("right", objects["right"], perception.clean_background(), env.arms["right"], False)
        cv2.imwrite(str(root / f"{label}_scene.png"), raw)
        cv2.imwrite(str(root / f"{label}_policy.png"), policy)
        panel = cv2.resize(policy, (320, 240))
        cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        panels.append(panel)
        metrics[label] = measured
        env.renderer.close()
    # Pick the training episode nearest the median layout, without touching held-out images.
    reference = min(stats["episodes"], key=lambda ep: sum(abs(ep["samples"][0][r][k] - target[r][k]) for r in ("obj1", "obj2") for k in ("cx", "cy", "width", "height")))
    path = Path(reference["samples"][0]["frame"]) / "rgb_WoArm_WArmObjKpts.png"
    real = cv2.resize(cv2.imread(str(path)), (320, 240))
    cv2.putText(real, "Training reference", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    comparison = np.vstack([np.hstack([real, *panels[:2]]), np.hstack(panels[2:])])
    cv2.imwrite(str(root / "comparison.png"), comparison)
    write_json(root / "visual_metrics.json", {"reference": str(path), "metrics": metrics})
    return metrics


def report(root: Path, n: int) -> None:
    groups = {label: [json.loads(p.read_text()) for p in sorted((root / "episodes" / label).glob("*/result.json"))
                      if int(p.parent.name.removeprefix("ep")) < n] for label in "ABCDE"}
    visual = json.loads((root / "visual_metrics.json").read_text())["metrics"]
    summary = {}
    for label, rows in groups.items():
        valid = [row for row in rows if "error" not in row]
        k = sum(row["success"] for row in valid)
        summary[label] = {"completed": len(rows), "valid": len(valid), "errors": len(rows) - len(valid),
                          "successes": k, "success_rate": k / max(len(valid), 1), "wilson95": wilson(k, len(valid)),
                          "grasp_flag": sum(row["per_side"]["right"]["grasped"] for row in valid),
                          "lifted": sum(row["physical_lift"] for row in valid),
                          "transported": sum(row["physical_transport"] for row in valid),
                          "inside": sum(row["per_side"]["right"]["inside_container"] for row in valid),
                          "mean_distance": float(np.mean([row["per_side"]["right"]["dist_to_container_center"] for row in valid])) if valid else None,
                          "visual": visual[label]}
    comparisons = {}
    for left, right in (("B", "A"), ("C", "A"), ("D", "A"), ("D", "B"), ("D", "C"), ("E", "D")):
        # Execution errors are reported separately and excluded pairwise.
        a = [row for row in groups[left] if "error" not in row]
        b = [row for row in groups[right] if "error" not in row]
        comparisons[f"{left}-{right}"] = paired(a, b)
    # Holm correction across the six pre-specified comparisons.
    ordered = sorted(comparisons, key=lambda key: comparisons[key].get("mcnemar_p", 1))
    previous = 0.0
    for rank, key in enumerate(ordered):
        adjusted = max(previous, min(1.0, comparisons[key].get("mcnemar_p", 1) * (len(ordered) - rank)))
        comparisons[key]["holm_p"] = adjusted
        previous = adjusted
    write_json(root / "summary.json", {"planned_per_group": n, "groups": summary, "paired": comparisons})
    lines = ["# pick_cube Real2Sim A–E 消融实验", "", f"每组预先编号的前 {n} 个场景；超过该范围的试验不进入统计。", "",
             "固定 checkpoint、控制器、推理步数和每个配对场景的推理噪声。",
             "A–D 共用位置 ±1.5 cm、yaw ±8° 的标准化扰动；几何组使用各自拟合的名义布局，因此绝对坐标并不相同。",
             "E 在 D 上额外改变相机高度 ±2 cm、俯角 ±3°、光强 ±8%、色调 ±0.025；实体尺寸、质量和摩擦保持固定。", "",
             "| 组 | 完成/计划 | 执行错误 | 成功（95% Wilson CI） | 抬起 | 搬运 | 最终在盒内 | 输入几何 MAE | 物体 RGB MAE |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for label, value in summary.items():
        lo, hi = value["wilson95"]
        lines.append(f"| {label} | {value['completed']}/{n} | {value['errors']} | {value['successes']}/{value['valid']} ({100*value['success_rate']:.1f}%, {100*lo:.1f}–{100*hi:.1f}%) | {value['lifted']} | {value['transported']} | {value['inside']} | {value['visual']['geometry_mae']:.4f} | {value['visual']['object_rgb_mae']:.1f} |")
    lines += ["", "![训练参考与 A–E 模型输入](comparison.png)", "", "| 配对比较 | 成功率差（百分点） | 配对 bootstrap 95% CI | McNemar p | Holm p |", "|---|---:|---|---:|---:|"]
    for name, value in comparisons.items():
        if not value.get("n_scenes"):
            continue
        lo, hi = value["boot95"]
        lines.append(f"| {name} | {100*value['diff']:+.1f} | {100*lo:+.1f}–{100*hi:+.1f} | {value['mcnemar_p']:.4g} | {value['holm_p']:.4g} |")
    lines += ["", "## 解释与限制", "", "- 尺寸来自过滤后的训练重建，不是尺量真值；桌高以红块最低轮廓近似，仍有系统误差。",
              "- 视觉指标来自名义初始场景，比较训练集逐 episode 中位数；它们是输入接近程度的诊断量，不是策略效果的替代。",
              "- 严格成功要求抓取标志触发、方块完全在实际内腔中、释放且静止；圆角锥壁组另检查真实内腔边界。",
              "- 抬起为相对初始高度增加超过 2.5 cm；搬运为抬起且水平移动超过 5 cm。",
              "- A–D 含尺寸、构图和接触形状的几何因素；C 的效果不能单独归因于相机。E 衡量额外扰动的稳健性，不代表重新训练。",
              "- 当前仅在本地可用训练 episode 上拟合；该实验是固定场景分布的仿真比较，不能直接推断真实部署成功率。",
              "- 无论结果如何，不修改控制器或按结果重新选择场景。原始配置保留在 baseline/，每次试验保存 episode.json 和 result.json。", ""]
    diagnostics = grasp_diagnostics(root, n)
    if diagnostics:
        threshold = next(iter(diagnostics.values()))["threshold"]
        cutoff = np.log(threshold / (1 - threshold))
        lines += ["## 抓取解码诊断", "", "训练 dataloader 把 grasp 目标直接二值化为 0/1，flow matching 在该空间学习；当前推理对积分后的 grasp 分量施加 sigmoid。",
                  f"所以理想正例 1 会变成 0.7311，而冻结的抓取阈值是 {threshold:g}，相当于要求原始 flow 输出超过 {cutoff:.4f}。",
                  "当阈值高于 0.7311 时，理想正例也不能触发闭合；更低的阈值允许部分正例触发。所有组统一使用该轮冻结的解码和阈值。", "",
                  f"| 组 | 记录到的最大抓取输出 | 曾超过当前阈值的试验 | 仅逆变换后超过 {threshold:g} 的试验 |", "|---|---:|---:|---:|"]
        for label, value in diagnostics.items():
            lines.append(f"| {label} | {value['max_decoded_grasp']:.3f} | {value['current_gate_trials']}/{value['n']} | {value['raw_gate_trials']}/{value['n']} |")
        lines += ["", "最后一列仅对已记录的预测做逆 sigmoid 诊断，不是修复后的闭环成功率。应先统一修正训练/推理解码语义，再独立复测，不能把本轮失败全部归因于视觉域差异。", ""]
    if all(value["valid"] == n and value["successes"] == 0 for value in summary.values()):
        lines += ["## 本轮结论", "", "五组均未观察到严格任务成功。外观和构图指标改善尚未转化为本轮闭环成功。",
                  f"每组只有 {n} 次且没有成功事件，无法据此确定组间等效或场景改造无效；零差值的配对 bootstrap 会退化为 [0,0]，不应解释为真实效果没有不确定性。", ""]
    (root / "report.md").write_text("\n".join(lines))


def grasp_diagnostics(root: Path, n: int) -> dict:
    diagnostics = {}
    path = root / "A.yaml"
    threshold = float(load_config(str(path))["control"]["grasp_threshold"]) if path.exists() else 0.75
    for label in "ABCDE":
        maxima = []
        for index in range(n):
            path = root / "episodes" / label / f"ep{index:03d}" / "episode.json"
            if not path.exists():
                continue
            episode = json.loads(path.read_text())
            values = [np.asarray(step["sides"]["right"]["grasp"]).max() for step in episode["policy"]]
            if values:
                maxima.append(float(max(values)))
        if maxima:
            values = np.asarray(maxima)
            raw = np.log(np.clip(values, 1e-6, 1 - 1e-6) / np.clip(1 - values, 1e-6, 1))
            diagnostics[label] = {"n": len(values), "max_decoded_grasp": float(values.max()),
                                  "threshold": threshold,
                                  "median_max_decoded_grasp": float(np.median(values)),
                                  "current_gate_trials": int((values > threshold).sum()),
                                  "raw_gate_trials": int((raw > threshold).sum())}
    if diagnostics:
        write_json(root / "grasp_diagnostics.json", diagnostics)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        report(args.output, args.n)
        return
    render_comparison(args.output)
    if args.render_only:
        return
    protocol = {"n_per_group": args.n, "seed": args.seed, "workers": args.workers, "intro": False,
                "config_sha256": {label: hashlib.sha256((args.output / f"{label}.yaml").read_bytes()).hexdigest() for label in "ABCDE"},
                "groups": {"A": "frozen current scene", "B": "appearance only", "C": "geometry, layout and camera only", "D": "B+C", "E": "D + bounded view/light variation"}}
    manifest = args.output / "protocol.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        fixed_keys = ("seed", "intro", "config_sha256", "groups")
        if any(previous[key] != protocol[key] for key in fixed_keys):
            raise ValueError("protocol/config changed; choose a new output directory")
        if previous != protocol:
            history = args.output / "protocol_history.json"
            revisions = json.loads(history.read_text()) if history.exists() else []
            revisions.append(previous)
            write_json(history, revisions)
    write_json(manifest, protocol)
    jobs = [(label, index, args.seed) for label in "ABCDE" for index in range(args.n)
            if not (args.output / "episodes" / label / f"ep{index:03d}" / "result.json").exists()]
    print(f"[ablation] remaining={len(jobs)} workers={args.workers}", flush=True)
    context = mp.get_context("spawn")
    started = time.time()
    with context.Pool(args.workers, initializer=init_worker, initargs=(str(args.output), args.device)) as pool:
        for completed, value in enumerate(pool.imap_unordered(run_trial, jobs), 1):
            if completed % 5 == 0 or "error" in value:
                print(f"[ablation] {completed}/{len(jobs)} {value['variant']}:{value['idx']} success={value.get('success')} error={value.get('error')} wall={time.time()-started:.1f}s", flush=True)
    report(args.output, args.n)
    print(f"[ablation] complete: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
