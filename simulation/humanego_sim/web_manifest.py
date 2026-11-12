# -*- coding: utf-8 -*-
"""
The web viewer's data/manifest.json: config list, results table, notes and tuning plot.

Every number on the page comes from data, never from hand-typed text:
  results table   success of the build's own rollouts (build_web.py evaluations), per config
  held-out        configs/tuned/<robot>.json (autotune_validate.py on seeds 7777 / 8888)
  tuning plot     autotune_runs/<run>/iterations + final.json of the run behind the Trossen profile
validate() is the schema the viewer (web/js/app.js) relies on; tests/test_web_build.py runs it.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

SCHEMA = 2
SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_TEXT = {
    "production": "the real rig's controller settings (Control.yaml)",
    "tuned": "its self-tuned controller with the grasp skill",
}


def pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def side_rate(rollouts: List[dict], side: str) -> Optional[float]:
    v = [float(r["per_side"][side]["success"]) for r in rollouts if side in r.get("per_side", {})]
    return sum(v) / len(v) if v else None


def summary_rows(configs: List[dict]) -> List[dict]:
    rows = []
    for c in configs:
        if c.get("scene_only"):
            continue
        rs = c["rollouts"]
        prof = c.get("profile") or {"profile": "production"}
        row = {"robot": c["robot"], "camera": c["camera"], "mode": c["mode"], "n": len(rs),
               "right": side_rate(rs, "right"), "left": side_rate(rs, "left"),
               "both": (sum(float(r["success"]) for r in rs) / len(rs)) if rs else None,
               "profile": prof.get("profile", "production")}
        if c.get("task") == "pick_cube":
            row["task"] = "pick_cube"
        if prof.get("autotune_run"):
            row["autotune_run"] = prof["autotune_run"]
        ho = [prof[k] for k in ("final", "held_out", "held_out_2") if k in prof]
        if ho:
            row["held_out"] = [{"seed": h["seed"], "n_scenes": h["n_scenes"], "per_arm": h["per_arm"],
                                "right": h["per_side"].get("right"), "left": h["per_side"].get("left"),
                                "both": h["both"]} for h in ho]
        rows.append(row)
    return rows


def results_note(configs: List[dict], robot_labels: Dict[str, str]) -> str:
    """Protocol + where the numbers come from, generated from the configs' profiles."""
    by_robot = {}
    for c in configs:
        if c.get("scene_only") or c.get("task") == "pick_cube":
            continue
        by_robot.setdefault(c["robot"], c.get("profile") or {"profile": "production"})
    if not by_robot:
        if any(c.get("task") == "pick_cube" and not c.get("scene_only") for c in configs):
            return "pick_cube: local trained policy rollouts. Success means the grasped block was released " \
                   "and is resting fully inside the container. These are individual runs, not a benchmark."
        return "Scene previews only. No policy was run and no task success rate was evaluated."
    parts = []
    for robot, prof in by_robot.items():
        name = robot_labels.get(robot, robot)
        p = prof.get("profile", "production")
        txt = f"<b>{name}</b>: {PROFILE_TEXT.get(p, p)}"
        ho = [prof[k] for k in ("final", "held_out", "held_out_2") if k in prof]
        if ho:
            txt += (f" (<code>{prof.get('autotune_run', '?')}</code>; held-out per arm "
                    + " / ".join(f"<b>{pct(h['per_arm'])}</b> on {h['n_scenes']} scenes, seed {h['seed']}" for h in ho)
                    + ")")
        parts.append(txt)
    return ("Every rollout is a closed-loop run of the released checkpoint from a randomized initial condition "
            "(bread–plate offset, depth and bread yaw as in the paper's protocol), with <b>soft bread</b> held by "
            "friction alone. <b>Two arms</b>: one single-hand model drives both arms; the left arm sees a mirrored "
            "image, exactly as on the real robot. <b>Served</b> = released and resting on the plate (not touching "
            "the table). Controllers: " + "; ".join(parts) + ". The table counts the rollouts on this page; the "
            "held-out numbers come from <code>simulation/configs/tuned/&lt;robot&gt;.json</code>.")


def tuning_section(tuned: Optional[dict], runs_dir: Optional[str] = None) -> Optional[dict]:
    """Tuning plot of the autotune run behind a tuned profile (local autotune_runs/), with a note from
    the tuned file's held-out numbers. None without the run. (The run's own baseline trial is not
    quoted: older final.json files do not record which fixed overrides it ran with.)"""
    if not tuned or not tuned.get("autotune_run"):
        return None
    runs_dir = runs_dir or os.path.join(SIM_ROOT, "autotune_runs")
    d = os.path.join(runs_dir, tuned["autotune_run"])
    its = sorted(glob.glob(os.path.join(d, "iterations", "iter_*.json")))
    if not its:
        return None
    batches = []
    for p in its:
        it = json.load(open(p))
        batches.append({"scores": [round(c["stats"]["per_side_success_rate"], 3) for c in it["candidates"]],
                        "action": it.get("decision", {}).get("action")})
    h = tuned.get("final") or tuned["held_out"]      # the reserved test set when it has been run
    note = (f"Grey dots: candidates in each batch (per-arm success on shared random scenes); orange: best so far. "
            f"On {h['n_scenes']} held-out scenes (seed {h['seed']}) the <code>{tuned['autotune_run']}</code> controller "
            f"with its <b>grasp skill</b> serves <b>{pct(h['per_arm'])}</b> per arm (right {pct(h['per_side'].get('right'))}, "
            f"left {pct(h['per_side'].get('left'))}), both breads {pct(h['both'])}")
    if "final" in tuned and "final_production" in tuned:
        note += (f"; the production controller settings serve {pct(tuned['final_production']['per_arm'])} per arm "
                 f"on the same scenes")
    elif "held_out_2" in tuned:
        h2 = tuned["held_out_2"]
        note += f"; {pct(h2['per_arm'])} per arm on a second set (seed {h2['seed']})"
    return {"run": tuned["autotune_run"], "robot": tuned.get("robot"), "batches": batches, "note": note + "."}


def assemble(configs: List[dict], robot_labels: Dict[str, str], camera_labels: Dict[str, str],
             default: dict, tuning: Optional[dict], extra: Optional[dict] = None) -> dict:
    for c in configs:
        for r in c["rollouts"]:
            r.pop("featured", None)
            r["scene"] = {k: [round(float(v), 4) for v in p] for k, p in r["scene"].items()}
        good = [r for r in c["rollouts"] if r["success"]]
        if good:
            good[0]["featured"] = True
    man = {"schema": SCHEMA, "robots": robot_labels, "cameras": camera_labels, "configs": configs,
           "default": default, "summary": summary_rows(configs), "summary_note": results_note(configs, robot_labels),
           "tuning": tuning}
    man.update(extra or {})
    return man


def validate(man: dict, web: Optional[str] = None) -> List[str]:
    """Schema of the manifest the viewer reads; with `web`, also that every referenced file exists."""
    err = []

    def need(cond, msg):
        if not cond:
            err.append(msg)
    for k in ("robots", "cameras", "configs", "default", "summary", "summary_note"):
        need(k in man, f"missing {k}")
    if err:
        return err
    need(isinstance(man["configs"], list) and man["configs"], "configs must be a non-empty list")
    d = man["default"]
    need(any(c["robot"] == d.get("robot") and c["camera"] == d.get("camera") and c["mode"] == d.get("mode")
             and c.get("task", "serve_bread") == d.get("task", "serve_bread")
             for c in man["configs"]), "default config not in configs")
    ids = set()
    for c in man["configs"]:
        tag = f"{c.get('robot')}/{c.get('camera')}/{c.get('mode')}"
        for k in ("robot", "camera", "mode", "model", "rollouts", "live"):
            need(k in c, f"{tag}: missing {k}")
        need(c.get("robot") in man["robots"], f"{tag}: robot without a label")
        need(c.get("camera") in man["cameras"], f"{tag}: camera without a label")
        need(c.get("mode") in ("bimanual", "single"), f"{tag}: bad mode")
        sides = {"right"} if c.get("mode") == "single" else {"right", "left"}
        need(sum(1 for r in c.get("rollouts", []) if r.get("featured")) <= 1, f"{tag}: several featured rollouts")
        for r in c.get("rollouts", []):
            rid = r.get("id", "?")
            need(rid not in ids, f"duplicate rollout id {rid}")
            ids.add(rid)
            need(rid.startswith(c.get("model", "")), f"{rid}: id does not start with the model key")
            if c.get("scene_only"):
                need(r.get("success") is None and r.get("per_side") == {}, f"{rid}: preview must be unevaluated")
                names = list(c.get("objects", {}))
                need(bool(names), f"{rid}: preview objects missing")
            else:
                need(set(r.get("per_side", {})) == sides, f"{rid}: per_side {set(r.get('per_side', {}))} != {sides}")
                need(isinstance(r.get("success"), bool), f"{rid}: success must be a bool")
                need(r.get("success") == all(v["success"] for v in r.get("per_side", {}).values()),
                     f"{rid}: success != all sides served")
                names = (["container", "cube"] if c.get("task") == "pick_cube"
                         else ["plate"] + [f"bread_{s}" for s in sides])
            for name in names:
                need(len(r.get("scene", {}).get(name, [])) == 3, f"{rid}: scene.{name} must be [x, y, yaw]")
            need(r.get("fmt", "json") in ("json", "bin"), f"{rid}: bad fmt")
            if web:
                ext = ".bin" if r.get("fmt") == "bin" else ".json"
                need(os.path.exists(os.path.join(web, "data", "rollouts", rid + ext)), f"{rid}{ext} missing")
        if web and "model" in c:
            mp = os.path.join(web, "data", "models", c["model"] + ".json")
            need(os.path.exists(mp), f"{tag}: model {mp} missing")
            if os.path.exists(mp):
                model = json.load(open(mp))
                assets = {g["asset"] for g in model.get("geoms", []) if g.get("asset")}
                assets |= {s["asset"] for s in model.get("soft", {}).values()}
                assets |= {m["map"] for m in model.get("materials", {}).values() if m.get("map")}
                for a in sorted(assets):
                    need(os.path.exists(os.path.join(web, a)), f"{tag}: asset {a} missing")
    for row in man["summary"]:
        for k in ("right", "left", "both"):
            v = row.get(k)
            need(v is None or 0.0 <= v <= 1.0, f"summary {row.get('robot')}: {k}={v} out of [0, 1]")
        for h in row.get("held_out", []):
            need(0.0 <= h["per_arm"] <= 1.0 and h["n_scenes"] > 0, f"summary {row.get('robot')}: bad held_out")
    t = man.get("tuning")
    if t is not None:
        need(isinstance(t.get("batches"), list) and isinstance(t.get("note"), str), "tuning: batches + note")
        for b in t.get("batches", []):
            need(all(0.0 <= s <= 1.0 for s in b.get("scores", [])), "tuning: score out of [0, 1]")
    for k, v in (man.get("media") or {}).items():
        if web and isinstance(v, str):
            need(os.path.exists(os.path.join(web, v)), f"media {k}: {v} missing")
    return err


def readme_results() -> str:
    """The README's results table, from configs/tuned/*.json (tests/test_web_build.py keeps the README
    in sync: python -m humanego_sim.web_manifest prints it). Headline: the reserved test set (seeds
    9001+, never used to pick parameters), tuned vs production on the same scenes; then the dev seeds."""
    from .profiles import TUNED_DIR
    from .robots import ROBOTS
    rows = ["| Robot | Controller (`--profile tuned`) | Per arm, reserved test [95 % CI] | Right / left | Both served "
            "| Production controller, same scenes (per arm / both) | Dev seeds (per arm) |",
            "|---|---|---|---|---|---|---|"]
    for robot in ROBOTS:
        p = os.path.join(TUNED_DIR, f"{robot}.json")
        if not os.path.exists(p):
            continue
        t = json.load(open(p))
        dev = " / ".join(f"{pct(t[k]['per_arm'])} (seed {t[k]['seed']}, {t[k]['n_scenes']} scenes)"
                         for k in ("held_out", "held_out_2") if k in t)
        f, fp = t.get("final"), t.get("final_production")
        if f:
            ci = f.get("per_arm_wilson95")
            head = (f"**{pct(f['per_arm'])}**" + (f" [{pct(ci[0])}, {pct(ci[1])}]" if ci else "")
                    + f" ({f['n_scenes']} scenes, seed {f['seed']})")
            side = f"{pct(f['per_side'].get('right'))} / {pct(f['per_side'].get('left'))}"
            both = pct(f["both"])
            prod = f"{pct(fp['per_arm'])} / {pct(fp['both'])}" if fp else "—"
        else:
            h = t["held_out"]
            head, side, both, prod = "—", f"{pct(h['per_side'].get('right'))} / {pct(h['per_side'].get('left'))}", pct(h["both"]), "—"
        rows.append(f"| {ROBOTS[robot].label} | `{t['autotune_run']}` + grasp skill | {head} | {side} | {both} | {prod} | {dev} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(readme_results())
