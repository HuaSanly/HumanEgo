# -*- coding: utf-8 -*-
"""
Named config profiles, applied by `env.load_config` between the per-robot calibration and
explicit overrides:

    serve_bread.yaml  <  per_robot.<robot>  <  profile  <  overrides (--set, autotune candidates)

  production  the real rig's Control.yaml settings (serve_bread.yaml as is; the default, so
              nothing changes silently)
  tuned       the robot's shipped controller parameters from configs/tuned/<robot>.json
              (autotune + held-out validation) with the grasp skill they were tuned with on

Select one with `--profile tuned` on the scripts, or `{"profile": "tuned"}` in the overrides
dict (it then also reaches the evaluation workers).

    from humanego_sim.env import load_config
    cfg = load_config(None, {"robot": "ur5e", "profile": "tuned"})
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNED_DIR = os.path.join(SIM_ROOT, "configs", "tuned")

PROFILES = ("production", "tuned")
DEFAULT_PROFILE = "production"


def tuned_path(robot: str) -> str:
    return os.path.join(TUNED_DIR, f"{robot}.json")


def load_tuned(robot: str) -> dict:
    """configs/tuned/<robot>.json: {params, requires, held_out, held_out_2, autotune_run, ...}."""
    p = tuned_path(robot)
    if not os.path.exists(p):
        have = sorted(f[:-5] for f in os.listdir(TUNED_DIR) if f.endswith(".json"))
        raise FileNotFoundError(f"no tuned profile for robot '{robot}' ({p}); tuned robots: {have}")
    with open(p) as f:
        return json.load(f)


def dotted_to_nested(flat: Dict[str, object], base_cfg: Optional[dict] = None) -> dict:
    """{'control.alpha': 0.3, 'physics.bread_friction.0': 1.5} -> nested overrides.
    A trailing integer indexes a list, filled in from `base_cfg` (same rule as autotune.params)."""
    ov: dict = {}
    for k, v in flat.items():
        parts = k.split(".")
        if parts[-1].isdigit():
            if base_cfg is None:
                raise ValueError(f"list-element key {k} needs the base config")
            cur_base = base_cfg
            for p in parts[:-1]:
                cur_base = cur_base[p]
            lst = list(cur_base)
            lst[int(parts[-1])] = v
            parts, v = parts[:-1], lst
        cur = ov
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return ov


def profile_overrides(profile: Optional[str], robot: str, base_cfg: Optional[dict] = None) -> dict:
    """Nested overrides of a profile for one robot ({} for production)."""
    from .env import deep_update
    profile = profile or DEFAULT_PROFILE
    if profile not in PROFILES:
        raise ValueError(f"unknown profile '{profile}', choose from {PROFILES}")
    if profile == "production":
        return {}
    if (base_cfg or {}).get("task", "serve_bread") != "serve_bread":
        raise ValueError("The tuned profiles are for serve_bread; use --profile production for pick_cube "
                         "and override its inference parameters with --set")
    t = load_tuned(robot)
    ov = dotted_to_nested(t["params"], base_cfg)
    return deep_update(ov, dotted_to_nested(t.get("requires", {}), base_cfg))


def profile_info(profile: Optional[str], robot: str) -> dict:
    """Provenance for logs and the web manifest: where the numbers come from."""
    profile = profile or DEFAULT_PROFILE
    if profile == "production":
        return {"profile": "production", "source": "configs/serve_bread.yaml"}
    t = load_tuned(robot)
    out = {"profile": profile, "source": os.path.relpath(tuned_path(robot), SIM_ROOT)}
    for k in ("autotune_run", "chosen", "note", "held_out", "held_out_2", "final", "final_production"):
        if k in t:
            out[k] = t[k]
    return out


def add_profile_arg(ap, default: str = DEFAULT_PROFILE):
    ap.add_argument("--profile", default=default, choices=PROFILES,
                    help="production = real-rig Control.yaml; tuned = configs/tuned/<robot>.json + grasp skill")
    return ap
