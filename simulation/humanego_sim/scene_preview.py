# -*- coding: utf-8 -*-
"""Record a scene's physics while the existing robot servos hold their start pose."""
from __future__ import annotations

import json
import os


def record_scene(env, log_dir: str, scene: dict = None, duration: float = 5.0) -> dict:
    if duration <= 0:
        raise ValueError("Preview duration must be positive")
    env.reset(scene=scene)
    scene_init = env.default_scene()
    scene_init.update(scene or {})
    frames = [env.snapshot()]
    next_frame = 1 / 30
    while env.t < duration:
        env.step()
        if env.t >= next_frame:
            frames.append(env.snapshot())
            next_frame += 1 / 30
    result = {"success": None, "per_side": {}, "evaluated": False,
              "duration": env.t, "n_policy_steps": 0}
    meta = dict(env.meta, task=env.cfg["task"], scene_only=True, scene_init=scene_init, result=result)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "episode.json"), "w") as f:
        json.dump({"meta": meta, "frames": frames, "policy": []}, f)
    return result
