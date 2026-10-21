# -*- coding: utf-8 -*-
"""Render an MP4 from a recorded episode (third-person view + the policy's input image)."""
from __future__ import annotations

import json
import os

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np


def _put(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_episode_video(env, log_dir: str, out_path: str = None, camera: str = "front_view",
                         size=(960, 720), fps: int = 30):
    with open(os.path.join(log_dir, "episode.json")) as f:
        ep = json.load(f)
    frames, plog = ep["frames"], ep["policy"]
    out_path = out_path or os.path.join(log_dir, "episode.mp4")
    W, H = size
    r = mujoco.Renderer(env.m, H, W)
    pol_times = np.array([p["t"] for p in plog]) if plog else np.zeros(0)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    m, d = env.m, env.d
    last_imgs = {}
    for fr in frames:
        if "qfull" in fr:
            d.qpos[:] = fr["qfull"]
        else:
            for s, q in fr["qpos"].items():
                arm = env.arms[s]
                d.qpos[arm.qadr] = q[:len(arm.qadr)]
                d.qpos[arm.g_qadr] = q[len(arm.qadr)]
            for name, pq in fr["objects"].items():
                env.place_object(name, pq)
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=camera)
        main = np.ascontiguousarray(r.render()[:, :, ::-1])
        # latest policy images at this time
        k = int(np.searchsorted(pol_times, fr["t"], side="right")) - 1
        if k >= 0:
            for s, e in plog[k]["sides"].items():
                if "image" in e:
                    p = os.path.join(log_dir, "policy_view", e["image"])
                    if os.path.exists(p):
                        last_imgs[s] = cv2.imread(p)
        tw = W // 3
        th = int(tw * 3 / 4)
        y = 8
        for s in ("right", "left"):
            if s in last_imgs:
                im = cv2.resize(last_imgs[s], (tw, th))
                x = W - tw - 8
                main[y:y + th, x:x + tw] = im
                cv2.rectangle(main, (x, y), (x + tw, y + th), (255, 255, 255), 1)
                _put(main, f"policy input ({s}{', mirrored' if s == 'left' else ''})", (x + 6, y + 18), 0.45)
                y += th + 6
        _put(main, f"t = {fr['t']:5.1f} s", (12, 28), 0.7)
        caption = (f"HumanEgo {ep['meta']['task']} scene preview" if ep["meta"].get("scene_only")
                   else "HumanEgo policy in simulation (released serve_bread checkpoint)")
        _put(main, caption, (12, H - 14), 0.5)
        writer.append_data(main[:, :, ::-1])
    writer.close()
    r.close()
    return out_path
