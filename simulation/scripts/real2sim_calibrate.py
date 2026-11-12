"""Measure available training episodes and prepare isolated pick_cube A–E configs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanego_sim.env import ServeBreadSim, deep_update, load_config


def region_stats(image: np.ndarray, mask: np.ndarray) -> dict:
    n, labels, stats, centers = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8))
    if n < 2:
        raise ValueError("empty object mask")
    index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h, area = stats[index]
    height, width = mask.shape
    core = cv2.erode((labels == index).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return {"width": float(w / width), "height": float(h / height),
            "cx": float(centers[index, 0] / width), "cy": float(centers[index, 1] / height),
            "area": float(area / (width * height)),
            "rgb": np.median(image[core][:, ::-1], axis=0).tolist()}


def measure_episode(path: Path) -> dict:
    frames = sorted((path / "preprocess/all_data").glob("*/training_data.json"))
    tri = path / "preprocess/camtriangulator_results.json"
    if not frames or not tri.exists():
        raise ValueError("missing training frames or triangulation")
    samples = []
    for index in sorted(set([0, len(frames) // 2, len(frames) - 1])):
        frame = frames[index].parent
        image = cv2.imread(str(frame / "rgb.png"))
        masks = {role: cv2.imread(str(frame / f"mask_{role}.png"), 0) for role in ("obj1", "obj2")}
        if image is None or any(mask is None or not mask.any() for mask in masks.values()):
            continue
        sample = {role: region_stats(image, mask) for role, mask in masks.items()}
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        free = (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 90)
        free[:image.shape[0] // 3] = False
        free[:, int(image.shape[1] * 0.8):] = False
        for mask in masks.values():
            free[cv2.dilate(mask, np.ones((15, 15), np.uint8)) > 0] = False
        arm = cv2.imread(str(frame / "mask_arm.png"), 0)
        if arm is not None:
            free[arm > 0] = False
        sample["table_rgb"] = np.median(image[free][:, ::-1], axis=0).tolist()
        sample["frame"] = str(frame)
        sample["fraction"] = index / max(len(frames) - 1, 1)
        samples.append(sample)
    if not samples or samples[0]["fraction"] != 0:
        raise ValueError("missing first-frame object masks")
    data = json.loads(frames[0].read_text())
    reconstruction = json.loads(tri.read_text())
    T = np.asarray(data["metadata"]["c2w"])
    forward = T[:3, 2].copy()
    forward[2] = 0
    forward /= np.linalg.norm(forward)
    left = np.cross([0, 0, 1], forward)
    clouds = {key: np.asarray(value["points_3d_world"])
              for key, value in reconstruction["objects"].items()}
    table_z = np.percentile(clouds["obj2"][:, 2], 5)
    geometry = {"height": float(T[2, 3] - table_z),
                "pitch": float(np.rad2deg(np.arcsin(-T[2, 2]))),
                "fx": data["metadata"]["k"][0][0] / data["metadata"]["w"],
                "fy": data["metadata"]["k"][1][1] / data["metadata"]["h"]}
    for role, points in clouds.items():
        centered = points[:, :2] - points[:, :2].mean(0)
        values, vectors = np.linalg.eigh(centered.T @ centered)
        extent = np.ptp(centered @ vectors, axis=0)
        delta = points.mean(0) - T[:3, 3]
        axis = np.r_[vectors[:, -1], 0]
        yaw = np.rad2deg(np.arctan2(axis @ left, axis @ forward)) % 180
        geometry[role] = {"x": float(delta @ forward), "y": float(delta @ left),
                          "long": float(max(extent)), "short": float(min(extent)),
                          "height": float(np.ptp(points[:, 2])), "yaw": float(yaw)}
    cube = geometry["obj2"]
    valid = (0.35 < geometry["height"] < 0.75 and 30 < geometry["pitch"] < 65
             and 0.02 < cube["long"] < 0.06 and 0.015 < cube["short"] < 0.05
             and 0.015 < cube["height"] < 0.06 and 0.18 < geometry["obj1"]["long"] < 0.30
             and 0.06 < geometry["obj1"]["height"] < 0.15)
    return {"episode": path.name, "geometry": geometry, "geometry_valid": valid, "samples": samples}


def prepare(output: Path, checkpoint: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("evaluation has started; calibrate into a new directory")
    if not (output / "A.yaml").exists():
        (output / "A.yaml").write_text(yaml.safe_dump(load_config("simulation/configs/pick_cube.yaml"), sort_keys=False))
    model = json.loads((checkpoint.parent / "config.json").read_text())
    episodes, excluded = [], []
    for item in model["MPS_PATHS_TRAIN"]:
        try:
            episodes.append(measure_episode(Path(item)))
        except ValueError as error:
            excluded.append({"episode": item, "reason": str(error)})
    valid = [episode for episode in episodes if episode["geometry_valid"]]
    if len(valid) < 5:
        raise ValueError(f"only {len(valid)} reliable reconstructions; need at least five")
    def median(getter):
        return np.median([getter(ep) for ep in valid], axis=0)
    geometry = {key: float(median(lambda ep: ep["geometry"][key])) for key in ("height", "pitch", "fx", "fy")}
    geometry["container_size"] = [float(median(lambda ep: ep["geometry"]["obj1"][axis]))
                                  for axis in ("short", "long", "height")]
    # Silhouette points are not complete box vertices; use one robust common cube side.
    side = float(np.median([ep["geometry"]["obj2"][k] for ep in valid for k in ("long", "short", "height")]))
    geometry["cube_size"] = [side] * 3
    for role, name in (("obj1", "container"), ("obj2", "cube")):
        geometry[name + "_pose"] = [float(median(lambda ep: ep["geometry"][role][axis])) for axis in ("x", "y")]
        # Container long side follows world Y; the symmetric cube starts without yaw.
        geometry[name + "_pose"].append(0.0)
    targets = {}
    for role in ("obj1", "obj2"):
        targets[role] = {key: np.median([ep["samples"][0][role][key] for ep in episodes], axis=0).tolist()
                         for key in ("width", "height", "cx", "cy", "rgb")}
    targets["table_rgb"] = np.median([ep["samples"][0]["table_rgb"] for ep in episodes], axis=0).tolist()
    stats = {"configured_train": len(model["MPS_PATHS_TRAIN"]), "available": len(episodes),
             "valid_geometry": len(valid), "excluded": excluded, "episodes": episodes,
             "geometry_fit": geometry, "image_targets": targets,
             "method": "Training-only per-episode medians. First/middle/last available training frames; "
                       "initial frame geometry. Reject implausible cube extents, table heights and container extents. "
                       "Sizes are triangulation estimates, not tape measurements; cube lowest contour approximates table plane."}
    (output / "training_statistics.json").write_text(json.dumps(stats, indent=2))
    baseline = load_config(str(output / "A.yaml"))
    appearance = {"appearance": {"keypoint_resolution": [1232, 1624],
                                "table": {"rgba": [0.75, 0.72, 0.65, 1], "specular": 0.04},
                                "lighting": {"ambient": [0.55, 0.55, 0.55], "key": [0.25, 0.25, 0.25],
                                             "fill": [0.08, 0.08, 0.08], "direction": [0.5, -0.5, -1]},
                                "wall": {"x": 0.95, "rgba": [0.66, 0.64, 0.59, 1]},
                                "markers": [[0.17, 0.22], [0.15, 0], [0.17, -0.20]]},
                  "scene": {"objects": {"container": {"rgba": [0.36, 0.53, 0.66, 1], "specular": 0.17, "shininess": 0.25},
                                          "cube": {"rgba": [0.60, 0.10, 0.09, 1], "specular": 0.03, "shininess": 0.1}}}}
    shape = {"camera_intrinsics": {"fx": geometry["fx"] * 640, "fy": geometry["fy"] * 480,
                                    "cx": 319.5, "cy": 239.5},
             "rig": {"camera_pose": {"position": [0, 0, geometry["height"]], "pitch_deg": geometry["pitch"]}},
             "scene": {"objects": {"container": {"size": geometry["container_size"], "pose": geometry["container_pose"],
                                                    "shape": "rounded", "radius": 0.018, "taper": 0.84, "rim": 0.005},
                                     "cube": {"size": geometry["cube_size"], "pose": geometry["cube_pose"]}}}}
    # Fit rendered region medians. Images include illumination, so source RGB is not albedo.
    fitted = deep_update(baseline, appearance)
    for _ in range(3):
        env = ServeBreadSim(fitted)
        env.reset()
        image, masks = env.render_rgb(hide_robot=True), env.render_segmentation()
        for role, name in (("obj1", "container"), ("obj2", "cube")):
            rendered = np.asarray(region_stats(image, masks[name])["rgb"])
            color = np.asarray(fitted["scene"]["objects"][name]["rgba"][:3])
            fitted["scene"]["objects"][name]["rgba"] = np.clip(color * np.asarray(targets[role]["rgb"]) / np.maximum(rendered, 1), 0.01, 0.98).tolist() + [1.0]
        # Use a clear patch on the tabletop, avoiding props, marks, and background.
        rendered = np.median(image[350:410, 250:390, ::-1].reshape(-1, 3), axis=0)
        color = np.asarray(fitted["appearance"]["table"]["rgba"][:3])
        fitted["appearance"]["table"]["rgba"] = np.clip(color * np.asarray(targets["table_rgb"]) / np.maximum(rendered, 1), 0.01, 0.98).tolist() + [1.0]
        env.renderer.close()
    appearance["appearance"] = fitted["appearance"]
    for name in ("container", "cube"):
        appearance["scene"]["objects"][name]["rgba"] = fitted["scene"]["objects"][name]["rgba"]
    configs = {"A": baseline, "B": deep_update(baseline, appearance), "C": deep_update(baseline, shape),
               "D": deep_update(deep_update(baseline, shape), appearance)}
    configs["E"] = deep_update(configs["D"], {"ablation_randomization": True})
    for name, config in configs.items():
        (output / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "fitted_overrides.json").write_text(json.dumps({"appearance": appearance, "geometry": shape}, indent=2))
    print(json.dumps({"available": len(episodes), "valid_geometry": len(valid), "geometry": geometry, "targets": targets}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/pick_cube/server_0.1/latest.pt"))
    args = parser.parse_args()
    prepare(args.output, args.checkpoint)
