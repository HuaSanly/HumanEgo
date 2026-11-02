"""Training canvas and local -Y virtual gripper convention (BGR)."""

import cv2
import numpy as np

from .data_types import SceneState

TRAINING_RENDERING = dict(
    canvas_hw=[1232, 1624],
    width_open_m=0.18,
    width_closed_m=0.05,
    finger_length_m=0.08,
    root_depth_m=0.08,
    points_per_segment=6,
    hand_radius=4,
    object_radius=5,
    white_core_radius=1,
    open_colors=[[0, 255, 255], [0, 160, 255], [0, 60, 255], [0, 0, 255]],
    closed_colors=[[255, 255, 255], [255, 255, 255], [255, 0, 255], [255, 0, 255]],
    object_colors={"obj1": [255, 255, 0], "obj2": [0, 255, 0]},
)
CANVAS_HW = tuple(TRAINING_RENDERING["canvas_hw"])


def canvas_intrinsics(
    K: np.ndarray, source_hw=(480, 640), canvas_hw=CANVAS_HW
) -> np.ndarray:
    sy, sx = np.asarray(canvas_hw) / source_hw
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]]) @ K


def gripper_points(gripper: float, settings=None):
    opts = TRAINING_RENDERING if settings is None else settings
    closed = gripper > 0.5
    width = opts["width_closed_m"] if closed else opts["width_open_m"]
    finger, depth = opts["finger_length_m"], opts["root_depth_m"]
    tip_l, tip_r = np.array([-width / 2, 0, 0]), np.array([width / 2, 0, 0])
    base_l, base_r = tip_l + [0, -finger, 0], tip_r + [0, -finger, 0]
    root = np.array([0, -(finger + depth), 0])
    tip, base, palm, root_color = opts["closed_colors" if closed else "open_colors"]
    segments = [
        (root, base_l, root_color, palm),
        (root, base_r, root_color, palm),
        (base_l, tip_l, base, tip),
        (base_r, tip_r, base, tip),
    ]
    count = opts["points_per_segment"]
    return [
        (
            point,
            tuple(int(ca[j] + (cb[j] - ca[j]) * index / (count - 1)) for j in range(3)),
        )
        for a, b, ca, cb in segments
        for index, point in enumerate(np.linspace(a, b, count))
    ]


def _draw(image, points, pose, K, radius, colors, core):
    camera = np.asarray(points).reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
    stats = {
        "points_total": len(camera),
        "points_drawn": 0,
        "depth_clipped": 0,
        "outside_image": 0,
        "camera_xyz_min_m": camera.min(axis=0).tolist() if len(camera) else None,
        "camera_xyz_max_m": camera.max(axis=0).tolist() if len(camera) else None,
    }
    projected = []
    for p, color in zip(camera, colors, strict=True):
        if p[2] <= 0.1:
            stats["depth_clipped"] += 1
            continue
        uv = K @ p
        projected.append((uv[:2] / uv[2]).tolist())
        x, y = (uv[:2] / uv[2]).astype(int)
        if (
            -radius <= x < image.shape[1] + radius
            and -radius <= y < image.shape[0] + radius
        ):
            cv2.circle(
                image, (x, y), radius, tuple(int(c) for c in color), -1, cv2.LINE_AA
            )
            cv2.circle(image, (x, y), core, (255, 255, 255), -1, cv2.LINE_AA)
            stats["points_drawn"] += 1
        else:
            stats["outside_image"] += 1
    stats["uv_min_px"] = np.min(projected, axis=0).tolist() if projected else None
    stats["uv_max_px"] = np.max(projected, axis=0).tolist() if projected else None
    return stats


def render_scene(
    background_bgr: np.ndarray,
    scene: SceneState,
    K: np.ndarray,
    settings=None,
    *,
    diagnostics=None,
) -> np.ndarray:
    """K matches the 640x480 input; output uses the training reference canvas."""
    opts = TRAINING_RENDERING if settings is None else settings
    image = cv2.resize(background_bgr, tuple(opts["canvas_hw"][::-1]))
    K = canvas_intrinsics(K, canvas_hw=opts["canvas_hw"])
    points, colors = zip(*gripper_points(scene.gripper, opts))
    hand_stats = _draw(
        image,
        points,
        scene.hand_in_cam,
        K,
        opts["hand_radius"],
        colors,
        opts["white_core_radius"],
    )
    if diagnostics is not None:
        diagnostics["hand"] = hand_stats
    for key, obj in scene.objects.items():
        object_stats = _draw(
            image,
            obj.kpts_local,
            obj.T_in_cam,
            K,
            opts["object_radius"],
            [opts["object_colors"][key]] * len(obj.kpts_local),
            opts["white_core_radius"],
        )
        if diagnostics is not None:
            diagnostics[key] = object_stats
    return image
