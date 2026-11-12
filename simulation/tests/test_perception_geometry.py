# -*- coding: utf-8 -*-
"""Released perception (KptsSelector -> multi-frame depth lift -> MAD/DBSCAN -> PCA frame), the
mask split and the left-arm mirroring, on a synthetic top-down scene with known geometry.

Camera 0.62 m above the table looking straight down; two 4 cm tall breads (ellipses in the
image, long axis along image u) left and right of the image centre, a plate in the middle.
"""
import types

import cv2
import numpy as np
import pytest

from humanego_sim.cameras import get_camera
from humanego_sim.perception import SimPerception, postprocess_kpts_3d, split_mask_by_x

Z_TABLE, Z_BREAD, Z_PLATE = 0.62, 0.58, 0.61
# camera optical frame in the world: x_cam = -y_world, y_cam = -x_world, z_cam = -z_world
T_CAM_IN_WORLD = np.array([[0.0, -1.0, 0.0, 0.45], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, Z_TABLE],
                           [0.0, 0.0, 0.0, 1.0]])
BREADS = {"bread_right": ((450, 250), (70, 32)), "bread_left": ((190, 230), (66, 30))}   # centre (u, v), semi-axes
PLATE = ((320, 330), 70)


class FakeEnv:
    def __init__(self):
        cam = get_camera("realsense_d405")
        self.K, self.W, self.H = cam.K.copy(), cam.width, cam.height
        self.T_cam_in_world = T_CAM_IN_WORLD
        self.masks = {n: self._ellipse(c, ax) for n, (c, ax) in BREADS.items()}
        self.masks["plate"] = self._ellipse(PLATE[0], (PLATE[1], PLATE[1]))
        z = np.full((self.H, self.W), Z_TABLE, np.float32)
        z[self.masks["plate"] > 0] = Z_PLATE
        for n in BREADS:
            z[self.masks[n] > 0] = Z_BREAD
        self.depth = z
        self.rng = np.random.default_rng(0)
        # what SimPerception's robustness layer reads (all perturbations off by default)
        from humanego_sim.env import load_config
        self.cfg = load_config()
        self.obj_names = list(self.masks)
        self.distractors = []
        self.meta = {}
        import mujoco
        self.m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="policy_cam"/></worldbody></mujoco>')

    def _ellipse(self, c, ax):
        m = np.zeros((self.H, self.W), np.uint8)
        cv2.ellipse(m, c, ax, 0, 0, 360, 255, -1)
        return m

    def render_segmentation(self):
        return {k: v.copy() for k, v in self.masks.items()}

    def render_depth(self, noise=True, artifacts=None):
        return self.depth + (self.rng.normal(0, 4e-4, self.depth.shape).astype(np.float32) if noise else 0)

    def cam_point(self, uv, z):
        return np.array([(uv[0] - self.K[0, 2]) * z / self.K[0, 0], (uv[1] - self.K[1, 2]) * z / self.K[1, 1], z])


@pytest.fixture(scope="module")
def perceived():
    env = FakeEnv()
    per = SimPerception(env, n_init_frames=4, depth_noise=True)
    return env, per.estimate_objects(["right", "left"], shared_context=True)


def _main_axis(P):
    X = P - P.mean(0)
    return np.linalg.eigh(X.T @ X)[1][:, -1]


def test_split_mask_by_x():
    env = FakeEnv()
    both = env.masks["bread_right"] | env.masks["bread_left"]
    sp = split_mask_by_x(both, env.W // 2)
    assert (sp["right"] > 0).sum() == (env.masks["bread_right"] > 0).sum()
    assert (sp["left"] > 0).sum() == (env.masks["bread_left"] > 0).sum()
    # one blob across the split line is cut at it
    one = sp["right"].copy()
    cv2.rectangle(one, (300, 240), (460, 260), 255, -1)
    sp1 = split_mask_by_x(one, env.W // 2)
    assert sp1["left"][:, env.W // 2:].max() == 0 and sp1["right"][:, :env.W // 2].max() == 0


def test_postprocess_drops_outliers():
    rng = np.random.default_rng(0)
    pts = list(rng.normal([0.1, 0.0, 0.6], 0.01, (20, 3))) + [np.array([0.5, 0.5, 1.2]), None,
                                                              np.array([np.nan, 0, 0])]
    out = postprocess_kpts_3d(pts)
    assert 17 <= len(out) <= 20 and np.abs(out - [0.1, 0.0, 0.6]).max() < 0.05      # the far point is gone


def test_bread_keypoints_lie_on_the_bread(perceived):
    env, objs = perceived
    (u, v), (a, b) = BREADS["bread_right"]
    k3d = np.asarray(objs["right"].objs_meta["obj1"]["kpts_3d"])
    assert len(k3d) >= 15                                    # 20 contour points, a few may be filtered
    np.testing.assert_allclose(k3d[:, 2], Z_BREAD, atol=0.004)
    c = env.cam_point((u, v), Z_BREAD)
    np.testing.assert_allclose(k3d[:, :2].mean(0), c[:2], atol=0.006)
    # contour points: on the ellipse x^2/A^2 + y^2/B^2 = 1 (A, B in metres), within a few pixels
    A, B = a * Z_BREAD / env.K[0, 0], b * Z_BREAD / env.K[1, 1]
    r = np.sqrt(((k3d[:, 0] - c[0]) / A) ** 2 + ((k3d[:, 1] - c[1]) / B) ** 2)
    assert np.all(np.abs(r - 1.0) < 0.15)


def test_pca_frame_follows_the_long_axis(perceived):
    env, objs = perceived
    m = objs["right"].objs_meta["obj1"]
    T, k3d = np.asarray(m["T_obj_in_cam_static"]), np.asarray(m["kpts_3d"])
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6)
    ax = _main_axis(k3d)
    assert abs(ax @ [1.0, 0.0, 0.0]) > 0.98                  # image u == camera x
    assert np.max(np.abs(T[:3, :3].T @ ax)) > 0.98           # one frame axis is the long axis
    # local keypoints reproduce the camera-frame ones through the frame
    kl = np.asarray(m["kpts_local"])
    np.testing.assert_allclose((T[:3, :3] @ kl.T).T + T[:3, 3], k3d, atol=1e-9)


def test_left_arm_sees_the_mirrored_world(perceived):
    env, objs = perceived
    (u, v), _ = BREADS["bread_left"]
    k3d = np.asarray(objs["left"].objs_meta["obj1"]["kpts_3d"])
    c = env.cam_point((u, v), Z_BREAD)
    np.testing.assert_allclose(k3d[:, :2].mean(0), [-c[0], c[1]], atol=0.006)   # x flipped
    # the plate is shared context (not split): both arms see it, mirrored for the left one
    pr = np.asarray(objs["right"].objs_meta["obj2"]["kpts_3d"]).mean(0)
    pl = np.asarray(objs["left"].objs_meta["obj2"]["kpts_3d"]).mean(0)
    np.testing.assert_allclose(pl, pr * [-1, 1, 1], atol=0.004)
    np.testing.assert_allclose(pr[:2], env.cam_point(PLATE[0], Z_PLATE)[:2], atol=0.006)


def test_runner_bread_geometry_in_world(perceived):
    """runner._perceived_geometry: what the grasp skill knows (centre, long axis, extent, place height)."""
    runner = pytest.importorskip("humanego_sim.runner")
    env, objs = perceived
    fake = types.SimpleNamespace(env=env, cfg={}, anchor_key="obj1", task="serve_bread")
    for side, name in (("right", "bread_right"), ("left", "bread_left")):
        g = runner.EpisodeRunner._perceived_geometry(fake, side, objs[side])
        (u, v), (a, b) = BREADS[name]
        cw = (T_CAM_IN_WORLD @ np.r_[env.cam_point((u, v), Z_BREAD), 1.0])[:3]
        top = Z_TABLE - Z_BREAD
        br = g["bread"]
        np.testing.assert_allclose(br["c"][:2], cw[:2], atol=0.006)
        assert br["c"][2] == pytest.approx(0.5 * top, abs=0.003)     # half the top height
        # long axis: image u == camera x == world -y
        assert abs(np.dot(br["a"], [0.0, 1.0, 0.0])) > 0.98 and br["a"][2] == 0.0
        assert br["half_len"] == pytest.approx(a * Z_BREAD / env.K[0, 0], rel=0.15)
        assert br["half_wid"] == pytest.approx(b * Z_BREAD / env.K[1, 1], rel=0.25)
        # place height = bread centre height + plate floor (rim keypoints - 1.35 cm)
        assert g["place_z"] == pytest.approx(br["c"][2] + max((Z_TABLE - Z_PLATE) - 0.0135, 0.0), abs=0.003)
