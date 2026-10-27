"""CPU-only checks for configuration readiness, input boundaries and imports."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import cv2
import yaml

from lerobot_policy_humanego.config import load_config, validate_runtime_config
from lerobot_policy_humanego.data_types import (
    ObjectState,
    Observation,
    RobotState,
    SceneState,
    StereoFrame,
)
from lerobot_policy_humanego.perception import Perception
from lerobot_policy_humanego.processor_humanego import (
    HumanEgoProcessor,
    validate_observation,
)


ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "cfg" / "inference" / "params.yaml"


class ScaffoldTests(unittest.TestCase):
    def test_paths_resolve_from_params_outside_repository_cwd(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                config = load_config(PARAMS)
            finally:
                os.chdir(previous)
        self.assertEqual(
            Path(config.policy["ckpt"]),
            ROOT / "runs/pick_cube/HumanEgo_anchor/latest.pt",
        )
        self.assertEqual(
            Path(config.camera["intrinsics_path"]), PARAMS.parent / "intrinsics.yaml"
        )
        self.assertEqual(
            Path(config.camera["extrinsics_path"]), PARAMS.parent / "extrinsics.yaml"
        )
        self.assertEqual(config.policy["num_inference_steps"], 20)

    def test_templates_report_missing_calibration(self):
        config = load_config(PARAMS)
        raw = yaml.safe_load(
            (PARAMS.parent / "stereo_raw_calibration.yaml").read_text()
        )
        self.assertAlmostEqual(
            config.extrinsics["T_left_from_right"][0][3],
            np.linalg.norm(raw["translation_left_to_right_m"]),
            places=12,
        )
        self.assertEqual(
            config.extrinsics["T_left_from_right"],
            [
                [1.0, 0.0, 0.0, 0.060039595259997],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        _, _, p1, p2, *_ = cv2.stereoRectify(
            np.asarray(raw["left"]["K"], dtype=np.float64),
            np.asarray(raw["left"]["distortion"], dtype=np.float64),
            np.asarray(raw["right"]["K"], dtype=np.float64),
            np.asarray(raw["right"]["distortion"], dtype=np.float64),
            (640, 480),
            np.asarray(raw["rotation_left_to_right"], dtype=np.float64),
            np.asarray(raw["translation_left_to_right_m"], dtype=np.float64),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        np.testing.assert_allclose(config.intrinsics["left"]["K"], p1[:, :3])
        np.testing.assert_allclose(config.intrinsics["right"]["K"], p2[:, :3])
        self.assertAlmostEqual(
            -p2[0, 3] / p2[0, 0], config.extrinsics["T_left_from_right"][0][3]
        )
        with self.assertRaises(ValueError) as error:
            validate_runtime_config(config)
        self.assertIn("extrinsics.T_EE_from_hand", str(error.exception))
        self.assertNotIn("intrinsics.left.K", str(error.exception))
        self.assertNotIn("intrinsics.right.K", str(error.exception))
        self.assertNotIn("extrinsics.T_left_from_right", str(error.exception))

    def make_complete_config(self, directory):
        directory = Path(directory)
        for source in PARAMS.parent.glob("*.yaml"):
            shutil.copyfile(source, directory / source.name)
        params_path = directory / "params.yaml"
        params = yaml.safe_load(params_path.read_text())
        params["policy"]["ckpt"] = "model/latest.pt"
        params["deployment"]["diagnostic_dir"] = "diagnostics"
        params_path.write_text(yaml.safe_dump(params))
        model = directory / "model"
        model.mkdir()
        # Only exercise file readiness; these are not loadable model artifacts.
        for name in ("latest.pt", "config.json", "dataset_stats.json"):
            (model / name).touch()
        config = load_config(params_path)
        K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        config.intrinsics["left"]["K"] = K
        config.intrinsics["right"]["K"] = K
        baseline = np.eye(4)
        baseline[0, 3] = 0.06
        config.extrinsics["T_left_from_right"] = baseline.tolist()
        config.extrinsics["T_EE_from_hand"] = np.eye(4).tolist()
        return config

    def test_ready_config_requires_checkpoint_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_complete_config(directory)
            validate_runtime_config(config)
            self.assertEqual(
                Path(config.deployment["diagnostic_dir"]),
                Path(directory) / "diagnostics",
            )
            (Path(directory) / "model/dataset_stats.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "dataset_stats.json"):
                validate_runtime_config(config)

    def test_export_refuses_nonempty_directory_before_loading_weights(self):
        from lerobot_policy_humanego.export import export_policy

        with tempfile.TemporaryDirectory() as directory:
            config = self.make_complete_config(directory)
            output = Path(directory) / "export"
            output.mkdir()
            (output / "keep.txt").write_text("owned result")
            with self.assertRaisesRegex(FileExistsError, "nonempty"):
                export_policy(config, output)
            self.assertEqual((output / "keep.txt").read_text(), "owned result")

    def test_runtime_checks_metric_units_and_stereo_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_complete_config(directory)
            config.extrinsics["translation_unit"] = "mm"
            with self.assertRaisesRegex(ValueError, "translation_unit"):
                validate_runtime_config(config)
            config.extrinsics["translation_unit"] = "m"
            config.extrinsics["T_left_from_right"] = np.eye(4).tolist()
            with self.assertRaisesRegex(ValueError, "baseline"):
                validate_runtime_config(config)

    def test_runtime_rejects_nonrigid_tool_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_complete_config(directory)
            config.extrinsics["T_EE_from_hand"][0][0] = 2.0
            with self.assertRaisesRegex(ValueError, "T_EE_from_hand.*rigid"):
                validate_runtime_config(config)

    def test_calibration_must_match_transmitted_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_complete_config(directory)
            path = Path(config.camera["intrinsics_path"])
            values = yaml.safe_load(path.read_text())
            values["image_size_hw"] = [1200, 1600]
            path.write_text(yaml.safe_dump(values))
            with self.assertRaisesRegex(ValueError, "transmitted image size"):
                load_config(config.source)

    def test_malformed_calibration_names_the_invalid_field(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_complete_config(directory)
            config.intrinsics["left"]["K"] = [["unknown"]]
            with self.assertRaisesRegex(ValueError, "intrinsics.left.K.*numeric"):
                validate_runtime_config(config)
            path = Path(config.camera["intrinsics_path"])
            values = yaml.safe_load(path.read_text())
            values["left"] = None
            path.write_text(yaml.safe_dump(values))
            with self.assertRaisesRegex(ValueError, "intrinsics.left.*mapping"):
                load_config(config.source)

    def test_lightweight_imports_do_not_import_model_or_hardware_runtimes(self):
        code = """
import importlib, sys
sys.path.insert(0, sys.argv[1])
for name in ('config', 'data_types', 'stereo', 'perception', 'rendering',
             'export'):
    importlib.import_module('inference.' + name)
for name in ('torch', 'transformers', 'onnxruntime', 'pyrealsense2', 'lerobot'):
    assert name not in sys.modules, name
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", code, str(ROOT)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_object_state_preserves_positional_constructor_and_independent_defaults(
        self,
    ):
        first, second = ObjectState(np.eye(4)), ObjectState(np.eye(4))
        self.assertEqual(first.kpts_local.shape, (0, 3))
        self.assertEqual(first.kpts_local.dtype, np.float32)
        self.assertIsNot(first.kpts_local, second.kpts_local)
        points = np.ones((3, 3))
        self.assertIs(ObjectState(np.eye(4), points).kpts_local, points)

    def test_input_validation_and_reset_preserve_perception_owner(self):
        config = load_config(PARAMS)
        from unittest.mock import Mock

        perception = Mock(spec=Perception)
        processor = HumanEgoProcessor(config, perception)
        image = np.zeros((480, 640, 3), np.uint8)
        observation = Observation(StereoFrame(image, image), RobotState(np.eye(4), 0.0))
        validate_observation(observation)
        processor.scene = SceneState({}, ObjectState(np.eye(4)), np.eye(4), 0.0)
        processor.reset()
        self.assertIsNone(processor.scene)
        self.assertIs(processor.perception, perception)
        bad = Observation(StereoFrame(image[:240], image), observation.robot)
        with self.assertRaisesRegex(ValueError, "left image"):
            validate_observation(bad)
        with self.assertRaisesRegex(ValueError, "gripper"):
            validate_observation(
                Observation(observation.frame, RobotState(np.eye(4), 2.0))
            )


if __name__ == "__main__":
    unittest.main()
