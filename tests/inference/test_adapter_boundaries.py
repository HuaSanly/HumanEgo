"""Boundary failures use a stub; neural/ODE parity is checked by run_gpu_validation."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot_policy_humanego.data_types import (
    ObjectState,
    PreparedObservation,
    SceneState,
    StereoFrame,
)
from lerobot_policy_humanego.modeling_humanego import HumanEgoPolicy


def adapter_input():
    adapter = HumanEgoPolicy.__new__(HumanEgoPolicy)
    torch.nn.Module.__init__(adapter)
    original = Mock()
    original.device = "cpu"
    original.build_ict.return_value = (None, None)
    original.decode_ee_in_cam.return_value = np.eye(4, dtype=np.float32)
    pos = np.zeros((50, 3), np.float32)
    rot = np.tile([1, 0, 0, 1, 0, 0], (50, 1)).astype(np.float32)
    grasp = np.full((50, 1), 0.64321, np.float32)
    original.infer.return_value = ({"right": (pos, rot, grasp)}, 0.2)
    adapter.policy = original
    adapter.T_align = np.eye(4)
    image = np.zeros((480, 640, 3), np.uint8)
    obj = ObjectState(np.eye(4))
    scene = SceneState({"obj1": obj, "obj2": obj}, obj, np.eye(4), 0.0)
    prepared = PreparedObservation(StereoFrame(image, image), scene, image, np.eye(3))
    return adapter, prepared, grasp


@pytest.mark.parametrize("kind", ["grasp", "done"])
def test_nonfinite_predictions_fail(kind):
    adapter, prepared, grasp = adapter_input()
    if kind == "grasp":
        grasp[0] = np.nan
    else:
        prediction, _ = adapter.policy.infer.return_value
        adapter.policy.infer.return_value = (prediction, np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        adapter.predict_action_chunk(prepared)


def test_continuous_grasp_is_not_thresholded():
    adapter, prepared, grasp = adapter_input()
    action = adapter.predict_action_chunk(prepared)
    np.testing.assert_array_equal(action[0, :, -1].numpy(), grasp[:, 0])
    assert action.shape == (1, 50, 10)
    assert (
        adapter.policy.build_ict.call_args.kwargs["reference_anchor"]
        is prepared.scene.reference_anchor
    )
    assert adapter.policy.decode_ee_in_cam.call_count == 50
