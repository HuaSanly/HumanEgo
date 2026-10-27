"""Reload a relocated deployment with an installed wheel, outside the repository."""

import argparse
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import shutil


CHILD = r"""
import builtins,sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from lerobot_policy_humanego.modeling_humanego import HumanEgoPolicy
from lerobot_policy_humanego.data_types import ObjectState,SceneState,StereoFrame,PreparedObservation
from lerobot_policy_humanego.configuration_humanego import HumanEgoConfig
from lerobot_policy_humanego.processor_humanego import make_humanego_pre_post_processors_from_pretrained
import lerobot_policy_humanego,training.FlowMatchingModel
import numpy as np
import torch
assert Path(lerobot_policy_humanego.__file__).is_relative_to(sys.argv[1])
assert Path(training.FlowMatchingModel.__file__).is_relative_to(sys.argv[1])
root=Path(sys.argv[2]);forbidden=Path(sys.argv[3]).resolve()
original_open=builtins.open

def checked_open(file,*args,**kwargs):
    if isinstance(file,(str,bytes,Path)) and Path(file).resolve().is_relative_to(forbidden):
        raise AssertionError('Serving attempted to access original training tree: '+str(file))
    return original_open(file,*args,**kwargs)
builtins.open=checked_open
config=HumanEgoConfig.from_pretrained(root)
config.pretrained_path=root
policy=HumanEgoPolicy.from_pretrained(root,config=config)
policy.to('cpu').to('cuda').eval()
assert policy.policy.model is policy.model
assert policy.policy.noise is policy.fixed_noise
assert policy.fixed_noise.device == next(policy.model.parameters()).device
assert not policy.model.training
pre,post=make_humanego_pre_post_processors_from_pretrained(config,root,
    preprocessor_overrides={'device_processor':{'device':'cuda'},'rename_observations_processor':{'rename_map':{}}})
step=next(s for s in pre.steps if hasattr(s,'params_path'))
assert Path(step.params_path)==root/'assets/params.yaml'
# Reset must not construct GPU perception unnecessarily.
pre.reset();assert step.processor is None and not step.startup_probe_allowed
frame=StereoFrame(np.zeros((480,640,3),np.uint8),np.zeros((480,640,3),np.uint8))
T=np.eye(4,dtype=np.float32);T[:3,3]=[.1,.2,.7]
anchor=ObjectState(T.copy());hand=T.copy();hand[0,3]+=.12
scene=SceneState({'obj1':anchor,'obj2':ObjectState(hand.copy())},ObjectState(T.copy()),hand,0.)
prepared=PreparedObservation(frame,scene,np.zeros((1232,1624,3),np.uint8),np.array([[1050,0,812],[0,1050,616],[0,0,1]]))
first=policy.predict_action_chunk(prepared)
policy.reset()
second=policy.predict_action_chunk(prepared)
torch.testing.assert_close(first,second,rtol=0,atol=0)
assert first.shape==(1,50,10) and torch.isfinite(first).all()
print(json.dumps({'wheel':lerobot_policy_humanego.__file__,'relocated_bundle':str(root),
 'shape':list(first.shape),'checks':['installed wheel outside repository','original training tree access denied',
 'relocated processor paths','one original model','cpu/cuda noise follows model','eval mode','fixed-noise replay']}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="humanego-relocation-") as directory:
        moved = Path(directory) / "bundle"
        shutil.copytree(args.bundle, moved)
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                CHILD,
                str(args.installed.resolve()),
                str(moved),
                str(args.training_root.resolve()),
            ],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        )
    report = json.loads(result.stdout.splitlines()[-1])
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
