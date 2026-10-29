# HumanEgo in Simulation

Closed-loop simulation of the **released HumanEgo `serve_bread` policy** on a physics
replica of the real deployment rig: two Trossen WidowX AI arms and a top-mounted
Intel RealSense D405, placed by the **real hand-eye calibration**. The network is never
touched — the simulator shares [`inference/policy.py`](../inference/policy.py)'s
`ICTPolicy` and `ObjectState` with the new remote-serving scaffold. Its original
RGB-D lifting helpers and camera configuration live in
`humanego_sim/legacy_inference/`; the remote-serving scaffold does not own hardware
drivers. The simulation policy and controller behavior are unchanged.

**Interactive page:** <https://humanego-ai.github.io/sim/> — replay pre-computed rollouts
in 3D, or switch to **Live** and run the policy *in your browser* (MuJoCo WebAssembly +
ONNX Runtime Web) from any initial condition you drag into place.

| | |
|---|---|
| Robots | Trossen WidowX AI (real rig) · Franka Panda · UR5e + Robotiq 2F-85 |
| Cameras | RealSense D405 (real rig) · RealSense D435 · ZED 2i · ZED Mini |
| Arms | two (independent bimanual, as in the teaser) · one (as in the paper) |
| Objects | **soft** croissant and sugar danish loaf (MuJoCo flex, they squash in the gripper) modelled after the breads in the real rollouts, light-blue plate, maple butcher-block table; sizes measured from the released dataset |

## Quick start

```bash
conda create -n humanego-sim python=3.11 -y && conda activate humanego-sim
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r simulation/requirements.txt
huggingface-cli download Leo-TX/HumanEgo --include "serve_bread/*" --local-dir ./checkpoints

python simulation/scripts/build_assets.py                      # once: procedural bread / plate / table assets
python simulation/scripts/run_episode.py --video               # real-rig controller (production profile)
python simulation/scripts/run_episode.py --profile tuned --video   # the shipped tuned controller + grasp skill
```

Other rigs are one flag away:

```bash
python simulation/scripts/fetch_robots.py franka_panda ur5e        # large Menagerie meshes, fetched on demand
python simulation/scripts/fit_robot_poses.py franka_panda ur5e     # start pose = the real Trossen start pose (IK)
python simulation/scripts/run_episode.py --profile tuned --robot franka_panda --camera zed2i --video
python simulation/scripts/run_episode.py --mode single --set control.alpha=0.3
```

Headless rendering uses `MUJOCO_GL=egl` (set automatically).

## pick_cube scene

`configs/pick_cube.yaml` is the standalone configuration for a blue open plastic
container (`obj1`) and a red rigid block (`obj2`), matching
`cfg/preprocess/tasks/pick_cube.yaml`. It contains the fixed D scene, robot rig,
table, camera calibration, appearance, physics, perception and grasp controller;
it does not inherit `serve_bread.yaml` or another YAML. Prop dimensions, masses,
friction and initial `[x, y, yaw_deg]` poses are editable under `scene.objects`.
The default container has rounded, tapered collision walls; dimensions were
estimated from training data and have not been confirmed by physical measurements.

The default is a **closed-loop run of your trained pick_cube model**, using
`runs/pick_cube/server_0.1/latest.pt` and the `config.json` / `dataset_stats.json`
next to it. It uses the right arm, as in training (`single_hand: true`,
`absolute` actions). The checkpoint's `config.json` selects `camera_frame` or
`anchor_frame` automatically; do not change this flag to reinterpret trained weights.
No public model download is needed. Perception-guided grasp assistance overwrites
the grasp point and orientation, waits before closing, and permits one retry.
The existing 5 cm automatic lift is explicit under `control.forced_lift_*`.
Bread physics and plate-based idle completion are omitted. Success requires
the grasped block to be released, settled and fully inside the container.
The optional `control.release_assist` is enabled in the active pick_cube config.
At 10 Hz it checks true rigid-body poses, cube contacts with both finger pads,
the complete oriented block's gravity-drop path into the rounded/tapered cavity,
and low block/hand velocity relative to the container. The drop region has a
5 mm interior margin and the block's lowest point must be at most 8 cm above
the rim. After these conditions hold continuously for 1 s of simulation time
while the policy still commands closed, the runner halts the hand, mutes policy
control and opens the gripper. Leaving the region, losing contact, excessive
motion or a policy open command resets the timer. A tilted container (>5 degrees)
is excluded. This uses simulator ground truth, not deployable perception.

The assist waits for the measured gripper opening, then requires 0.5 s of
continuously successful placement before going home (up to 2 s for settling).
It triggers at most once per episode and cannot be overridden by late policy
commands or re-grasping. A release that starts before the episode time limit
gets bounded extra time for this sequence. Triggering is not itself success:
normal physical placement checks still apply, and failure to open is a failure.
`meta.result.per_side.right.release_assist` in `episode.json` records the settings,
trigger time, wait duration, last blocking reason and outcome (`placed`,
`open_timeout`, `settle_timeout` or `not_triggered`). The generic default is off;
serve_bread is unchanged. To compare with the unassisted policy:

```bash
python simulation/scripts/run_episode.py --config simulation/configs/pick_cube.yaml \
  --set control.release_assist.enabled=false --seed 0 --video --out pick_cube_no_release_assist
```

Historical A–E and grasp-study configs remain frozen under `simulation/runs/`;
active pick_cube parameters are maintained only in `configs/pick_cube.yaml`.

```bash
# Activate an environment containing the project's simulation dependencies.
conda activate GlassEgo
python simulation/scripts/build_assets.py --table-only
python simulation/scripts/run_episode.py --config simulation/configs/pick_cube.yaml --video --out pick_cube_model

# Choose a different checkpoint and inference settings. --set values use JSON syntax.
python simulation/scripts/run_episode.py --config simulation/configs/pick_cube.yaml \
  --set 'policy.ckpt="./runs/pick_cube/HumanEgo/latest.pt"' \
  --set policy.num_inference_steps=20 --set control.policy_period=0.3 \
  --set control.max_episode_time=40 --device cuda --seed 0 --video

# Use an anchor-frame checkpoint directly from GlassEgo (no model copying).
python simulation/scripts/run_episode.py --config simulation/configs/pick_cube.yaml \
  --set 'policy.ckpt="~/GlassEgo/runs/Block Jamming real gournd/new_03/latest.pt"' \
  --set policy.num_inference_steps=10 --video --out pick_cube_anchor

# Run the trained model and export its rollout for the web viewer.
python simulation/scripts/build_web.py --config simulation/configs/pick_cube.yaml
python simulation/web/tools/serve.py 8791 simulation/web
```

Open `http://127.0.0.1:8791/?task=pick_cube&mode=single` and press **Replay** to
view the trained policy's predictions, input images and actual physics outcome.
This is recorded Python inference; browser ONNX Live inference is not enabled
for pick_cube. Videos and episode logs are in `simulation/runs/<out>/`.
`--profile production` is the default; `tuned` profiles belong to serve_bread
and are rejected for pick_cube. Use `--set` for your own inference/controller
parameters. Same-seed reruns are intended for reproducing a run; task success
must be measured rather than inferred from a model loading successfully.

For `anchor_frame`, each side freezes its own initial perceived `obj1` pose as
the reference, matching training's `virtual_static_anchor`. In pick_cube this is
the blue container. ICT object poses can still update independently; the reference
used to encode geometry and decode trajectories stays fixed throughout the episode.
Left-arm perception already supplies a mirrored reference, so it is not mirrored
again. Missing anchors, fewer than three finite 3D keypoints or invalid reference
matrices stop the run instead of falling back to camera coordinates. `camera_frame`
retains its existing input and decoding path. Both modes retain the existing
gripper decoding, grasp assistance, physics and controller settings, and require
absolute actions. Anchor-frame support covers Python inference and recorded
replay, not browser ONNX Live inference.

For geometry inspection only, add `--set scene_only=true --duration 5`; it skips
the policy. Physics-only previews can also be built with
`build_web.py --config simulation/configs/pick_cube.yaml --set scene_only=true`.
Those retain the **Live physics** prop editor and may use either arm mode.

## Results

### Reproducible pick_cube batch evaluation

```bash
# 10 randomized layouts × 3 policy-noise seeds = 30 trials; one GPU worker by default.
python simulation/scripts/evaluate.py --config simulation/configs/pick_cube.yaml \
  --n 10 --scene-seed 7777 --policy-seeds 0 1 2 --out pick_cube_eval_dev

# Freeze inputs without loading model weights into the GPU or running inference.
python simulation/scripts/evaluate.py --config simulation/configs/pick_cube.yaml \
  --n 50 --scene-seed 8888 --policy-seeds 0 1 2 --out pick_cube_eval_150 --prepare-only

# Resume using frozen inputs, even if the live task YAML has since changed.
python simulation/scripts/evaluate.py --out pick_cube_eval_150 --resume --workers 1

# Retry infrastructure errors explicitly; ordinary task failures remain completed trials.
python simulation/scripts/evaluate.py --out pick_cube_eval_150 --resume --retry-errors

# Rebuild the report and render failed recordings, without running the policy again.
python simulation/scripts/evaluate.py --out pick_cube_eval_dev --report-only --videos failures
# Or choose zero-based trial indices: --report-only --video-trials 0 3

# Compare another controller on the exact same layouts, environment seeds and policy noise.
python simulation/scripts/evaluate.py --config simulation/configs/pick_cube.yaml \
  --trials simulation/runs/pick_cube_eval_dev/trials.json \
  --set control.grasp_assist.enabled=false --set control.forced_lift_height=0 \
  --set control.forced_lift_frames=0 --out pick_cube_eval_autonomous_grasp
```

`--n` counts **scenes**, not total trials. The default pick_cube batch is 50
scenes × noise seeds `[0, 1, 2]` = 150 trials. `--scene-seed` defaults to `--seed`
(0). Scene-set seeds 9001–9099 require `--final` and should be reserved for final
evaluation rather than parameter selection. Policy seeds are explicit seeds of
the same checkpoint's fixed per-episode flow noise; they are not training seeds.
Every scene uses the same environment/depth-noise seed across its policy repeats.
Worker count and execution order do not enter seed generation. GPU execution can
still have small numerical differences; reproducibility is not a bitwise guarantee.

Sampling settings live in `configs/eval_pick_cube.yaml` (`--sampling` selects
another file). The default independently jitters cube and container x/y by ±3 cm;
yaw jitter is configurable and defaults to zero. Full object footprints, including
container handles, must fit the table, declared task workspace and camera view,
with no initial object overlap. The workspace is a geometric envelope, **not an
IK guarantee for arbitrary policy orientations**. Rejected candidate counts are
saved; policy failures never cause a scene to be replaced. Batch episodes start
from settled states without the decorative intro drop.

The output `simulation/runs/<out>/` (or absolute `--out` directory) contains:

- `config.yaml`, `sampling.yaml`, `trials.json`: frozen effective inputs and every trial's poses/seeds.
- `protocol.json`, `source.patch`: checkpoint/training-metadata hashes, source-code hash,
  dependency versions, Git revision/local changes, device and all assist settings.
- `episodes/trial_XXXXXX/result.json` and `episode.json`: atomic per-trial results and physics recordings.
- `results.csv`, `summary.json`, `report.md`: trial table, aggregate statistics and readable report.

The batch score requires the existing released/inside/settled placement checks,
plus a real secure grasp: both pads exert >0.1 N on a block lifted >2.5 cm for at
least 0.1 s, and the measured gripper is at least 90% open at the end. The original
runner score is retained as `runner_success`. Contact/lift observations do not
change policy inputs or control. Reports distinguish lift, secure grasp, transport,
release-assist use, IK failures and task failure modes. Errors count as failures
in the final denominator and are listed separately; incomplete batches show a
completed-trial rate but withhold the final rate and confidence interval.
The primary 95% interval bootstraps **whole scenes**, keeping their policy repeats
together. Per-policy-seed Wilson intervals and rates are also saved.

Completed successes, task failures and errors are skipped on resume; only
`--retry-errors` retries errors. A trial interrupted before publishing its atomic
`result.json` is rerun. Changed frozen inputs, source code, dependencies, device,
checkpoint contents or training metadata require a new output directory. Reporting
alone does not require the original checkpoint or runtime. Videos are off by
default and are reconstructed from the existing 10 Hz physics logs without inference.

Per-arm success = bread released and resting on the plate, on randomized scenes (paper protocol,
soft bread, sourced physics, table height measured from the real video). The headline column is the
reserved test set (seeds 9001+, 100 scenes per robot, never used to pick parameters), with the
production controller on the same scenes for comparison; the dev seeds are where the parameters were
selected. Numbers from `configs/tuned/<robot>.json` (this table is generated:
`python -m humanego_sim.web_manifest`; a test fails when it drifts from the files). How the upgrade of
2026-10-18 got here: [`docs/upgrade/report.md`](docs/upgrade/report.md).

<!-- results:start -->
| Robot | Controller (`--profile tuned`) | Per arm, reserved test [95 % CI] | Right / left | Both served | Production controller, same scenes (per arm / both) | Dev seeds (per arm) |
|---|---|---|---|---|---|---|
| Trossen WidowX AI | `trossen_v7` + grasp skill | **80%** [74%, 85%] (100 scenes, seed 9001) | 88% / 73% | 64% | 60% / 27% | 83% (seed 7777, 32 scenes) / 89% (seed 8888, 32 scenes) |
| Franka Panda | `franka_panda_v4` + grasp skill | **76%** [69%, 81%] (100 scenes, seed 9001) | 88% / 63% | 57% | 46% / 17% | 73% (seed 7777, 24 scenes) / 73% (seed 8888, 24 scenes) |
| UR5e + Robotiq 2F-85 | `ur5e_v3` + grasp skill | **76%** [70%, 81%] (100 scenes, seed 9001) | 99% / 53% | 53% | 57% / 22% | 83% (seed 7777, 24 scenes) |
<!-- results:end -->

The real rig's production controller (`--profile production`) scores much lower on the same
scenes; most of the gain comes from the grasp skill and the tuned servo timing. The comparisons
are in [Grasp skill](#grasp-skill-in-the-low-level-controller) and
[Self-tuning loop](#self-tuning-loop-autotune). Read rates with their sampling error:
see [Evaluation protocol](#evaluation-protocol). The interactive page shows the same
controllers on its own rollouts.

## How it works

1. **Sense.** A virtual camera at the real hand-eye calibration renders RGB-D, cropped like `CamRS`.
2. **Perceive.** The released perception runs on ground-truth masks (in place of DINO-SAM):
   keypoints, multi-frame depth lift, PCA object frames; the robot is removed from the image
   (perfect inpainting) and the virtual gripper drawn, as on the robot.
3. **Act.** `ICTPolicy` on the released checkpoint predicts a 50-step hand trajectory; one
   right-hand model drives both arms, the left arm sees a mirrored image.
4. **Move.** The production controller (temporal ensembling, safety cage, EMA, look-ahead) feeds
   a Trossen-driver emulator (quintic re-targeting + IK) and MuJoCo, where the bread is a soft
   flex body held by friction alone.

Settings come in **profiles** (`humanego_sim/profiles.py`, `--profile` on every script):
`production` is the real rig's `Control.yaml` (the default); `tuned` adds the robot's tuned
parameters from `configs/tuned/<robot>.json` and the grasp skill. Profiles sit between the
per-robot calibration and explicit `--set` overrides. Details below:
[what is simulated](#what-is-simulated-and-how-faithfully), [grasp skill](#grasp-skill-in-the-low-level-controller),
[Live mode](#live-mode-the-policy-in-your-browser), [web viewer](#web-viewer-web).

## Reproduce

```bash
# success rate with 95 % intervals (development seeds 7777 / 8888; 9001-9099 are reserved)
python simulation/scripts/evaluate.py --profile tuned --n 32 --seed 7777 --workers 8
# production vs tuned on the same scenes, paired test
python simulation/scripts/compare_params.py --sets production=- tuned=simulation/configs/tuned/trossen_wxai.json \
    --set 'control.grasp_assist={"enabled": true}' --seed 7777 8888
python simulation/scripts/evaluate.py --profile tuned --robot ur5e --n 24 --seed 7777
# the interactive page: build every config, pack, record the videos (never deploys)
python simulation/scripts/rebuild_site.py --workers 2
# tests: CPU subset (what CI runs), then the closed-loop smoke episodes (checkpoint + GPU)
pytest simulation/tests -m "not sim"
pytest simulation/tests -m sim
```

Engineering logs of the upgrade rounds (what was tried, measured, kept or reverted):
[`docs/upgrade/`](docs/upgrade/) — evaluation and robustness ([robust.md](docs/upgrade/robust.md)),
physics ([physics.md](docs/upgrade/physics.md)), engineering and the website ([eng.md](docs/upgrade/eng.md)).

## Evaluation protocol

A success rate here is a proportion over a few dozen random scenes, so it carries sampling
error. The scripts report it (`humanego_sim/stats.py`):

* **Confidence intervals.** `evaluate.py` and `compare_params.py` print each rate with its 95 %
  Wilson score interval. Per-arm rates count two arms per scene, and those two outcomes are
  not independent. `compare_params.py` therefore also prints a scene-bootstrap interval that
  resamples whole scenes; when the two intervals differ, quote the bootstrap one.
* **Paired comparisons.** `compare_params.py` runs every set on the same scenes and tests each
  against the first set. It reports an exact McNemar test on the arms whose outcome differs
  (served with A and failed with B, or the reverse), and a paired scene-bootstrap 95 %
  interval of the per-arm difference. "A beats B" means McNemar p < 0.05 *and* a bootstrap
  interval that excludes 0. Repeated runs of the same settings differ by 3–4 points because
  the policy is nondeterministic on the GPU. The paired test counts that noise as discordant
  arms, so it does not report it as an effect.
* **Scene sets.** `--seed` (with `--n`) selects a scene set. Seeds **7777 and 8888 are for
  development**: select and tune parameters on both. Seeds **9001–9099 are reserved for final
  reports**. Never pick parameters on them, and never re-tune after seeing their results. The
  scripts refuse these seeds unless you pass `--final`. `--seed 7777 8888` runs both
  development sets and pools them for the paired test.
* **Sample size.** Headline numbers use **≥ 100 scenes per robot** (200 arms). The half-width
  of the 95 % interval on the per-arm rate is:

  | scenes (arms) | at 60 % | at 80 % | at 90 % |
  |---|---|---|---|
  | 24 (48) | ±13 | ±11 | ±9 |
  | 32 (64) | ±12 | ±10 | ±8 |
  | 64 (128) | ±8 | ±7 | ±5 |
  | 100 (200) | ±7 | ±6 | ±4 |

  So a 32-scene comparison of two independent runs resolves only differences of about 15
  points. Smaller differences need the paired test on the same scenes.

```bash
GA='control.grasp_assist={"enabled": true}'
T=simulation/configs/tuned/trossen_wxai.json
python simulation/scripts/compare_params.py --sets tuned=$T fused=$T@'{"perception": {"mask_frames": 5}}' \
    --set "$GA" --seed 7777 8888
```

A set is `name=params.json`, `name=-` (production settings), `name=@'<JSON overrides>'`, or a
params file plus overrides (`name=params.json@'<JSON>'`), as above.

## What is simulated, and how faithfully

| Real system | Simulator |
|---|---|
| Trossen WidowX AI ×2, hand-eye calibrated (`T_base_in_cam`) | MuJoCo Menagerie `trossen_wxai`; bases placed from the **real calibration** |
| `trossen_arm` driver: non-blocking Cartesian goals re-planned at 10 Hz with `goal_time` | `SimArm`: quintic re-targeting from the current interpolator state (velocity continuous), DLS IK every physics step, reachability errors, `safe_z_min` floor, EE correction frame |
| RealSense D405, 1280×720 → centre crop → 640×480 | pinhole camera with the same crop/resize and a depth-noise model |
| DINO-SAM masks | ground-truth segmentation (same mask semantics: obj1 = every bread, obj2 = plate) |
| LaMa arm inpainting | render without the robot |
| KptsSelector → multi-frame depth lift → PCA pose → VisualKpts | **the released modules themselves** |
| `Inference.run_bimanual` (mirrored left arm, latching, forced lift, idle-done, wrap-up) | `runner.py`, same logic on a deterministic sim clock |
| `AsyncRobotController` (ensembling, safety cage, EMA/Slerp, look-ahead, stride) | `controller.py`, same logic on a sim clock |

Notes on the modelling choices (all in `configs/serve_bread.yaml`):

* **Scene geometry.** The camera sits 0.436 m above the table: the real driver's
  `safe_z_min = −0.20 m` "protects the table", which puts the table just below the arm
  bases. Objects then lie ~0.6 m from the camera, inside the depth range of the human
  training data. Bread yaw ~90° (long axis left–right in the image) as in the demos.
* **Physics from measurements, not from the policy's score.** Every physical parameter, its
  defensible range and its source (food-science papers, datasheets, the real rollout videos,
  the released dataset) is in [`docs/physics_params.md`](docs/physics_params.md); policy-free
  tests check the model against them (`scripts/physics_compress.py`, `physics_pinch.py`,
  `physics_drop.py`, `physics_gripper.py`, `drop_test.py`; log: [`docs/upgrade/physics.md`](docs/upgrade/physics.md)).
* **Object sizes and masses** come from the released dataset's triangulated contour
  keypoints (croissant ≈ 13.8 × 6.2 × 5 cm, plate ⌀ 24.5 cm, confirmed by a circle fit to the
  Aria SLAM points of 42 demos), the real rollout video for the danish loaf (≈ 14 × 7.4 × 6 cm,
  measured against the plate) and the plate's rim height (3 cm, three rollout views), and
  real weights (croissant 50 g, danish 75 g, plate 450 g).
* **Breads that look like the real ones.** `scripts/bread_gen.py` builds the straight rolled
  croissant of the demos (chevron bands, pale dough in the grooves, glossy egg wash) and the
  laminated danish loaf with its sugar crust as parametric surfaces with painted PBR maps.
  As in the real bimanual run, the right arm gets the croissant and the left arm the danish.
* **Soft bread** (`humanego_sim/softbody.py`). Each bread is a MuJoCo flex: its collision
  surface is interpolated from a 4 × 2 × 2 trilinear grid of point masses with volumetric
  elasticity (E = 20 kPa, ν = 0.25: brioche crumb is 20 ± 3 kPa), and the textured mesh is
  bound to the same grid as a skin, so the crust visibly squashes under the fingers and
  springs back. The contact is stiff (`solimp` 0.995): the fingers' indentation comes from the
  crumb's modulus, not from the contact sinking in (with the earlier `solimp` 0.95 a crumb cube
  of E = 20 kPa responded like 6.8 kPa). The grip holds by friction alone; each surface has its
  own friction (crust on the silicone finger pads 0.8, on the plastic plate 0.4, on the maple
  table 0.45, on crust 0.4; no published value exists for bread, so these are proxies from food
  on rubber / plastic / wood). Grippers squeeze as their hardware does: the Trossen through the
  system-identified Menagerie servo (kp 339 N/m: 4–8 N per finger on the breads), the 2F-85 at its
  minimum force setting (~20 N per pad) and the Franka Hand with a 20 N grasp. Flex elasticity
  needs MuJoCo's `discrete` integrator (implicit in position, stable at the 2 ms step);
  `impratio = 100` keeps a held bread from creeping (0.1–0.3 mm/s under its weight).
  `physics.soft.plastic` adds the crumb's permanent dent (off by default, Python only).
  `physics.bread_model: rigid` restores the earlier rigid-bread + finger-adhesion proxy.
* **Crumb that lands dead.** Real crumb dissipates almost all impact energy: a dropped bread
  does not bounce (the real croissant, released 6 cm above the plate, is at rest within
  0.27 s). Flex damping 0.03 s (loss tangent ≈ 0.17 at the grasp time scale) and over-damped
  bread contacts (`solref` damping ratio 3) reproduce that: coefficient of restitution
  0.16–0.19 (`physics_drop.py`). In `scripts/drop_test.py` (danish dropped 4–9 cm onto a
  croissant lying on the plate, random offset and tilt) it stays on the plate in 100 % of drops.
* **Served = resting on the plate.** An arm succeeds when it released its bread and the bread
  rests on the plate: its centre is over the plate and no part touches the table (the
  plate's floor is 8.5 mm above the table). The earlier rule, centre within R − 2 cm, failed
  breads that lie fully on the plate's rim zone; it is still reported as `success_strict`.
* **Grasp state.** The real loop flags a grasp when the measured gripper opening drops
  below 50 %; in simulation a gripper that has finished closing and stalled on the bread
  also counts (`control.grasp_state_source`). The stall rule stands in for squash the sim
  bread does not reproduce: the real fingers pinch the bread's end flat (real video), while
  the sim crumb under the Trossen's 4–8 N stays above 50 % opening. With the real rule alone
  (`grasp_state_source: measured`) the flag never fires, the bread is never latched to the
  hand, and success collapses (Trossen dev seed 7777: production 52 → 6 %, tuned 80 → 11 %
  per arm). The real robot's 95 % means its flag does fire.
* **Release.** The wrap-up follows the real code: two arms halt, open over 1.0 s (blocking)
  and then go home (`Inference._wrap_up_side`); one arm opens non-blocking and starts home
  after 0.5 s (`Inference.run`). `control.release.open_time` / `home_after` override it. The
  earlier 0.5 s open with home after 0.8 s scored the same (dev seeds, paired: production
  47 vs 46 %, tuned 86 vs 88 %, single arm 55 vs 55 %). The release height still differs from
  the real video: see `docs/real_vs_sim.md` §2.

### Why grasps land above the bread's centre

Measured over 32 grasps (`scripts/grasp_height.py`, `scripts/grasp_test.py`), the
fingertip point closes ≈ 1.6–2.2 cm above the bread's centre (the croissant is 5 cm tall).
The offset adds up from three parts of the real pipeline, all reproduced faithfully:

| | above centre |
|---|---|
| perception: keypoints are lifted from depth on the visible top surface | +0.2–0.6 cm |
| policy: the predicted grasp point relative to those keypoints (the human demos pinch 1.2 cm above the contour centroid) | +0.5 cm |
| execution: the driver re-plans a quintic with `goal_time = 1.2 s` from the current state every 0.1 s (no feed-forward, as `RobotArmTrossen.move_p`), so the arm trails the descending target by ≈ 2 cm; the gripper closes on the `any` grasp signal (0.5 s) and the post-grasp lift starts before the arm catches up | +1.0–1.2 cm |

What does and does not move it (same 16 scenes, soft bread, per-arm success in brackets):

| change | fingertips above centre at closing | per-arm success |
|---|---|---|
| production controller | +1.6 cm | 44 % |
| `look_ahead_offset` 45 → 30 | +2.2 cm | 38 % |
| `duration_smooth_param` 12 → 6 (goal time 0.6 s) | +1.3 cm | 44 % |
| close only if the grasp is ≤ 20 steps ahead (`grasp_k_any` 20) | +1.4 cm | 41 % |
| goal feed-forward velocity (`feedforward_gain` 1, not in the real controller) | +1.8 cm | 34 % |
| real grasp flag only (opening < 50 %, no stall detection) | +1.6 cm | 16 % |

The gripper closes on the early `any` signal while the arm is still descending; the stall
on the bread then counts as a grasp and the lift starts. With friction-only soft bread this matters: a pinch holds when the
fingertips are ≤ ≈ 1.9 cm above the centre and slips out above that (the pads then press
on the rounded top, above the bread's widest section), independently of grid resolution,
crumb stiffness or friction (`scripts/grasp_test.py` replays recorded grasp poses with any physics). The
rigid + adhesion proxy hid this.

## Grasp skill in the low-level controller

The production controller only follows the plan and toggles the gripper on the policy's
grasp signal. An audit of every arm (`scripts/grasp_audit.py`) showed where that loses bread:
the left arm pinched the 7.4 cm danish near its end, high or off to the side and it slipped
out when lifted; some grasps closed on air; a few breads stayed wedged in the opened hand.
`control.grasp_assist` adds a grasp skill that uses only what the real robot has, the
gripper reading and its own perception of the bread at t = 0 (Python and browser):

1. **Refine** the plan's pinch point: pull it onto the perceived bread's long axis, cap its
   offset from the bread's centre along the axis and its height above the centre.
2. **Close on arrival**: keep the gripper open until the arm has reached that point, and hold
   still while the fingers close.
3. **Check the grip**: if the gripper closed on air or the bread slipped out on the way up
   (the gripper reading collapses), reopen and let the policy re-approach; each retry aims
   lower and closer to the bread's middle. The empty check reads the gripper joint, or the
   gap between the pads for the underactuated 2F-85, which closes its driver joint almost
   fully around a bread (the real one reports object detection).

Same 32 held-out scenes, `soft_v5` settings:

| | per arm | right | left | both served |
|---|---|---|---|---|
| production controller behaviour | 70 % | 91 % | 50 % | 50 % |
| + refine & close on arrival & re-grasp | 77 % | 91 % | 63 % | 56 % |
| + adaptive re-grasp (lower, more central) | 80 % | 94 % | 66 % | 63 % |
| + skill and controller re-tuned together (`soft_v6`) | **81–83 %** | 88 % | **75–78 %** | **69 %** |

`soft_v6` searched the skill's knobs together with the controller's (one batch reached the
loop's target; its candidates were then re-checked on the held-out scenes, a second
independent set of 32 scenes gives 78 % per arm). It grasps more top-down (gripper held 10°
steeper than the plan), pinches at the perceived centre height, within 3.8 cm of the
bread's middle, and tracks fast (goal time 0.2 s). The same skill on the production
controller settings scores 62 %; without it 41 %.

Releasing differently (setting the bread down first, lifting or backing the open hand out)
did not help and is left off (`control.release`).

The other hands got the same treatment: a finer collision skin for the Franka's small pads
(`physics.soft.surface: fine`), pad-gap grip sensing for the 2F-85, then the skill and
controller re-tuned per robot (`autotune_runs/franka_v2`, `ur5e_v2`). Per arm on 24 held-out
scenes (seed 7777):

| | `*_v1` tuned, no skill | production settings + skill | **`*_v2` (skill + per-robot tuning)** |
|---|---|---|---|
| Franka Panda | 33 % | 25 % | **60 %** (right 75 %, left 46 %) |
| UR5e + 2F-85 | 38 % | 52 % | **71 %** (right 96 %, left 46 %) |

The Franka settled on a slightly steeper gripper (2.7°), the tool point 1 cm shallower
and up to 3 re-grasps; the UR5e on a fast EMA (0.53), a 9° more level gripper and a tight
pull onto the bread's axis (gain 0.95, within 1.7 cm of its middle). The left arm (mirrored
view, wider danish) is the weak side on every robot.

### The left arm

With those settings the left arm served 75 % of its breads on the Trossen and 46 % on the
Franka and UR5e, against 88–96 % on the right. Every left-arm episode was logged and audited
(`compare_params.py --log`, `scripts/grasp_audit.py`), and each suspect was swapped out on
the same held-out scenes:

| suspect | test | left arm |
|---|---|---|
| mirrored perception | perceived bread (mirrored back) vs ground truth | long-axis error 7.5° (right arm 8.0°), centre offsets mirror-symmetric: not it |
| asymmetric start pose (gripper 38° below horizontal, right arm 25°) | `rig.mirror_left_start` (Trossen); capping the grasp pitch at 27–30° (Franka) | 69 % vs 75 %; 33–38 % vs 46 %: not it |
| the 7.4 cm danish | a croissant for the left arm too | Trossen 69 % (no change), Franka 46 → 67 %, UR5e 50 → 67 % |
| release onto the croissant | where the danish ends up | UR5e: 11 of 12 left failures. It drops from 10–12 cm onto the croissant and slides off the plate, or the move home drags it |

Two causes, two fixes:

1. **Pinch height on the taller danish** (Trossen: the left arm lost breads in transit, 16
   of 160 episodes against 1 on the right). The 6 cm, 75 g danish holds when pinched at or
   below its centre (97 % of lifts) and slips out above it (68 % at 0 to +0.5 cm, ~30 % above
   +1 cm). The skill caps the pinch height relative to the *perceived* centre. That was the
   mean height of the contour keypoints, but they are lifted from depth on the visible upper
   surface, so it sat 0.25 cm above the croissant's centre and 0.5 cm above the danish's.
   A bread rests on the table, so its centre is half its perceived top height (within 1 mm
   on both breads; `runner._perceived_geometry`, Python and browser;
   `perception.bread_centre_height: mean` restores the old estimate). With that estimate the
   Trossen pinch cap moves from 0.2 cm to 0.6 cm below the centre (`grasp_assist.height_max`
   −0.006). Each change alone does nothing on the left arm; together they do
   (Trossen, left arm, 64 held-out scenes, two seeds):

   | centre estimate | pinch cap | left arm |
   |---|---|---|
   | keypoint mean (before) | −0.2 cm | 80 % |
   | half the top height | −0.2 cm | 81 % |
   | keypoint mean | −0.6 cm | 81 % |
   | **half the top height** | **−0.6 cm** | **92 %** |
   | half the top height | −0.9 cm | 88 % |

2. **Release onto the croissant** (UR5e). The UR5e opens its hand ~11 cm above the plate,
   on top of the croissant that is already there. `control.release.place_down` first sets
   the danish down to 3 cm above its resting height, then opens, lifts the open hand 8 cm
   straight up and only then goes home. Without the lift the move home drags the bread off
   (6 % per arm). The Trossen already releases from ~8 cm, and place-down lowers its score
   and the Franka's, so only the UR5e uses it (`configs/tuned/ur5e.json`). The UR5e and
   Franka were then re-tuned (`autotune_runs/ur5e_v3`, `franka_v3`).

Per arm / right / left / both breads served, on the held-out scenes of seed 7777 and on a
second set (seed 8888); 32 scenes for the Trossen, 24 for the others:

| | seed 7777 | seed 8888 |
|---|---|---|
| Trossen, before (`soft_v6`) | 81 / 88 / 75 / 69 % | 83 / 81 / 84 / 75 % |
| **Trossen, after** | **91 / 94 / 88 / 84 %** | **94 / 94 / 94 / 91 %** |
| UR5e, before (`ur5e_v2`) | 73 / 96 / 50 / 46 % | 71 / 100 / 42 / 42 %\* |
| UR5e, `ur5e_v2` + place-down | 85 / 96 / 75 / 71 % | 85 / 100 / 71 / 71 % |
| **UR5e, after (`ur5e_v3` + place-down)** | **88 / 92 / 83 / 79 %** | **81 / 96 / 67 / 63 %** |
| Franka, before (`franka_v2`) | 60 / 75 / 46 / 33 % | 50 / 71 / 29 / 21 %\* |
| **Franka, after (`franka_v3`)** | **63 / 88 / 38 / 33 %** | **60 / 88 / 33 / 33 %** |

\* measured with the new centre estimate, which leaves the UR5e and Franka unchanged
(seed 7777: 75 / 96 / 54 % and 58 / 71 / 46 %). Repeated runs of the same scenes differ by
3–4 points (GPU nondeterminism in the policy), so the two seeds are reported separately.

The Franka's left arm did not improve. Its hand opens 8 cm and the danish is 7.4 cm wide:
the pinch holds only when it lands within ~1.5 cm of the bread's axis, and the perceived axis
is itself 1.0–1.4 cm off. With a croissant on the left the same settings serve 67 %, so this
is a limit of the hand, not of the controller.

Tried and not adopted: latching the grasp flag once the gripper stalls (the flag flickers
while the danish slips, but the flicker is a symptom: latched, the Trossen left arm served
78 % instead of 84 %, the Franka's 33 % instead of 46 %); capping the grasp pitch (Franka);
place-down on the Trossen and Franka; the −0.6 cm pinch cap on the UR5e and Franka (UR5e
left arm 50–71 % instead of 67–88 %); lifting the open hand without placing first.

## Robustness

The numbers above use idealised sensing: ground-truth masks, Gaussian depth noise, a perfect
hand-eye calibration, a fresh image every step, and one nominal scene. `humanego_sim/robustness.py`
adds the errors of a real rig, and `scripts/robust_report.py` measures each one alone and all of
them together, on the same scenes as the clean run (paired test, README "Evaluation protocol").
Everything except mask fusion is off by default.

| perturbation | key | model at the "realistic" level used here |
|---|---|---|
| segmentation errors | `perception.mask_noise` | per object, an erosion/dilation of ±2 px that persists through the episode; 1.5 px smooth boundary jitter; in each detector frame, 5 % missed detections per object and 10 % spurious blobs |
| real segmenter | `perception.mask_source: dinosam` | the released DINO-SAM (Grounding-DINO tiny + SAM2 tiny) on the rendered camera image, run by `scripts/dinosam_server.py` in an env with transformers + sam2 (`HUMANEGO_DINOSAM_PYTHON`) |
| clutter taken for bread | `perception.distractor_confusion` | per episode, a banana is segmented as bread 30 % of the time, the yellow box 8 % |
| depth artefacts | `perception.depth_artifacts` | in a 2 px band at depth steps over 8 mm, flying pixels (50 %) and holes (25 %); 3 % of pixels in hole blobs (70 % of the pattern persistent); depth scale error σ 0.5 % |
| hand-eye calibration | `perception.calib_error` | true camera vs calibration: σ 0.3° and 3 mm per axis |
| timing | `perception.latency` | the policy image is 66–99 ms old; 5 % of frames are not refreshed |
| scene variation | `--variation` (`sampling.sample_scenes`) | bread size ±12 %; bread yaw ±15° and plate position ±3 cm beyond the paper's spread; table ±1 cm (the robot assumes the nominal height); light ±35 %, colour tint, key light tilted up to 25°; 1–3 visual-only distractors behind the task area |

How the levels were set:

* **Segmentation.** On these renders the real DINO-SAM is close to the ground truth
  (`scripts/robust_mask_calib.py`, 32 scenes × 4 conditions): bread IoU 0.97, boundary within
  about 1 px, plate IoU 0.94 (it includes the plate's shadow), no missed objects. It does take
  clutter for bread: the banana in 3 of 10 scenes and the yellow box in 1 of 13. The synthetic
  `mask_noise` is therefore a stress level for real images, which are harder than renders.
* **Depth.** The spec of the RealSense D405.
* **Calibration.** A typical, not best-case, checkerboard hand-eye calibration.
* **Latency.** A 30 fps USB camera.

**Development seeds, Trossen**, measured on the physics before v2 (commit dfb58b9), with tuned
settings + grasp skill and single-frame masks as released (32 scenes on seed 7777, plus 32 on
seed 8888 for clean / masks / all). Difference vs clean on the
same scenes, scene-bootstrap 95 % CI; p is McNemar:

| condition | per arm [95 % CI] | vs clean | p |
|---|---|---|---|
| clean | 92 % [83, 97] · 8888: 91 % [81, 96] | — | — |
| segmentation errors | **80 %** [68, 88] · 8888: **80 %** [68, 88] | −12 [−25, −2] · −11 [−22, −2] | 0.04 · 0.04 |
| depth artefacts | 86 % [75, 92] | −6 [−14, 0] | 0.29 |
| calibration | 88 % [77, 94] | −5 [−14, +3] | 0.38 |
| latency | 92 % [83, 97] | 0 | 1 |
| bread size | 89 % [79, 95] | −3 [−14, +6] | 0.75 |
| layout | 86 % [75, 92] | −6 [−16, +2] | 0.22 |
| table height | 91 % [81, 96] | −2 [−6, +3] | 1 |
| lighting | 91 % [81, 96] | −2 [−8, +3] | 1 |
| distractors | 89 % [79, 95] | −3 [−9, 0] | 0.5 |
| **all together** | **70 %** [58, 80] · 8888: **80 %** [68, 88] | −22 [−39, −6] · −11 [−23, +2] | 0.004 · 0.09 |

* **Only segmentation errors cost significant success alone; the policy shrugs off the rest.**
  Lighting, latency and table height change nothing measurable. Depth artefacts, calibration
  error and layout each cost about 5 points, within noise at this n.
* **What loses the arms is missed detections.** On seed 7777 (`scripts/robust_audit.py`,
  `scripts/robust_perception.py`), 8 of the 10 arms lost to segmentation errors had a bread or
  the plate missed in the one DINO-SAM frame. The object then has no keypoints, its pose is
  the identity, and the policy closes on air. A missed plate costs both arms. The boundary
  errors move the perceived bread by only 1–3 mm.
* **Combined,** the remaining losses are clutter taken for bread (the box: the right arm's bread
  perceived 16 and 33 cm off) and a spread of calibration, layout and size cases.

**Perception-side mitigations** (development seeds 7777 + 8888 pooled, 64 scenes, paired):

| mitigation | segmentation errors | all together | clean |
|---|---|---|---|
| **mask fusion over 5 frames** (`perception.mask_frames: 5`) | 80 → **88 %**, +9 [+1, +16], p = 0.04 | 75 → **83 %**, +8 [−1, +16], p = 0.08 | identical (fused GT masks are the same mask) |
| lift from a 3 px eroded mask (`lift_reject.erode_px`) | no change open-loop | — | moves the clean perception: not adopted |
| drop keypoints on the table plane (`lift_reject.table_margin`) | never fires | — | — |

Mask fusion runs the detector on 5 frames, drops empty frames and frames whose area is off the
median by more than 35 %, and keeps the pixels most frames agree on. It removes transient misses
and blobs; persistent errors such as clutter taken for bread stay. It is **on by default**
(`configs/serve_bread.yaml`, `perception.mask_frames: 5`). On the real rig it costs four more
DINO-SAM calls at t = 0. The keypoint lift is not where arms are lost, so the lift options stay
off.

```bash
# development: one robot, every condition, same scenes (the scripts refuse reserved seeds without --final)
python simulation/scripts/robust_report.py --robot trossen_wxai --seed 7777 --n 32 --log
python simulation/scripts/robust_report.py --table trossen_wxai_7777
python simulation/scripts/robust_report.py --compare trossen_dev7777:trossen_dev7777_fuse5 --conditions masks all
python simulation/scripts/robust_audit.py trossen_wxai_7777 masks all       # what lost the arms
python simulation/scripts/robust_perception.py --n 32 --conditions masks depth calib   # open-loop perception error
```

The final report on the reserved scene set (100 scenes per robot) is run once the simulator's
physics is final; see `docs/upgrade/robust.md`.

## Further grasp-skill options (off)

`control.grasp_assist` also has options for real-robot work, all off by default and none adopted in the
tuned profiles (no significant gain on physics v2, both dev seeds; the slow approach costs the UR5e 20
points). Details and numbers: `docs/upgrade/controller.md`.

* `reperceive`: before a re-grasp, re-run the released perception once the bread is visible and still
  (it may have been pushed or dropped elsewhere); a bread found on the plate counts as served.
* `timeout_near` / `approach_max`: the approach timeout counts only near the pinch point.
* `grip_check`: `q` (default: gripper opening / pad gap), `effort` (the gripper motor's effort, what
  trossen_arm and the 2F-85 report) or `touch` (finger-pad contact force: a sensor the real grippers do not
  have), with one-pad and slip (force collapse) detection.
* `yaw`: turn the hand so the fingers close across the bread's short axis (`align`, or `fit` the hand's opening).
* `slow_dist` / `slow_speed`: slow final approach. `release.open_time` / `release.settle`: gentler release.

## Live mode: the policy in your browser

`web/js/live/` is a line-by-line port of the Python stack to JavaScript:

| | |
|---|---|
| physics | official MuJoCo WebAssembly build (`@mujoco/mujoco` 3.16, same version as Python) on a physics-only MJCF (`live_export.py`) |
| policy | the whole 20-step flow-ODE solve exported as **one ONNX graph** (context + cross-attention K/V computed once, Euler steps unrolled; `scripts/export_onnx.py`, verified against `ICTPolicy.infer` to ~1e-7). WebGPU: fp32, ~0.1 s per re-plan; CPU fallback: fp16 weights, 10 steps |
| perception | three.js renders from the calibrated policy camera: robot-free clean image, object-ID masks, metric depth → the released KptsSelector / lift / PCA / VisualKpts logic |
| control | `driver.js` (quintic re-targeting + DLS IK on `mj_jacSite`), `controller.js`, `runner.js` |

The simulation advances in lock-step with the network (it pauses while a re-plan is
computed), so behaviour does not depend on how fast your machine is.

### Python ↔ browser parity (`scripts/parity_test.py`)

```bash
python simulation/scripts/parity_test.py --robot ur5e --seed 7777 --scene 0 --modes sync free full
```

runs one scene (depth noise off) through the Python stack with a trace on, then runs the browser
stack in headless Chrome (Playwright, MuJoCo WASM, WebGPU) from the same physics state with the same
flow-ODE initial noise, and compares, per control tick, the EE targets, the driver goals, the gripper
commands, the grasp-skill state and events, the release phase and the bread poses, plus the ICT, anchor uv
and grasp flag of every re-plan. It reports the first divergence and the max errors
(`runs/parity/<name>/report.json`). Modes:

* `sync`: Python's perception and policy outputs, physics re-synced to Python's before every control
  tick. Isolates the ported logic (`runner.js`, `controller.js`, `driver.js`).
* `free`: the same, with free-running WASM physics.
* `full`: the browser on its own: perception from the three.js renders, the ONNX policy.

Seed 7777, scene 0, tuned settings with the grasp skill:

| | Trossen | UR5e (place-down release) | Franka |
|---|---|---|---|
| `sync`: EE targets / driver goals / ICT, max difference | 0.000 mm / 0.000 mm / 1e-6 | 0.000 mm / 0.002 mm / 1e-6 | 0.000 mm / 0.000 mm / 1e-6 |
| `sync`: gripper, skill, release, latching mismatches | none | none | none |
| `sync`: WASM vs native physics, one 0.1 s tick (bread / EE) | 0.005 / 0.002 mm | 0.07 / 0.01 mm | 0.005 / 0.0001 mm |
| `free`: first divergence | left bread dropped on the plate, t = 20 s (12 mm) | none (bread ≤ 0.2 mm) | left bread dropped, t = 22 s (5 mm) |
| `free`: outcome Python / browser | ✓✓ / ✓✓ | ✓✓ / ✓✓ | ✓✗ / ✓✗ |

The commands are identical in both modes; the physics drifts by micrometres per tick until a bread is
dropped (contact chaos), with the same outcome. In `full` mode (Trossen) the browser's perception
agrees with Python's to 0.3 mm (bread centre), 0.2° (long axis) and 0.8 mm (plate), and the ONNX graph
matches `ICTPolicy.infer` to 6e-7 on identical inputs, but the policy-input images differ (PSNR 21 dB:
the viewer's studio backdrop and the table texture's mapping differ from MuJoCo's render), so the
re-plans and the trajectories differ from t = 0.9 s on (same outcome on this scene).

## Self-tuning loop (`autotune/`)

An agent-in-the-loop population search over simulator and low-level controller
parameters, after the population loop of
[recursive-tamp](https://github.com/ColinYu1/recursive-tamp) (re-implemented here):

```
repeat:
  candidates  <- rank-weighted diagonal CEM (unit cube, per-parameter bounds)
  evaluate every candidate on the SAME randomized scenes (common random numbers)
  evidence    <- success, failure modes (grasp_missed / dropped_in_transit / rim),
                 Spearman(parameter, score), bound saturation
  decision    <- continue_local | retune_search | revise_global | validate_and_stop
  revise_global -> inbox/request_NNN.json; an agent (e.g. Claude Code) answers with
                   inbox/response_NNN.json (new bounds, frozen values, candidates, stop)
```

```bash
python simulation/scripts/autotune.py --name my_rig --pop 8 --scenes 14 --workers 12
```

Searched parameters (`autotune/params.py`): the controller knobs of
`cfg/inference/serve_bread/Control.yaml` (EMA, look-ahead, stride, goal-time factor,
safety cage, grasp threshold and look-ahead window, idle-done release, forced lift),
and the TCP offset. Physics is not searched: its parameters come from measurements and the
literature (`docs/physics_params.md`); `autotune.params.PHYSICS` (bounded by those ranges) is for
sensitivity studies only. (Runs up to `soft_v5` also searched the crumb modulus, damping, crust
friction and gripper force cap; the shipped `configs/tuned/*.json` contain no physics.) Then

```bash
python simulation/scripts/autotune_validate.py --name my_rig --top 4 --min_iter 5 --n 32
```

re-evaluates the best candidates and the production settings on the same held-out scenes
and writes `final.json` (what `build_web.py --params` uses; the shipped results are copied
to `configs/tuned/`).

**Runs `soft_v4` → `soft_v5` (soft bread, two arms; per-arm success on the same 32 held-out scenes):**

| | per-arm | right (croissant) | left (danish) | both served |
|---|---|---|---|---|
| production `Control.yaml`, near-elastic crumb | 28 % | 28 % | 28 % | — |
| `soft_v4` tuned, near-elastic crumb | 58 % | 84 % | 31 % | — |
| production, lossy crumb + physical success | 41 % | 41 % | 41 % | 13 % |
| `soft_v4` tuned, lossy crumb | 66 % | 91 % | 41 % | 41 % |
| **`soft_v5` tuned (adds tool pitch), lossy crumb** | **70 %** | **91 %** | **50 %** | **50 %** |

`soft_v5`: driver goal time 0.37 s (`duration_smooth_param` 3.7), EMA 0.37, look-ahead 40,
tool point 0.9 cm deeper and the gripper held 7.6° more level than the commanded EE pose,
crumb E = 17.5 kPa, μ = 0.95, 13.8 N. Per robot (24 held-out scenes, own tuning runs):

| | production | tuned | calibration found |
|---|---|---|---|
| Franka Panda | 25 % | 33 % | tool point 1.25 cm *shallower* (keeps the palm off the bread), pitch 2.5° |
| UR5e + 2F-85 | 31 % | 38 % | tool point 1.3 cm deeper, pitch 5°, 21 N |

`soft_v4`'s tuned controller changed three things, all pointing at the grasp-height mechanism above:
the driver goal time drops from 1.2 s to 0.49 s (`duration_smooth_param` 12 → 4.9) and the
position EMA from 0.5 to 0.29, so the arm follows the descending plan instead of trailing it,
and the tool point moves 0.9 cm deeper along the (downward-pitched) approach
(`tcp_offset` −0.009). Everything else stays near production (look-ahead 46, grasp
threshold 0.58, `any` closing). The bread physics stays inside real-crumb ranges
(E = 17 kPa, μ = 0.92, 12.7 N). The left arm stays weak: the danish is 7.4 cm wide against
an 8.8 cm maximum gripper opening, and the left arm starts from the real rig's asymmetric
pose and sees a mirrored image. The agent's decisions (`inbox/response_*.json`) record the
reasoning between batches.

**Other robots.** The policy grasps like a human hand: end-on along the bread, fingertips
3–4.5 cm past its centre, approach pitched 20–35° down. The Trossen's 7.9 cm finger plates
(8.4 cm from fingertip to carriage, 6 cm tall) straddle half a croissant. The other hands
are built differently (measured in the EE frame of each model):

| | finger pads (approach × height) | fingertip → palm | what goes wrong with the human-style grasp |
|---|---|---|---|
| Trossen WidowX AI | 7.0 × 6.0 cm plates | 8.4 cm | — |
| Franka Hand | 1.7 × 1.7 cm pads on 5 cm fingers | ~5 cm | the palm housing lands on the bread and squashes it; thin fingers can tunnel into the crumb |
| Robotiq 2F-85 | 3.8 × 2.2 cm pads | 5.5 cm | shallow: a high pinch catches only the crust |

A second, simulation-specific problem: small, sharp-cornered pads (2F-85, Franka) can tunnel
into a surface-only flex while squeezing, and the bread then stays hooked on the opened
gripper and is carried away (most "dropped in transit" failures were this). A volumetric flex
(`physics.soft.dim: 3`, tetrahedra to the bread's centre) keeps the pads out: UR5e 17 % → 54 %
per arm on 12 scenes. It also makes dropped bread bouncier and costs the Trossen rig ~10
points, so it is set for the UR5e only (`per_robot:`); a finer collision skin
(`physics.soft.surface: fine`) helped less.

So each hand gets its own calibration, as when moving a policy to a new gripper:
`control.tool_pitch_deg` holds the gripper more level than the commanded EE pose (about the
spread axis, through the tool point) and `control.tcp_offset` moves the tool point along
the approach; together with the tracking speed and the gripper's servo they are tuned per
robot (`autotune_runs/franka_v1`, `ur5e_v1`). Per-robot values go in `per_robot:` of the
config.

Earlier findings on the rigid-bread proxy: the production grasp timing (`any` over the whole
chunk) is near-optimal there, and most failures were grasps, not placing.

## Web viewer (`web/`)

```bash
python simulation/scripts/export_onnx.py                   # policy for Live mode
python simulation/scripts/rebuild_site.py --workers 2      # all site configs, each with its profile; pack; videos
python simulation/scripts/build_web.py --configs ur5e:realsense_d405:bimanual:10:tuned   # one config
python simulation/web/tools/serve.py 8791 simulation/web --gzip   # http://127.0.0.1:8791, like GitHub Pages
python simulation/scripts/deploy_web.py ../humanego-ai.github.io/sim --teaser
```

The viewer replays recorded `qpos` (30 Hz, interpolated to the display rate) through the
MuJoCo kinematic tree in three.js, with the policy's predicted 50-step trajectory, the
perception keypoints riding on the objects, and the exact images the policy saw.

The manifest (`data/manifest.json`, `humanego_sim/web_manifest.py`) is generated: the results
table counts the page's own rollouts, the held-out numbers and the tuning note come from
`configs/tuned/*.json`. After a build, `humanego_sim/web_pack.py` packs the site in place:
rollouts as bit-exact binary tracks (`data/rollouts/<id>.bin`, decoded by `web/js/rollout.js`),
policy-input videos at crf 30, quantized robot GLBs (KHR_mesh_quantization), WebP textures;
the soft breads keep float positions because the viewer deforms them on the CPU.
`deploy_web.py` ships only what the packed site loads.

Tools in `web/tools/`: `record.py` (teaser / robot videos), `measure_load.py` (first-load bytes and
time in headless Chrome, throttled, phone, without WebGPU, Live mode), `compare_shots.py`
(screenshot diff of two builds at fixed rollout times), `check_rollouts.py` (packed rollouts
decode in the browser to their JSON), `shot.py`.

## Layout

```
simulation/
  configs/serve_bread.yaml     rig calibration, scene, physics, controller (real values)
  humanego_sim/
    scene.py robots.py cameras.py   MJCF builder, robot / camera registries
    softbody.py                     soft bread: flex on a trilinear grid + skin
    driver.py env.py                SimArm (Trossen-driver behaviour + IK), physics + rendering
    perception.py controller.py     released perception modules, production controller
    runner.py evaluate.py sampling.py   closed loop, parallel evaluation, initial conditions
    profiles.py                     named config profiles (production | tuned)
    web_export.py live_export.py replay.py glb_tools.py
    web_manifest.py web_pack.py     generated web manifest; packing for the web
  autotune/                    population search + decision rules + agent inbox
  scripts/                     build_assets, bread_gen, run_episode, evaluate, autotune, grasp_test,
                               diagnose_episode, fetch_robots, fit_robot_poses, export_onnx,
                               build_web, rebuild_site, deploy_web, smoke_test
  configs/tuned/<robot>.json   shipped tuned controller + held-out scores (--profile tuned)
  tests/                       pytest suite (CPU subset in CI: .github/workflows/sim-tests.yml)
  docs/upgrade/                engineering logs of the upgrade rounds
  web/                         index.html, css/, js/ (viewer.js, app.js, rollout.js, live/), tools/
```

## Credits

Robot models: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Trossen WidowX AI, Franka Panda, UR5e, Robotiq 2F-85). Breads, plate and tabletop are
procedural (`scripts/bread_gen.py`, `scripts/build_assets.py`).
Physics: [MuJoCo](https://mujoco.org); in-browser inference: ONNX Runtime Web; rendering: three.js.
