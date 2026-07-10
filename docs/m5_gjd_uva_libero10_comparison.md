# M5 GJD vs UVA LIBERO-10 Comparison

This page records the June 2026 comparison between Open-WAM M5 generalist joint
denoising (GJD) and the UVA baseline in `previous_works/unified_video_action`.

The main comparison uses the merged #168 HAIC M5 GJD artifacts. The checkpoint
family is the mode-token counterfactual M5 GJD run trained with joint, IDM, and
FDM modes. The simulator success score comes from the mirrored HAIC evaluation
summary. Offline FDM/IDM metrics were then run locally from the copied step-40000
checkpoint on `simurgh4`.

## Compared Artifacts

| Model | Source artifact | Eval set |
| --- | --- | --- |
| M5 GJD mode-token CF step40000 | `haic_mode_cf_step40000_500ep_success_at_k_20260611_234808/success_at_k_summary.json` | LIBERO-10, 500 rollouts, success@k summary |
| M5 GJD mode-token CF step40000 | `checkpoints/haic_mode_token_cf_m5_gjd_step40000/checkpoint_step_40000` | LIBERO-10 offline FDM/IDM, 20 selected windows |
| M5 GJD mode-token CF step40000 | `checkpoints/haic_mode_token_cf_m5_gjd_step40000/checkpoint_step_40000` | Encoded LIBERO-10 counterfactual validation latents, 20 target-only branch samples |
| UVA baseline | `previous_works/unified_video_action/outputs/uva_sim_libero10_test5_20260608_002653/eval_log_libero10.ckpt.json` | LIBERO-10, 10 tasks x 5 seeds |
| UVA baseline | `previous_works/unified_video_action/outputs/uva_modes_libero10_val10/uva_mode_eval_metrics.json` | LIBERO-10 offline modes, 10 batches x batch size 2 |

The Open-WAM and UVA task orders are not the same. Results below are aligned by
task name/language, not by numeric task id.

## Result Summary

| Comparison | Success |
| --- | ---: |
| M5 GJD mixed mode-token CF, task-aligned episodes 0-4 | 45/50 = 90.0% |
| UVA baseline, same 10 tasks x 5 seeds | 38/50 = 76.0% |
| M5 GJD mixed mode-token CF, full HAIC eval success@1 | 461/500 = 92.2% |
| M5 GJD mixed mode-token CF, full HAIC eval success@2 | 461/500 = 92.2% |
| M5 GJD mixed mode-token CF, full HAIC eval success@3 | 461/500 = 92.2% |

No failed first-attempt M5 rollout was recovered by retry in the HAIC
success@k tree.

## Offline FDM/IDM Metrics

The corrected debug-video comparison now uses one source contract per row and
then projects both model outputs into the same metric space.

Paths below use `<EVAL_ROOT>` for the local evaluation artifact root and
`<OPEN_WAM_ROOT>` for the repository checkout. These are intentionally
placeholders; the numeric summaries are the portable part of this comparison.

- Real-demo rows use Open-WAM's real-demo LeRobot latent/action source. Open-WAM
  consumes those latents/actions natively. UVA receives its matched native HDF5
  `agentview` frames/actions/states. Strict FDM scoring should use the matched
  raw HDF5 `agentview_rgb` target, not the Open-WAM decoded-latent target,
  because the WAN decoder itself introduces reconstruction error.
- Counterfactual rows use the encoded CF validation sidecars. Open-WAM consumes
  the CF latents/actions natively. UVA receives the same raw `agentview` frames
  plus Open-WAM raw7 actions converted into UVA's 10D absolute-action input
  convention. Both models are scored against the same raw CF target.
- FDM video metrics score the shared UVA-selected future timesteps
  `t+4,t+8,t+12,t+16` in raw-orientation `agentview`, resized to `128 x 128`,
  `[0,1]` RGB.
- FVD is different from the frame-wise FDM metrics. UVA can only provide its
  native sparse future keyframes here, so those four frames are repeated before
  I3D as in UVA's evaluator. Open-WAM can decode the full dense 16-frame future
  chunk, so Open-WAM FVD uses those 16 continuous frames directly. FVD clips are
  always `uint8 agentview` at `128 x 128`, but the orientation is model-native:
  Open-WAM clips use decoded-agentview orientation, raw HDF5 targets are flipped
  into that orientation for Open-WAM FVD, and UVA clips use UVA's native FVD
  orientation/keyframe projection.
- IDM metrics score dense future low-level action steps `0..15` in Open-WAM
  raw7 delta-OSC. UVA predictions are converted back from 10D absolute EEF
  rot6d with the same state-conditioned inverse used to build its inputs.
- The preferred physical IDM metric is simulator-compounded EEF trajectory
  error. For each window, restore the exact LIBERO simulator state at the IDM
  start, execute the GT Open-WAM raw7 actions to define the target trajectory,
  restore the same state again, execute the predicted raw7 actions, and compare
  the resulting post-action EEF states. This avoids guessing an analytic
  `state + action` rule for OSC commands.
- Debug-video commands do not compute FVD by default. Pass `--compute-fvd` only
  when the UVA I3D checkpoint is available and the extra metric pass is desired.
  FVD is distributional and sample-sensitive; do not compare a 10-clip
  diagnostic directly against the paper-scale UVA number around `51`.

Corrected debug-video run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/debug_gjd_uva_mode_videos.py \
  --episodes 10 \
  --sources real_demo,counterfactual \
  --horizon-frames 4 \
  --skip-sim-replay \
  --run-id selected_fdm_idm_real_cf10_openwam_native_20260615 \
  --runtime-device cuda:0 \
  --decode-device cuda:0
```

IDM pose-space debug run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/debug_gjd_uva_mode_videos.py \
  --episodes 10 \
  --sources real_demo,counterfactual \
  --horizon-frames 4 \
  --skip-sim-replay \
  --run-id selected_fdm_idm_pose_real_cf10_openwam_native_20260615 \
  --runtime-device cuda:0 \
  --decode-device cuda:0
```

Output:

```text
<EVAL_ROOT>/debug_videos/selected_fdm_idm_real_cf10_openwam_native_20260615/
<EVAL_ROOT>/debug_videos/selected_fdm_idm_pose_real_cf10_openwam_native_20260615/
```

Simulator-compounded IDM reruns:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
uv run python scripts/debug_gjd_uva_mode_videos.py \
  --episodes 10 \
  --sources real_demo \
  --models openwam,uva \
  --real-demo-contract matched_native_pair \
  --horizon-frames 4 \
  --skip-sim-replay \
  --compute-idm-sim-metrics \
  --run-id idm_sim_matched_native_plus_uva_native_real10_20260618 \
  --runtime-device cuda:0 \
  --decode-device cuda:0
```

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
uv run python scripts/debug_gjd_uva_mode_videos.py \
  --episodes 10 \
  --sources real_demo,counterfactual \
  --models openwam,uva \
  --real-demo-contract hdf5_semantic_pair \
  --horizon-frames 4 \
  --skip-sim-replay \
  --compute-idm-sim-metrics \
  --run-id idm_sim_openwam_uva_native_cf10_20260618 \
  --runtime-device cuda:0 \
  --decode-device cuda:0
```

Output:

```text
<EVAL_ROOT>/debug_videos/idm_sim_matched_native_plus_uva_native_real10_20260618/
<EVAL_ROOT>/debug_videos/idm_sim_openwam_uva_native_cf10_20260618/
```

The matched-native real-demo run produced 50 videos and no skipped common or
UVA-native simulator-compounded IDM rows. The counterfactual same-window run
produced 40 videos and no skipped common or UVA-native simulator-compounded IDM
rows. Real-demo windows restore `demo["states"][raw_action_start]` with
`env.set_init_state`. Counterfactual windows restore the stored flattened
`simulator_state` with `env.sim.set_state_from_flattened`.

### Simulator-Compounded IDM Table

This is the controller-level IDM comparison for Open-WAM and a diagnostic for
UVA. The target is not an analytic EEF pose label. For each model, the target is
the EEF trajectory produced by LIBERO after executing that model's native GT
action contract from the exact same simulator state. Open-WAM uses native raw7
delta-OSC actions with the delta controller. The UVA rows below execute native
absolute EEF actions with `control_delta=False` through the same direct
Open-WAM-side LIBERO helper used for this diagnostic. That direct helper is not
paper-comparable for UVA; the raw-HDF5 replay check below shows that UVA's own
Robomimic env metadata path is required for paper-scale simulator grounding.
The table reports all 16 dense post-action steps, the selected UVA/FDM-aligned
steps `t+4,t+8,t+12,t+16`, and the final post-action drift.

Lower is better.

| Source | Model | Count | Mean EEF pos cm | Selected EEF pos cm | Final EEF pos cm | Mean rot deg | Selected rot deg | Final rot deg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real demo | Open-WAM native raw7 controller | 10 | 0.42 +/- 0.36 | 0.49 +/- 0.41 | 0.75 +/- 0.60 | 1.00 +/- 0.71 | 1.13 +/- 0.72 | 1.63 +/- 0.83 |
| Real demo | UVA Robomimic native abs7 controller | 10 | 1.82 +/- 0.71 | 2.00 +/- 0.78 | 2.20 +/- 1.08 | 1.56 +/- 0.52 | 1.69 +/- 0.55 | 1.99 +/- 0.75 |
| Real demo | UVA native absolute controller | 10 | 1.76 +/- 2.28 | 1.82 +/- 2.29 | 1.88 +/- 2.52 | 8.57 +/- 14.70 | 9.63 +/- 16.47 | 12.09 +/- 20.59 |
| Counterfactual | Open-WAM native raw7 controller | 10 | 0.61 +/- 0.47 | 0.68 +/- 0.53 | 0.80 +/- 0.65 | 0.81 +/- 0.52 | 0.94 +/- 0.60 | 1.33 +/- 0.92 |
| Counterfactual | UVA native absolute controller | 10 | 1.55 +/- 2.19 | 1.81 +/- 2.53 | 2.53 +/- 3.51 | 5.56 +/- 8.13 | 6.22 +/- 9.00 | 6.96 +/- 11.16 |

Corrected UVA-native IDM output:

```text
<EVAL_ROOT>/uva_idm_sim_robomimic_env_real10_20260618.json
```

The corrected UVA row uses the same 10 matched-native real-demo identities as
the Open-WAM row, but executes UVA's released IDM predictions through UVA's own
Robomimic env metadata path. This is the appropriate simulator-grounded route
after the raw-HDF5 replay self-consistency check below.

UVA raw-HDF5 replay self-consistency:

Before interpreting UVA simulator-grounded IDM, I checked whether the released
UVA HDF5 data itself replays accurately under UVA's own simulator stack. The
check restores `demo["states"][raw_action_start]`, executes the raw HDF5
absolute EEF actions with UVA's Robomimic env metadata and
`controller_configs.control_delta=False`, and compares the resulting EEF states
against HDF5 `obs/ee_states` over the same 16-action windows.

Output:

```text
<EVAL_ROOT>/uva_hdf5_gt_replay_self_consistency_uva_env_20260618.json
```

| Contract | Count | Mean EEF pos cm | Selected EEF pos cm | Final EEF pos cm | Mean rot deg | Selected rot deg | Final rot deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UVA HDF5 GT replay through UVA Robomimic env | 10 | 0.73 +/- 0.31 | 0.76 +/- 0.30 | 0.69 +/- 0.29 | 0.64 +/- 0.27 | 0.64 +/- 0.29 | 0.67 +/- 0.42 |
| UVA HDF5 GT replay through direct Open-WAM `ControlEnv` shortcut | 10 | 13.73 +/- 4.48 | 16.89 +/- 5.68 | 25.06 +/- 9.30 | 36.70 +/- 9.43 | 46.01 +/- 12.43 | 74.20 +/- 25.89 |

The UVA Robomimic-env replay matches the scale reported by the UVA paper
(`~0.75 cm`, `~1 deg`) as a raw-data replay floor. The direct Open-WAM
`ControlEnv` shortcut does not pass this check because LIBERO's `ControlEnv`
constructor ignores the HDF5 Robomimic controller config and does not reproduce
UVA's exact env metadata path. Paper-comparable UVA simulator IDM should
therefore use the UVA Robomimic env contract above.

Interpretation:

- Open-WAM remains better on both real-demo and counterfactual under native
  simulator grounding.
- UVA's corrected Robomimic-env IDM row is much more stable than the direct
  `ControlEnv` diagnostic, especially for rotation. On this 10-window matched
  comparison, Open-WAM is still lower: `0.42 cm` vs `1.82 cm` mean position and
  `1.00 deg` vs `1.56 deg` mean rotation.
- The direct UVA simulator diagnostic above is retained only to document the
  route mismatch; it should not be used as the paper-comparable UVA simulator
  metric.
- HDF5 absolute actions must not be executed directly as Open-WAM rollout
  actions. A quick controller check showed direct HDF5 actions drift by roughly
  `16 cm` mean / `30 cm` final over 16 steps, while shifted Open-WAM raw7 actions
  reproduce the GT raw-action rollout at roughly `0.22 cm` mean / `0.50 cm`
  final. This is why the IDM comparison first projects both predictions into
  Open-WAM raw7, then lets LIBERO define the compounding.
- For reference, forcing UVA through the Open-WAM raw7 bridge gave real-demo
  `3.24 +/- 1.27 cm` mean / `6.07 +/- 2.94 cm` final and counterfactual
  `2.39 +/- 0.78 cm` mean / `4.60 +/- 1.93 cm` final. These are now bridge
  diagnostics, not UVA-native simulator scores.
- For reference, forcing Open-WAM through the HDF5 semantic-pair bridge gave
  `3.74 +/- 1.34 cm` mean / `6.93 +/- 2.08 cm` final real-demo position error.
  That is a bridge stress test, not the preferred Open-WAM-native simulator
  metric.

### Real-Demo FDM Table

| Model | Count | Shared agentview MSE | MAE | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open-WAM M5 GJD step40000 | 10 | 0.00063 +/- 0.00032 | 0.00998 | 32.85 | 0.9929 |
| UVA baseline | 10 | 0.01747 +/- 0.00646 | 0.06303 | 18.08 | 0.7757 |

### Simulator-Rendered FDM Target Diagnostic

This target is generated by restoring the exact LIBERO simulator state at the
FDM context boundary, executing the model-native GT action contract, rendering
future `agentview` observations, and comparing each model's predicted video to
those rendered observations. Open-WAM uses raw7 delta-OSC actions with the delta
controller. UVA uses native absolute EEF actions with `control_delta=False`.

This path is intentionally diagnostic-only for now. It is not reported as a
headline FDM comparison because the real-demo replay target can diverge from the
stored dataset images, and the current counterfactual interpretation depends on
which native action contract produced the rendered target. Use it to debug
simulator replay parity before treating any numbers as model-quality metrics.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
uv run python scripts/debug_gjd_uva_mode_videos.py \
  --episodes 10 \
  --sources real_demo,counterfactual \
  --models openwam,uva \
  --real-demo-contract matched_native_pair \
  --horizon-frames 4 \
  --skip-sim-replay \
  --compute-fdm-sim-metrics \
  --run-id fdm_sim_target_real_cf10_20260618 \
  --runtime-device cuda:0 \
  --decode-device cuda:0
```

The CSV `shared_fdm_sim_metrics.csv` records `sim_target_vs_stored_*`
diagnostics so replay mismatch can be separated from model prediction error.
Do not include simulator-rendered FDM numbers in the comparison tables until
the replay target is proven pixel-aligned with the stored training/eval data.

### Real-Demo IDM Table

| Model | Count | Shared raw7 MSE | MAE | L2 |
| --- | ---: | ---: | ---: | ---: |
| Open-WAM M5 GJD step40000 | 10 | 0.00054 +/- 0.00031 | 0.01522 | 0.05396 |
| UVA baseline | 10 | 3.62255 +/- 1.94424 | 1.32359 | 4.87628 |

### Real-Demo IDM EEF Pose Table

This table converts both IDM outputs from Open-WAM raw7 delta OSC into absolute
EEF target pose using the matched dense `observation.state` sequence, then
scores the same selected future indices used by FDM: `t+4,t+8,t+12,t+16`.
`Target pos` compares predicted absolute EEF command target against the GT
absolute EEF command target. `State pos` compares predicted absolute EEF command
target against the observed future EEF state at the same selected timestep.

This is an Open-WAM-source adaptation test, not UVA's paper-native LIBERO IDM
metric. UVA is fed Open-WAM-decoded `agentview` frames and Open-WAM raw7 actions
converted into UVA's 10D absolute-action convention. UVA's own HDF5-native IDM
sanity check is reported below.

Lower is better.

| Model | Count | Target pos L2 cm | State pos L2 cm | Target rot deg | State rot deg | Gripper abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open-WAM M5 GJD step40000 | 10 | 0.24 +/- 0.05 | 2.64 +/- 1.11 | 0.56 +/- 0.25 | 2.74 +/- 1.49 | 0.0034 |
| UVA baseline, Open-WAM-adapted input | 10 | 14.59 +/- 8.83 | 14.35 +/- 8.41 | 102.84 +/- 13.47 | 103.04 +/- 13.27 | 1.1027 |

### UVA Native LIBERO IDM Sanity Check

To check the paper-scale claim, I also ran the released UVA checkpoint through
UVA's own HDF5 dataloader and `inverse_model` path. This uses the native input
contract: original HDF5 `agentview_rgb`, CLIP text tokens, and HDF5 absolute
EEF actions converted to 10D rotation-6D labels by UVA's dataset.

Commands:

```bash
PYTHONPATH=previous_works/unified_video_action:previous_works/unified_video_action/.uva_pydeps \
CUDA_VISIBLE_DEVICES=0 uv run python previous_works/unified_video_action/eval_uva_modes.py \
  -c previous_works/unified_video_action/checkpoints/libero10.ckpt \
  -o <EVAL_ROOT>/uva_native_inverse_check_20260615 \
  -d cuda:0 \
  --dataset-dir <OPEN_WAM_ROOT>/previous_works/unified_video_action/data/libero_10 \
  --modes inverse_model \
  --split train \
  --batch-size 2 \
  --num-workers 0 \
  --max-batches 10 \
  --max-vis-examples 0
```

Native UVA position-only IDM error:

| Split | Batches | All future pos L2 | Selected `t+4,t+8,t+12,t+16` pos L2 | First-4 pos L2 | Native action L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 10 | 2.27 +/- 0.42 cm | 2.36 +/- 0.48 cm | 2.44 +/- 0.50 cm | 0.1310 |
| val | 10 | 2.52 +/- 0.18 cm | 2.58 +/- 0.44 cm | 2.43 +/- 0.28 cm | 0.0583 |

This is close to the paper's reported centimeter scale and confirms that the
large `14-16 cm` number above measures cross-contract adaptation, not UVA's
native LIBERO IDM capability.

### Combined Native-Physical Aligned Summary

The cleanest combined comparison keeps each model on the input/action contract
it was trained on, then reports physical or scale-normalized quantities:

- UVA real-demo uses its native HDF5 `agentview_rgb` and absolute EEF 10D
  action labels.
- Open-WAM real-demo uses its native LeRobot latent/action rows from the M5 GJD
  eval.
- Cross-contract Open-WAM raw7 metrics are reported only when the target action
  adapter roundtrip validates. This rejects the HDF5 real-demo raw7 projection
  and accepts the counterfactual Open-WAM OSC-to-UVA projection.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_uva_openwam_aligned.py \
  --max-real-samples 10 \
  --max-cf-samples 10 \
  --cf-protocols native_consecutive \
  --run-id native_physical_real_cf10_20260615 \
  --device cuda:0
```

Output:

```text
<EVAL_ROOT>/aligned_uva_openwam/native_physical_real_cf10_20260615/
```

UVA native-contract metrics:

| Source | Count | FDM MSE `[-1,1]` | FDM MSE `[0,1]` equiv | IDM pos cm selected | IDM rot deg selected | Action L2 native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| real-demo HDF5 native | 10 | 0.00665 | 0.00166 | 2.50 | 2.38 | 0.115 |
| counterfactual adapted to UVA | 10 | 0.04690 | 0.01173 | 17.82 | 84.51 | 2.192 |

Open-WAM native-contract metrics on the selected M5 GJD eval rows:

| Source | Count | FDM MSE `[0,1]` first4 | FDM MSE `[0,1]` full16 | IDM raw7 MSE first4 | IDM raw7 MSE full16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| real-demo LeRobot native | 10 | 0.000512 | 0.000681 | 0.000274 | 0.000278 |
| counterfactual encoded native | 10 | 0.01810 | 0.01751 | 0.08543 | 0.05443 |

Adapter validity:

| Source | Count | Target roundtrip MSE | Target roundtrip max abs | Open-WAM raw7 score valid? |
| --- | ---: | ---: | ---: | --- |
| real-demo HDF5 -> Open-WAM raw7 | 10 | 1.4025 | 3.1438 | no |
| counterfactual Open-WAM raw7 -> UVA abs10 -> raw7 | 10 | 5.59e-8 | 0.00106 | yes |

Interpretation: UVA is healthy under its native real-demo contract, while
Open-WAM is healthy under its native real-demo contract. The single-source
Open-WAM-adapted UVA real-demo score is still useful as a stress test, but it is
not the primary fair score because the HDF5 absolute-action target does not
roundtrip into Open-WAM raw7.

### HDF5 Semantic-Pair Input Comparison

I then ran a stricter semantic-pair comparison. The source window is the same
LIBERO task/demo/timestep semantics for both models:

- UVA receives its native HDF5 `agentview_rgb` and HDF5 absolute EEF action
  window.
- Open-WAM receives the corresponding HDF5 `agentview_rgb | eye_in_hand_rgb`
  canvas encoded through the Open-WAM visual tower, plus HDF5 state-derived
  proprio.
- FDM is scored against the same HDF5 agentview target frames.
- IDM is projected into absolute EEF target position using the same HDF5 current
  state sequence.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/debug_gjd_uva_mode_videos.py \
  --episodes 10 \
  --sources real_demo \
  --real-demo-contract hdf5_semantic_pair \
  --horizon-frames 4 \
  --skip-sim-replay \
  --run-id hdf5_semantic_pair_real10_axisfix_20260617 \
  --runtime-device cuda:0 \
  --decode-device cuda:0
```

Output:

```text
<EVAL_ROOT>/debug_videos/hdf5_semantic_pair_real10_axisfix_20260617/
```

FDM on shared HDF5 agentview frames:

| Model | Count | MSE `[0,1]` | MAE | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open-WAM M5 GJD step40000 | 10 | 0.11136 +/- 0.07604 | 0.25610 | 10.64 | -0.0685 |
| UVA baseline | 10 | 0.00267 +/- 0.00078 | 0.02080 | 26.14 | 0.9665 |

IDM projected to shared absolute EEF target position:

| Model | Count | Target pos L2 cm | Target rot deg | State pos L2 cm | Raw7 MSE diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open-WAM M5 GJD step40000 | 10 | 2.27 +/- 0.91 | 2.23 +/- 1.11 | 1.32 +/- 1.03 | 0.228 |
| UVA baseline | 10 | 2.50 +/- 0.67 | 3.16 +/- 0.86 | 4.07 +/- 1.44 | 0.0743 |

This table is the source of the earlier "~2 cm GJD IDM" statement. It is useful
as a semantic-pair diagnostic, but it is not the final Open-WAM-native GJD IDM
score. In this contract, Open-WAM is fed HDF5-derived visual/state inputs and is
then projected into the HDF5 absolute EEF target-position metric. That makes it
closer to UVA's native HDF5 contract, but it is still a cross-contract adapter
test for Open-WAM.

The previous `90.32 +/- 0.65 deg` Open-WAM rotation row came from this diagnostic
before applying the LIBERO HDF5 action-frame bridge. Native LIBERO HDF5 absolute
actions are offset from the EEF/state rotation frame by about `+90 deg` around
Z; the current run converts those actions through the same LIBERO-specific bridge
used by the counterfactual UVA adapter. After that fix, Open-WAM's HDF5
semantic-pair rotation is `2.23 +/- 1.11 deg`. This table is still a bridge
stress test, not the native Open-WAM benchmark, because Open-WAM is being forced
onto HDF5 visual/action conventions.

### Matched-Native Single-Agentview Comparison

The most useful real-demo comparison is matched-native: the selected windows
use the same LIBERO task, demo id, raw action start, and future horizon, but
each model receives the representation it was trained on.

- Open-WAM receives its native LeRobot latent/action/proprio row.
- UVA receives its native HDF5 `agentview_rgb`, absolute EEF actions, and
  states.
- FDM outputs and targets are all projected into one camera view:
  `agentview`, raw LIBERO orientation, `128 x 128`, RGB.
- IDM outputs are converted into absolute EEF target position as before.

I verified the source identity before scoring: all 500 Open-WAM LeRobot episodes
map to the corresponding UVA HDF5 task file and `demo_{episode_index % 50}`;
all 500 trajectory lengths match exactly; proprio/state arrays match to
floating-point noise. The action arrays are intentionally different because
Open-WAM stores delta/OSC-style actions while the UVA HDF5 rows store absolute
EEF action targets.

While building this comparison, I found one visual preprocessing issue: Open-WAM
decoded `agentview` is vertically flipped relative to raw HDF5 `agentview`.
The FDM projection now flips Open-WAM decoded streams into raw LIBERO
orientation before resizing to `128 x 128`.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/debug_gjd_uva_mode_videos.py \
  --episodes 10 \
  --sources real_demo \
  --real-demo-contract matched_native_pair \
  --horizon-frames 4 \
  --skip-sim-replay \
  --run-id matched_native_pair_agentview128_oriented_real10_20260616 \
  --runtime-device cuda:0 \
  --decode-device cuda:0
```

Output:

```text
<EVAL_ROOT>/debug_videos/matched_native_pair_agentview128_oriented_real10_20260616/
```

Matched-native FDM, each model scored against its native GT projected to
single-agentview `128 x 128`:

| Model | Count | MSE `[0,1]` | MAE | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open-WAM M5 GJD step40000 | 10 | 0.00063 +/- 0.00032 | 0.00998 | 32.85 | 0.9929 |
| UVA baseline | 10 | 0.00267 +/- 0.00078 | 0.02080 | 26.14 | 0.9665 |

Matched-native IDM projected to absolute EEF target pose:

| Model | Count | Target pos L2 cm | Target rot deg | State pos L2 cm | State rot deg | Raw7 MSE diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open-WAM M5 GJD step40000 | 10 | 0.24 +/- 0.05 | 0.56 +/- 0.25 | 2.64 +/- 1.11 | 2.74 +/- 1.49 | 0.00054 |
| UVA baseline | 10 | 2.50 +/- 0.67 | 3.16 +/- 0.86 | 4.07 +/- 1.44 | 90.18 +/- 2.06 | 0.0745 |

Native GT bridge after orientation normalization:

| Bridge | Count | MSE `[0,1]` | MAE | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open-WAM decoded GT vs UVA raw HDF5 GT | 10 | 0.00249 +/- 0.00072 | 0.02787 | 26.54 | 0.9710 |

This bridge explains the previous contradictory FDM result. Before orientation
normalization, Open-WAM decoded GT and HDF5 raw GT were compared upside-down,
which made the bridge look very poor. After normalizing both into one
`agentview 128 x 128` space, the native visual targets are close enough for the
matched-native FDM table to be interpretable.

### Historical Raw HDF5 Target FDM Diagnostic

This scalar diagnostic used a stricter raw-target contract:

- Open-WAM keeps its native LeRobot latent/action/proprio inputs.
- UVA keeps its native HDF5 `agentview/actions/states` inputs.
- Both FDM outputs are scored against the same matched raw HDF5 `agentview_rgb`
  target frames at the UVA-selected future timesteps.

This matters because the Open-WAM LeRobot copy does not store raw images; it
stores WAN latents. Decoding those latents back to RGB is useful for debugging
but should not be treated as the ground-truth image target when comparing to UVA
on raw HDF5 observations.

Historical Open-WAM raw-HDF5-target run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
.venv/bin/python3 scripts/eval_openwam_fdm_fvd.py \
  --real-demo-contract openwam_native_hdf5_video_target \
  --max-videos 10 \
  --output-dir <EVAL_ROOT>/openwam_native_hdf5_video_target_fvd_real10_20260616 \
  --runtime-device cuda:0 \
  --decode-device cuda:0 \
  --fvd-device cuda:0 \
  --overwrite \
  --progress-interval 1
```

Output:

```text
<EVAL_ROOT>/openwam_native_hdf5_video_target_fvd_real10_20260616/
```

Scalar FDM metrics from the saved aligned clips:

| Model | Target | Count | MSE `[0,1]` | MAE | PSNR | SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Open-WAM M5 GJD step40000 | raw HDF5 GT | 10 | 0.00152 +/- 0.00012 | 0.02338 +/- 0.00092 | 28.42 | 0.9691 |
| UVA baseline | raw HDF5 GT | 10 | 0.00267 +/- 0.00078 | 0.02080 +/- 0.00315 | 26.14 | 0.9665 |

On this 10-clip scalar diagnostic, Open-WAM is better on MSE, PSNR, and SSIM;
UVA is better on MAE.

Do not use this historical raw-HDF5 command as the current Open-WAM FVD
benchmark. Current FVD uses dense model-native clips; raw-HDF5 targets remain a
bridge diagnostic because they include the HDF5-to-WAN-decoded target-domain
gap.

FVD diagnostics:

| Model | Target | Count | Clip frames before repeat | FVD |
| --- | --- | ---: | ---: | ---: |
| UVA paper FVD path | UVA native LIBERO full-dynamics target | 500 | 4 | 51.10 |
| UVA local released checkpoint, `full_dynamic_model,num_iter=8` | UVA native LIBERO full-dynamics target | 500 | 4 | 58.18 |
| UVA local released checkpoint, `dynamic_model,num_iter=8` | UVA native LIBERO action-conditioned FDM target | 500 | 4 | 92.72 |
| Open-WAM M5 GJD step40000 | Open-WAM native decoded target | 500 | 16 | 8.12 |
| Open-WAM M5 GJD step40000 | raw HDF5 GT | 10 | 4 | 485.96 |
| Open-WAM M5 GJD step40000 | raw HDF5 GT | 500 | 4 | 312.63 |

The raw-HDF5 Open-WAM rows are legacy sparse-repeated diagnostics and are now
superseded. They repeated the same four UVA-selected metric frames before I3D,
which hides Open-WAM's dense 16-frame FDM output and also mixes in the
raw-HDF5-to-WAN-decoded target bridge. The current FVD utility stores dense
Open-WAM clips directly, while keeping the four selected frames only for scalar
MSE/SSIM alignment.

The corrected 500-clip Open-WAM dense run is:

```text
runs/openwam_fdm_fvd_dense_native500_20260617/openwam_fdm_fvd.json
```

The corresponding 500-clip dense RGB metrics were MSE `0.000457` and MAE
`0.00815`.

Do not interpret 10-clip or 100-clip FVD as paper-comparable. FVD estimates a
feature distribution and is unstable at small sample counts. Recomputing FVD
from the same stored Open-WAM 500 dense clips produced `7.91` on all clips, but
100-clip subsets ranged from `23.92` to `29.39`:

```text
runs/openwam_fdm_fvd_dense_native500_20260617/subset_fvd_audit_20260618.json
```

The current paper-scale result is that UVA's released checkpoint reproduces the
paper-scale full-dynamics result (`58.18` vs reported `51.10`). The
action-conditioned UVA FDM path is `dynamic_model`; with the same `num_iter=8`
setting its native 500-clip FVD is `92.72`. Open-WAM's dense native
action-conditioned FDM FVD is `8.12` under its native decoded-target contract.

### Full Real/Counterfactual Mode Table

This table uses the corrected one-camera projection and model-native input
contracts:

- Real-demo rows use matched native inputs: Open-WAM native LeRobot
  latents/actions/proprio and UVA native HDF5 `agentview/actions/states`.
- Counterfactual rows use Open-WAM CF sidecars. For UVA, CF frames are first
  converted from Open-WAM decoded orientation back into raw HDF5 `agentview`
  orientation before UVA's own preprocessing. Open-WAM raw7 delta-OSC actions
  are also rotated into UVA's native LIBERO absolute-action frame before UVA
  IDM inference, then rotated back into the Open-WAM EEF/state frame for shared
  scoring. FDM targets/predictions are all projected into raw
  `agentview 128 x 128`.
- FDM metric is RGB MSE/SSIM in that single-agentview space.
- IDM metrics are absolute EEF target-position L2 in centimeters and target
  rotation geodesic error in degrees.

Real-demo output:

```text
<EVAL_ROOT>/debug_videos/matched_native_pair_agentview128_oriented_real10_20260616/
```

Counterfactual output:

```text
<EVAL_ROOT>/debug_videos/cf_agentview128_idm_framefix_20260616/
```

### Final 10-Episode Tables

These are the current final comparison tables for the selected 10 LIBERO-10
episodes/windows. They supersede earlier cross-contract diagnostics in this
note.

IDM, simulator-grounded real demo:

- Open-WAM uses the M5 GJD step-40000 checkpoint under its native raw7
  delta-OSC action contract. For each window, the exact LIBERO simulator state
  is restored, GT raw7 actions are executed to define the target EEF trajectory,
  and predicted raw7 IDM actions are executed from the same state.
- UVA uses the released UVA checkpoint under its native HDF5 abs7 action
  contract. For each matched window, UVA's native HDF5 Robomimic env metadata
  is loaded, `controller_configs.control_delta=False` is set before env
  construction, GT abs7 actions define the target EEF trajectory, and predicted
  UVA IDM abs7 actions are executed from the same HDF5 simulator state.
- Metrics average all 16 dense post-action simulator steps. The selected column
  uses the shared FDM-aligned timesteps `t+4,t+8,t+12,t+16`. Lower is better.

| Metric | Open-WAM native IDM | UVA corrected native IDM |
| --- | ---: | ---: |
| Mean EEF pos | 0.42 +/- 0.36 cm | 1.82 +/- 0.71 cm |
| Selected EEF pos | 0.49 +/- 0.41 cm | 2.00 +/- 0.78 cm |
| Final EEF pos | 0.75 +/- 0.60 cm | 2.20 +/- 1.08 cm |
| Mean rot | 1.00 +/- 0.71 deg | 1.56 +/- 0.52 deg |
| Selected rot | 1.13 +/- 0.72 deg | 1.69 +/- 0.55 deg |
| Final rot | 1.63 +/- 0.83 deg | 1.99 +/- 0.75 deg |

IDM result files:

```text
<EVAL_ROOT>/debug_videos/idm_sim_matched_native_plus_uva_native_real10_20260618/shared_idm_sim_summary.json
<EVAL_ROOT>/uva_idm_sim_robomimic_env_real10_20260618.json
```

FDM, native stored-target:

- Open-WAM real-demo rows use native Open-WAM LeRobot latents/actions/proprio
  and native decoded stored targets. Counterfactual rows use Open-WAM CF
  sidecars and targets.
- UVA real-demo rows use native HDF5 `agentview/actions/states`.
  Counterfactual rows convert Open-WAM CF frames into UVA's raw HDF5
  `agentview` orientation and convert raw7 actions into UVA's native LIBERO
  absolute-action frame before UVA inference.
- MSE/SSIM rows are the selected 10-window frame metrics in single-camera
  `agentview 128 x 128` RGB space. MSE is in `[0,1]`; lower MSE and higher
  SSIM are better.
- FVD is a separate paper-scale real-demo distributional run over 500 clips, not
  the 10-window diagnostic. Open-WAM FVD uses dense native 16-frame clips; UVA
  FDM FVD uses its native sparse keyframe path repeated to the UVA evaluator's
  16-frame I3D input convention. UVA's `58.18` number is the
  `full_dynamic_model` path, not the action-conditioned FDM model.

| Source | Metric | Open-WAM M5 GJD step40000 | UVA baseline |
| --- | --- | ---: | ---: |
| Real demo | MSE `[0,1]` lower | 0.00063 +/- 0.00032 | 0.00267 +/- 0.00078 |
| Real demo | SSIM higher | 0.9929 | 0.9665 |
| Real demo | FVD lower, 500 clips | 8.12 | 92.72 |
| Counterfactual | MSE `[0,1]` lower | 0.01095 +/- 0.00685 | 0.01593 +/- 0.00883 |
| Counterfactual | SSIM higher | 0.8801 | 0.8050 |
| Counterfactual | FVD | not reported; no 500-clip CF FVD run | not reported; no 500-clip CF FVD run |

FDM result files:

```text
<EVAL_ROOT>/debug_videos/fdm_sim_target_real_cf10_20260618/shared_fdm_summary.json
runs/openwam_fdm_fvd_dense_native500_20260617/openwam_fdm_fvd.json
runs/uva_native_fvd_500_dynamic_model_numiter8_20260618/native_fvd_500_dynamic_model_numiter8.json
runs/uva_native_fvd_500_numiter8_20260616/native_fvd_500_numiter8.json  # full_dynamic_model reference
```

### Final 100-Episode Real-Demo Tables

This is the expanded real-demo pass using the same native contracts as the
10-episode table above. The selector samples 10 successful matched windows per
LIBERO-10 task, for 100 total windows. Open-WAM and UVA use the same task text,
Open-WAM LeRobot episode index, UVA HDF5 demo id
`episode_index % 50`, and `raw_action_start` for each row.

Output:

```text
<EVAL_ROOT>/real100_matched_native_openwam_uva_20260618/summary.json
<EVAL_ROOT>/real100_matched_native_openwam_uva_20260618/selections.csv
<EVAL_ROOT>/real100_matched_native_openwam_uva_20260618/fdm_rows.csv
<EVAL_ROOT>/real100_matched_native_openwam_uva_20260618/idm_sim_rows.csv
<EVAL_ROOT>/real100_matched_native_openwam_uva_fvd_20260618/summary.json
<EVAL_ROOT>/real100_matched_native_openwam_uva_fvd_20260618/fdm_rows.csv
<EVAL_ROOT>/real100_matched_native_openwam_uva_fvd_20260618/fdm_fvd_metrics.csv
<EVAL_ROOT>/real100_matched_native_uva_dynamic_numiter8_fvd_20260618/summary.json
```

Selection check: 10 rows per each of the 10 LIBERO-10 tasks.

IDM, simulator-grounded real demo:

- Open-WAM uses native raw7 delta-OSC predicted/GT actions executed through the
  Open-WAM LIBERO controller path from the exact HDF5 simulator state.
- UVA uses native abs7 predicted/GT actions executed through UVA's Robomimic
  env metadata path with `control_delta=False` from the exact HDF5 simulator
  state.
- Metrics average all 16 dense post-action simulator steps. The selected column
  uses `t+4,t+8,t+12,t+16`. Lower is better.

| Metric | Open-WAM native IDM | UVA corrected native IDM |
| --- | ---: | ---: |
| Mean EEF pos | 0.43 +/- 0.65 cm | 2.03 +/- 0.77 cm |
| Selected EEF pos | 0.48 +/- 0.73 cm | 2.21 +/- 0.83 cm |
| Final EEF pos | 0.66 +/- 0.87 cm | 2.36 +/- 1.02 cm |
| Mean rot | 1.10 +/- 4.03 deg | 1.60 +/- 0.55 deg |
| Selected rot | 1.26 +/- 4.59 deg | 1.74 +/- 0.61 deg |
| Final rot | 1.68 +/- 5.63 deg | 1.97 +/- 0.96 deg |

FDM, native stored-target real demo:

- Open-WAM uses native LeRobot latents/actions/proprio and native decoded stored
  targets.
- UVA uses native HDF5 `agentview/actions/states`.
- Metrics use single-camera `agentview 128 x 128` RGB at the selected shared
  timesteps. MSE is in `[0,1]`; lower MSE/MAE and higher PSNR/SSIM are better.
- FVD uses UVA's released I3D feature model. The scalar rows below are the
  100-window matched diagnostic, but FVD is reported from the stable native
  500-clip diagnostic because 100-clip FVD is noisy. For the action-conditioned
  UVA FDM comparison, UVA must run `dynamic_model` with `num_iter=8`; the
  earlier `216.95` matched-100 FVD was a stale `num_iter=1` debug value and
  should not be used. Lower FVD is better.

| Metric | Open-WAM M5 GJD step40000 | UVA baseline |
| --- | ---: | ---: |
| MSE `[0,1]` | 0.00058 +/- 0.00039 | 0.00279 +/- 0.00179 |
| MAE `[0,1]` | 0.00933 +/- 0.00213 | 0.02016 +/- 0.00505 |
| PSNR | 33.27 +/- 2.29 | 26.51 +/- 2.29 |
| SSIM | 0.9934 +/- 0.0030 | 0.9645 +/- 0.0232 |
| FVD, native 500-clip diagnostic | 8.12 | 92.72 |

Matched-100 FVD is useful only as a smoke diagnostic. On the same stored
Open-WAM predictions, all-500 FVD was `7.91`, while 100-clip subsets were
`27.52`, `26.70`, `27.68`, `29.39`, `23.92`, and `28.84`. The matched-100
Open-WAM/UVA action-conditioned diagnostic was `29.05` vs `153.23`; it should
not replace the native 500-clip FVD row above.

### Final 100-Episode Counterfactual Tables

This is the expanded counterfactual pass using the same model contracts as the
real-demo table. The selector uses 10 eligible target-only CF windows per
LIBERO-10 task, for 100 total windows. Open-WAM and UVA receive the same CF
source window identity; each model then uses its native input/action convention.

Output:

```text
<EVAL_ROOT>/cf100_openwam_uva_fdm_idm_fvd_20260618/summary.json
<EVAL_ROOT>/cf100_openwam_uva_fdm_idm_fvd_20260618/selections.csv
<EVAL_ROOT>/cf100_openwam_uva_fdm_idm_fvd_20260618/fdm_rows.csv
<EVAL_ROOT>/cf100_openwam_uva_fdm_idm_fvd_20260618/idm_sim_rows.csv
```

Selection check: 10 rows per each of the 10 LIBERO-10 tasks.

IDM, simulator-grounded counterfactual:

- Open-WAM uses native raw7 delta-OSC predicted/GT actions executed through the
  Open-WAM LIBERO controller path from the CF flattened simulator state.
- UVA uses native abs7 predicted/GT actions executed through the UVA-style
  absolute EEF controller path from the same CF flattened simulator state.
- Metrics average all 16 dense post-action simulator steps. The selected column
  uses `t+4,t+8,t+12,t+16`. Lower is better.

| Metric | Open-WAM native IDM | UVA corrected native IDM |
| --- | ---: | ---: |
| Mean EEF pos | 1.11 +/- 1.66 cm | 2.24 +/- 3.51 cm |
| Selected EEF pos | 1.26 +/- 1.97 cm | 2.57 +/- 4.00 cm |
| Final EEF pos | 1.75 +/- 3.39 cm | 3.62 +/- 6.27 cm |
| Mean rot | 2.98 +/- 6.24 deg | 10.72 +/- 19.65 deg |
| Selected rot | 3.44 +/- 7.20 deg | 12.43 +/- 22.65 deg |
| Final rot | 4.64 +/- 9.70 deg | 17.90 +/- 33.78 deg |

FDM, native stored-target counterfactual:

- Open-WAM uses native CF latents/actions/proprio and the CF raw target frames.
- UVA uses the matched CF dense window converted into UVA's native LIBERO
  image/action/state contract.
- Metrics use single-camera `agentview 128 x 128` RGB at the selected shared
  timesteps. MSE is in `[0,1]`; lower MSE/MAE and higher PSNR/SSIM are better.
- Counterfactual FVD is intentionally not reported here. The maintained FVD
  diagnostic is real-demo only until we add a paper-scale CF distributional
  protocol.

| Metric | Open-WAM M5 GJD step40000 | UVA baseline |
| --- | ---: | ---: |
| MSE `[0,1]` | 0.01195 +/- 0.01094 | 0.01647 +/- 0.00892 |
| MAE `[0,1]` | 0.04067 +/- 0.01937 | 0.05175 +/- 0.01415 |
| PSNR | 20.72 +/- 3.45 | 18.46 +/- 2.17 |
| SSIM | 0.8775 +/- 0.0638 | 0.7973 +/- 0.0822 |

Native-contract table:

This is a target-pose/action diagnostic, not the controller-compounded metric.
For physical action compounding, use the simulator-compounded IDM table above.

| Source | Mode | Metric | Open-WAM M5 GJD step40000 | UVA baseline |
| --- | --- | --- | ---: | ---: |
| Real demo | FDM | MSE `[0,1]` lower | 0.00063 +/- 0.00032 | 0.00267 +/- 0.00078 |
| Real demo | FDM | SSIM higher | 0.9929 | 0.9665 |
| Real demo | FDM | FVD lower, 500 clips | 8.12 | 58.18 |
| Real demo | IDM | EEF target pos cm lower | 0.24 +/- 0.05 | 2.50 +/- 0.67 |
| Real demo | IDM | EEF target rot deg lower | 0.56 +/- 0.25 | 3.16 +/- 0.86 |
| Counterfactual | FDM | MSE `[0,1]` lower | 0.01095 +/- 0.00685 | 0.01593 +/- 0.00883 |
| Counterfactual | FDM | SSIM higher | 0.8801 | 0.8050 |
| Counterfactual | FDM | FVD | not reported; no 500-clip CF FVD run | not reported; no 500-clip CF FVD run |
| Counterfactual | IDM | EEF target pos cm lower | 0.28 +/- 0.31 | 2.31 +/- 0.85 |
| Counterfactual | IDM | EEF target rot deg lower | 0.34 +/- 0.21 | 2.94 +/- 1.06 |

Open-WAM-on-HDF5 diagnostic table:

This table is not the native Open-WAM benchmark. FDM rows score Open-WAM
native inputs against raw HDF5 video targets. IDM rows use the HDF5 semantic-pair
bridge because action targets require a state/action convention.

| Source | Mode | Metric | Open-WAM onto HDF5 | UVA native HDF5 |
| --- | --- | --- | ---: | ---: |
| Real demo | FDM | MSE `[0,1]` lower, selected frames | 0.00152 +/- 0.00012 | 0.00267 +/- 0.00078 |
| Real demo | FDM | SSIM higher, selected frames | 0.9691 | 0.9665 |
| Real demo | FDM | FVD lower, 500 clips | 341.26 | 58.18 |
| Real demo | FDM | Dense RGB MSE `[0,1]`, 500 clips | 0.00349 | not applicable; UVA emits sparse keyframes |
| Real demo | IDM | EEF target pos cm lower | 2.27 +/- 0.91 | 2.50 +/- 0.67 |
| Real demo | IDM | EEF target rot deg lower | 2.23 +/- 1.11 | 3.16 +/- 0.86 |

The real and counterfactual rows now have the expected qualitative shape: the
same model is stable across real/CF inputs, and Open-WAM is better on this
Open-WAM-generated CF diagnostic while UVA remains in a plausible centimeter
range after the orientation fix.

### IDM Contract Provenance

The old "~2 cm" IDM number and the final matched-native IDM number answer
different questions:

| Contract | Model | IDM target pos | IDM target rot | Meaning |
| --- | --- | ---: | ---: | --- |
| UVA native HDF5 sanity | UVA | 2.36-2.58 cm | not recomputed here | Released UVA checkpoint under its paper-style HDF5 input/action contract. |
| HDF5 semantic-pair diagnostic | Open-WAM | 2.27 +/- 0.91 cm | 2.23 +/- 1.11 deg | Open-WAM forced through HDF5-derived input/state projection with the LIBERO action-frame bridge; useful diagnostic, not the native Open-WAM score. |
| Matched-native real demo | Open-WAM | 0.24 +/- 0.05 cm | 0.56 +/- 0.25 deg | Open-WAM under its native LeRobot latent/action/proprio contract, projected to EEF target pose for comparison. |
| Matched-native real demo | UVA | 2.50 +/- 0.67 cm | 3.16 +/- 0.86 deg | UVA under its native HDF5 `agentview/actions/states` contract. |
| Matched-native counterfactual | Open-WAM | 0.28 +/- 0.31 cm | 0.34 +/- 0.21 deg | Open-WAM under native CF sidecar latents/actions/proprio. |
| Matched-native counterfactual | UVA | 2.31 +/- 0.85 cm | 2.94 +/- 1.06 deg | UVA on CF frames converted into raw HDF5 `agentview` orientation and CF raw7 actions converted through UVA's native LIBERO action frame, then projected to the same EEF target-pose metric. |
| Simulator-compounded real demo | Open-WAM | 0.42 +/- 0.36 cm mean, 0.75 +/- 0.60 cm final | 1.00 +/- 0.71 deg mean, 1.63 +/- 0.83 deg final | Execute predicted Open-WAM-native raw7 actions in LIBERO from the exact matched HDF5 start state and compare to the Open-WAM GT-action rollout. |
| Simulator-compounded real demo | UVA Robomimic native | 1.82 +/- 0.71 cm mean, 2.20 +/- 1.08 cm final | 1.56 +/- 0.52 deg mean, 1.99 +/- 0.75 deg final | Execute predicted UVA-native abs7 actions and raw HDF5 GT abs7 actions through UVA's Robomimic env metadata path with `control_delta=False`; this is the corrected paper-style UVA simulator-grounded route. |
| Simulator-compounded real demo | UVA direct diagnostic | 1.76 +/- 2.28 cm mean, 1.88 +/- 2.52 cm final | 8.57 +/- 14.70 deg mean, 12.09 +/- 20.59 deg final | Execute predicted UVA-native absolute EEF actions through the direct Open-WAM-side LIBERO helper with `control_delta=False`; diagnostic only, because UVA raw-data replay requires its Robomimic env metadata path for paper-scale grounding. |
| UVA raw-HDF5 replay self-consistency | UVA GT actions | 0.73 +/- 0.31 cm mean, 0.69 +/- 0.29 cm final | 0.64 +/- 0.27 deg mean, 0.67 +/- 0.42 deg final | Replay released UVA raw HDF5 GT actions through UVA's own Robomimic env metadata path; this is the simulator floor before evaluating UVA IDM in that env. |
| Simulator-compounded counterfactual | Open-WAM | 0.61 +/- 0.47 cm mean, 0.80 +/- 0.65 cm final | 0.81 +/- 0.52 deg mean, 1.33 +/- 0.92 deg final | Execute predicted raw7 actions in LIBERO from the exact CF start state and compare to the GT-action rollout. |
| Simulator-compounded counterfactual | UVA direct diagnostic | 1.55 +/- 2.19 cm mean, 2.53 +/- 3.51 cm final | 5.56 +/- 8.13 deg mean, 6.96 +/- 11.16 deg final | Execute predicted UVA-native absolute EEF actions through the direct Open-WAM-side LIBERO helper with `control_delta=False` after converting CF raw7 targets into UVA's LIBERO absolute-action frame. |

Use the simulator-compounded rows when the question is physical IDM behavior
after controller execution. Use the matched-native pose rows when the question
is action-target prediction under each model's own data contract. Raw action
MSE/L2 numbers are retained only as diagnostics because Open-WAM and UVA do not
share a raw action convention: Open-WAM uses delta/OSC-style raw7, while UVA's
native LIBERO data uses absolute EEF targets represented as 10D position plus
rotation 6D plus gripper.

### Counterfactual FDM Table

| Model | Count | Shared agentview MSE | MAE | PSNR | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open-WAM M5 GJD step40000 | 10 | 0.01095 +/- 0.00685 | 0.03999 | 20.41 | 0.8801 |
| UVA baseline | 10 | 0.01593 +/- 0.00883 | 0.05095 | 18.66 | 0.8050 |

### Counterfactual IDM Table

| Model | Count | Shared raw7 MSE | MAE | L2 |
| --- | ---: | ---: | ---: | ---: |
| Open-WAM M5 GJD step40000 | 10 | 0.00535 +/- 0.01039 | 0.02031 | 0.08302 |
| UVA baseline | 10 | 0.17169 +/- 0.18082 | 0.18642 | 0.85287 |

### Counterfactual IDM EEF Pose Table

The counterfactual pose metric uses the same selected future indices and the
same state-conditioned Open-WAM raw7 to absolute EEF conversion.

Lower is better.

| Model | Count | Target pos L2 cm | State pos L2 cm | Target rot deg | State rot deg | Gripper abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open-WAM M5 GJD step40000 | 10 | 0.28 +/- 0.31 | 2.77 +/- 1.54 | 0.34 +/- 0.21 | 0.72 +/- 0.62 | 0.0038 |
| UVA baseline, oriented CF input | 10 | 2.31 +/- 0.85 | 3.49 +/- 1.32 | 2.94 +/- 1.06 | 2.62 +/- 0.51 | 0.4539 |

### Invalidated HDF5-Source Diagnostic

I also tested a real-demo HDF5-source contract and invalidated it for Open-WAM
model comparison. Encoding HDF5 RGB into Open-WAM is possible, but converting
HDF5 absolute EEF actions into Open-WAM raw7 produces out-of-distribution
rotation deltas around `3.1`, while Open-WAM/CF raw rotation channels are near
zero. Example HDF5-derived raw7 first action:
`[0.1278, -0.0975, 0.4208, 0.0829, -0.1036, -3.1055, 1.0]`.

The invalidated run is kept only as a diagnostic:

```text
<EVAL_ROOT>/debug_videos/selected_fdm_idm_real_cf10_hdf5_source_20260615/
```

## Task-Aligned 50-Rollout Breakdown

| Open-WAM task id | Task | M5 GJD ep0-4 | UVA ep0-4 | M5 failed episodes |
| ---: | --- | ---: | ---: | --- |
| 0 | put both the alphabet soup and the tomato sauce in the basket | 4/5 | 3/5 | 4 |
| 1 | put both the cream cheese box and the butter in the basket | 4/5 | 0/5 | 0 |
| 2 | turn on the stove and put the moka pot on it | 5/5 | 5/5 | - |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 5/5 | 4/5 | - |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 4/5 | 4/5 | 2 |
| 5 | pick up the book and place it in the back compartment of the caddy | 5/5 | 5/5 | - |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 4/5 | 5/5 | 2 |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 5/5 | 4/5 | - |
| 8 | put both moka pots on the stove | 4/5 | 3/5 | 2 |
| 9 | put the yellow and white mug in the microwave and close it | 5/5 | 5/5 | - |

## Current Main Route Smoke

I also ran a fresh local smoke rollout on current `main` to verify that the M5
GJD rollout path matches the training-parity contract. This used the locally
available pure-joint M5 GJD mode-token step-20000 checkpoint, so it validates the
runtime route but is not the mixed joint/IDM/FDM comparison score.

Smoke result:

```text
checkpoint: <compute-host-openwam-exp-root>/checkpoints/haic_fresh_pure_joint_m5_gjd_mode_token_step20000_20260609_224338/checkpoint_step_20000/model_state.pt
task: LIBERO-10 task 2, episode 0
success: true
env_timestep: 251
chunks: 16
actions: 246
```

Runtime parity fields observed in the logs:

- Native packed M5 coupling route.
- `mot_generalist_mode_text_token=joint`.
- `startup_model_obs_frames=1`.
- First chunk `generation_frame_start=1`.
- Four invalid startup action tokens are masked out.
- Sixteen executable low-level actions per full generated chunk.
- Per-chunk proprio frames are appended during packed-history warmup.

Local smoke outputs were written under the operator's local eval root:

```text
<local-openwam-exp-root>/evals/m5_gjd_vs_uva_20260612_170822/local_pure_joint_route_smoke_t2_ep0/
```

## Reproduction Notes

The comparison artifact generated during this audit lives at:

```text
<local-openwam-exp-root>/evals/m5_gjd_vs_uva_20260612_170822/m5_gjd_mode_cf_step40000_vs_uva_libero10_ep0_4.json
```

The full mixed M5 GJD simulator score comes from the mirrored HAIC summary
rather than a new local 500-rollout run. If a stricter single-machine simulator
comparison is needed, rerun the same 10 x 5 episode set through
`scripts/run_gjd_libero.sh rollout` or `scripts/run_libero_mot_batch_visualization.py`
with the #168 M5 GJD settings.

The UVA score is the simulator policy-model rollout score. UVA's IDM and FDM
paths are offline prediction metrics, not robot-control simulator rollout
policies.
