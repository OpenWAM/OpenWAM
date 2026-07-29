# M5 GJD And UVA LIBERO-10 Results

This card records the final corrected June 2026 comparison between Open-WAM M5
generalist joint denoising (GJD) and the released UVA LIBERO model. It preserves
the result and its limitations without treating the private comparison harness
as a supported Open-WAM interface.

## Artifacts

- Open-WAM: M5 GJD mode-token checkpoint at step 40,000, trained with real and
  counterfactual joint, FDM, and IDM modes.
- UVA: released `libero10.ckpt`.
- Rollout set: all ten LIBERO-10 tasks, episodes 0 through 4.
- Offline real-demo set: ten successful matched windows per task, 100 total.
- Offline counterfactual set: ten eligible target-only windows per task, 100
  total.

The original scripts, detailed manifests, and intermediate tables are retained
in Git history at PRs #169 and #171 and in the immutable refactor reference
checkout.

## Rollout Success

| Evaluation | Open-WAM M5 GJD | UVA |
| --- | ---: | ---: |
| Task-aligned episodes 0-4 | 45/50 (90.0%) | 38/50 (76.0%) |
| Full Open-WAM success@1 run | 461/500 (92.2%) | not run |

The Open-WAM full run reported the same `461/500` at success@2 and success@3;
no failed first attempt was recovered by retry.

## Real-Demo Dynamics

FDM uses each model's native stored target and temporal support. The metrics
therefore measure native-contract health; they are not a strict same-target
ranking. Both routes project selected frames to single-camera `128 x 128`
agentview RGB. Lower MSE/FVD and higher PSNR/SSIM are better.

| Metric | Open-WAM M5 GJD | UVA |
| --- | ---: | ---: |
| MSE `[0,1]`, 100 windows | 0.00058 +/- 0.00039 | 0.00279 +/- 0.00179 |
| MAE `[0,1]`, 100 windows | 0.00933 +/- 0.00213 | 0.02016 +/- 0.00505 |
| PSNR, 100 windows | 33.27 +/- 2.29 | 26.51 +/- 2.29 |
| SSIM, 100 windows | 0.9934 +/- 0.0030 | 0.9645 +/- 0.0232 |
| Native 500-clip FVD | 8.12 | 92.72 |

The FVD comparison uses the same released I3D feature extractor but different
model-native clips. Open-WAM supplies dense 16-frame action-conditioned
predictions. UVA's action-conditioned `dynamic_model` supplies sparse
keyframes repeated to the evaluator's 16-frame input. Small matched subsets
were unstable, so the card reports the separate 500-clip diagnostics.

For simulator-compounded IDM, each model executes predicted and ground-truth
actions from the same native simulator state under its own action/controller
contract. Metrics average all 16 post-action steps.

| Metric | Open-WAM native IDM | UVA native IDM |
| --- | ---: | ---: |
| Mean EEF position | 0.43 +/- 0.65 cm | 2.03 +/- 0.77 cm |
| Final EEF position | 0.66 +/- 0.87 cm | 2.36 +/- 1.02 cm |
| Mean rotation | 1.10 +/- 4.03 deg | 1.60 +/- 0.55 deg |
| Final rotation | 1.68 +/- 5.63 deg | 1.97 +/- 0.96 deg |

## Counterfactual Dynamics

Both models use the same counterfactual source identity, then consume it under
their native image/action convention. No paper-scale counterfactual FVD was
run.

| FDM metric, 100 windows | Open-WAM M5 GJD | UVA |
| --- | ---: | ---: |
| MSE `[0,1]` | 0.01195 +/- 0.01094 | 0.01647 +/- 0.00892 |
| MAE `[0,1]` | 0.04067 +/- 0.01937 | 0.05175 +/- 0.01415 |
| PSNR | 20.72 +/- 3.45 | 18.46 +/- 2.17 |
| SSIM | 0.8775 +/- 0.0638 | 0.7973 +/- 0.0822 |

| Simulator-compounded IDM metric | Open-WAM native IDM | UVA adapted IDM |
| --- | ---: | ---: |
| Mean EEF position | 1.11 +/- 1.66 cm | 2.24 +/- 3.51 cm |
| Final EEF position | 1.75 +/- 3.39 cm | 3.62 +/- 6.27 cm |
| Mean rotation | 2.98 +/- 6.24 deg | 10.72 +/- 19.65 deg |
| Final rotation | 4.64 +/- 9.70 deg | 17.90 +/- 33.78 deg |

## Interpretation

- The rollout table is directly task-aligned but covers only five episodes per
  task for the cross-model comparison.
- FDM rows compare model-native targets. WAN decode drift relative to UVA raw
  HDF5 targets is large enough that cross-target MSE is misleading.
- Raw action MSE is not comparable because Open-WAM uses delta-OSC raw7 while
  UVA uses absolute EEF actions.
- Simulator-compounded IDM is the preferred physical diagnostic, but each model
  still runs through its native controller/data contract.
- These results support model-health and benchmark comparisons, not a claim
  that every offline scalar is an apples-to-apples measure of model quality.

Use `scripts/run_gjd_libero.sh` and
`scripts/run_libero_mot_visualization.py` for maintained Open-WAM GJD rollout.
Use `open_wam.evals.dynamics` for Open-WAM-native FDM/IDM diagnostics.
