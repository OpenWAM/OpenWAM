# Research Dynamics Diagnostics

These modules implement checkout-only FDM/IDM analyses. They are deliberately
outside `src/open_wam` and are not installed in the production wheel. Reusable
policy, rollout, data, and counterfactual-action contracts remain owned by the
`open_wam` package.

Use the stable repository scripts rather than importing this directory as a
public API:

- `scripts/run_joint_denoising_fdm_ablation.py`
- `scripts/run_joint_denoising_fdm_counterfactual.py`
- `scripts/encode_libero_fdm_counterfactual_dataset.py`

The commands require explicit checkpoint and local data inputs. Run each with
`--help` for its complete contract. This keeps private machine paths out of the
repository and makes a checkout's external dependencies visible at launch.
