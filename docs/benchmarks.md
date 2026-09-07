# Benchmarks And Data

Open-WAM keeps benchmark-specific loading behind adapters while exposing one
uniform model-facing batch contract.

## Supported Sources

| Source | Status | Primary use |
| --- | --- | --- |
| Public tiny synthetic | Checked-in fixture | CI-safe loader/eval contract smoke tests. |
| LIBERO | Dataset and simulator paths | Manipulation policy training, evaluation, and realtime rollout experiments. |
| RoboTwin | Dataset and simulator adapter path | Simulated robotic manipulation with configurable action schema. |
| CALVIN | Dataset and simulator adapter path | Simulated language-conditioned manipulation with 7D relative actions. |
| LeRobot consortium | Heterogeneous multi-repository dataset adapter | Mixed-source training with explicit camera, action, and sampling contracts. |

Private datasets, local simulator checkouts, and large checkpoints should be
provided through the local path registry, not hard-coded in public configs.

### LIBERO Training Prerequisites

Maintained latent-local LIBERO training configs need inputs that a latent dataset
root does not contain on its own.

| Input | Registry key | Required by | Where it comes from |
| --- | --- | --- | --- |
| Pre-encoded latent root | `paths.datasets.libero_root` | all | Wan-VAE encoding of the LIBERO episodes. |
| Previous-frame condition latents | inside each latent payload | shipped policy programs and GJD | `scripts/augment_lerobot_latents_with_single_frame_condition.py --source-frame-offset -1`. |
| Empty text embedding | `paths.datasets.empty_text_embedding` | all | Negative-prompt embedding shared by latent-local datasets. |
| Replay-status labels | `paths.datasets.libero_replay_status_path` | configs with `require_replay_status: true` | Simulator replay labeling, merged by `scripts/build_libero_replay_metadata.py`. **Not shipped inside the dataset root.** |

Shipped policy-program and GJD configs use `include_all` replay rows and set
both replay-status requirements to false. Historical exact-backend compatibility
profiles may still require replay labels for their successful/failure split;
check the selected config rather than assuming either behavior.

The shipped sequence contract sets `condition_source_frame_offset=-1` and
fails closed when the payload lacks matching condition latents. Existing video
latents are reused; the augmentation command adds only the deterministic
previous-frame condition field. Run a bounded `--max-files` smoke with
`--sanity-check` before processing a full root.

See `configs/local_paths.sample.yaml` for the generating command and its
argument-shape caveats.

Dataset adapters may declare required files and directories through the shared
artifact-preflight contract. Training checks those requirements before model
construction and reports the owning config field, expected filesystem shape,
and remediation. Extensions can register the same resolver contract alongside
their raw or latent dataset builder; benchmark checks do not belong in the
trainer.

### LeRobot Consortium Snapshot

The consortium adapter validates configured remote repository IDs against a
bounded metadata snapshot. Wheels carry that snapshot as read-only package
resources, so validation works without a source checkout. A discrepancy warns
about missing or stale metadata; it never writes into `site-packages`.

Maintainers refreshing the index from a source checkout use
`scripts/build_lerobot_consortium_index.py`. To work against a separate writable
snapshot, set `OPEN_WAM_CONSORTIUM_INDEX_ROOT=/absolute/index/root` before the
adapter is imported. The directory must contain the repo-ID list, inventory
CSV/Markdown, and contracts JSON under their canonical filenames. Dataset and
video content are still resolved by the adapter's normal local/remote source
configuration; the packaged index is metadata only.

## Action Dimensions

Benchmarks expose different native action spaces. The model-facing action
dimension is configured separately from the source action dimension.

| Benchmark | Common source action | Model-facing examples |
| --- | --- | --- |
| LIBERO | 7D EEF delta plus gripper | 7D or sparse 30D mapping depending on config. |
| RoboTwin | 16D or 30D modes | Native 16D, native 30D, or mapped sparse 30D. |
| CALVIN | 7D `rel_actions` | Native 7D or sparse 30D compatibility mapping. |

Action mapping should be explicit in the dataset adapter/config. A model should
not infer missing dimensions silently.

## Visual Layout

The data layer builds canonical RGB layouts before the visual backbone sees the
batch. Public configs should make these choices visible:

- camera names
- camera count
- frame window
- target image size
- layout policy
- channel order

This keeps visual packing controlled across methods and benchmarks.

## Public Fixture

The public tiny synthetic fixture exists to test infrastructure, not model
quality. It is useful for:

- static config validation
- loader construction
- CPU eval smoke checks
- artifact manifest layout validation
- new contributor onboarding

Run it with:

```bash
open-wam-validate-config \
  configs/examples/public_tiny_synthetic_contract.yaml \
  configs/evals/public_tiny_synthetic_contract.yaml

open-wam-eval \
  --cfg configs/evals/public_tiny_synthetic_contract.yaml \
  --device cpu \
  --max-batches 1
```

Publish policy-quality claims through versioned experiment cards with current
artifacts and complete evaluation contracts.
