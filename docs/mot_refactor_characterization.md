# MoT Refactor Characterization

The MoT characterization suite is an opt-in numerical regression gate for
changes near the shared visual runtime, policy variants, attention programs,
training execution, checkpoint loading, and streaming inference. It runs real
LIBERO latent samples through real trained checkpoints on CUDA. It is separate
from the public synthetic smoke suite because its inputs and roughly 30 GB
checkpoints are machine-local.

## Coverage

The static matrix covers the six maintained non-GJD M5 couplings:

- `video_then_action`
- `action_then_video`
- `joint`
- `decoupled_same_step`
- `video_noisy_to_action`
- `action_noisy_to_video`

The exact numerical matrix currently uses the four available strict
checkpoints for VTA, ATV, joint, and decoupled. VNA and ANV remain in the
whole-config, command, training-scenario, and inference-contract tests, but do
not require historical checkpoints. Add them to the numerical matrix only
when checkpoints produced by the supplied strict commands are available.

Each checkpoint-backed non-GJD coupling runs a forward and backward pass for
both ground-truth training geometries:

- random segment: 64-256 latent frames, stride 4, randomized length and start
- full segment: a fixed 1000/1000 latent-frame request with W64 and fixed start

`segment_min_frames` and `segment_max_frames` are latent-frame units in the
local latent adapter. With `require_full_segment=true`, the 1000/1000 request
selects the complete available trajectory when a demonstration is shorter
than 1000 latent frames; it does not pad a short demonstration to 1000. The
frozen fixture records the resolved segment length and corresponding
four-actions-per-latent action span.

Both use replacement sampling, the legacy-prefix single-frame per-chunk proprio
contract, `noisy_video_condition_prob=0.5`, include-all replay, uniform sample
weighting, no sample loss reweighting, and the supplied model-only checkpoint
policy with interval 500 and retention 3. The characterization worker disables
checkpoint writes and experiment logging after validating that source contract.
For each geometry, a whole-config test verifies that the six methods differ
only in experiment name and `current_block_coupling`; their real-data fixture is
therefore intentionally shared. Phase 1 resolves the config-default W30 because
the supplied command does not override `training.window_size`, while phase 2
explicitly resolves W64.

The GJD syntax and semantics matrix covers `vanilla`, `pure_joint`, and
`mode_token`. No checkpoint currently exists for the corrected mixed-dynamics
vanilla contract, so vanilla is intentionally excluded from the default exact
numerical checkpoint run. Its config, five training scenarios, and three
inference routes remain statically asserted. Mode-token checkpoint training
exercises real joint, real FDM, real IDM, counterfactual FDM, and
counterfactual IDM buckets. Pure-joint exercises the real joint bucket.
Inference exercises joint, FDM, and IDM routing for each available GJD
checkpoint. The default exact checkpoint matrix is therefore six entries:
VTA, ATV, joint, decoupled, GJD pure-joint, and GJD mode-token.

The inference fixture uses the maintained streaming contract:

- LingBot streaming VAE
- one startup model observation and five simulator initialization steps
- W30 model inference
- four generated latent frames
- four actions per latent frame, for a 16-action model horizon
- all 16 actions selected for execution by the supplied non-GJD command
- non-GJD rollout limits of 800 timesteps / 50 chunks
- GJD rollout limits of 1500 timesteps / 100 chunks

The component worker executes three consecutive model chunks. Chunk zero
encodes the frozen real startup observation through the streaming VAE. Chunks
one and two consume the previous predicted latent chunk while carrying the
same `PolicyInferState`; this isolates recurrent policy/cache parity from
simulator variation. A separate LIBERO gate runs the real
streaming-observation replacement and packed-history warmup path.

Two shared-infrastructure sentinels cover behavior that does not need to be
multiplied across the method matrix:

- joint M5 and mode-token GJD joint inference run 17 recurrent chunks to cross
  the W30 packed-cache retention boundary
- mode-token GJD compares an uninterrupted update against save, teardown,
  full-state FSDP restore, and the same continuation update

W30 is an alternating action/video block window, not a 30-latent-frame cache.
With four latent frames per chunk it exposes 60 history frames and retains the
current four-frame chunk, for a 64-frame cache. Chunk 16 fills that cache and
chunk 17 proves eviction while the rollout cursor continues advancing.
Mode-token GJD is the resume sentinel because its full state contains the
shared video backbone, action expert, learned mode token, AdamW moments,
scheduler, strategy state, and train counters.

## Assertions

The training worker uses `TrainingRuntime` and the configured FSDP strategy. It
checks:

- the checkpoint has every current trained MoT parameter and no stale trained
  MoT parameters
- scalar loss is positive and finite
- model outputs are finite and retain exact full-tensor content hashes
- every populated gradient group is fully finite
- action-expert and video-backbone gradients are independently nonzero
- mode-token gradients are independently nonzero for the mode-token ablation
- action-expert, video-backbone, decoder, and mode-token gradient summaries
  remain numerically close to the recorded baseline
- one AdamW update runs through unscale, clipping, optimizer-state dtype
  normalization, optimizer step, and scheduler step
- post-update parameter summaries and high-gradient parameter probes change in
  every required trainable group

The optimizer probe reuses the final scenario's already-computed gradient. One
unscaled backward is mathematically equivalent to accumulating the same frozen
microbatch for the configured accumulation cycle. The actual training CLI
smoke separately enters the runtime microstep/update branch and requires
exactly one train-metrics record.

For GJD FDM and IDM, both real and counterfactual fixtures must use the
rollout-style target-only layout: one visible `t0`, `loss_frame_start=1`,
`singleton_chunk_frame=0`, `chunk_origin_frame=1`, and future chunk-size
randomization in 1-4. The four action rows aligned with the non-supervised `t0`
latent must be masked, while every post-`t0` action row remains valid. The
assertion applies only to conditional GJD scenarios; joint denoising retains
its full-history layout and action mask.

The inference worker builds `VariantPipeline` through the production factory,
loads the checkpoint without accepting missing trained parameters, restores the
configured MoT inference backend, and runs the streaming frontend plus three
recurrent policy chunks. It records:

- encoded startup latents
- predicted `16 x 7` actions
- four predicted future video latents
- policy metrics
- cache and variant-state schemas
- selected cache/state tensor fingerprints at every chunk
- exact `step_index` progression `1, 2, 3`
- monotonically advancing rollout cursors

The cache-rollover report additionally requires latent history to plateau at
64 frames and action history at 256 action steps while the cursor advances
beyond the retained window. The full-state report hashes every local FSDP model
and optimizer-state byte before save and immediately after restore. Restore is
byte-exact on every rank. It then requires the resumed continuation to match
the uninterrupted loss and output hashes exactly, the scheduler, counters, and
state schema exactly, and distributed gradient/update aggregates within narrow
documented tolerances. BF16/NCCL backward reductions are not byte-stable after
process teardown, so post-update model and AdamW hashes are not asserted equal;
their complete tensor/state structure is.

The persisted resume golden keeps full output fingerprints for the first
update, before any optimizer nondeterminism can accumulate. Post-update reports
retain output shape, dtype, element count, and finiteness while the stronger
uninterrupted-versus-resumed numerical comparison remains an in-run assertion.
This prevents an independent job from failing only because a low-bit reduction
changed while still catching altered inputs, routing, outputs, optimizer
schema, scheduler behavior, or resume counters.

The gate also verifies two checkpoint details that ordinary model-only loading
cannot cover:

- rank 0 deserializes the full payload once, then broadcasts and reshards it
  across the FSDP2 mesh
- sparse AdamW state remains sparse, so a parameter unused before the
  checkpoint receives its true first optimizer step after resume

The FSDP2 hierarchy wraps packed video/action blocks bottom-up and the pipeline
root last. The root owns trainable projections, conditioning encoders, and the
learned GJD mode token that sit outside packed blocks; without the root unit,
those parameters would update independently on each rank and no single logical
model could be resumed exactly.

This strict checkpoint preflight is intentional. A legacy checkpoint that
lacks the current full-proprio projections is not a valid substitute for a
checkpoint trained by the current command, even if `strict=False` loading
would otherwise initialize those projections randomly.

## Private Assets

Create a machine-local manifest from
`tests/characterization/mot_assets.example.yaml`. The manifest separates:

- LIBERO data
- LingBot VAE base assets
- the step-3500 video-only transformer
- empty text embedding
- counterfactual train and validation latent roots
- one checkpoint for each of the six current exact numerical entries

VNA, ANV, and vanilla GJD do not require manifest entries. Explicit asset
selection remains supported after a matching checkpoint entry is added.

Paths are never embedded in frozen fixtures or committed goldens as authority.
The checkpoint must itself satisfy the current key contract. In particular,
use the checkpoints associated with the strict six-mode commands, not older
fixed-128 VNA/ANV checkpoints that predate full proprio conditioning.

Every checkpoint entry must also resolve to the original
`resolved_config.yaml`. A checkpoint directory is sufficient when it contains
both files. For separately copied weights, use the explicit mapping shown in
the example manifest. Before any model allocation, the runner hashes that file
and compares behavior-defining fields against the exercised contract. This
catches architecture-compatible semantic drift such as an ATV checkpoint
trained with `parallel_sequence_contract=default`, or a pre-CF GJD checkpoint
trained with `generalist_training_paradigm=demo_only`. A multi-asset run
collects all source-contract failures before staging any checkpoint.

When an approved trained checkpoint is used only as the numerical weight
golden, its manifest entry may declare
`accepted_origin_mismatch_fields`. The list must exactly equal the differences
in the original `resolved_config.yaml`: an extra, missing, or newly introduced
mismatch fails preflight. The exercised runtime and frozen data still use the
authoritative command contract. This mechanism is suitable for the approved
ATV/joint/decoupled `successful_only` continuation checkpoints; it must not
allowlist attention, sequence, proprio, topology, or conditional-layout drift.

`resolved_config.yaml` proves saved-config parity, not the source-code revision
that originally interpreted it. Historical runs in this repository do not
consistently record a Git SHA. For contracts whose meaning changed without a
config-field change, establish goldens from a checkpoint with independently
known revision provenance. The frozen fixtures and runtime assertions verify
the current checkout's data/layout behavior, but they cannot retroactively
prove how an old checkpoint was trained.

## Freeze Inputs

Freeze deterministic inputs once from the real dataset:

```bash
uv run python -m tests.characterization.run_mot_refactor_characterization \
  fixtures \
  --assets /path/to/mot_assets.yaml \
  --output-root /path/to/frozen_fixtures
```

Fixtures use safetensors plus JSON metadata and SHA-256 verification. They
include source indices, resolved sampling contracts, alignment metadata, and
the real source-row metadata needed to diagnose a changed result. CF fixtures
prefer a deterministic strong axis-pulse branch instead of the unperturbed
ground-truth branch, so the gate exercises actual counterfactual tensors.

Rebuild the complete fixture set and require every output byte to match:

```bash
uv run python -m tests.characterization.run_mot_refactor_characterization \
  replay-fixtures \
  --assets /path/to/mot_assets.yaml \
  --fixture-root /path/to/frozen_fixtures
```

This verifies deterministic source indexing, segment geometry, CF branch
selection, collation, metadata, and safetensors serialization. It is stronger
than merely reloading the frozen tensors.

## Record A Baseline

Run one asset while developing:

```bash
uv run python -m tests.characterization.run_mot_refactor_characterization \
  record \
  --assets /path/to/mot_assets.yaml \
  --fixture-root /path/to/frozen_fixtures \
  --output-root /path/to/current_reports \
  --golden-root /path/to/golden_reports \
  --stage-root /local/nvme/openwam_mot_stage \
  --asset-id mot_joint \
  --cuda-devices 0,1,2,3
```

Omit `--asset-id` only when deliberately running all six currently required
checkpoints. The training worker defaults to four GPUs and FSDP CPU offload because a strict
full-segment backward pass is near the memory limit of a 48 GB GPU. Use
`--no-fsdp-cpu-offload` only on hardware with enough verified headroom.

On hosts where NCCL shared-memory transport is unavailable, add
`--disable-nccl-shm`. This sets `NCCL_SHM_DISABLE=1` and
`NCCL_CUMEM_HOST_ENABLE=0` for worker processes.

Characterization subprocesses default `TORCHINDUCTOR_COMPILE_THREADS=1` to
avoid leaving one rank-local async compiler pool per GPU after a completed
report. Set the environment variable explicitly to benchmark another compiler
parallelism level; it is not changed by production training commands.

For diagnosis only, `--allow-checkpoint-provenance-mismatch` records a stale
checkpoint and includes every mismatch in the report. Do not use that option
to establish strict refactor goldens: it proves execution compatibility, not
parity with the supplied training commands.

## Verify A Refactor

Record new reports without replacing the goldens, then compare:

```bash
uv run python -m tests.characterization.run_mot_refactor_characterization \
  verify \
  --actual-root /path/to/current_reports \
  --golden-root /path/to/golden_reports \
  --asset-id mot_joint
```

Structure, shapes, dtypes, checkpoint compatibility, configs, losses, model
outputs, cache schemas, action execution, state progression, optimizer schema,
scheduler values, and all other non-distributed fields compare exactly,
including floating-point values. Tensor fingerprints include SHA-256 over all
normalized tensor bytes in addition to diagnostic statistics and probes.
Explicit absolute/relative tolerance applies to gradient summaries, gradient
norms, and selected gradient probes, where NCCL reduction order can change the
final low bits. All-reduced post-step parameter-group aggregates use a fixed
absolute bound of `0.25`; subtracting two roughly `10^7`-scale BF16 summaries
is not byte-stable across process teardown. Inputs, outputs, losses, optimizer
schema, scheduler state, and every field outside those named distributed
numeric paths remain exact. Raw nonzero gradient counts remain in reports,
while verification compares their density rounded to one part per million
because near-underflow values can cross exact zero. Checkpoint paths and peak
CUDA allocation are excluded because staging location and allocator behavior
are not model semantics.

The pytest entrypoint is deliberately gated:

```bash
OPEN_WAM_RUN_MOT_GPU_CHARACTERIZATION=1 \
OPEN_WAM_MOT_CHARACTERIZATION_ASSETS=/path/to/mot_assets.yaml \
OPEN_WAM_MOT_CHARACTERIZATION_FIXTURES=/path/to/frozen_fixtures \
OPEN_WAM_MOT_CHARACTERIZATION_GOLDENS=/path/to/golden_reports \
OPEN_WAM_MOT_CHARACTERIZATION_ASSET_ID=mot_joint \
uv run pytest -q tests/test_mot_refactor_gpu_characterization.py
```

Set `OPEN_WAM_MOT_CHARACTERIZATION_STAGE_ROOT` for local staging and
`OPEN_WAM_MOT_CHARACTERIZATION_DISABLE_NCCL_SHM=1` on affected hosts. The
explicit asset ID prevents an ordinary pytest invocation from launching the
entire multi-hour matrix.

## End-To-End Gates

Record the shared cache and checkpoint infrastructure goldens separately from
the per-method matrix:

```bash
uv run python -m tests.characterization.run_mot_refactor_characterization \
  record \
  --assets /path/to/mot_assets.yaml \
  --fixture-root /path/to/frozen_fixtures \
  --output-root /path/to/cache_rollover_reports \
  --golden-root /path/to/infrastructure_goldens \
  --phase cache_rollover \
  --asset-id mot_joint \
  --asset-id gjd_mode_token \
  --cuda-devices 0

uv run python -m tests.characterization.run_mot_refactor_characterization \
  record \
  --assets /path/to/mot_assets.yaml \
  --fixture-root /path/to/frozen_fixtures \
  --output-root /path/to/resume_reports \
  --golden-root /path/to/infrastructure_goldens \
  --phase resume \
  --asset-id gjd_mode_token \
  --training-world-size 4 \
  --cuda-devices 0,1,2,3 \
  --no-fsdp-cpu-offload
```

The resume sentinel uses one frozen microbatch per optimizer update. Ordinary
training characterization still records the source accumulation contract, and
the production CLI smoke exercises its runtime microstep boundary. Restricting
this checkpoint test to one update keeps it focused on serialization and
continuation rather than duplicating method-specific gradient coverage. Point
`--output-root` at local scratch with at least 150 GB free: the current
mode-token full-state and model-only files together occupy about 123 GB.

The corresponding opt-in pytest gates use:

```bash
OPEN_WAM_RUN_MOT_CACHE_ROLLOVER=1 \
OPEN_WAM_RUN_MOT_FULL_STATE_RESUME=1 \
OPEN_WAM_MOT_CHARACTERIZATION_ASSETS=/path/to/mot_assets.yaml \
OPEN_WAM_MOT_CHARACTERIZATION_FIXTURES=/path/to/frozen_fixtures \
OPEN_WAM_MOT_INFRASTRUCTURE_GOLDENS=/path/to/infrastructure_goldens \
uv run pytest -q tests/test_mot_refactor_gpu_characterization.py
```

Exercise the actual distributed training CLI for one selected checkpoint:

```bash
uv run python -m tests.characterization.run_mot_refactor_characterization \
  training-cli-smoke \
  --assets /path/to/mot_assets.yaml \
  --output-root /path/to/cli_smoke \
  --asset-id mot_joint \
  --cuda-devices 0,1,2,3
```

The command stages a model-only symlink without sibling training counters,
launches `open_wam.training.train` through four-rank `torchrun`, performs one
real optimizer update with gradient accumulation set to one, disables
checkpoint writes and W&B, and verifies the JSONL train metric at step one.
This gate covers the production update lifecycle. The component worker
separately records the source command's configured accumulation count and
checks the corresponding optimizer and scheduler state transition.

The resume sentinel disables FSDP CPU offload to match the maintained GJD
launcher environment. Its frozen one-batch workload fits on four 48 GB GPUs.
The broader full-segment training matrix keeps CPU offload enabled by default
because those 1000-frame characterization paths require more headroom.

Run one full LIBERO rollout for each of the six checkpoint-backed methods,
sharded across four GPUs:

```bash
uv run python -m tests.characterization.run_mot_refactor_characterization \
  libero-rollout \
  --assets /path/to/mot_assets.yaml \
  --output-root /path/to/current_rollouts \
  --golden-root /path/to/rollout_goldens \
  --cuda-devices 0,1,2,3 \
  --task-id 0 --episode-idx 0 --seed 0
```

Non-GJD methods retain 800/50 limits and GJD retains 1500/100. Each report
requires the summary, action JSONL, per-chunk logs, and comparison video, then
fingerprints the exact action bytes, chunk semantics, and video bytes. Verify
after a refactor with:

```bash
uv run python -m tests.characterization.run_mot_refactor_characterization \
  verify-libero-rollout \
  --actual-root /path/to/current_rollouts \
  --golden-root /path/to/rollout_goldens
```

For a shorter wiring check, select one `--asset-id` and pass
`--max-timestep 64 --max-chunks 3`. This still exercises startup, recurrent
policy state, streaming VAE updates, simulator actions, packed-history warmup,
and video/debug artifact writing.

## Refactor Use

Before changing a core runtime boundary:

1. Record the affected assets from the pre-refactor commit.
2. Make the refactor without changing the experiment contract.
3. Record the same frozen fixtures and checkpoints.
4. Verify the numerical reports.
5. Run the training CLI smoke and relevant LIBERO rollout gate.

When an intentional semantic change modifies a report, review the tensor,
gradient, and state differences before recording a new baseline. Do not update
goldens only to make the gate pass.
