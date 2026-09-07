# Generated Video To Action Composition

Status: implemented. This note defines the maintained inference contract and
the parity gates that protect it.

## Purpose

The composition runtime connects two independent policy sessions:

1. a video producer emits a future latent-video artifact;
2. an action consumer processes its own observed context plus that artifact;
3. LIBERO executes the resulting actions;
4. each session updates or reconciles its own recurrent history from the real
   observations and executed actions.

The interface is model-neutral. A causal video model, a selective
`video_then_action` policy, or another policy with a declared video output can
be the producer. A native VTA action stage, strict IDM policy, or routed GJD IDM
stage can be the consumer when it declares the matching composition capability.

The ordinary one-model VTA route is unchanged. Composition is an ablation that
materializes two models; it is not an optimization of native VTA and does not
resume an in-flight call in the producer.

## Contracts

The repository keeps its normal model boundary:

`ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder`

Composition uses four contracts around that boundary:

- `PolicyInferenceCapabilities` declares native outputs, optional selective
  outputs, accepted composition artifacts, recurrent-history policy, and RNG
  policy.
- `PolicyGeneratedVideo` is a future-only `[B, C, T, H, W]` tensor with an
  absolute latent-frame origin and a content-derived latent-space identity.
  It never contains the observed condition frame.
- `PolicyVideoGenerationRequest` asks a producer for an exact number of future
  latent frames without supplying future observations. The surrounding
  `PolicyInferContext.temporal_geometry` owns the session attention window.
- `PolicyVideoConditionedActionRequest` transfers the generated artifact to a
  separate action-policy call. Observed video, prompt, proprioception, executed
  actions, and recurrent state stay on the consumer's ordinary context/session.

The generic helpers live in
`open_wam.pipelines.video_action_composition`. Benchmark code owns model
loading, observation adaptation, action execution, and recurrent commits; it
does not invoke private policy forwards or inspect checkpoint names.

## Information Flow

The orchestrator makes all context available to the consumer:

```text
consumer observed window
consumer task/negative text embeddings
consumer proprio state
consumer recurrent state and executed-action history
producer PolicyGeneratedVideo
temporal origin and execution commit
```

The consumer owns visibility. The adapter does not pre-emptively erase context
to emulate a particular method.

| Consumer | Observed history visible to action queries | Text | Past action history | Generated future |
| --- | --- | --- | --- | --- |
| native VTA action stage | normal VTA recurrent video window | retained | masked by the maintained `video_only` history contract | inserted as clean timestep-zero video |
| strict IDM / GJD IDM route | latest clean video frame only | zeroed | hidden by conditional video-only history semantics | clean conditioning video |

For VTA, the independent consumer first builds its own cache from real observed
video, then forwards the transferred future video at timestep zero to construct
its own clean future K/V, and finally runs the unchanged VTA action denoiser.
Every clean and predicted video token is therefore processed by the consumer
model. The orchestrator still supplies prompt, proprioception, and executed
actions, but the policy's maintained sequence contract decides which streams
are visible. Producer cache internals never cross the boundary.

For strict IDM, DualExpert maps the neutral request to its existing
`video_conditioned_action` runtime plan. That plan preserves the maintained
text-free conditional semantics used by standalone IDM and GJD IDM: only the
latest clean history frame is visible, while the transferred future keeps its
configured chunk size (four latent frames and sixteen actions by default). This
mapping is policy-owned; the generic composition layer does not know about
dynamics objectives.

## Producer Requirements

A producer is usable when it:

- declares `video` in its native outputs;
- optionally declares a selective video-only request;
- publishes a non-empty future-only `PolicyGeneratedVideo`;
- publishes its absolute latent-frame origin and latent-space identity;
- declares how speculative history is replaced by real observations.

A GJD producer must also have a positive `joint` training route. Composition
invokes GJD's default joint rollout objective; an FDM/IDM-only checkpoint has
not trained that route and is rejected before simulator construction.

Selective output is an optimization, not a compatibility requirement. A
multimodal producer may run its full native program and have its unused action
output discarded.

The maintained causal-video policy and DualExpert VTA satisfy this contract.
For `chunked_conditioned_video`, the session temporal geometry's attention
window physically bounds committed real history as well as attention
visibility. With W30 and four-frame chunks, each denoise call retains at most
60 target-history frames, one external condition frame, and the current
four-frame block. After partial execution, the producer discards the unexecuted
speculative tail, reconstructs the interrupted noisy block from committed
history, and publishes only the remaining suffix. Target-local chunk phase and
absolute temporal positions are kept separate throughout reconciliation.

These are producer-owned recurrence semantics. The composition layer only
passes the typed request and execution commit; it contains no causal-video or
DualExpert cache arithmetic. Parallel Stream remains fail-closed until it
publishes the same typed artifact and recurrent-history behavior; no model-name
exception exists in the generic composition API.

Latent-first training checkpoints may deliberately record
`load_wan_vae_frontend: false` and `load_text_conditioning: false`. An online
RGB rollout must explicitly enable both assets. Without the VAE, the producer
cannot publish an artifact-backed latent-space identity, so composition fails
before simulator construction instead of accepting an unverifiable handoff.

## Consumer Requirements

A consumer declares a `video -> action` `PolicyCompositionCapability`. The
declaration owns two independent choices:

- the transferable input/output modalities;
- how stochastic sampling obtains its RNG stream.

Two RNG policies are maintained:

- `caller_stream`: used by VTA decomposition. The action stage continues the
  exact random stream left by video generation. If producer and consumer use
  different CUDA devices, the composition plan copies the producer generator
  state to the consumer before action inference and returns the advanced state
  to the producer afterward.
- `isolated_step_seed`: used by strict IDM/GJD. The action consumer is seeded
  independently for each chunk, and the caller RNG state is restored afterward.

The consumer must also declare either next-observation replacement or explicit
observed-history reconciliation. Producer and consumer keep separate policy,
visual-frontend, text, and cache states throughout the rollout.

## Compatibility Checks

Loading fails before simulator construction unless:

- the producer and consumer declare compatible artifact capabilities;
- both stages agree on packed camera layout, latent channels/spatial geometry,
  temporal stride, patch geometry, and generated chunk size;
- both frontends resolve to the same content-derived VAE identity;
- each generated chunk and consumer output report the same frame origin;
- the consumer program is VTA, fixed IDM, or GJD with an active IDM route;
- each runtime declares a supported recurrent-history policy;
- packed DualExpert action modules share their runtime device.

Action schemas are not compared between producer and consumer. Video-only
producers do not own executable-action semantics; the consumer owns action
normalization, decoding, horizon, and device placement.

## Programmatic Use

```python
from open_wam.sdk.policy import (
    PolicyInferContext,
    PolicyTemporalGeometry,
    PolicyVideoGenerationRequest,
)
from open_wam.pipelines import (
    build_video_conditioned_action_context,
    require_generated_video,
    resolve_policy_video_action_consumer_plan,
    resolve_policy_video_producer_plan,
)

producer_plan = resolve_policy_video_producer_plan(
    producer.policy_variant
)
consumer_plan = resolve_policy_video_action_consumer_plan(
    consumer.policy_variant
)
generation_request = PolicyVideoGenerationRequest(
    frame_count=4,
)

video_step = producer_runner.infer_prepared_step(
    session=producer_session,
    context=PolicyInferContext(
        state=producer_state,
        output_request=producer_plan.output_request,
        video_generation=generation_request,
        temporal_geometry=PolicyTemporalGeometry(
            frame_chunk_size=4,
            attention_window_size=30,
        ),
    ),
    visual_outputs=producer_visual_outputs,
)
generated = require_generated_video(
    video_step.infer_output,
    request=generation_request,
)

with consumer_plan.rng_stream(
    producer_device=producer_device,
    consumer_device=consumer_device,
):
    action_step = consumer_runner.infer_prepared_step(
        session=consumer_session,
        context=build_video_conditioned_action_context(
            PolicyInferContext(state=consumer_state),
            generated,
        ),
        visual_outputs=consumer_visual_outputs,
    )
```

This excerpt shows the artifact handoff. For `isolated_step_seed` consumers, a
caller must also resolve the step seed, snapshot and restore the caller RNG,
and seed the consumer call. In production, use the LIBERO adapter instead of
reproducing seed, frontend, execution-commit, and history-reconciliation logic
around this low-level flow.

## LIBERO CLI

The canonical route is `generated_video_then_action`:

Like every maintained online LIBERO route, composition activates the
`online_rollout` renderer profile (EGL) before simulator startup. Renderer
selection depends on the simulator workload, not the producer or consumer
policy semantics.

```bash
uv run --extra sim python \
  scripts/run_libero_policy_video_action_visualization.py \
  --cfg /path/to/video_producer/resolved_config.yaml \
  --checkpoint /path/to/video_producer/checkpoint_step_N \
  --set backbone.load_wan_vae_frontend=true \
  --set backbone.load_text_conditioning=true \
  --action-route generated_video_then_action \
  --action-consumer-cfg /path/to/action_consumer/resolved_config.yaml \
  --action-consumer-checkpoint /path/to/action_consumer/checkpoint_step_M \
  --benchmark libero_10 \
  --task-id 0 \
  --episode-idx 0 \
  --frontend-encode-mode lingbot_streaming_vae \
  --startup-model-obs-frames 1 \
  --startup-env-init-steps 5 \
  --dual-expert-inference-window-size 30 \
  --runtime-device cuda:0 \
  --action-device cuda:0 \
  --frontend-device cuda:0 \
  --decode-device cuda:0 \
  --action-consumer-runtime-device cuda:1 \
  --action-consumer-action-device cuda:1 \
  --action-consumer-frontend-device cuda:1 \
  --seed 0 \
  --output-dir outputs/generated_video_then_action
```

Consumers with isolated random streams, including strict IDM and GJD IDM,
require an explicit rollout seed. Native VTA decomposition continues the
producer's random stream and does not require one.

Use the same VTA config/checkpoint for both roles to run the manually split VTA
ablation. Use a fixed IDM config or a GJD config with positive IDM routing for
strict IDM. The batch entrypoint is
`scripts/run_libero_policy_video_action_batch_visualization.py`.

## Golden Parity

The always-on CPU gate runs two recurrent chunks through three independently
materialized VTA pipelines: native VTA, selective VTA video producer, and VTA
action consumer. It requires exact generated latents, decoded actions, cursor
state, and every action-cache K/V tensor.

The opt-in real gate runs native VTA and two-model VTA on the same checkpoint.
By default the producer/native model uses `cuda:0` and the consumer uses
`cuda:1`, exercising the cross-device RNG bridge. It requires byte-identical
action JSONL and comparison MP4, plus exact per-chunk policy debug payloads
after asserting and removing the intentionally different device labels.

```bash
export OPEN_WAM_RUN_VTA_COMPOSITION_PARITY=1
export OPEN_WAM_VTA_COMPOSITION_CONFIG=/path/to/vta/resolved_config.yaml
export OPEN_WAM_VTA_COMPOSITION_CHECKPOINT=/path/to/vta/checkpoint
# Set this only when the checkpoint does not bundle `transformer/` and the
# resolved config points to a transformer location unavailable on this host.
export OPEN_WAM_VTA_COMPOSITION_TRANSFORMER_DIR=/path/to/transformer
export OPEN_WAM_VTA_COMPOSITION_DATASET_ROOT=/path/to/libero_10
export OPEN_WAM_VTA_COMPOSITION_BASE_MODEL_ROOT=/path/to/lingbot-va-base
export OPEN_WAM_LIBERO_REPO_ROOT=/path/to/LIBERO
export OPEN_WAM_VTA_COMPOSITION_PRODUCER_DEVICE=cuda:0
export OPEN_WAM_VTA_COMPOSITION_CONSUMER_DEVICE=cuda:1

uv run pytest -q \
  tests/test_policy_video_action_golden.py::test_real_vta_native_and_two_model_rollouts_are_bitwise_equal
```

The VTA-to-strict-IDM route is protected by a deterministic Task 0 / episode 0
/ seed 0 golden at
`tests/characterization/goldens/vta_external_idm_task0_ep0_seed0.json`. It
checks the exact first-chunk action trace and complete rollout contract.

```bash
export OPEN_WAM_RUN_VTA_IDM_GOLDEN=1
export OPEN_WAM_VTA_IDM_GOLDEN_SCOPE=first_chunk  # or full_rollout
export OPEN_WAM_VTA_IDM_PRODUCER_CONFIG=/path/to/vta/resolved_config.yaml
export OPEN_WAM_VTA_IDM_PRODUCER_CHECKPOINT=/path/to/vta/checkpoint
export OPEN_WAM_VTA_IDM_CONSUMER_CONFIG=/path/to/idm/resolved_config.yaml
export OPEN_WAM_VTA_IDM_CONSUMER_CHECKPOINT=/path/to/idm/checkpoint
# These are optional when each checkpoint bundles `transformer/` or its
# resolved config already names a valid local transformer.
export OPEN_WAM_VTA_IDM_PRODUCER_TRANSFORMER_DIR=/path/to/producer/transformer
export OPEN_WAM_VTA_IDM_CONSUMER_TRANSFORMER_DIR=/path/to/consumer/transformer
export OPEN_WAM_VTA_IDM_DATASET_ROOT=/path/to/libero_10
export OPEN_WAM_VTA_IDM_BASE_MODEL_ROOT=/path/to/lingbot-va-base
export OPEN_WAM_LIBERO_REPO_ROOT=/path/to/LIBERO

uv run pytest -q \
  tests/test_policy_video_action_golden.py::test_real_vta_external_idm_rollout_matches_golden
```

These gates protect inference only. Training programs, losses, data sampling,
checkpoint formats, and default native rollout routes are unchanged.
