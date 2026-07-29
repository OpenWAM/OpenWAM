# Method Families

Open-WAM uses method families to compare where policy logic attaches to a
shared world-action-model visual path. The method name describes the policy
semantics, not a separate top-level training stack.

## Current Families

| Family | Variant name | Main idea |
| --- | --- | --- |
| Method 1 | `parallel_stream` | Exact LingBot-compatible visual/action diffusion semantics through shared-backbone runtime programs. |
| Method 2 | `parallel_stream` (action-conditioned) | Joint-denoise profile on the exact shared LingBot-compatible runtime. |
| Method 4 | `post_latent` / `post_decoded` | Feature-attached action heads over latent or decoded visual representations. |
| Method 5 | `mot` | Multi-object-token style action modeling with explicit typed state. |
| Video-only | `causal_video_prediction` | Visual prediction baseline without an action decoder. |

Traditional Method 2 `register_attached` and the experimental Method 3
`video_sequence_policy`/VPP path have been retired. Method 2 keeps explicit
compatibility stubs; Method 3 had no public checkpoint contract and is
recoverable from Git history.

## Shared Execution

All method families should remain trainable and inferable through the same
high-level commands:

```bash
open-wam-train --cfg configs/experiments/<experiment>.yaml
open-wam-eval --cfg configs/evals/<eval>.yaml
open-wam-sanity --cfg configs/examples/<example>.yaml --device cpu
open-wam-sim-rollout --cfg configs/examples/<example>.yaml
```

The trainer and evaluator should not branch on method names. Method-specific
behavior belongs in:

- the selected `PolicyVariant`
- the selected `ActionDecoder`
- runtime-program selection
- visual-stage requirements
- config-driven cache and scheduler policy

## Adding A Method

Use [Add a New Method](cookbooks/new_method.md) as the starting recipe. A new
method normally needs:

- one enum-backed policy-variant config choice
- one `PolicyVariant` implementation
- one action decoder if the output/loss differs from existing decoders
- one small smoke config
- one config/static validation test
- one focused runtime or variant-pipeline test

If the method only changes the final action loss, prefer adding an
`ActionDecoder` rather than a new policy variant.

## Compatibility Rule

The shared visual tower is the controlled variable. A new method should not
silently swap the backbone, canonical RGB layout, action mapping, or scheduler
family unless the config says so explicitly and the experiment card documents
the change.
