# Causal Video Training Golden

`training_step_v1.safetensors` records the loss, predicted latents, every named
gradient, and every named trainable parameter after one SGD update.
`multichunk_rollout_v1.safetensors` records the complete generated history and
targeted first-chunk metric for the deterministic two-chunk inference path.

Do not regenerate this file while refactoring. A replacement requires an
explicitly reviewed numerical-contract change and a new schema/version name.
