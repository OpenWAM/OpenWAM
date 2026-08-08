# Extension Method Template

This directory is a runnable out-of-tree policy and decoder extension. Place
the files in an installed application package, replace the `example.*`
identifiers, customize either tensor implementation, and expose
`extension.register_open_wam`.

Policy and decoder extensions require `open-wam[torch]`; use the `train` or
`eval` extra when invoking those runtimes. Declare that dependency in the
application package rather than importing from an Open-WAM source checkout.

The extension boundary is:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

The example requests visual-core tokens, normalizes them in the policy, and
uses a small linear action decoder with masked MSE supervision. It is intended
to prove the extension boundary, not to serve as a quality robot policy.

Run one synthetic training step from a source checkout:

```bash
uv run --extra train open-wam-train \
  --cfg templates/extension_method/config.yaml \
  --extension templates.extension_method.extension
```

Suggested customization workflow:

1. Parse `ExtensionPolicyConfig.options` and
   `ExtensionActionDecoderConfig.options` into application-owned dataclasses.
2. Replace the example tensor operations in `policy_variant.py` and/or
   `action_decoder.py`.
3. Register both builders in `extension.py`.
4. Load `config.yaml` with `--extension your_package.extension`.
5. Add config, construction, gradient, inference-state, and checkpoint tests.

Do not add method branches to Open-WAM's generic trainer or visual tower. Add a
shared core contract only when the SDK contracts cannot express a reusable
capability.
