# Extension Method Template

This directory is the runnable policy and decoder extension scaffold shipped
with Open-WAM. It is installed under the same module path used by a source
checkout, so the command below also exercises wheel packaging. Copy the files
into an application package, replace the `example.*` identifiers, customize
either tensor implementation, and expose `extension.register_open_wam`.

The `open_wam.templates.*` module name belongs only to the shipped scaffold.
Production extensions use their application-owned package name and import
Open-WAM contracts from `open_wam.sdk.*`.

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

Run one synthetic training step:

```bash
uv run --extra train open-wam-train \
  --cfg templates/extension_method/config.yaml \
  --extension open_wam.templates.extension_method \
  --save-root runs/extension-method-smoke
```

Suggested customization workflow:

1. Parse `ExtensionPolicyConfig.options` and
   `ExtensionActionDecoderConfig.options` into application-owned dataclasses.
2. Replace the example tensor operations in `policy_variant.py` and/or
   `action_decoder.py`.
3. Register both builders in `extension.py`.
4. Install the application package and load `config.yaml` with
   `--extension your_package.extension`.
5. Add config, construction, gradient, inference-state, and checkpoint tests.

Do not add method branches to Open-WAM's generic trainer or visual tower. Add a
shared core contract only when the SDK contracts cannot express a reusable
capability.
