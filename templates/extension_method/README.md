# Extension Method Template

This directory is an out-of-tree scaffold for a policy and decoder extension.
Place the files in an installed application package, replace the `example.*`
identifiers, implement the tensor operations, and expose
`extension.register_open_wam`.

`config.yaml` is a loadable contract template. It is not runnable until the
`NotImplementedError` methods in the policy and decoder are implemented.

Policy and decoder extensions require `open-wam[torch]`; use the `train` or
`eval` extra when invoking those runtimes. Declare that dependency in the
application package rather than importing from an Open-WAM source checkout.

The extension boundary is:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

Suggested workflow:

1. Parse `ExtensionPolicyConfig.options` and
   `ExtensionActionDecoderConfig.options` into application-owned dataclasses.
2. Implement the typed methods in `policy_variant.py` and `action_decoder.py`.
3. Register both builders in `extension.py`.
4. Load `config.yaml` with `--extension your_package.extension`.
5. Add config, construction, gradient, inference-state, and checkpoint tests.

Do not add method branches to Open-WAM's generic trainer or visual tower. Add a
shared core contract only when the SDK contracts cannot express a reusable
capability.
