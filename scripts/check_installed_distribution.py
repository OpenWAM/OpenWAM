from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check an installed Open-WAM wheel.")
    parser.add_argument("--expected-package-root", type=Path)
    args = parser.parse_args(argv)

    import open_wam

    package_path = Path(open_wam.__file__).resolve()
    if args.expected_package_root is not None:
        expected_root = args.expected_package_root.expanduser().resolve()
        if not package_path.is_relative_to(expected_root):
            raise SystemExit(
                f"Expected installed package under {expected_root}, imported {package_path}."
            )

    package_resources = files("open_wam")
    required_resources = (
        "py.typed",
        "resources/configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml",
        "resources/consortium/lerobot_consortium_hf_repo_ids.txt",
        "resources/consortium/lerobot_consortium_hf_dataset_inventory.csv",
        "resources/consortium/lerobot_consortium_hf_dataset_inventory.md",
        "resources/consortium/lerobot_consortium_hf_dataset_contracts.json",
        "resources/templates/extension_method/config.yaml",
    )
    missing = tuple(path for path in required_resources if not package_resources.joinpath(path).is_file())
    if missing:
        raise SystemExit(f"Installed wheel is missing package resources: {missing}")

    from open_wam.data import lerobot_consortium_catalog
    from open_wam.models.visual_tower.reference_loader import load_internal_wan_transformer_class

    if lerobot_consortium_catalog._CONSORTIUM_INDEX_MUTABLE:
        raise SystemExit("Installed consortium snapshots must be read-only package resources.")
    if not all(
        path.is_file()
        for path in (
            lerobot_consortium_catalog._CONSORTIUM_INDEX_REPO_IDS_PATH,
            lerobot_consortium_catalog._CONSORTIUM_INDEX_INVENTORY_CSV_PATH,
            lerobot_consortium_catalog._CONSORTIUM_INDEX_INVENTORY_MD_PATH,
            lerobot_consortium_catalog._CONSORTIUM_INDEX_CONTRACTS_JSON_PATH,
        )
    ):
        raise SystemExit("Installed consortium adapter did not resolve its packaged snapshots.")
    transformer_class = load_internal_wan_transformer_class()
    if transformer_class.__name__ != "WanTransformer3DModel":
        raise SystemExit(f"Unexpected installed LingBot transformer class: {transformer_class!r}")

    _run_packaged_extension_smoke()
    print(f"installed distribution ok: {package_path}")


def _run_packaged_extension_smoke() -> None:
    import torch

    from open_wam.configs import load_experiment_config, resolve_config_reference
    from open_wam.data import build_synthetic_batch
    from open_wam.extensions import load_extension_module
    from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
    from open_wam.pipelines import build_variant_pipeline_from_config

    torch.manual_seed(7)
    load_extension_module("open_wam.resources.templates.extension_method.extension")
    config = load_experiment_config(resolve_config_reference("templates/extension_method/config.yaml"))
    pipeline = build_variant_pipeline_from_config(config)
    batch = build_synthetic_batch(config.data, batch_size=1)
    train_batch = PolicyTrainBatch(
        actions=batch.actions,
        action_mask=batch.action_mask,
        state=batch.state,
        extra={"task_text": batch.task_text},
    )
    train_output = pipeline.forward_train(batch.views, train_batch)
    train_output.decoder_output.loss.backward()
    infer_output = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
    )
    if train_output.decoder_output.action_pred.shape != (1, 2, 4):
        raise SystemExit("Packaged extension produced an invalid training action shape.")
    if infer_output.decoder_output.action_pred.shape != (1, 2, 4):
        raise SystemExit("Packaged extension produced an invalid inference action shape.")
    if not any(parameter.grad is not None for parameter in pipeline.policy_variant.parameters()):
        raise SystemExit("Packaged extension policy did not receive gradients.")
    if not any(parameter.grad is not None for parameter in pipeline.action_decoder.parameters()):
        raise SystemExit("Packaged extension decoder did not receive gradients.")


if __name__ == "__main__":
    main()
