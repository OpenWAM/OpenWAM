from __future__ import annotations

import ast
from pathlib import Path

import torch

from open_wam.configs import load_experiment_config
from open_wam.data import build_synthetic_batch
from open_wam.extensions import load_extension_module
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.pipelines import build_variant_pipeline_from_config


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = REPO_ROOT / "templates" / "extension_method"


def test_extension_template_depends_only_on_public_sdk() -> None:
    for filename in ("policy_variant.py", "action_decoder.py", "extension.py"):
        module = ast.parse((TEMPLATE_ROOT / filename).read_text(encoding="utf-8"))
        open_wam_imports: list[str] = []
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.module:
                open_wam_imports.append(node.module)
            elif isinstance(node, ast.Import):
                open_wam_imports.extend(alias.name for alias in node.names)
        open_wam_imports = [
            name for name in open_wam_imports if name.startswith("open_wam")
        ]
        assert all(name.startswith("open_wam.sdk") for name in open_wam_imports)


def test_extension_template_runs_train_gradient_and_inference() -> None:
    torch.manual_seed(7)
    load_extension_module("templates.extension_method.extension")
    config = load_experiment_config(TEMPLATE_ROOT / "config.yaml")
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

    assert train_output.decoder_output.action_pred.shape == (1, 2, 4)
    assert torch.isfinite(train_output.decoder_output.loss)
    assert infer_output.decoder_output.action_pred.shape == (1, 2, 4)
    assert infer_output.policy_output.next_state.step_index == 1
    assert any(parameter.grad is not None for parameter in pipeline.policy_variant.parameters())
    assert any(parameter.grad is not None for parameter in pipeline.action_decoder.parameters())


def test_package_declares_pep561_typing_marker() -> None:
    assert (REPO_ROOT / "src" / "open_wam" / "py.typed").is_file()
