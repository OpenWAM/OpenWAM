"""Policy and dynamics batches preserve independent B1 semantics and precision."""

from dataclasses import replace
import os

import pytest
import torch

from open_wam.configs import (
    BatchingMode,
    DynamicsObjective,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from open_wam.data.latent_contracts import collate_latent_wam_samples
from tests.test_dual_expert_generalist_training import _dynamics_sample_metadata
from tests.test_variable_batch_pipeline import (
    PROGRAMS,
    _tiny_pipeline,
    _samples,
    _collate,
)


def dynamics_samples(objectives, base=None):
    if base is None:
        base = _samples(VideoActionProgram.GENERALIST_JOINT_DENOISING)
    result = []
    for index, objective in enumerate(objectives):
        sample = base[index % len(base)]
        mask = (
            torch.ones_like(sample.actions)
            if sample.action_mask is None
            else sample.action_mask.clone()
        )
        if objective.is_conditional:
            mask[: int(sample.metadata.get("action_tokens_per_frame", 2))] = 0
        result.append(
            replace(
                sample,
                condition_latents=None
                if objective.is_conditional
                else sample.condition_latents,
                action_mask=mask,
                metadata={
                    **sample.metadata,
                    **_dynamics_sample_metadata(
                        objective,
                        frame_count=sample.video_latents.shape[1],
                        source="counterfactual_dynamics"
                        if index % 2 and objective.is_conditional
                        else "real_demo",
                    ),
                    "sampled_chunk_size": index % 4 + 1,
                    "sampled_window_size": 4 + index * 3,
                },
            )
        )
    return result


@pytest.mark.parametrize("batching", [BatchingMode.PADDED, BatchingMode.PACKED])
@pytest.mark.parametrize(
    "device,precision",
    [
        ("cpu", "float32"),
        pytest.param("cuda", "float32", marks=pytest.mark.gpu),
        pytest.param("cuda", "bfloat16", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    "program,token,objectives",
    [
        (VideoActionProgram.GENERALIST_JOINT_DENOISING, token, tuple(DynamicsObjective))
        for token in (False, True)
    ]
    + [
        (
            VideoActionProgram.FORWARD_DYNAMICS,
            False,
            (DynamicsObjective.ACTION_CONDITIONED_VIDEO,) * 2,
        ),
        (
            VideoActionProgram.INVERSE_DYNAMICS,
            False,
            (DynamicsObjective.VIDEO_CONDITIONED_ACTION,) * 2,
        ),
    ]
    + [(program, False, ()) for program in PROGRAMS],
)
def test_dynamics_batch_matches_independent_predictions_masks_metrics_and_gradients(
    batching,
    program,
    token,
    objectives,
    device,
    precision,
):
    if device == "cuda" and (
        not torch.cuda.is_available() or os.getenv("OPEN_WAM_RUN_GPU_SANITY") != "1"
    ):
        pytest.skip("Requires an explicitly allocated GPU.")
    if device == "cuda":
        # These cases instantiate different models, unlike steps of one run.
        torch.compiler.reset()
    _check_dynamics_batch(batching, program, token, objectives, device, precision)


def _check_dynamics_batch(
    batching, program, token, objectives, device, precision="float32"
):
    torch.manual_seed(71)
    pipeline, executor = _tiny_pipeline(
        program,
        generalist_mode_text_token=token,
        activation_checkpointing=True,
        attention_head_dim=32 if device == "cuda" else 8,
        sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
    )
    pipeline.to(device)
    samples = dynamics_samples(objectives) if objectives else _samples(program)
    objectives = objectives or (None,) * len(samples)
    move = lambda batch: executor.batch_adapter.move_to_device(
        batch, torch.device(device)
    )
    rng_state = (
        torch.cuda.get_rng_state if device == "cuda" else torch.random.get_rng_state
    )
    mixed_precision = precision == "bfloat16"
    with torch.autocast(device, dtype=torch.bfloat16, enabled=mixed_precision):
        torch.manual_seed(901)
        together = executor.forward_train(move(_collate(samples, batching)))
        rng_after = rng_state()
        torch.manual_seed(901)
        singles = [
            executor.forward_train(move(collate_latent_wam_samples([sample])))
            for sample in samples
        ]
    assert torch.equal(rng_after, rng_state())
    tolerance = (
        dict(atol=3e-4, rtol=3e-3) if device == "cuda" else dict(atol=3e-6, rtol=3e-4)
    )
    gradient_tolerance = dict(atol=8e-4, rtol=8e-3) if device == "cuda" else tolerance

    def tensors(output):
        return [
            output.decoder_output.action_pred,
            output.decoder_output.aux["predicted_latents"],
            *output.decoder_output.metrics.values(),
        ]

    actual_values, expected_values = (
        [together.loss],
        [torch.stack([item.loss for item in singles]).mean()],
    )
    for actual, expected, objective in zip(
        together.output.sample_outputs, singles, objectives, strict=True
    ):
        assert actual.policy_output.aux == expected.output.policy_output.aux
        if objective is not None:
            assert (
                actual.policy_output.aux["dual_expert_generalist_text_dropped"]
                == objective.is_conditional
            )
        actual_values.extend(tensors(actual))
        expected_values.extend(tensors(expected.output))
        left = actual.policy_output.decoder_artifacts.payload
        right = expected.output.policy_output.decoder_artifacts.payload
        torch.testing.assert_close(left.action.action_mask, right.action.action_mask)
        torch.testing.assert_close(
            left.video.future_loss_mask, right.video.future_loss_mask
        )
        if objective is not None and objective.is_conditional:
            assert not left.action.action_mask[:, :2].any()
            assert not left.video.future_loss_mask[:, :, :1].any()
    parameters = tuple(pipeline.parameters())
    expected_gradients = torch.autograd.grad(
        expected_values[0], parameters, allow_unused=True
    )
    actual_gradients = torch.autograd.grad(together.loss, parameters, allow_unused=True)
    oracle_values = oracle_gradients = None
    if mixed_precision:
        # Batched GEMMs and sparse vs dense attention need not round identically
        # in BF16. Bound their error by B1's measured error against the same FP32
        # model, rather than weakening the FP32 or existing kernel parity gates.
        with torch.autocast(device, enabled=False):
            torch.manual_seed(901)
            oracle = [
                executor.forward_train(move(collate_latent_wam_samples([sample])))
                for sample in samples
            ]
        oracle_loss = torch.stack([item.loss for item in oracle]).mean()
        oracle_values = [
            oracle_loss,
            *(value for item in oracle for value in tensors(item.output)),
        ]
        oracle_gradients = torch.autograd.grad(
            oracle_loss, parameters, allow_unused=True
        )
    for index, (actual, expected) in enumerate(
        zip(actual_values, expected_values, strict=True)
    ):
        if mixed_precision:
            _assert_precision_error(actual, expected, oracle_values[index], tolerance)
        else:
            torch.testing.assert_close(actual, expected, **tolerance)
    for index, (expected, actual) in enumerate(
        zip(expected_gradients, actual_gradients, strict=True)
    ):
        if expected is None:
            assert actual is None
            if mixed_precision:
                assert oracle_gradients[index] is None
        elif mixed_precision:
            _assert_precision_error(
                actual, expected, oracle_gradients[index], gradient_tolerance
            )
        else:
            torch.testing.assert_close(actual, expected, **gradient_tolerance)


def _assert_precision_error(actual, reference, oracle, tolerance):
    actual, reference, oracle = (
        torch.as_tensor(value).float() for value in (actual, reference, oracle)
    )
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    baseline_error = reference - oracle
    batch_error = actual - oracle
    for reduction in (
        lambda value: value.abs().max(),
        lambda value: value.square().mean().sqrt(),
    ):
        budget = (
            2 * reduction(baseline_error)
            + tolerance["atol"]
            + tolerance["rtol"] * reduction(oracle)
        )
        assert reduction(batch_error) <= budget
