"""Capture the blocking LIBERO driver with real reduced policy models.

Only RGB encoding, simulator I/O, and disk rendering are substituted. Inference,
composition, proprio, action conversion and observed-history commits are real.
The frozen signatures were recorded before the RolloutEngine migration.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import numpy as np
import torch

from open_wam.configs import (
    ActionSchemaConfig,
    ActionTargetConfig,
    ActionTargetStateEncoding,
    DualExpertActionDecoderConfig,
    DualExpertPolicyConfig,
    ExperimentConfig,
    InferenceConfig,
    ProprioContextMode,
    RobotWinDataConfig,
    TrainingConfig,
    VideoActionProgram,
    VideoActionSequenceContract,
    DynamicsObjective,
    ParallelStreamPolicyConfig,
    ParallelStreamActionDecoderConfig,
    CausalVideoPredictionPolicyConfig,
    CausalVideoProgram,
    VideoOnlyActionDecoderConfig,
)
from open_wam.evals import libero_policy_inputs as inputs
from open_wam.evals import libero_policy_rollout as rollout
from open_wam.evals import libero_policy_planner as planner
from open_wam.evals.libero_rollout_artifacts import extract_predicted_latents
from open_wam.evals.libero_policy_composition import VideoActionComposition
from open_wam.evals.libero_policy_runtime import LiberoPolicyRuntime, PolicyActionRoute
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.training_provenance import PolicyTrainingProvenance
from open_wam.pipelines import (
    VariantRolloutRunner,
    build_variant_pipeline_from_config,
    resolve_policy_video_action_consumer_plan,
    resolve_policy_video_producer_plan,
)
from open_wam.utils import seed_everywhere
from open_wam.contracts import identify_video_latent_space


def _signature(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(k): _signature(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_signature(v) for v in value]
    return value


def capture(
    monkeypatch,
    *,
    program="video_then_action",
    streaming=True,
    consumer=None,
    limit="chunks",
    prefix=None,
    reset=False,
    seed=83,
    startup=1,
    route="native",
    architecture="dual_expert",
    episode_module=rollout,
):
    trace = {"frontend": [], "predictions": [], "commits": []}

    def runtime(name):
        torch.manual_seed(1901)
        config = ExperimentConfig(
            data=RobotWinDataConfig(
                num_frames=4,
                action_target=ActionTargetConfig(
                    state_encoding=ActionTargetStateEncoding.EEF_POS_AXISANGLE_GRIPPER_2D
                ),
                action_schema=ActionSchemaConfig(
                    action_dim=7, action_horizon=8, state_dim=8, state_horizon=1
                ),
            ),
            backbone=SharedVideoTransformerConfig(
                implementation="shared_transformer",
                hidden_size=32,
                num_layers=2,
                num_heads=4,
                attention_head_dim=8,
                ffn_dim=64,
                text_dim=16,
                freq_dim=8,
                load_reference_core_weights=False,
                load_text_conditioning=False,
                load_wan_vae_frontend=False,
            ),
            policy_variant=DualExpertPolicyConfig(
                hidden_size=32,
                program=VideoActionProgram.VIDEO_THEN_ACTION
                if name == "causal"
                else VideoActionProgram(name),
                num_action_layers=2,
                proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
                sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
            ),
            action_decoder=DualExpertActionDecoderConfig(
                hidden_size=32, action_dim=7, action_horizon=8
            ),
            training=TrainingConfig(chunk_size=2, window_size=8),
            inference=InferenceConfig(
                frame_chunk_size=2,
                attention_window_size=4,
                video_num_inference_steps=2,
                action_num_inference_steps=2,
            ),
        )
        if name == "causal":
            config = replace(
                config,
                policy_variant=CausalVideoPredictionPolicyConfig(
                    hidden_size=32,
                    program=CausalVideoProgram.CHUNKED_CONDITIONED_VIDEO,
                    noisy_video_condition_prob=0.5,
                ),
                action_decoder=VideoOnlyActionDecoderConfig(
                    hidden_size=32, action_dim=7
                ),
            )
        elif architecture == "parallel_stream":
            config = replace(
                config,
                data=replace(
                    config.data,
                    action_schema=replace(config.data.action_schema, action_horizon=16),
                ),
                policy_variant=ParallelStreamPolicyConfig(
                    hidden_size=32,
                    program=VideoActionProgram(name),
                    frame_chunk_size=2,
                    action_per_frame=4,
                    proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
                ),
                action_decoder=ParallelStreamActionDecoderConfig(
                    hidden_size=32, action_dim=7, action_horizon=16
                ),
            )
        pipeline = build_variant_pipeline_from_config(config).eval()
        pipeline.policy_variant.initialize_for_training(pipeline.visual_tower)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}\n")
            (root / "model.safetensors").write_bytes(b"test encoder")
            pipeline.visual_tower.frontend.reference_assets.latent_space_identity = (
                identify_video_latent_space(
                    root, encoder_family="fixture", encoding_contract="fixture.stride4"
                )
            )
        runner = VariantRolloutRunner(pipeline)
        original_infer, original_commit = (
            runner.infer_prepared_step,
            runner.reconcile_observed_history,
        )

        def infer(**kwargs):
            output = original_infer(**kwargs)
            trace["predictions"].append(
                _signature(
                    {
                        "model": name,
                        "state": kwargs["context"].state,
                        "video": extract_predicted_latents(output.infer_output),
                        "action": None
                        if output.action_plan is None
                        else output.action_plan.actions,
                        "origin": output.infer_output.policy_output.generation_frame_start,
                    }
                )
            )
            return output

        def commit(**kwargs):
            history = kwargs["history"]
            output = original_commit(**kwargs)
            trace["commits"].append(
                _signature(
                    {
                        "model": name,
                        "video": history.video_latents,
                        "actions": history.action_history,
                        "proprio": history.proprio_history,
                        "applied": output.applied,
                        "debug": output.debug,
                        "cursor": output.session.policy_state.cursor.current_start_frame,
                    }
                )
            )
            return output

        monkeypatch.setattr(runner, "infer_prepared_step", infer)
        monkeypatch.setattr(runner, "reconcile_observed_history", commit)
        return LiberoPolicyRuntime(
            config,
            Path("/checkpoint"),
            pipeline,
            runner,
            {},
            *([torch.device("cpu")] * 4),
            17,
            startup,
            5,
            streaming,
        )

    primary = runtime(program)
    composition = None
    if consumer is not None:
        secondary = runtime(consumer)
        composition = VideoActionComposition(
            secondary,
            resolve_policy_video_producer_plan(primary.pipeline.policy_variant),
            resolve_policy_video_action_consumer_plan(
                secondary.pipeline.policy_variant,
                training=PolicyTrainingProvenance(
                    frozenset({DynamicsObjective.VIDEO_CONDITIONED_ACTION})
                ),
            ),
            {},
        )
        route = PolicyActionRoute.GENERATED_VIDEO_THEN_ACTION.value

    def frontend(pipeline, *, views, preserve_stream_cache=False, **kwargs):
        # Exercise RNG isolation and cache ordering without checkpoint-sized VAEs.
        rgb = next(iter(views.values()))
        count = rgb.shape[0]
        values = rgb[:, 0, 0, 0].float()
        indices = (
            list(range(3, count, 4))
            if preserve_stream_cache
            else list(range(0, count, 4))
        )
        if not indices:
            indices = [count - 1]
        video = (
            values[indices][None, None, :, None, None].expand(1, 48, -1, 4, 4).clone()
        )
        video += torch.rand(()) * 0.01
        trace["frontend"].append(
            _signature(
                {
                    "raw": values,
                    "stream": kwargs["use_streaming_frontend"],
                    "preserve": preserve_stream_cache,
                    "video": video,
                }
            )
        )
        return pipeline.prepare_visual_outputs_from_latents(
            video,
            text_context=torch.ones(1, 3, 16),
            negative_text_context=torch.zeros(1, 3, 16),
        )

    monkeypatch.setattr(inputs, "_prepare_policy_visual_outputs", frontend)
    monkeypatch.setattr(planner, "_prepare_policy_visual_outputs", frontend)
    if episode_module is not rollout:
        monkeypatch.setattr(episode_module, "_prepare_policy_visual_outputs", frontend)

    class Environment:
        def __init__(self):
            self.env = self
            self.timestep = 0
            self.done = False
            self.closed = False
            self.actions = []
            self.position = np.zeros(3, dtype=np.float32)

        def reset(self):
            pass

        def set_init_state(self, state):
            pass

        def step(self, action):
            action = np.asarray(action, dtype=np.float32)
            self.timestep += 1
            self.actions.append(action.copy())
            self.position += action[:3] * 0.01
            frame = np.full((4, 4, 3), self.timestep, dtype=np.uint8)
            obs = {
                "agentview_image": frame,
                "robot0_eye_in_hand_image": frame + 1,
                "robot0_eef_pos": self.position.copy(),
                "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "robot0_gripper_qpos": np.asarray([0.01, -0.01], dtype=np.float32),
            }
            self.done = (limit == "terminal" and self.timestep == 16) or (
                limit == "startup_terminal" and self.timestep >= 5
            )
            return obs, 0.0, limit == "success" and self.timestep == 16, {}

        def close(self):
            self.closed = True

    def persist(**kwargs):
        payload = kwargs["payload"]
        trace.update(
            _signature(
                {
                    "observations": payload.real_observations,
                    "actions": payload.action_trace,
                    "videos": payload.predicted_latent_chunks,
                    "events": payload.chunk_events,
                }
            )
        )
        summary = dict(kwargs["summary"])
        summary.pop("checkpoint_file")
        summary.pop("video_action_composition", None)
        summary["libero_renderer"].pop("source_path")
        return SimpleNamespace(summary=summary)

    monkeypatch.setattr(episode_module, "persist_libero_rollout_artifacts", persist)
    env = Environment()
    options = episode_module.LiberoPolicyEpisodeOptions(
        "libero_10",
        2,
        3,
        {"time": 16, "startup_time": 5}.get(limit, 100),
        0 if limit == "empty" else 3,
        prefix,
        None,
        None,
        4,
        False,
        route,
        reset,
        None,
        "/output",
        "parity",
        20.0,
        seed,
        save_rollout_video=True,
    )
    task = episode_module.LiberoPolicyTaskResources(None, "move the object", [None])
    seed_everywhere(178)
    trace["summary"] = episode_module.run_libero_policy_episode(
        options,
        primary,
        task,
        env,
        include_episode_coordinates=True,
        close_env_after_rollout=True,
        video_action_composition=composition,
    )
    trace["rng"] = _signature(torch.get_rng_state())
    trace["closed"] = env.closed
    return trace


CASES = {
    **{
        name: {"program": name}
        for name in (
            "joint",
            "video_then_action",
            "action_then_video",
            "decoupled_same_step",
            "video_noisy_to_action",
            "action_noisy_to_video",
            "generalist_joint_denoising",
        )
    },
    "offline": {"streaming": False},
    "prefix": {"prefix": 4},
    "reset": {"reset": True},
    "unseeded": {"seed": None},
    "startup": {"startup": 3},
    **{
        name: {"limit": name}
        for name in (
            "time",
            "success",
            "terminal",
            "empty",
            "startup_terminal",
            "startup_time",
        )
    },
    "two_vta": {"consumer": "video_then_action"},
    "two_vta_offline": {"consumer": "video_then_action", "streaming": False},
    "two_vta_prefix": {"consumer": "video_then_action", "prefix": 4},
    "two_vta_terminal": {"consumer": "video_then_action", "limit": "terminal"},
    "strict_idm": {"consumer": "inverse_dynamics"},
    "gjd_idm": {"consumer": "generalist_joint_denoising"},
    "causal_vta": {"program": "causal", "consumer": "video_then_action"},
    "causal_idm": {"program": "causal", "consumer": "inverse_dynamics"},
    "parallel_vta": {"architecture": "parallel_stream"},
    "parallel_joint": {"architecture": "parallel_stream", "program": "joint"},
    "joint_idm": {
        "program": "generalist_joint_denoising",
        "route": "joint_video_then_idm",
    },
}


if __name__ == "__main__":
    import argparse
    import pytest

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-driver", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = {}
    episode_module = rollout
    if args.reference_driver is not None:
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "libero_reference", args.reference_driver
        )
        episode_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = episode_module
        spec.loader.exec_module(episode_module)
    for name, case in CASES.items():
        with pytest.MonkeyPatch.context() as patch:
            records[name] = capture(patch, **case, episode_module=episode_module)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
