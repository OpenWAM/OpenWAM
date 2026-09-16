"""Real reduced policies at blocking and externally driven simulator boundaries."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from open_wam.runtime.policy_planner import PolicyPlanner
from open_wam.configs import (
    ActionNormalizationConfig,
    ActionSchemaConfig,
    ActionTargetConfig,
    CalvinDataConfig,
    DualExpertActionDecoderConfig,
    DualExpertPolicyConfig,
    ExperimentConfig,
    InferenceConfig,
    ParallelStreamActionDecoderConfig,
    ParallelStreamPolicyConfig,
    SharedVideoTransformerConfig,
    TrainingConfig,
    ViewLayoutConfig,
)
from open_wam.integrations.calvin_env import (
    OpenWAMCalvinCustomModel,
    _materialize_calvin_control,
)
from open_wam.pipelines import VariantRolloutRunner, build_variant_pipeline_from_config
from open_wam.simulators.contracts import (
    EpisodeSpec,
    SimulatorCapabilities,
    SimulatorObservation,
)
from open_wam.runtime.control import ControlTransition
from open_wam.simulators.policy_adapter import SimulatorPolicyAdapter
from open_wam.simulators.rollout import (
    run_closed_loop_sim_rollout,
    run_zero_control_smoke,
    summarize_sim_rollout,
)


def _pipeline(architecture, density=1, *, proprio="per_chunk_additive"):
    torch.manual_seed(13)
    data = CalvinDataConfig(
        num_frames=2,
        canonical_height=32,
        canonical_width=32,
        view_layout=tuple(
            ViewLayoutConfig(
                source_name=name,
                canonical_name=name,
                top=i * 16,
                left=0,
                height=16,
                width=32,
            )
            for i, name in enumerate(("rgb_static", "rgb_gripper"))
        ),
        action_schema=ActionSchemaConfig(
            action_dim=7, action_horizon=2 * density, state_dim=4, state_horizon=1
        ),
        action_target=ActionTargetConfig(
            normalization=ActionNormalizationConfig(
                mode="gaussian", mean=(2.0,) * 7, std=(3.0,) * 7
            )
        ),
    )
    policy = (
        DualExpertPolicyConfig(
            hidden_size=128, num_action_layers=2, program="video_then_action"
        )
        if architecture == "dual_expert"
        else ParallelStreamPolicyConfig(
            hidden_size=128,
            frame_chunk_size=2,
            action_per_frame=density,
            program="video_then_action",
        )
    )
    policy = replace(
        policy, program="video_then_action", proprio_context_mode=proprio
    )
    decoder = (
        DualExpertActionDecoderConfig
        if architecture == "dual_expert"
        else ParallelStreamActionDecoderConfig
    )
    config = ExperimentConfig(
        data=data,
        policy_variant=policy,
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            attention_head_dim=32,
            ffn_dim=256,
            text_dim=16,
            freq_dim=8,
            load_reference_core_weights=False,
            load_text_conditioning=False,
            load_wan_vae_frontend=False,
        ),
        action_decoder=decoder(
            hidden_size=128, action_dim=7, action_horizon=2 * density
        ),
        training=TrainingConfig(chunk_size=2, window_size=4),
        inference=InferenceConfig(
            frame_chunk_size=2,
            video_num_inference_steps=2,
            action_num_inference_steps=2,
            attention_window_size=4,
        ),
    )
    pipeline = build_variant_pipeline_from_config(config).eval()
    # A real causal strided frontend, without a heavyweight external VAE asset.
    frontend = pipeline.visual_tower.frontend
    frontend.latentizer.stride = (density, *frontend.latentizer.stride[1:])
    return pipeline, data


class ObservedEnvironment:
    benchmark_name = "observed_test"
    capabilities = SimulatorCapabilities(action_step_semantics="single_env_step")

    def __init__(self, stop_after=4):
        self.actions = []
        self.spec = None
        self.stop_after = stop_after

    def observation(self):
        index = len(self.actions)
        # Spatial variation survives the frontend's group normalization.
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, : 4 + index * 3] = 200
        return SimulatorObservation(
            views={name: image.copy() for name in ("rgb_static", "rgb_gripper")},
            state=np.full(4, index, dtype=np.float32),
            task_text="move the block",
        )

    def reset(self, spec):
        self.spec = spec
        self.actions.clear()
        return self.observation()

    def task_text(self):
        return "move the block"

    def materialize_control(self, source_action, *, data_config):
        return _materialize_calvin_control(source_action)

    def step(self, action):
        self.actions.append(action.copy())
        return ControlTransition(
            observation=self.observation(),
            done=len(self.actions) == self.stop_after,
            success=False,
            reward=-1.0,
            info={"index": len(self.actions)},
        )

    def render_frame(self, observation):
        return observation.views["rgb_static"]


def test_zero_control_smoke_uses_source_conversion_without_a_fake_policy():
    data = CalvinDataConfig(
        action_schema=ActionSchemaConfig(action_dim=7, action_horizon=4, state_dim=4),
        action_target=ActionTargetConfig(normalization=ActionNormalizationConfig(
            mode="gaussian", mean=(2.,) * 7, std=(3.,) * 7)),
    )
    environment = ObservedEnvironment()
    result = run_zero_control_smoke(
        adapter=environment, data_config=data, device=torch.device("cpu"),
        task_id=1, episode_idx=2, seed=3, max_steps=9,
    )
    assert result.steps == 4 and not result.success
    assert result.policy_action_shapes == ()
    assert result.mean_policy_step_s is None
    assert result.live_wall_time_s == result.wall_time_s
    assert len(result.video_frames) == 4
    np.testing.assert_array_equal(np.stack(environment.actions), np.tile([2.] * 6 + [1.], (4, 1)))


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize("commit", ["first_frame", "full_chunk"])
def test_public_rollout_reconciles_real_observations_controls_and_proprio(
    architecture, commit, monkeypatch
):
    pipeline, data = _pipeline(architecture)
    runner = VariantRolloutRunner(pipeline)
    environment = ObservedEnvironment()
    commits, outputs = [], []
    reconcile, infer = runner.reconcile_observed_history, runner.infer_prepared_step

    def capture_commit(*, session, history):
        update = reconcile(session=session, history=history)
        assert update.applied
        commits.append(history)
        return update

    def capture_infer(**kwargs):
        output = infer(**kwargs)
        outputs.append(output)
        return output

    monkeypatch.setattr(runner, "reconcile_observed_history", capture_commit)
    monkeypatch.setattr(runner, "infer_prepared_step", capture_infer)
    result = run_closed_loop_sim_rollout(
        adapter=environment,
        rollout_runner=runner,
        data_config=data,
        device=torch.device("cpu"),
        task_id=1,
        episode_idx=2,
        seed=3,
        max_steps=9,
        action_commit_mode=commit,
    )
    assert result.steps == 4 and not result.success
    assert environment.spec == EpisodeSpec(task_id=1, episode_idx=2, seed=3)
    assert result.termination.reason.value == "env_terminal"
    assert result.termination.transition.info == {"index": 4}
    assert result.planner_teardown.error is None
    assert result.achieved_action_hz == result.steps / result.live_wall_time_s
    assert result.wall_time_s >= result.live_wall_time_s + result.planner_teardown.drain_time_s
    assert len(result.video_frames) == 4
    assert len(commits) == (3 if commit == "first_frame" else 1)
    assert all(record["reward"] == -1.0 for record in result.action_records)
    source = pipeline.action_adapter.to_source(outputs[0].action_plan.actions)[
        0
    ].numpy()
    np.testing.assert_array_equal(
        environment.actions[0], _materialize_calvin_control(source).action
    )
    actual_model = pipeline.action_adapter.to_model(
        torch.from_numpy(np.stack(environment.actions))
    )
    position = 0
    for history in commits:
        count = history.action_history.shape[1]
        torch.testing.assert_close(
            history.action_history[0],
            actual_model[position : position + count],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            history.proprio_history,
            torch.arange(position + 1, position + count + 1)
            .float()[:, None]
            .expand(-1, 4),
            rtol=0,
            atol=0,
        )
        # Independently encode the actual interval, not the speculative model output.
        env = ObservedEnvironment()
        observations = []
        for index in range(position, position + count + 1):
            env.actions = [None] * index
            observations.append(env.observation())
        adapter = SimulatorPolicyAdapter(
            runner, data, torch.device("cpu"), _materialize_calvin_control
        )
        with torch.no_grad():
            visual, _ = adapter.prepare(
                observations, runner.reset(task_text=(environment.task_text(),))
            )
        torch.testing.assert_close(
            history.video_latents,
            visual.frontend.video_latents[:, :, 1:],
            rtol=0,
            atol=0,
        )
        position += count


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
def test_simulator_throughput_excludes_unused_planner_drain(architecture, monkeypatch):
    from threading import Event, Timer

    from open_wam.configs.enums import RealtimePlannerMode
    from open_wam.runtime import rollout_engine
    from open_wam.runtime.planner_executor import PlannerExecutor

    pipeline, data = _pipeline(architecture)
    environment = ObservedEnvironment(stop_after=2)
    started, release = Event(), Event()
    prepare = SimulatorPolicyAdapter.prepare
    step = environment.step
    close = PlannerExecutor.close
    options = rollout_engine.RolloutOptions
    prepares = 0

    def delayed_prepare(adapter, *args):
        nonlocal prepares
        prepares += 1
        if prepares == 2:
            started.set()
            assert release.wait(5)
            raise RuntimeError("unused terminal plan")
        return prepare(adapter, *args)

    def terminal_step(action):
        transition = step(action)
        if transition.done:
            assert started.wait(5)
        return transition

    def drain(worker):
        # Hold a real unused planner until after the live loop has terminated.
        timer = Timer(0.1, release.set)
        timer.start()
        try:
            return close(worker)
        finally:
            release.set()
            timer.join()

    monkeypatch.setattr(SimulatorPolicyAdapter, "prepare", delayed_prepare)
    monkeypatch.setattr(environment, "step", terminal_step)
    monkeypatch.setattr(PlannerExecutor, "close", drain)
    monkeypatch.setattr(
        rollout_engine, "RolloutOptions",
        lambda **kwargs: options(
            **kwargs, planner_mode=RealtimePlannerMode.ASYNC_HISTORY_FIRST,
            replan_low_watermark_actions=1,
        ),
    )
    try:
        result = run_closed_loop_sim_rollout(
            adapter=environment, rollout_runner=VariantRolloutRunner(pipeline),
            data_config=data, device=torch.device("cpu"), task_id=None,
            episode_idx=None, seed=None, max_steps=9, action_commit_mode="full_chunk",
        )
    finally:
        release.set()
    assert result.steps == 2
    assert result.termination.reason.value == "env_terminal"
    assert result.planner_teardown.error == "RuntimeError: unused terminal plan"
    assert result.planner_teardown.drain_time_s > 0
    assert result.wall_time_s >= result.live_wall_time_s + result.planner_teardown.drain_time_s
    assert result.achieved_action_hz == result.steps / result.live_wall_time_s
    assert result.achieved_action_hz > result.steps / result.wall_time_s
    summary = summarize_sim_rollout(result)
    assert summary["live_wall_time_s"] == result.live_wall_time_s
    assert summary["planner_drain_time_s"] == result.planner_teardown.drain_time_s


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize("density", [1, 2, 4])
def test_external_calvin_and_blocking_driver_have_identical_policy_lifecycle(
    architecture,
    density,
):
    pipeline, data = _pipeline(architecture, density)
    runner = VariantRolloutRunner(pipeline)
    blocking = ObservedEnvironment(stop_after=4 * density)
    torch.manual_seed(902)
    run_closed_loop_sim_rollout(
        adapter=blocking,
        rollout_runner=runner,
        data_config=data,
        device=torch.device("cpu"),
        task_id=None,
        episode_idx=None,
        seed=None,
        max_steps=4 * density,
    )
    external = OpenWAMCalvinCustomModel(
        rollout_runner=runner, data_config=data, device=torch.device("cpu")
    )
    environment = ObservedEnvironment(stop_after=4 * density)
    torch.manual_seed(902)
    try:
        for _ in range(4 * density):
            obs = environment.observation()
            action = external.step(
                {**obs.views, "robot_obs": obs.state}, goal=obs.task_text
            )
            environment.step(action)
    finally:
        external.reset()
    np.testing.assert_array_equal(
        np.stack(environment.actions), np.stack(blocking.actions)
    )
    receipt = external.last_rollout_result
    assert receipt.termination.reason.value == "cancelled"
    # The external API has not supplied the final executed action's observation.
    assert receipt.termination.control_count == 4 * density - 1
    assert receipt.planner_teardown.error is None
    external.reset()
    assert external.last_rollout_result is receipt
    # Reset must also make a fresh episode safe on the SAME pipeline.
    torch.manual_seed(902)
    initial = ObservedEnvironment().observation()
    try:
        first = external.step(
            {**initial.views, "robot_obs": initial.state}, goal=initial.task_text
        )
        np.testing.assert_array_equal(first, blocking.actions[0])
    finally:
        external.reset()
    assert external.last_rollout_result is not receipt


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("bad_state", [None, np.ones(2), np.ones((1, 4)), np.full(4, np.nan)])
def test_proprio_contract_rejects_missing_or_malformed_observations(
    architecture, index, bad_state
):
    pipeline, data = _pipeline(architecture)
    runner = VariantRolloutRunner(pipeline)
    adapter = SimulatorPolicyAdapter(
        runner, data, torch.device("cpu"), _materialize_calvin_control
    )
    observations = [ObservedEnvironment().observation()] * 3
    observations[index] = replace(observations[index], state=bad_state)
    session = runner.reset()
    with pytest.raises(ValueError, match="proprio|Proprio"):
        adapter.prepare(tuple(observations), session)
    assert session.policy_state is None


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize("density", [1, 2, 4])
def test_proprio_history_never_compresses_a_missing_middle_boundary(architecture, density):
    from open_wam.models.policy_variants import PolicyTemporalSpan

    pipeline, data = _pipeline(architecture, density)
    runner = VariantRolloutRunner(pipeline)
    adapter = SimulatorPolicyAdapter(
        runner, data, torch.device("cpu"), _materialize_calvin_control
    )
    observations = [ObservedEnvironment().observation()] * (3 * density + 1)
    visual, _ = adapter.prepare(tuple(observations), runner.reset())
    observations[2 * density] = replace(observations[2 * density], state=None)
    with pytest.raises(ValueError, match="observation 1"):
        adapter.observed_history(
            tuple(observations), (np.zeros(7),) * (3 * density), visual,
            PolicyTemporalSpan(1, 3), runner.reset(),
        )


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
def test_no_proprio_policy_accepts_missing_state_through_the_real_lifecycle(architecture):
    pipeline, data = _pipeline(architecture, proprio="none")

    class NoStateEnvironment(ObservedEnvironment):
        def observation(self):
            return replace(super().observation(), state=None)

    result = run_closed_loop_sim_rollout(
        adapter=NoStateEnvironment(), rollout_runner=VariantRolloutRunner(pipeline),
        data_config=data, device=torch.device("cpu"), task_id=None, episode_idx=None,
        seed=None, max_steps=4,
    )
    assert result.steps == 4


@pytest.mark.parametrize("architecture", ["dual_expert", "parallel_stream"])
@pytest.mark.parametrize("density", [1, 2, 4])
def test_blocking_paced_and_external_drivers_publish_identical_lifecycles(architecture, density):
    from open_wam.runtime.rollout_engine import RolloutEngine, RolloutOptions

    results = []
    for driver in ("blocking", "paced", "external"):
        pipeline, data = _pipeline(architecture, density)
        runner = VariantRolloutRunner(pipeline)
        environment = ObservedEnvironment(stop_after=4 * density)
        adapter = SimulatorPolicyAdapter(
            runner, data, torch.device("cpu"), _materialize_calvin_control
        )
        engine = RolloutEngine(PolicyPlanner(runner, adapter), adapter, RolloutOptions(
            max_actions=5 * density,
            target_action_hz=10000 if driver == "paced" else None,
            execute_prefix_actions=density,
        ))
        session = runner.reset(task_text=(environment.task_text(),))
        torch.manual_seed(902)
        if driver == "external":
            stream = engine.control_stream(environment.observation(), session)
            try:
                action = next(stream)
                while True:
                    transition = environment.step(action)
                    try:
                        action = stream.send(transition)
                    except StopIteration as stopped:
                        result = stopped.value
                        break
            finally:
                stream.close()
        else:
            result = engine.run(environment.observation(), session, step=environment.step)
        results.append(result)

    reference = results[0]
    for actual in results[1:]:
        assert actual.lifecycle.revision == reference.lifecycle.revision
        assert actual.lifecycle.next_control_index == reference.lifecycle.next_control_index
        assert actual.lifecycle.observed_control_end == reference.lifecycle.observed_control_end
        assert actual.termination.reason is reference.termination.reason
        assert actual.termination.transition.info == reference.termination.transition.info
        assert actual.session.policy_state.cursor == reference.session.policy_state.cursor
        assert actual.session.policy_state.revision == reference.session.policy_state.revision
        for key in ("action", "source_action", "done", "success"):
            assert [r.to_record()[key] for r in actual.actions] == [r.to_record()[key] for r in reference.actions]
        for key in ("accepted", "base_revision", "accepted_revision", "acceptance_action_index"):
            assert [getattr(r, key) for r in actual.replans] == [getattr(r, key) for r in reference.replans]
