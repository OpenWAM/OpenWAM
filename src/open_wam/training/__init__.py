"""Training entrypoints and runtime components inside the source package."""

from .checkpoints import CheckpointManager
from .cli import (
    TrainCliOverrides,
    apply_config_overrides,
    apply_train_cli_overrides,
    build_train_arg_parser,
    load_training_cli_config,
    parse_override_assignments,
    parse_train_cli,
    resolve_experiment_config_path,
)
from .controls import (
    TrainabilityReport,
    apply_training_component_controls,
    normalize_component_selectors,
)
from .launch import (
    DistributedLaunchContext,
    LaunchEnvironment,
    validate_training_launch,
)
from .logging import (
    CompositeLogSink,
    ConsoleLogSink,
    JsonlLogSink,
    NoopLogSink,
    WandBLogSink,
)
from .loop_policies import EpochLoopPolicy, StepLoopPolicy
from .optim import build_optimizer, build_scheduler
from .runtime import TrainingRuntime
from .state import TrainState
from .step_executor import (
    LatentBatchAdapter,
    PipelineTrainStepExecutor,
    ViewBatchAdapter,
    build_batch_adapter,
)
from .strategies import (
    DistributedStrategy,
    SingleDeviceStrategy,
    build_training_strategy,
)

__all__ = [
    "CheckpointManager",
    "CompositeLogSink",
    "ConsoleLogSink",
    "DistributedLaunchContext",
    "DistributedStrategy",
    "EpochLoopPolicy",
    "JsonlLogSink",
    "LatentBatchAdapter",
    "LaunchEnvironment",
    "NoopLogSink",
    "PipelineTrainStepExecutor",
    "SingleDeviceStrategy",
    "StepLoopPolicy",
    "TrainCliOverrides",
    "TrainState",
    "TrainabilityReport",
    "TrainingRuntime",
    "ViewBatchAdapter",
    "WandBLogSink",
    "apply_config_overrides",
    "apply_train_cli_overrides",
    "apply_training_component_controls",
    "build_batch_adapter",
    "build_optimizer",
    "build_scheduler",
    "build_train_arg_parser",
    "build_training_strategy",
    "load_training_cli_config",
    "normalize_component_selectors",
    "parse_override_assignments",
    "parse_train_cli",
    "resolve_experiment_config_path",
    "validate_training_launch",
]
