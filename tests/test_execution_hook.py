from dataclasses import dataclass, field
from enum import Enum

from vllm_runtime_sim.backend import AnalyticalBackend
from vllm_runtime_sim.execution_hook import (
    ModelRunnerSimulationHook,
    SyntheticModelRunnerOutput,
    SyntheticOutputFactory,
)


class Mode(Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2


@dataclass
class HF:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 2
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128


@dataclass
class DType:
    itemsize: int = 2


@dataclass
class ModelConfig:
    hf_text_config: object = field(default_factory=HF)
    dtype: object = field(default_factory=DType)


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 4


@dataclass
class VllmConfig:
    model_config: object = field(default_factory=ModelConfig)
    parallel_config: object = field(default_factory=ParallelConfig)


@dataclass
class NewReq:
    req_id: str
    num_computed_tokens: int
    prompt_token_ids: list[int]

    @property
    def prompt_len(self):
        return len(self.prompt_token_ids)


@dataclass
class Cached:
    req_ids: list[str]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]


@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list
    scheduled_cached_reqs: Cached
    num_scheduled_tokens: dict
    scheduled_spec_decode_tokens: dict
    finished_req_ids: set


@dataclass
class Batch:
    num_tokens: int
    num_reqs: int | None = 1
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0


@dataclass
class ForwardContext:
    cudagraph_runtime_mode: Mode
    batch_descriptor: Batch | None
    ubatch_slices: list | None = None


@dataclass
class ExecutionDescriptor:
    cg_mode: Mode
    num_tokens: int
    num_reqs: int | None
    uniform_token_count: int | None = None
    max_query_len: int | None = None
    num_active_loras: int = 0


class Model:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        self.calls += 1
        return "real"


class CudaGraphManager:
    def __init__(self):
        self.full_calls = 0
        self.pw_calls = 0

    def run_fullgraph(self, desc):
        self.full_calls += 1
        return "full-real"

    def run_pw_graph(self, *args, **kwargs):
        self.pw_calls += 1
        return "pw-real"


class Runner:
    def __init__(self, mode, context):
        self.vllm_config = VllmConfig()
        self.model = Model()
        self.cudagraph_manager = CudaGraphManager()
        self.mode = mode
        self.context = context

    def execute_model(self, scheduler_output):
        if self.mode is Mode.FULL:
            return self.cudagraph_manager.run_fullgraph(
                ExecutionDescriptor(Mode.FULL, 8, 1, 1)
            )
        if self.mode is Mode.PIECEWISE:
            return self.cudagraph_manager.run_pw_graph(self.model, {})
        return self.model(x=1)


def output(prompt_len=4, scheduled=4):
    return SchedulerOutput(
        scheduled_new_reqs=[NewReq("A", 0, list(range(prompt_len)))],
        scheduled_cached_reqs=Cached([], [], []),
        num_scheduled_tokens={"A": scheduled},
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
    )


def hook_runner(mode, context):
    runner = Runner(mode, context)
    hook = ModelRunnerSimulationHook(
        runner,
        AnalyticalBackend(),
        forward_context_getter=lambda: context,
        output_factory=SyntheticOutputFactory(
            output_cls=SyntheticModelRunnerOutput
        ),
    ).install()
    return runner, hook


def test_eager_cut_returns_synthetic_and_skips_model():
    context = ForwardContext(Mode.NONE, None)
    runner, hook = hook_runner(Mode.NONE, context)
    result = runner.execute_model(output())

    assert result.sampled_token_ids == [[1]]
    assert runner.model.calls == 0
    assert hook.last_result.step.runtime.cudagraph_mode.value == "none"


def test_piecewise_cut_uses_live_forward_context():
    context = ForwardContext(Mode.PIECEWISE, Batch(8, 1))
    runner, hook = hook_runner(Mode.PIECEWISE, context)
    result = runner.execute_model(output())

    assert result.sampled_token_ids == [[1]]
    assert runner.cudagraph_manager.pw_calls == 0
    assert hook.last_result.step.runtime.execution_tokens == 8
    assert hook.last_result.step.runtime.cudagraph_mode.value == "piecewise"


def test_full_cut_uses_execution_descriptor_without_forward_context():
    context = ForwardContext(Mode.NONE, None)
    runner, hook = hook_runner(Mode.FULL, context)
    result = runner.execute_model(output())

    assert result.sampled_token_ids == [[1]]
    assert runner.cudagraph_manager.full_calls == 0
    runtime = hook.last_result.step.runtime
    assert runtime.execution_tokens == 8
    assert runtime.cudagraph_mode.value == "full"
    assert runtime.uniform_token_count == 1


def test_partial_prefill_emits_no_token():
    context = ForwardContext(Mode.NONE, None)
    runner, _hook = hook_runner(Mode.NONE, context)
    result = runner.execute_model(output(prompt_len=8, scheduled=4))

    assert result.sampled_token_ids == [[]]


def test_hook_passthrough_outside_execute_model():
    context = ForwardContext(Mode.NONE, None)
    runner, _hook = hook_runner(Mode.NONE, context)

    assert runner.model() == "real"
    assert runner.model.calls == 1
    desc = ExecutionDescriptor(Mode.FULL, 8, 1, 1)
    assert runner.cudagraph_manager.run_fullgraph(desc) == "full-real"
    assert runner.cudagraph_manager.full_calls == 1
