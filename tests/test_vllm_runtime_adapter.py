from dataclasses import dataclass
from enum import Enum

from vllm_runtime_sim import (
    CudaGraphMode,
    QWEN3_8B,
    RuntimeContextAdapter,
    SchedulerOutputAdapter,
    lower_dense_decoder,
)


class FakeCudaGraphMode(Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2


@dataclass
class FakeNewReq:
    req_id: str
    num_computed_tokens: int


@dataclass
class FakeCached:
    req_ids: list[str]
    num_computed_tokens: list[int]


@dataclass
class FakeSchedulerOutput:
    scheduled_new_reqs: list[FakeNewReq]
    scheduled_cached_reqs: FakeCached
    num_scheduled_tokens: dict[str, int]
    scheduled_spec_decode_tokens: dict[str, list[int]]
    finished_req_ids: set[str]


@dataclass
class FakeBatchDescriptor:
    num_tokens: int
    num_reqs: int | None
    uniform: bool
    has_lora: bool = False
    num_active_loras: int = 0


@dataclass
class FakeForwardContext:
    cudagraph_runtime_mode: FakeCudaGraphMode
    batch_descriptor: FakeBatchDescriptor | None
    ubatch_slices: list[object] | None = None


@dataclass
class FakeParallelConfig:
    tensor_parallel_size: int


@dataclass
class FakeDType:
    itemsize: int


@dataclass
class FakeModelConfig:
    dtype: FakeDType


@dataclass
class FakeVllmConfig:
    parallel_config: FakeParallelConfig
    model_config: FakeModelConfig


class FakeAttentionBackend:
    @staticmethod
    def get_name():
        return "FLASH_ATTN"


@dataclass
class FakeModelRunner:
    attention_backend: object


def make_logical():
    output = FakeSchedulerOutput(
        scheduled_new_reqs=[
            FakeNewReq("A", 4096),
            FakeNewReq("B", 2048),
            FakeNewReq("C", 1024),
        ],
        scheduled_cached_reqs=FakeCached([], []),
        num_scheduled_tokens={"A": 1, "B": 1, "C": 512},
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
    )
    return SchedulerOutputAdapter().extract(output)


def test_runtime_adapter_consumes_cudagraph_batch_descriptor():
    logical = make_logical()
    ctx = FakeForwardContext(
        FakeCudaGraphMode.FULL,
        FakeBatchDescriptor(num_tokens=576, num_reqs=3, uniform=False),
    )
    cfg = FakeVllmConfig(FakeParallelConfig(4), FakeModelConfig(FakeDType(2)))
    runner = FakeModelRunner(FakeAttentionBackend())

    runtime = RuntimeContextAdapter().extract(
        logical, vllm_config=cfg, forward_context=ctx, model_runner=runner
    )

    assert runtime.logical_tokens == 514
    assert runtime.execution_tokens == 576
    assert runtime.cudagraph_mode is CudaGraphMode.FULL
    assert runtime.tp_size == 4
    assert runtime.dtype_bytes == 2
    assert runtime.num_requests == 3
    assert runtime.uniform_batch is False
    assert runtime.attention_backend == "FLASH_ATTN"

    dag = lower_dense_decoder(QWEN3_8B, logical, runtime)
    qkv = next(op for op in dag.operations if op.name == "layer0.qkv")
    attn = next(op for op in dag.operations if op.name == "layer0.attention")
    assert qkv.m == 576
    assert attn.query_lens == (1, 1, 512)


def test_runtime_adapter_preserves_ubatch_count_and_lora_key():
    logical = make_logical()
    ctx = FakeForwardContext(
        FakeCudaGraphMode.PIECEWISE,
        FakeBatchDescriptor(
            num_tokens=576,
            num_reqs=None,
            uniform=False,
            has_lora=True,
            num_active_loras=2,
        ),
        ubatch_slices=[object(), object()],
    )
    cfg = FakeVllmConfig(FakeParallelConfig(4), FakeModelConfig(FakeDType(2)))

    runtime = RuntimeContextAdapter().extract(
        logical, vllm_config=cfg, forward_context=ctx
    )
    assert runtime.cudagraph_mode is CudaGraphMode.PIECEWISE
    assert runtime.ubatch_count == 2
    assert runtime.has_lora is True
    assert runtime.num_active_loras == 2
    assert runtime.num_requests is None


def test_runtime_adapter_falls_back_to_eager_logical_shape():
    logical = make_logical()
    ctx = FakeForwardContext(FakeCudaGraphMode.NONE, None)
    cfg = FakeVllmConfig(FakeParallelConfig(1), FakeModelConfig(FakeDType(2)))

    runtime = RuntimeContextAdapter().extract(
        logical, vllm_config=cfg, forward_context=ctx
    )
    assert runtime.execution_tokens == logical.logical_tokens
    assert runtime.cudagraph_mode is CudaGraphMode.NONE
    assert runtime.tp_size == 1
