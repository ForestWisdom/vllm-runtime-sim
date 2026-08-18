from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class CudaGraphMode(str, Enum):
    NONE = "none"
    FULL = "full"
    PIECEWISE = "piecewise"


@dataclass(frozen=True)
class RequestWorkload:
    request_id: str
    num_computed_tokens: int
    num_scheduled_tokens: int
    speculative_tokens: int = 0
    prompt_len: int | None = None
    num_output_tokens: int = 0

    @property
    def query_len(self) -> int:
        return self.num_scheduled_tokens

    @property
    def seq_len_after_step(self) -> int:
        return self.num_computed_tokens + self.num_scheduled_tokens

    @property
    def needs_sample(self) -> bool:
        """Whether this step reaches a position where vLLM should emit a token."""
        if self.num_output_tokens > 0:
            return True
        return (
            self.prompt_len is not None
            and self.seq_len_after_step >= self.prompt_len
        )


@dataclass(frozen=True)
class LogicalWorkload:
    requests: tuple[RequestWorkload, ...]

    @property
    def logical_tokens(self) -> int:
        return sum(r.num_scheduled_tokens for r in self.requests)

    @property
    def query_lens(self) -> tuple[int, ...]:
        return tuple(r.query_len for r in self.requests)

    @property
    def past_kv_lens(self) -> tuple[int, ...]:
        return tuple(r.num_computed_tokens for r in self.requests)

    @property
    def seq_lens(self) -> tuple[int, ...]:
        return tuple(r.seq_len_after_step for r in self.requests)


@dataclass(frozen=True)
class RuntimeExecutionDescriptor:
    logical_tokens: int
    execution_tokens: int
    cudagraph_mode: CudaGraphMode = CudaGraphMode.NONE
    attention_backend: str = "unknown"
    tp_size: int = 1
    dtype_bytes: int = 2
    num_requests: int | None = None
    uniform_batch: bool | None = None
    has_lora: bool = False
    num_active_loras: int = 0
    ubatch_count: int = 1
    # MRV2 BatchExecutionDescriptor fields. They matter for CUDA Graph
    # specialization but are not available from the older BatchDescriptor.
    uniform_token_count: int | None = None
    max_query_len: int | None = None

    def __post_init__(self) -> None:
        if self.execution_tokens < self.logical_tokens:
            raise ValueError("execution_tokens must be >= logical_tokens")
        if self.tp_size < 1:
            raise ValueError("tp_size must be >= 1")
        if self.dtype_bytes < 1:
            raise ValueError("dtype_bytes must be >= 1")
        if self.ubatch_count < 1:
            raise ValueError("ubatch_count must be >= 1")


@dataclass(frozen=True)
class GemmOp:
    name: str
    m: int
    k: int
    n: int
    dtype: str = "bf16"


@dataclass(frozen=True)
class AttentionOp:
    name: str
    query_lens: tuple[int, ...]
    past_kv_lens: tuple[int, ...]
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    backend: str = "unknown"


@dataclass(frozen=True)
class CollectiveOp:
    name: str
    collective: str
    world_size: int
    payload_bytes: int
    group: str = "tp"


@dataclass(frozen=True)
class PointwiseOp:
    name: str
    elements: int
    kind: str


Operation = GemmOp | AttentionOp | CollectiveOp | PointwiseOp


@dataclass
class PhysicalDAG:
    operations: list[Operation] = field(default_factory=list)

    def extend(self, ops: Sequence[Operation]) -> None:
        self.operations.extend(ops)
