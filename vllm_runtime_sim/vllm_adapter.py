from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ir import (
    CudaGraphMode,
    LogicalWorkload,
    RequestWorkload,
    RuntimeExecutionDescriptor,
)


@dataclass
class RequestState:
    num_computed_tokens: int


def _enum_name(value: Any) -> str:
    if value is None:
        return "none"
    name = getattr(value, "name", None)
    if name is not None:
        return str(name).lower()
    text = str(value).split(".")[-1].lower()
    return text


def _cuda_graph_mode(value: Any) -> CudaGraphMode:
    name = _enum_name(value)
    if "piecewise" in name:
        return CudaGraphMode.PIECEWISE
    if "full" in name:
        return CudaGraphMode.FULL
    return CudaGraphMode.NONE


def _dtype_bytes(vllm_config: Any) -> int:
    model_config = getattr(vllm_config, "model_config", None)
    dtype = getattr(model_config, "dtype", None)
    if dtype is None:
        return 2
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    name = str(dtype).lower()
    if "float32" in name or "fp32" in name:
        return 4
    if "float16" in name or "bfloat16" in name or "fp16" in name or "bf16" in name:
        return 2
    if "int8" in name or "fp8" in name:
        return 1
    return 2


def _attention_backend_name(model_runner: Any, forward_context: Any) -> str:
    # Prefer the initialized runtime backend over static config. Different vLLM
    # versions expose it through slightly different objects, so keep this duck-typed.
    for obj in (model_runner, forward_context):
        for attr in ("attention_backend", "attn_backend", "backend"):
            value = getattr(obj, attr, None)
            if value is not None:
                name = getattr(value, "get_name", None)
                if callable(name):
                    try:
                        return str(name())
                    except TypeError:
                        pass
                enum_name = getattr(value, "name", None)
                if enum_name is not None:
                    return str(enum_name).lower()
                return value.__class__.__name__
    return "unknown"


class SchedulerOutputAdapter:
    """Convert a vLLM V1 SchedulerOutput-like object to LogicalWorkload.

    This adapter intentionally uses duck typing so the simulation core remains
    importable without vLLM installed. Worker-side cached state is maintained
    only for fields that vLLM sends incrementally across scheduling steps.
    """

    def __init__(self) -> None:
        self._requests: dict[str, RequestState] = {}

    def extract(self, output: Any) -> LogicalWorkload:
        finished = getattr(output, "finished_req_ids", set()) or set()
        for req_id in finished:
            self._requests.pop(req_id, None)

        for req in getattr(output, "scheduled_new_reqs", ()):
            self._requests[req.req_id] = RequestState(
                num_computed_tokens=int(req.num_computed_tokens)
            )

        cached = getattr(output, "scheduled_cached_reqs", None)
        if cached is not None:
            for req_id, computed in zip(
                getattr(cached, "req_ids", ()),
                getattr(cached, "num_computed_tokens", ()),
            ):
                self._requests[req_id] = RequestState(int(computed))

        scheduled = getattr(output, "num_scheduled_tokens", {})
        spec = getattr(output, "scheduled_spec_decode_tokens", {}) or {}
        requests: list[RequestWorkload] = []

        for req_id, num_tokens in scheduled.items():
            if req_id not in self._requests:
                raise KeyError(
                    f"missing cached request state for {req_id}; "
                    "adapter must observe the request's first scheduled step"
                )
            state = self._requests[req_id]
            requests.append(
                RequestWorkload(
                    request_id=req_id,
                    num_computed_tokens=state.num_computed_tokens,
                    num_scheduled_tokens=int(num_tokens),
                    speculative_tokens=len(spec.get(req_id, ())),
                )
            )

        return LogicalWorkload(tuple(requests))


class RuntimeContextAdapter:
    """Extract physical execution choices already made by vLLM.

    The intended call site is immediately after vLLM has prepared its runtime
    forward context / CUDA Graph dispatch decision and before GPU realization.
    The adapter consumes the real `ForwardContext.batch_descriptor` instead of
    reimplementing vLLM's CUDA Graph padding/dispatch policy.
    """

    def extract(
        self,
        logical: LogicalWorkload,
        *,
        vllm_config: Any,
        forward_context: Any,
        model_runner: Any | None = None,
    ) -> RuntimeExecutionDescriptor:
        batch = getattr(forward_context, "batch_descriptor", None)
        execution_tokens = logical.logical_tokens
        num_requests: int | None = len(logical.requests)
        uniform: bool | None = None
        has_lora = False
        num_active_loras = 0

        if batch is not None:
            execution_tokens = int(getattr(batch, "num_tokens", execution_tokens))
            num_requests = getattr(batch, "num_reqs", num_requests)
            if num_requests is not None:
                num_requests = int(num_requests)
            uniform = bool(getattr(batch, "uniform", False))
            has_lora = bool(getattr(batch, "has_lora", False))
            num_active_loras = int(getattr(batch, "num_active_loras", 0))

        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1))
        ubatches = getattr(forward_context, "ubatch_slices", None)
        ubatch_count = len(ubatches) if ubatches else 1

        return RuntimeExecutionDescriptor(
            logical_tokens=logical.logical_tokens,
            execution_tokens=execution_tokens,
            cudagraph_mode=_cuda_graph_mode(
                getattr(forward_context, "cudagraph_runtime_mode", None)
            ),
            attention_backend=_attention_backend_name(model_runner, forward_context),
            tp_size=tp_size,
            dtype_bytes=_dtype_bytes(vllm_config),
            num_requests=num_requests,
            uniform_batch=uniform,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
            ubatch_count=ubatch_count,
        )
