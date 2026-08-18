from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ir import (
    CudaGraphMode,
    LogicalWorkload,
    RequestWorkload,
    RuntimeExecutionDescriptor,
)
from .lowering import DenseDecoderConfig


@dataclass
class RequestState:
    num_computed_tokens: int
    prompt_len: int | None = None
    num_output_tokens: int = 0


def _enum_name(value: Any) -> str:
    if value is None:
        return "none"
    name = getattr(value, "name", None)
    if name is not None:
        return str(name).lower()
    return str(value).split(".")[-1].lower()


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
    for obj in (model_runner, forward_context):
        if obj is None:
            continue
        for attr in ("attention_backend", "attn_backend", "backend"):
            value = getattr(obj, attr, None)
            if value is None:
                continue
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


def _prompt_len(req: Any) -> int | None:
    value = getattr(req, "prompt_len", None)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    token_ids = getattr(req, "prompt_token_ids", None)
    if token_ids is not None:
        return len(token_ids)

    embeds = getattr(req, "prompt_embeds", None)
    shape = getattr(embeds, "shape", None)
    if shape:
        return int(shape[0])
    return None


class DenseDecoderConfigAdapter:
    """Extract dense decoder dimensions from vLLM's resolved HF text config."""

    def extract(self, vllm_config: Any) -> DenseDecoderConfig:
        model_config = getattr(vllm_config, "model_config", None)
        hf = getattr(model_config, "hf_text_config", None)
        if hf is None:
            hf = getattr(model_config, "hf_config", None)
        if hf is None:
            raise ValueError("vllm_config.model_config has no resolved HF config")

        hidden_size = int(getattr(hf, "hidden_size"))
        intermediate_size = int(getattr(hf, "intermediate_size"))
        num_layers = int(
            getattr(hf, "num_hidden_layers", getattr(hf, "num_layers", 0))
        )
        num_attention_heads = int(getattr(hf, "num_attention_heads"))
        num_kv_heads = int(
            getattr(hf, "num_key_value_heads", num_attention_heads)
        )
        head_dim = getattr(hf, "head_dim", None)
        if head_dim is None:
            if hidden_size % num_attention_heads:
                raise ValueError(
                    "cannot infer head_dim from hidden_size/num_attention_heads"
                )
            head_dim = hidden_size // num_attention_heads

        dtype = str(getattr(model_config, "dtype", "bf16")).lower()
        if "bfloat16" in dtype:
            dtype = "bf16"
        elif "float16" in dtype or "half" in dtype:
            dtype = "fp16"

        return DenseDecoderConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=int(head_dim),
            dtype=dtype,
        )


class SchedulerOutputAdapter:
    """Convert a vLLM V1 SchedulerOutput-like object to LogicalWorkload."""

    def __init__(self) -> None:
        self._requests: dict[str, RequestState] = {}

    def extract(self, output: Any) -> LogicalWorkload:
        finished = getattr(output, "finished_req_ids", set()) or set()
        for req_id in finished:
            self._requests.pop(req_id, None)

        for req in getattr(output, "scheduled_new_reqs", ()):
            self._requests[req.req_id] = RequestState(
                num_computed_tokens=int(req.num_computed_tokens),
                prompt_len=_prompt_len(req),
            )

        cached = getattr(output, "scheduled_cached_reqs", None)
        if cached is not None:
            req_ids = list(getattr(cached, "req_ids", ()))
            computed = list(getattr(cached, "num_computed_tokens", ()))
            output_counts = list(getattr(cached, "num_output_tokens", ()))
            for index, (req_id, num_computed) in enumerate(zip(req_ids, computed)):
                state = self._requests.get(
                    req_id, RequestState(num_computed_tokens=int(num_computed))
                )
                state.num_computed_tokens = int(num_computed)
                if index < len(output_counts):
                    state.num_output_tokens = int(output_counts[index])
                self._requests[req_id] = state

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
                    prompt_len=state.prompt_len,
                    num_output_tokens=state.num_output_tokens,
                )
            )

        return LogicalWorkload(tuple(requests))


class RuntimeContextAdapter:
    """Extract physical execution choices already made by vLLM.

    PIECEWISE/eager execution exposes a live ForwardContext. Current MRV2 FULL
    replay bypasses a live ForwardContext and instead passes a
    BatchExecutionDescriptor directly to the CUDA Graph manager, so this adapter
    accepts either representation.
    """

    def extract(
        self,
        logical: LogicalWorkload,
        *,
        vllm_config: Any,
        forward_context: Any | None = None,
        execution_descriptor: Any | None = None,
        model_runner: Any | None = None,
    ) -> RuntimeExecutionDescriptor:
        execution_tokens = logical.logical_tokens
        num_requests: int | None = len(logical.requests)
        uniform: bool | None = None
        has_lora = False
        num_active_loras = 0
        ubatch_count = 1
        uniform_token_count: int | None = None
        max_query_len: int | None = None
        mode_value = None

        if execution_descriptor is not None:
            execution_tokens = int(
                getattr(execution_descriptor, "num_tokens", execution_tokens)
            )
            num_requests = getattr(execution_descriptor, "num_reqs", num_requests)
            if num_requests is not None:
                num_requests = int(num_requests)
            uniform_token_count = getattr(
                execution_descriptor, "uniform_token_count", None
            )
            uniform = uniform_token_count is not None
            max_query_len = getattr(execution_descriptor, "max_query_len", None)
            num_active_loras = int(
                getattr(execution_descriptor, "num_active_loras", 0)
            )
            has_lora = num_active_loras > 0
            mode_value = getattr(execution_descriptor, "cg_mode", None)

        elif forward_context is not None:
            batch = getattr(forward_context, "batch_descriptor", None)
            if batch is not None:
                execution_tokens = int(
                    getattr(batch, "num_tokens", execution_tokens)
                )
                num_requests = getattr(batch, "num_reqs", num_requests)
                if num_requests is not None:
                    num_requests = int(num_requests)
                uniform = bool(getattr(batch, "uniform", False))
                has_lora = bool(getattr(batch, "has_lora", False))
                num_active_loras = int(getattr(batch, "num_active_loras", 0))

            mode_value = getattr(
                forward_context, "cudagraph_runtime_mode", None
            )
            ubatches = getattr(forward_context, "ubatch_slices", None)
            ubatch_count = len(ubatches) if ubatches else 1

        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1))

        return RuntimeExecutionDescriptor(
            logical_tokens=logical.logical_tokens,
            execution_tokens=execution_tokens,
            cudagraph_mode=_cuda_graph_mode(mode_value),
            attention_backend=_attention_backend_name(
                model_runner, forward_context
            ),
            tp_size=tp_size,
            dtype_bytes=_dtype_bytes(vllm_config),
            num_requests=num_requests,
            uniform_batch=uniform,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
            ubatch_count=ubatch_count,
            uniform_token_count=(
                int(uniform_token_count)
                if uniform_token_count is not None
                else None
            ),
            max_query_len=(int(max_query_len) if max_query_len is not None else None),
        )
