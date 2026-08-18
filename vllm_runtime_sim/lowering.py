from __future__ import annotations

from dataclasses import dataclass

from .ir import (
    AttentionOp,
    CollectiveOp,
    GemmOp,
    LogicalWorkload,
    PhysicalDAG,
    PointwiseOp,
    RuntimeExecutionDescriptor,
)


@dataclass(frozen=True)
class DenseDecoderConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str = "bf16"


QWEN3_8B = DenseDecoderConfig(
    hidden_size=4096,
    intermediate_size=12288,
    num_layers=36,
    num_attention_heads=32,
    num_kv_heads=8,
    head_dim=128,
)


def _require_divisible(value: int, divisor: int, name: str) -> int:
    if value % divisor:
        raise ValueError(f"{name}={value} must be divisible by tp_size={divisor}")
    return value // divisor


def lower_dense_decoder(
    model: DenseDecoderConfig,
    logical: LogicalWorkload,
    runtime: RuntimeExecutionDescriptor,
) -> PhysicalDAG:
    """Lower a dense decoder step into per-rank physical operations.

    Dense GEMMs use execution_tokens because CUDA Graph padding can change their
    physical M dimension. Attention retains logical ragged query/KV lengths.
    """
    if logical.logical_tokens != runtime.logical_tokens:
        raise ValueError("logical/runtime token count mismatch")

    tp = runtime.tp_size
    q_heads = _require_divisible(model.num_attention_heads, tp, "num_attention_heads")
    kv_heads = _require_divisible(model.num_kv_heads, tp, "num_kv_heads")
    q_width = q_heads * model.head_dim
    kv_width = kv_heads * model.head_dim
    qkv_n = q_width + 2 * kv_width
    local_intermediate = _require_divisible(
        model.intermediate_size, tp, "intermediate_size"
    )
    m = runtime.execution_tokens
    h = model.hidden_size
    dtype_bytes = runtime.dtype_bytes

    dag = PhysicalDAG()
    for layer in range(model.num_layers):
        prefix = f"layer{layer}"
        dag.extend(
            [
                PointwiseOp(f"{prefix}.input_norm", m * h, "rmsnorm"),
                GemmOp(f"{prefix}.qkv", m=m, k=h, n=qkv_n, dtype=model.dtype),
                PointwiseOp(
                    f"{prefix}.qk_norm_rope",
                    logical.logical_tokens * (q_width + kv_width),
                    "qk_norm_rope",
                ),
                AttentionOp(
                    name=f"{prefix}.attention",
                    query_lens=logical.query_lens,
                    past_kv_lens=logical.past_kv_lens,
                    num_q_heads=q_heads,
                    num_kv_heads=kv_heads,
                    head_dim=model.head_dim,
                    backend=runtime.attention_backend,
                ),
                GemmOp(
                    f"{prefix}.o_proj",
                    m=m,
                    k=q_width,
                    n=h,
                    dtype=model.dtype,
                ),
            ]
        )
        if tp > 1:
            dag.operations.append(
                CollectiveOp(
                    f"{prefix}.o_proj.tp_reduce",
                    collective="all_reduce",
                    world_size=tp,
                    payload_bytes=m * h * dtype_bytes,
                )
            )

        dag.extend(
            [
                PointwiseOp(f"{prefix}.post_attn_norm", m * h, "rmsnorm"),
                GemmOp(
                    f"{prefix}.gate_up",
                    m=m,
                    k=h,
                    n=2 * local_intermediate,
                    dtype=model.dtype,
                ),
                PointwiseOp(
                    f"{prefix}.silu_mul",
                    m * local_intermediate,
                    "silu_mul",
                ),
                GemmOp(
                    f"{prefix}.down",
                    m=m,
                    k=local_intermediate,
                    n=h,
                    dtype=model.dtype,
                ),
            ]
        )
        if tp > 1:
            dag.operations.append(
                CollectiveOp(
                    f"{prefix}.down.tp_reduce",
                    collective="all_reduce",
                    world_size=tp,
                    payload_bytes=m * h * dtype_bytes,
                )
            )
    return dag
