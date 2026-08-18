from vllm_runtime_sim import (
    AnalyticalBackend,
    CollectiveOp,
    CudaGraphMode,
    GemmOp,
    LogicalWorkload,
    QWEN3_8B,
    RequestWorkload,
    RuntimeExecutionDescriptor,
    lower_dense_decoder,
)


def make_workload():
    return LogicalWorkload(
        (
            RequestWorkload("A", 4096, 1),
            RequestWorkload("B", 2048, 1),
            RequestWorkload("C", 1024, 512),
        )
    )


def test_qwen3_tp4_logical_shapes():
    logical = make_workload()
    runtime = RuntimeExecutionDescriptor(
        logical_tokens=514,
        execution_tokens=514,
        cudagraph_mode=CudaGraphMode.NONE,
        attention_backend="flash_attn",
        tp_size=4,
    )
    dag = lower_dense_decoder(QWEN3_8B, logical, runtime)

    qkv = next(op for op in dag.operations if op.name == "layer0.qkv")
    assert isinstance(qkv, GemmOp)
    assert (qkv.m, qkv.k, qkv.n) == (514, 4096, 1536)

    o_proj = next(op for op in dag.operations if op.name == "layer0.o_proj")
    assert isinstance(o_proj, GemmOp)
    assert (o_proj.m, o_proj.k, o_proj.n) == (514, 1024, 4096)

    gate_up = next(op for op in dag.operations if op.name == "layer0.gate_up")
    assert isinstance(gate_up, GemmOp)
    assert (gate_up.m, gate_up.k, gate_up.n) == (514, 4096, 6144)

    down = next(op for op in dag.operations if op.name == "layer0.down")
    assert isinstance(down, GemmOp)
    assert (down.m, down.k, down.n) == (514, 3072, 4096)

    collectives = [op for op in dag.operations if isinstance(op, CollectiveOp)]
    assert len(collectives) == 72
    assert collectives[0].payload_bytes == 514 * 4096 * 2


def test_cudagraph_padding_changes_dense_m_and_collective_payload():
    logical = make_workload()
    runtime = RuntimeExecutionDescriptor(
        logical_tokens=514,
        execution_tokens=576,
        cudagraph_mode=CudaGraphMode.FULL,
        attention_backend="flash_attn",
        tp_size=4,
    )
    dag = lower_dense_decoder(QWEN3_8B, logical, runtime)

    qkv = next(op for op in dag.operations if op.name == "layer0.qkv")
    assert qkv.m == 576

    attn = next(op for op in dag.operations if op.name == "layer0.attention")
    assert attn.query_lens == (1, 1, 512)
    assert attn.past_kv_lens == (4096, 2048, 1024)

    reduce = next(op for op in dag.operations if op.name == "layer0.o_proj.tp_reduce")
    assert reduce.payload_bytes == 576 * 4096 * 2


def test_analytical_backend_returns_positive_latency():
    logical = make_workload()
    runtime = RuntimeExecutionDescriptor(514, 514, tp_size=4)
    dag = lower_dense_decoder(QWEN3_8B, logical, runtime)
    assert AnalyticalBackend().estimate_dag_us(dag) > 0
