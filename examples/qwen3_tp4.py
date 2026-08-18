from vllm_runtime_sim import (
    AnalyticalBackend,
    CudaGraphMode,
    LogicalWorkload,
    QWEN3_8B,
    RequestWorkload,
    RuntimeExecutionDescriptor,
    lower_dense_decoder,
)

logical = LogicalWorkload(
    (
        RequestWorkload("decode-a", 4096, 1),
        RequestWorkload("decode-b", 2048, 1),
        RequestWorkload("prefill-c", 1024, 512),
    )
)

runtime = RuntimeExecutionDescriptor(
    logical_tokens=logical.logical_tokens,
    execution_tokens=576,  # example CUDA Graph padded size
    cudagraph_mode=CudaGraphMode.FULL,
    attention_backend="flash_attn",
    tp_size=4,
)

dag = lower_dense_decoder(QWEN3_8B, logical, runtime)
backend = AnalyticalBackend()

print(f"logical tokens: {logical.logical_tokens}")
print(f"physical tokens: {runtime.execution_tokens}")
print(f"ops: {len(dag.operations)}")
print(f"estimated step latency: {backend.estimate_dag_us(dag):.2f} us")

for op in dag.operations[:10]:
    print(op)
