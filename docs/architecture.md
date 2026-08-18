# Architecture

## Goal

Run the real vLLM serving control plane while replacing device realization with simulation.

```text
request
  |
real vLLM scheduler
  |
SchedulerOutput
  |
LogicalWorkload
  |
real runtime decisions (TP / CUDA Graph / backend)
  |
RuntimeExecutionDescriptor
  |
PhysicalDAG
  |---- GemmOp
  |---- AttentionOp
  |---- CollectiveOp
  `---- future DMA/MoE ops
  |
Simulation backend
  |
virtual latency + synthetic model output
  |
real vLLM scheduler
```

## Key separation

### Logical workload

Represents serving semantics that should not depend on CUDA Graph padding:

- request IDs
- scheduled token counts
- computed/context token counts
- ragged query lengths
- KV lengths
- speculative token counts

### Runtime execution descriptor

Represents decisions made by the real serving runtime:

- tensor parallel degree
- physical execution token count after padding
- CUDA Graph mode
- attention backend
- dtype width

### Physical DAG

Represents the hardware work that receives a performance model. Dense GEMMs use the physical execution token count; attention keeps the valid ragged sequence metadata.

## CUDA Graph

The simulator should not reimplement vLLM's CUDA Graph dispatch policy. vLLM should decide the runtime graph mode and padded `BatchDescriptor`; the simulator consumes that decision and skips actual graph capture/replay.

Future versions will add execution regions so graph replay/launch overhead is modeled separately from kernel time.

## Tensor parallelism

For the Qwen3-8B reference configuration with TP=4:

- QKV: `[M,4096] x [4096,1536]`
- local attention: 8 Q heads, 2 KV heads
- O projection: `[M,1024] x [1024,4096]` + TP AllReduce
- Gate/Up: `[M,4096] x [4096,6144]`
- Down: `[M,3072] x [3072,4096]` + TP AllReduce

For BF16, each AllReduce payload is `M * 4096 * 2` bytes.

## Planned dynamic features

### Speculative decoding

Add separate fields for verification work and accepted-token feedback. Verification tokens determine current GPU work; accepted tokens determine future scheduler state.

### MoE

Add `RoutingOp`, `AllToAllOp`, and `GroupedGemmOp`. Since true router outputs require hidden states, routing will initially support trace replay and statistical synthetic routing.

### KV offload

Add H2D/D2H/DMA/compression/decompression nodes and a timeline engine that models overlap and contention.
