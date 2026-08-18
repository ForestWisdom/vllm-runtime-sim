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
- prompt completion state used to distinguish partial chunked prefill from a sampling step

### Runtime execution descriptor

Represents decisions made by the real serving runtime:

- tensor parallel degree
- physical execution token count after padding
- CUDA Graph mode
- attention backend
- dtype width
- request padding / uniform decode information
- LoRA specialization key
- uBatch count
- MRV2 `uniform_token_count` and `max_query_len`

### Physical DAG

Represents the hardware work that receives a performance model. Dense GEMMs use the physical execution token count; attention keeps the valid ragged sequence metadata.

## Realization cut

The current prototype installs a reversible hook on a live vLLM model runner. It preserves `execute_model()` until vLLM has selected the execution path, then intercepts one of three realization boundaries:

```text
                       real GPUModelRunner.execute_model
                                  |
                         runtime preparation
                                  |
                         CUDA Graph dispatch
                                  |
                +-----------------+------------------+
                |                 |                  |
             EAGER           PIECEWISE             FULL
                |                 |                  |
          model.forward     run_pw_graph       run_fullgraph(desc)
                |                 |                  |
                +-----------------+------------------+
                                  |
                           SIMULATION CUT
                                  |
                     Logical/Runtime -> DAG
                                  |
                        performance backend
                                  |
                   synthetic ModelRunnerOutput
```

For eager and PIECEWISE execution, the hook consumes the live `ForwardContext`. For current MRV2 FULL replay, it consumes `BatchExecutionDescriptor` directly because the FULL replay path does not require a live forward context.

The hook also avoids emitting an output token for an incomplete chunked-prefill step. This is necessary because a scheduled prefill chunk is real model work but does not necessarily correspond to a sampling point.

### Current limitation

This cut removes the expensive numerical model forward / CUDA Graph replay, but it is **not yet a completely GPU-free vLLM startup path**. A normal GPU worker still initializes CUDA, model-runner device buffers, KV-cache structures and other device-side state before the hook is reached. The next milestone is a simulation Platform/Worker (or a lower-level device virtualization layer) that allows those initialization paths to run without physical GPUs while preserving as much of vLLM's host-side runtime logic as possible.

## CUDA Graph

The simulator should not reimplement vLLM's CUDA Graph dispatch policy. vLLM should decide the runtime graph mode and padded execution descriptor; the simulator consumes that decision and skips actual graph capture/replay.

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

Add separate fields for verification work and accepted-token feedback. Verification tokens determine current simulated device work; accepted tokens determine future scheduler state.

### MoE

Add `RoutingOp`, `AllToAllOp`, and `GroupedGemmOp`. Since true router outputs require hidden states, routing will initially support trace replay and statistical synthetic routing.

### KV offload

Add H2D/D2H/DMA/compression/decompression nodes and a timeline engine that models overlap and contention.
