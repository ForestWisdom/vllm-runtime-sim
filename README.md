# vllm-runtime-sim

Runtime-native LLM serving simulation: execute the real vLLM control plane while replacing device realization with a composable simulation backend.

## Design principle

> vLLM decides **what** and **how** to execute; the simulator predicts **how long** that execution takes.

The project separates four layers:

1. **Logical workload IR** — requests, scheduled tokens, context/KV lengths, speculative tokens.
2. **Runtime execution descriptor** — TP sharding, CUDA Graph mode/padding, backend choices.
3. **Physical operator IR** — GEMM, attention, collectives, pointwise and future DMA/MoE operations.
4. **Simulation backend** — analytical now; profile-driven Vidur/NeuSight-style compute and SimAI-style communication backends later.

## Two integration modes

### Runtime hook: higher execution-policy fidelity

`ModelRunnerSimulationHook` lets a live vLLM GPUModelRunner perform host-side scheduling/runtime preparation and CUDA Graph dispatch, then cuts execution before eager model forward, PIECEWISE graph execution, or FULL graph replay.

This path preserves more of the real GPU runtime decisions, but vLLM still needs to initialize its normal GPU worker/device state before the cut.

### Simulation platform: GPU-free control-plane prototype

The package also exposes an opt-in `vllm.platform_plugins` entry point. It replaces the normal worker with `SimulationWorker`, which does not load model weights or allocate numerical KV cache pages.

```bash
pip install -e '.[vllm]'

VLLM_RUNTIME_SIM=1 \
VLLM_SIM_TARGET_TP=4 \
vllm serve Qwen/Qwen3-8B --tensor-parallel-size 1
```

The real vLLM EngineCore, scheduler and logical block manager still run. Before CPU-host normalization disables CUDA-only runtime behavior, `SimulationPlatform` snapshots the target CUDA Graph policy. The GPU-free worker then constructs a private policy-only vLLM `CudaGraphManager` and calls its real `dispatch()` implementation. This preserves vLLM's capture-size candidate generation, FULL/PIECEWISE/NONE selection and execution-token padding without allocating or replaying a CUDA graph.

The initial GPU-free worker is intentionally limited to a single actual vLLM worker and dense decoder models with `--kv-cache-dtype auto`; `VLLM_SIM_TARGET_TP` controls the physical TP degree used by the simulator IR/backend.

Useful environment variables:

- `VLLM_RUNTIME_SIM=1`: activate the simulation platform plugin.
- `VLLM_SIM_TARGET_TP=N`: model physical tensor parallelism without spawning N accelerator workers.
- `VLLM_SIM_KV_CACHE_BYTES=...`: logical KV-cache capacity exposed to the scheduler (default 2 GiB).
- `VLLM_SIM_SLEEP=1`: sleep for estimated device latency instead of returning immediately. Virtual time will replace this mode later.
- `VLLM_SIM_STRICT_CG_POLICY=1`: fail startup instead of falling back to eager if the private vLLM CUDA Graph policy API changes or cannot initialize.

## CUDA Graph boundary

GPU-free mode reuses **policy**, not realization:

```text
real vLLM CudaGraphManager.__init__/_init_candidates
                  |
                  v
             real dispatch()
                  |
                  v
        BatchExecutionDescriptor
     (mode / padded tokens / graph key)
                  |
                  v
              simulator

capture() / torch.cuda.CUDAGraph / replay()  --> skipped
```

This is the first step toward a shadow GPU runtime: progressively keep more real vLLM batch preparation/runtime policy while replacing only device resources and numerical kernels.

## Status

Current prototype supports dense-decoder logical/runtime IR, TP lowering, real vLLM CUDA Graph dispatch-policy reuse in GPU-free mode, a live model-runner realization cut, and an experimental GPU-free platform/worker. Speculative decoding, MoE/EP, detailed communication, KV offload and causality-preserving virtual time remain planned extensions.
