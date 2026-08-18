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

The real vLLM EngineCore, scheduler and logical block manager still run. The initial GPU-free worker is intentionally limited to a single actual vLLM worker and dense decoder models with `--kv-cache-dtype auto`; `VLLM_SIM_TARGET_TP` controls the physical TP degree used by the simulator IR/backend.

Useful environment variables:

- `VLLM_RUNTIME_SIM=1`: activate the simulation platform plugin.
- `VLLM_SIM_TARGET_TP=N`: model physical tensor parallelism without spawning N accelerator workers.
- `VLLM_SIM_KV_CACHE_BYTES=...`: logical KV-cache capacity exposed to the scheduler (default 2 GiB).
- `VLLM_SIM_SLEEP=1`: sleep for estimated device latency instead of returning immediately. Virtual time will replace this mode later.

## Status

Current prototype supports dense-decoder logical/runtime IR, TP lowering, CUDA Graph-aware execution descriptors, a live model-runner realization cut, and an experimental GPU-free platform/worker. Speculative decoding, MoE/EP, detailed communication, KV offload and causality-preserving virtual time remain planned extensions.
