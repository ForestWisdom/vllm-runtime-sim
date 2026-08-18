# vllm-runtime-sim

Runtime-native LLM serving simulation: execute the real vLLM control plane while replacing GPU realization with a composable simulation backend.

## Design principle

> vLLM decides **what** and **how** to execute; the simulator predicts **how long** that execution takes.

The project separates four layers:

1. **Logical workload IR** — requests, scheduled tokens, context/KV lengths, speculative tokens.
2. **Runtime execution descriptor** — TP sharding, CUDA Graph mode/padding, backend choices.
3. **Physical operator IR** — GEMM, attention, collectives, pointwise and future DMA/MoE operations.
4. **Simulation backend** — mock/statistical/analytical now; Vidur/SimAI-style backends later.

The first prototype targets dense decoder models such as Qwen3 and tensor parallelism, while keeping extension points for speculative decoding, MoE/EP, CUDA Graph regions, KV offload and virtual time.

## Status

Initial prototype under active development.
