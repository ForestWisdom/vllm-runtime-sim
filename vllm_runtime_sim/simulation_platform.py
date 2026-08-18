from __future__ import annotations

from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.interface import PlatformEnum


class SimulationPlatform(CpuPlatform):
    """CPU-hosted platform whose worker never executes model numerics.

    Reusing CpuPlatform gives vLLM a valid host device and Gloo-compatible
    control environment. The worker is replaced after CPU config normalization,
    so no CPU model runner is constructed and no model weights are loaded.

    This first GPU-free path intentionally disables CUDA Graph runtime fidelity;
    the separate ModelRunnerSimulationHook remains the high-fidelity path when
    the real GPUModelRunner can be initialized. A later device-virtualization
    layer will combine both properties.
    """

    _enum = PlatformEnum.OOT
    device_name = "runtime-sim"

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        # Normalize the configuration into a host-only setup first. In
        # particular, this disables CUDA graph capture and CUDA-only features.
        super().check_and_update_config(vllm_config)

        parallel = vllm_config.parallel_config
        parallel.worker_cls = "vllm_runtime_sim.simulation_worker.SimulationWorker"

        # The initial worker is deliberately single-process. Physical TP/EP are
        # represented in the simulator IR instead of spawning device workers.
        # This is sufficient to prove end-to-end control-plane execution; a
        # distributed simulation executor is a later milestone.
        if parallel.tensor_parallel_size != 1:
            raise ValueError(
                "The GPU-free SimulationPlatform currently requires "
                "--tensor-parallel-size 1. Use VLLM_SIM_TARGET_TP to model a "
                "different physical TP degree without spawning GPU workers."
            )
        parallel.distributed_executor_backend = "uni"

        if vllm_config.speculative_config is not None:
            raise ValueError(
                "The GPU-free SimulationPlatform does not yet support "
                "speculative decoding. The runtime hook/IR already has an "
                "extension point for synthetic acceptance; worker integration "
                "will follow."
            )
