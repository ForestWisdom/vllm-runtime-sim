from __future__ import annotations

from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.interface import PlatformEnum

from .shadow_cudagraph import persist_policy_snapshot


class SimulationPlatform(CpuPlatform):
    """CPU-hosted platform whose worker never executes model numerics.

    Reusing CpuPlatform gives vLLM a valid host device and Gloo-compatible
    control environment. Before CPU normalization mutates compilation settings,
    the target CUDA Graph policy is snapshotted for the shadow runtime.
    """

    _enum = PlatformEnum.OOT
    device_name = "runtime-sim"

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        # Preserve the target-device CUDA Graph policy before CpuPlatform clears
        # CUDA-only capture settings. The worker consumes this snapshot on a
        # private config clone, so the host-side vLLM lifecycle remains CPU-safe.
        persist_policy_snapshot(vllm_config)

        # Normalize the live config into a host-only setup. This prevents CUDA
        # capture/compile/device initialization in the actual worker lifecycle.
        super().check_and_update_config(vllm_config)

        parallel = vllm_config.parallel_config
        parallel.worker_cls = "vllm_runtime_sim.simulation_worker.SimulationWorker"

        # The initial worker is deliberately single-process. Physical TP/EP are
        # represented in the simulator IR instead of spawning device workers.
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
