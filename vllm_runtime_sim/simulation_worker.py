from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn as nn

from vllm.v1.kv_cache_interface import FullAttentionSpec, get_kv_quant_mode
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from .backend import AnalyticalBackend
from .bridge import SimulationStep
from .execution_hook import SyntheticOutputFactory
from .ir import CudaGraphMode, RuntimeExecutionDescriptor
from .lowering import lower_dense_decoder
from .vllm_adapter import DenseDecoderConfigAdapter, SchedulerOutputAdapter


class SimulationWorker(WorkerBase):
    """GPU-free vLLM worker for end-to-end control-plane simulation.

    This worker intentionally does *not* load model weights or allocate a
    numerical KV cache. It gives EngineCore a logical KV-cache specification so
    the real vLLM scheduler/block manager can run, then lowers each real
    SchedulerOutput into the simulator IR and returns a synthetic
    ModelRunnerOutput.

    The first version uses eager-shaped execution because no GPUModelRunner is
    instantiated. ``VLLM_SIM_TARGET_TP`` can nevertheless select the physical
    TP degree used by the operator/collective model.
    """

    def __init__(
        self,
        vllm_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.logical_adapter = SchedulerOutputAdapter()
        self.model_adapter = DenseDecoderConfigAdapter()
        self.backend = AnalyticalBackend()
        self.output_factory = SyntheticOutputFactory()
        self.logical_kv_cache_config: Any | None = None
        self.last_step: SimulationStep | None = None
        self.last_latency_us: float | None = None
        self._model = nn.Identity()

    def init_device(self) -> None:
        # A host torch device keeps vLLM's generic worker plumbing satisfied,
        # but no model runner, CUDA stream, or accelerator allocation is made.
        self.device = torch.device("cpu")

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        # Deliberately skip weight loading. Model dimensions come from the
        # already-resolved HF config in vllm_config.model_config.
        return

    def get_model(self) -> nn.Module:
        return self._model

    def get_supported_tasks(self):
        # v0 only models causal language generation.
        return ("generate",)

    def get_kv_cache_spec(self):
        """Build scheduler-visible KV specs without instantiating model layers."""
        model = self.model_adapter.extract(self.vllm_config)
        cache_dtype = self.cache_config.cache_dtype
        if cache_dtype != "auto":
            # Quantized KV layout needs backend-specific storage information;
            # keep the first GPU-free milestone intentionally conservative.
            raise NotImplementedError(
                "SimulationWorker currently supports --kv-cache-dtype auto only"
            )

        block_size = int(self.cache_config.block_size)
        kv_quant_mode = get_kv_quant_mode("auto")
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=model.num_kv_heads,
            head_size=model.head_dim,
            dtype=self.model_config.dtype,
            kv_quant_mode=kv_quant_mode,
        )
        # The exact layer keys are opaque to the scheduler for a uniform dense
        # model; they mainly identify how many cache-owning layers exist.
        return {f"layers.{i}.self_attn": spec for i in range(model.num_layers)}

    def determine_available_memory(self) -> int:
        # Logical capacity only: no bytes are allocated. Keep the default modest
        # so scheduler metadata does not grow unexpectedly on development hosts.
        return int(os.getenv("VLLM_SIM_KV_CACHE_BYTES", str(2 * 1024**3)))

    def initialize_from_config(self, kv_cache_config) -> None:
        self.logical_kv_cache_config = kv_cache_config

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # No kernels, compilation, CUDA Graph capture, or warmup.
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def execute_model(self, scheduler_output):
        logical = self.logical_adapter.extract(scheduler_output)
        model = self.model_adapter.extract(self.vllm_config)
        target_tp = int(
            os.getenv(
                "VLLM_SIM_TARGET_TP",
                str(self.parallel_config.tensor_parallel_size),
            )
        )
        runtime = RuntimeExecutionDescriptor(
            logical_tokens=logical.logical_tokens,
            execution_tokens=logical.logical_tokens,
            cudagraph_mode=CudaGraphMode.NONE,
            attention_backend=os.getenv("VLLM_SIM_ATTN_BACKEND", "simulated"),
            tp_size=target_tp,
            dtype_bytes=self.model_config.dtype.itemsize,
            num_requests=len(logical.requests),
        )
        dag = lower_dense_decoder(model, logical, runtime)
        step = SimulationStep(logical=logical, runtime=runtime, model=model, dag=dag)
        latency_us = float(self.backend.estimate_dag_us(dag))
        self.last_step = step
        self.last_latency_us = latency_us

        if os.getenv("VLLM_SIM_SLEEP", "0").lower() in {"1", "true", "yes", "on"}:
            time.sleep(latency_us / 1e6)

        return self.output_factory.build(step)

    def sample_tokens(self, grammar_output):
        raise RuntimeError(
            "SimulationWorker.execute_model returns ModelRunnerOutput directly; "
            "sample_tokens should not be called in the current synchronous mode"
        )

    def get_cache_block_size_bytes(self) -> int:
        specs = self.get_kv_cache_spec()
        if not specs:
            return 0
        return next(iter(specs.values())).page_size_bytes

    def add_lora(self, lora_request) -> bool:
        raise NotImplementedError("LoRA is not yet supported by SimulationWorker")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not yet supported by SimulationWorker")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not yet supported by SimulationWorker")

    def list_loras(self) -> set[int]:
        return set()
