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
from .shadow_cudagraph import ShadowCudaGraphPolicy
from .vllm_adapter import (
    DenseDecoderConfigAdapter,
    RuntimeContextAdapter,
    SchedulerOutputAdapter,
)


class SimulationWorker(WorkerBase):
    """GPU-free vLLM worker for end-to-end control-plane simulation.

    The worker keeps the real EngineCore/Scheduler/KV block manager but does not
    load weights or allocate numerical KV pages. CUDA Graph *capture/replay* is
    disabled in the live host config; a private ShadowCudaGraphPolicy reuses
    vLLM's real CudaGraphManager candidate generation and dispatch logic to
    recover target FULL/PIECEWISE/NONE decisions and padded execution sizes.
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
        self.runtime_adapter = RuntimeContextAdapter()
        self.model_adapter = DenseDecoderConfigAdapter()
        self.backend = AnalyticalBackend()
        self.output_factory = SyntheticOutputFactory()
        self.logical_kv_cache_config: Any | None = None
        self.last_step: SimulationStep | None = None
        self.last_latency_us: float | None = None
        self.last_cudagraph_policy_error: str | None = None
        self._model = nn.Identity()
        self.shadow_cudagraph: ShadowCudaGraphPolicy | None = None

    def init_device(self) -> None:
        self.device = torch.device("cpu")

        # Constructing the policy facade does not capture/replay any graph. It
        # only initializes vLLM's real dispatch candidate table on a private
        # config clone. Keep a graceful fallback for version churn in this
        # private vLLM API; strict mode is useful for fidelity testing.
        try:
            self.shadow_cudagraph = ShadowCudaGraphPolicy(self.vllm_config)
        except Exception as exc:  # pragma: no cover - exercised with live vLLM
            self.last_cudagraph_policy_error = f"{type(exc).__name__}: {exc}"
            if os.getenv("VLLM_SIM_STRICT_CG_POLICY", "0").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                raise
            self.shadow_cudagraph = None

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        return

    def get_model(self) -> nn.Module:
        return self._model

    def get_supported_tasks(self):
        return ("generate",)

    def get_kv_cache_spec(self):
        """Build scheduler-visible KV specs without instantiating model layers."""
        model = self.model_adapter.extract(self.vllm_config)
        cache_dtype = self.cache_config.cache_dtype
        if cache_dtype != "auto":
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
        return {f"layers.{i}.self_attn": spec for i in range(model.num_layers)}

    def determine_available_memory(self) -> int:
        return int(os.getenv("VLLM_SIM_KV_CACHE_BYTES", str(2 * 1024**3)))

    def initialize_from_config(self, kv_cache_config) -> None:
        self.logical_kv_cache_config = kv_cache_config

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def _runtime_from_policy(self, logical) -> RuntimeExecutionDescriptor:
        target_tp = int(
            os.getenv(
                "VLLM_SIM_TARGET_TP",
                str(self.parallel_config.tensor_parallel_size),
            )
        )

        if self.shadow_cudagraph is not None:
            desc = self.shadow_cudagraph.dispatch(logical)
            runtime = self.runtime_adapter.extract(
                logical,
                vllm_config=self.vllm_config,
                execution_descriptor=desc,
            )
            # Actual host worker TP is intentionally 1. Override only the target
            # hardware degree represented by the physical DAG.
            return RuntimeExecutionDescriptor(
                logical_tokens=runtime.logical_tokens,
                execution_tokens=runtime.execution_tokens,
                cudagraph_mode=runtime.cudagraph_mode,
                attention_backend=os.getenv(
                    "VLLM_SIM_ATTN_BACKEND", "simulated"
                ),
                tp_size=target_tp,
                dtype_bytes=runtime.dtype_bytes,
                num_requests=runtime.num_requests,
                uniform_batch=runtime.uniform_batch,
                has_lora=runtime.has_lora,
                num_active_loras=runtime.num_active_loras,
                ubatch_count=runtime.ubatch_count,
                uniform_token_count=runtime.uniform_token_count,
                max_query_len=runtime.max_query_len,
            )

        return RuntimeExecutionDescriptor(
            logical_tokens=logical.logical_tokens,
            execution_tokens=logical.logical_tokens,
            cudagraph_mode=CudaGraphMode.NONE,
            attention_backend=os.getenv("VLLM_SIM_ATTN_BACKEND", "simulated"),
            tp_size=target_tp,
            dtype_bytes=self.model_config.dtype.itemsize,
            num_requests=len(logical.requests),
        )

    def execute_model(self, scheduler_output):
        logical = self.logical_adapter.extract(scheduler_output)
        model = self.model_adapter.extract(self.vllm_config)
        runtime = self._runtime_from_policy(logical)

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
