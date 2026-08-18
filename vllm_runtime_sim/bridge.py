from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ir import LogicalWorkload, PhysicalDAG, RuntimeExecutionDescriptor
from .lowering import DenseDecoderConfig, lower_dense_decoder
from .vllm_adapter import (
    DenseDecoderConfigAdapter,
    RuntimeContextAdapter,
    SchedulerOutputAdapter,
)


@dataclass(frozen=True)
class SimulationStep:
    logical: LogicalWorkload
    runtime: RuntimeExecutionDescriptor
    model: DenseDecoderConfig
    dag: PhysicalDAG


class VllmRuntimeBridge:
    """Stateful bridge from live vLLM objects to the simulator IR."""

    def __init__(self) -> None:
        self.scheduler = SchedulerOutputAdapter()
        self.runtime = RuntimeContextAdapter()
        self.model = DenseDecoderConfigAdapter()

    def build_step(
        self,
        scheduler_output: Any,
        *,
        vllm_config: Any,
        forward_context: Any | None = None,
        execution_descriptor: Any | None = None,
        model_runner: Any | None = None,
    ) -> SimulationStep:
        logical = self.scheduler.extract(scheduler_output)
        runtime = self.runtime.extract(
            logical,
            vllm_config=vllm_config,
            forward_context=forward_context,
            execution_descriptor=execution_descriptor,
            model_runner=model_runner,
        )
        model = self.model.extract(vllm_config)
        dag = lower_dense_decoder(model, logical, runtime)
        return SimulationStep(
            logical=logical,
            runtime=runtime,
            model=model,
            dag=dag,
        )
