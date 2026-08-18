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
    """Stateful bridge from live vLLM objects to the simulator IR.

    A worker/executor hook can keep one bridge instance and call `build_step`
    for each vLLM scheduling iteration. The bridge deliberately does not import
    vLLM so it can be tested without a GPU or a vLLM installation.
    """

    def __init__(self) -> None:
        self.scheduler = SchedulerOutputAdapter()
        self.runtime = RuntimeContextAdapter()
        self.model = DenseDecoderConfigAdapter()

    def build_step(
        self,
        scheduler_output: Any,
        *,
        vllm_config: Any,
        forward_context: Any,
        model_runner: Any | None = None,
    ) -> SimulationStep:
        logical = self.scheduler.extract(scheduler_output)
        runtime = self.runtime.extract(
            logical,
            vllm_config=vllm_config,
            forward_context=forward_context,
            model_runner=model_runner,
        )
        model = self.model.extract(vllm_config)
        dag = lower_dense_decoder(model, logical, runtime)
        return SimulationStep(logical=logical, runtime=runtime, model=model, dag=dag)
