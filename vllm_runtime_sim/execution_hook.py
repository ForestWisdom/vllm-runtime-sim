from __future__ import annotations

import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable, Protocol

from .bridge import SimulationStep, VllmRuntimeBridge


class Backend(Protocol):
    def estimate_dag_us(self, dag: Any) -> float: ...


@dataclass(frozen=True)
class SyntheticModelRunnerOutput:
    """GPU-free stand-in matching the public ModelRunnerOutput field layout."""

    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]
    logprobs: Any = None
    prompt_logprobs_dict: dict[str, Any] | None = None
    pooler_output: Any = None
    kv_connector_output: Any = None
    ec_connector_output: Any = None
    num_nans_in_logits: Any = None
    cudagraph_stats: Any = None
    routed_experts: Any = None
    sampling_masks: Any = None


class SyntheticOutputFactory:
    """Create scheduler-consumable synthetic output without target logits.

    v0 emits one deterministic token whenever a request reaches a sampling
    position. Partial chunked-prefill steps emit no token. Later versions can
    replace this policy with speculative-acceptance and trace-driven token
    generators.
    """

    def __init__(self, token_id: int = 1, output_cls: Any | None = None):
        self.token_id = token_id
        self.output_cls = output_cls

    def build(self, step: SimulationStep) -> Any:
        req_ids = [request.request_id for request in step.logical.requests]
        sampled = [
            [self.token_id] if request.needs_sample else []
            for request in step.logical.requests
        ]

        cls = self.output_cls
        if cls is None:
            try:
                from vllm.v1.outputs import ModelRunnerOutput

                cls = ModelRunnerOutput
            except Exception:
                cls = SyntheticModelRunnerOutput

        return cls(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            prompt_logprobs_dict={},
        )


@dataclass(frozen=True)
class SimulationResult:
    step: SimulationStep
    latency_us: float


class SimulationCut(RuntimeError):
    """Internal non-error control transfer from realization to the hook."""

    def __init__(self, result: SimulationResult):
        super().__init__("vLLM GPU realization replaced by simulator")
        self.result = result


class ModelRunnerSimulationHook:
    """Intercept vLLM after runtime policy is resolved, before GPU realization.

    Current MRV2 has three realization paths:

    * eager: ``model.forward``;
    * PIECEWISE: ``cudagraph_manager.run_pw_graph``;
    * FULL: ``cudagraph_manager.run_fullgraph``.

    Eager and PIECEWISE execute inside ``set_forward_context`` and therefore
    expose the live ForwardContext. FULL replay uses a
    ``BatchExecutionDescriptor`` directly, so that descriptor is consumed
    instead. ``execute_model`` is wrapped only to scope the SchedulerOutput and
    to translate the internal SimulationCut into a synthetic ModelRunnerOutput.
    """

    def __init__(
        self,
        runner: Any,
        backend: Backend,
        *,
        forward_context_getter: Callable[[], Any],
        bridge: VllmRuntimeBridge | None = None,
        output_factory: SyntheticOutputFactory | None = None,
        sleep: bool = False,
    ):
        self.runner = runner
        self.backend = backend
        self.forward_context_getter = forward_context_getter
        self.bridge = bridge or VllmRuntimeBridge()
        self.output_factory = output_factory or SyntheticOutputFactory()
        self.sleep = sleep
        self.last_result: SimulationResult | None = None
        self._scheduler_output: Any | None = None
        self._installed = False
        self._originals: dict[str, Any] = {}

    def _simulate(
        self,
        *,
        forward_context: Any | None = None,
        execution_descriptor: Any | None = None,
    ) -> None:
        if self._scheduler_output is None:
            return

        step = self.bridge.build_step(
            self._scheduler_output,
            vllm_config=self.runner.vllm_config,
            forward_context=forward_context,
            execution_descriptor=execution_descriptor,
            model_runner=self.runner,
        )
        latency_us = float(self.backend.estimate_dag_us(step.dag))
        result = SimulationResult(step=step, latency_us=latency_us)
        self.last_result = result

        # Sleeping preserves wall-clock causality but does not accelerate the
        # workload. A virtual-time engine will replace this option later.
        if self.sleep and latency_us > 0:
            time.sleep(latency_us / 1e6)

        raise SimulationCut(result)

    def install(self) -> "ModelRunnerSimulationHook":
        if self._installed:
            return self

        runner = self.runner
        hook = self

        self._originals["execute_model"] = runner.execute_model
        original_execute = runner.execute_model

        def execute_wrapper(_runner, scheduler_output, *args, **kwargs):
            hook._scheduler_output = scheduler_output
            try:
                return original_execute(scheduler_output, *args, **kwargs)
            except SimulationCut as cut:
                return hook.output_factory.build(cut.result.step)
            finally:
                hook._scheduler_output = None

        runner.execute_model = MethodType(execute_wrapper, runner)

        model = getattr(runner, "model", None)
        if model is not None and hasattr(model, "forward"):
            self._originals["model_forward"] = model.forward
            original_forward = model.forward

            def forward_wrapper(_model, *args, **kwargs):
                if hook._scheduler_output is None:
                    return original_forward(*args, **kwargs)
                hook._simulate(forward_context=hook.forward_context_getter())

            model.forward = MethodType(forward_wrapper, model)

        manager = getattr(runner, "cudagraph_manager", None)
        if manager is not None:
            if hasattr(manager, "run_pw_graph"):
                self._originals["run_pw_graph"] = manager.run_pw_graph
                original_pw = manager.run_pw_graph

                def pw_wrapper(_manager, *args, **kwargs):
                    if hook._scheduler_output is None:
                        return original_pw(*args, **kwargs)
                    hook._simulate(forward_context=hook.forward_context_getter())

                manager.run_pw_graph = MethodType(pw_wrapper, manager)

            if hasattr(manager, "run_fullgraph"):
                self._originals["run_fullgraph"] = manager.run_fullgraph
                original_full = manager.run_fullgraph

                def full_wrapper(_manager, desc, *args, **kwargs):
                    if hook._scheduler_output is None:
                        return original_full(desc, *args, **kwargs)
                    hook._simulate(execution_descriptor=desc)

                manager.run_fullgraph = MethodType(full_wrapper, manager)

        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return

        self.runner.execute_model = self._originals["execute_model"]

        model = getattr(self.runner, "model", None)
        if model is not None and "model_forward" in self._originals:
            model.forward = self._originals["model_forward"]

        manager = getattr(self.runner, "cudagraph_manager", None)
        if manager is not None:
            if "run_pw_graph" in self._originals:
                manager.run_pw_graph = self._originals["run_pw_graph"]
            if "run_fullgraph" in self._originals:
                manager.run_fullgraph = self._originals["run_fullgraph"]

        self._installed = False


def install_vllm_model_runner_hook(
    runner: Any,
    backend: Backend,
    *,
    sleep: bool = False,
    output_factory: SyntheticOutputFactory | None = None,
) -> ModelRunnerSimulationHook:
    """Install the hook against an imported vLLM GPUModelRunner instance."""
    from vllm.forward_context import get_forward_context

    return ModelRunnerSimulationHook(
        runner,
        backend,
        forward_context_getter=get_forward_context,
        output_factory=output_factory,
        sleep=sleep,
    ).install()
