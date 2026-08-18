from __future__ import annotations

import os


def register_simulation_platform() -> str | None:
    """vLLM platform-plugin entry point.

    The plugin is opt-in so merely installing vllm-runtime-sim does not replace
    the user's normal CUDA/CPU platform. Set ``VLLM_RUNTIME_SIM=1`` before
    starting vLLM to activate it.
    """
    enabled = os.getenv("VLLM_RUNTIME_SIM", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    return "vllm_runtime_sim.simulation_platform.SimulationPlatform"
