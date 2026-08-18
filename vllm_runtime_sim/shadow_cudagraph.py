from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable


_POLICY_ENV = "VLLM_RUNTIME_SIM_CUDAGRAPH_POLICY"


@dataclass(frozen=True)
class CudaGraphPolicySnapshot:
    """Serializable CUDA Graph policy captured before host normalization.

    SimulationPlatform intentionally reuses CpuPlatform for host setup. The CPU
    platform clears CUDA Graph capture sizes, so the target policy must be saved
    before that normalization and restored only on a private config clone used
    by the shadow dispatcher.
    """

    mode: str | None
    capture_sizes: tuple[int, ...] | None
    max_capture_size: int | None

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "CudaGraphPolicySnapshot":
        compilation = getattr(vllm_config, "compilation_config", None)
        if compilation is None:
            return cls(None, None, None)

        mode = getattr(compilation, "cudagraph_mode", None)
        mode_name = getattr(mode, "name", None)
        if mode_name is None and mode is not None:
            mode_name = str(mode).split(".")[-1]

        sizes = getattr(compilation, "cudagraph_capture_sizes", None)
        capture_sizes = None if sizes is None else tuple(int(x) for x in sizes)
        max_size = getattr(compilation, "max_cudagraph_capture_size", None)
        if max_size is not None:
            max_size = int(max_size)

        return cls(
            mode=str(mode_name) if mode_name is not None else None,
            capture_sizes=capture_sizes,
            max_capture_size=max_size,
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "mode": self.mode,
                "capture_sizes": (
                    list(self.capture_sizes) if self.capture_sizes is not None else None
                ),
                "max_capture_size": self.max_capture_size,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str) -> "CudaGraphPolicySnapshot":
        obj = json.loads(payload)
        sizes = obj.get("capture_sizes")
        return cls(
            mode=obj.get("mode"),
            capture_sizes=None if sizes is None else tuple(int(x) for x in sizes),
            max_capture_size=(
                None
                if obj.get("max_capture_size") is None
                else int(obj["max_capture_size"])
            ),
        )

    def restore(self, vllm_config: Any) -> None:
        compilation = vllm_config.compilation_config
        if self.capture_sizes is not None:
            compilation.cudagraph_capture_sizes = list(self.capture_sizes)
        if self.max_capture_size is not None:
            compilation.max_cudagraph_capture_size = self.max_capture_size

        if self.mode is not None:
            # Keep this import lazy so the runtime-independent unit tests do not
            # require vLLM to be installed.
            from vllm.config.compilation import CUDAGraphMode

            compilation.cudagraph_mode = CUDAGraphMode[self.mode]


def persist_policy_snapshot(vllm_config: Any) -> CudaGraphPolicySnapshot:
    """Save target CUDA Graph policy for worker processes.

    The platform hook runs before CpuPlatform mutates the config. Environment
    propagation is used because worker processes receive the normalized config.
    """

    snapshot = CudaGraphPolicySnapshot.from_vllm_config(vllm_config)
    os.environ[_POLICY_ENV] = snapshot.to_json()
    return snapshot


def load_policy_snapshot() -> CudaGraphPolicySnapshot | None:
    payload = os.getenv(_POLICY_ENV)
    if not payload:
        return None
    return CudaGraphPolicySnapshot.from_json(payload)


def _uniform_token_count(query_lens: tuple[int, ...]) -> int | None:
    if not query_lens:
        return None
    first = query_lens[0]
    if all(q == first for q in query_lens):
        return first
    return None


class ShadowCudaGraphPolicy:
    """Policy-only facade over vLLM's real CUDA Graph dispatcher.

    It deliberately does not capture or replay CUDA graphs. The default manager
    factory constructs the current vLLM CudaGraphManager on a private config
    clone and lets its real `_init_candidates()` and `dispatch()` implement the
    selection/padding policy. Device/PP scaffolding needed only by the full
    manager lifecycle is shadowed during construction.
    """

    def __init__(
        self,
        vllm_config: Any,
        *,
        snapshot: CudaGraphPolicySnapshot | None = None,
        manager_factory: Callable[[Any, CudaGraphPolicySnapshot], Any] | None = None,
    ) -> None:
        self.snapshot = snapshot or load_policy_snapshot()
        if self.snapshot is None:
            self.snapshot = CudaGraphPolicySnapshot.from_vllm_config(vllm_config)

        factory = manager_factory or self._build_vllm_manager
        self.manager = factory(vllm_config, self.snapshot)

    @staticmethod
    def _build_vllm_manager(
        vllm_config: Any, snapshot: CudaGraphPolicySnapshot
    ) -> Any:
        import torch
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu import cudagraph_utils as cg_utils

        shadow_config = copy.deepcopy(vllm_config)
        snapshot.restore(shadow_config)

        mode = (
            CUDAGraphMode[snapshot.mode]
            if snapshot.mode is not None
            else CUDAGraphMode.NONE
        )

        # CudaGraphManager's policy setup only needs PP first/last-rank flags,
        # but its constructor asks the global parallel-state helper for them.
        # A GPU-free single-host policy model does not initialize those groups,
        # so shadow just this scaffolding while preserving the manager's real
        # constructor, candidate generation and dispatch implementation.
        old_get_pp_group = cg_utils.get_pp_group
        cg_utils.get_pp_group = lambda: SimpleNamespace(  # type: ignore[assignment]
            is_first_rank=True,
            is_last_rank=True,
        )
        try:
            # Speculative decoding is intentionally disabled by the current
            # SimulationPlatform, so decode_query_len=1 exactly matches normal
            # autoregressive decode. This becomes runtime-derived when spec
            # decode is enabled in the shadow model runner.
            return cg_utils.CudaGraphManager(
                shadow_config,
                torch.device("cpu"),
                mode,
                decode_query_len=1,
                lora_capture_cases=[0],
                varlen_decode=False,
            )
        finally:
            cg_utils.get_pp_group = old_get_pp_group

    def dispatch(self, logical: Any, *, num_active_loras: int = 0) -> Any:
        query_lens = tuple(int(x) for x in logical.query_lens)
        num_tokens = int(logical.logical_tokens)
        num_reqs = len(logical.requests)
        uniform = _uniform_token_count(query_lens)
        max_query_len = max(query_lens, default=None)

        return self.manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            uniform_token_count=uniform,
            num_active_loras=int(num_active_loras),
            max_query_len=max_query_len,
        )
