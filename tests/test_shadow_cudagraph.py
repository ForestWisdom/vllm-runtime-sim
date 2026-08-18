from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

from vllm_runtime_sim.shadow_cudagraph import (
    CudaGraphPolicySnapshot,
    ShadowCudaGraphPolicy,
    _uniform_token_count,
)


class Mode(Enum):
    FULL_AND_PIECEWISE = 1
    FULL = 2


@dataclass
class FakeCompilation:
    cudagraph_mode: object
    cudagraph_capture_sizes: list[int]
    max_cudagraph_capture_size: int


@dataclass
class FakeConfig:
    compilation_config: FakeCompilation


@dataclass
class FakeRequest:
    request_id: str


@dataclass
class FakeLogical:
    query_lens: tuple[int, ...]
    logical_tokens: int
    requests: tuple[FakeRequest, ...]


class RecordingManager:
    def __init__(self):
        self.kwargs = None

    def dispatch(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            cg_mode=Mode.FULL,
            num_tokens=576,
            num_reqs=8,
            uniform_token_count=kwargs["uniform_token_count"],
            max_query_len=kwargs["max_query_len"],
            num_active_loras=kwargs["num_active_loras"],
        )


def test_snapshot_round_trip_preserves_target_policy():
    cfg = FakeConfig(FakeCompilation(Mode.FULL_AND_PIECEWISE, [1, 8, 16, 576], 576))
    snapshot = CudaGraphPolicySnapshot.from_vllm_config(cfg)

    assert snapshot.mode == "FULL_AND_PIECEWISE"
    assert snapshot.capture_sizes == (1, 8, 16, 576)
    assert snapshot.max_capture_size == 576
    assert CudaGraphPolicySnapshot.from_json(snapshot.to_json()) == snapshot


def test_uniform_query_length_detection():
    assert _uniform_token_count((1, 1, 1)) == 1
    assert _uniform_token_count((5, 5)) == 5
    assert _uniform_token_count((1, 512)) is None
    assert _uniform_token_count(()) is None


def test_shadow_policy_forwards_real_batch_shape_to_manager():
    manager = RecordingManager()
    snapshot = CudaGraphPolicySnapshot("FULL_AND_PIECEWISE", (1, 576), 576)
    policy = ShadowCudaGraphPolicy(
        FakeConfig(FakeCompilation(Mode.FULL_AND_PIECEWISE, [1, 576], 576)),
        snapshot=snapshot,
        manager_factory=lambda _cfg, _snapshot: manager,
    )
    logical = FakeLogical(
        query_lens=(1, 1, 512),
        logical_tokens=514,
        requests=(FakeRequest("A"), FakeRequest("B"), FakeRequest("C")),
    )

    desc = policy.dispatch(logical)

    assert manager.kwargs == {
        "num_reqs": 3,
        "num_tokens": 514,
        "uniform_token_count": None,
        "num_active_loras": 0,
        "max_query_len": 512,
    }
    assert desc.num_tokens == 576


def test_shadow_policy_preserves_uniform_decode_key():
    manager = RecordingManager()
    snapshot = CudaGraphPolicySnapshot("FULL", (8, 16), 16)
    policy = ShadowCudaGraphPolicy(
        FakeConfig(FakeCompilation(Mode.FULL, [8, 16], 16)),
        snapshot=snapshot,
        manager_factory=lambda _cfg, _snapshot: manager,
    )
    logical = FakeLogical(
        query_lens=(1, 1, 1, 1),
        logical_tokens=4,
        requests=tuple(FakeRequest(str(i)) for i in range(4)),
    )

    policy.dispatch(logical)
    assert manager.kwargs["uniform_token_count"] == 1
    assert manager.kwargs["max_query_len"] == 1
