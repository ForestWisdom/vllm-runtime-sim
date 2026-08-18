from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ir import LogicalWorkload, RequestWorkload


@dataclass
class RequestState:
    num_computed_tokens: int


class SchedulerOutputAdapter:
    """Convert a vLLM V1 SchedulerOutput-like object to LogicalWorkload.

    This adapter intentionally uses duck typing so the simulation core remains
    importable without vLLM installed. Worker-side cached state is maintained
    only for fields that vLLM sends incrementally across scheduling steps.
    """

    def __init__(self) -> None:
        self._requests: dict[str, RequestState] = {}

    def extract(self, output: Any) -> LogicalWorkload:
        finished = getattr(output, "finished_req_ids", set()) or set()
        for req_id in finished:
            self._requests.pop(req_id, None)

        for req in getattr(output, "scheduled_new_reqs", ()):
            self._requests[req.req_id] = RequestState(
                num_computed_tokens=int(req.num_computed_tokens)
            )

        cached = getattr(output, "scheduled_cached_reqs", None)
        if cached is not None:
            for req_id, computed in zip(
                getattr(cached, "req_ids", ()),
                getattr(cached, "num_computed_tokens", ()),
            ):
                self._requests[req_id] = RequestState(int(computed))

        scheduled = getattr(output, "num_scheduled_tokens", {})
        spec = getattr(output, "scheduled_spec_decode_tokens", {}) or {}
        requests: list[RequestWorkload] = []

        for req_id, num_tokens in scheduled.items():
            if req_id not in self._requests:
                raise KeyError(
                    f"missing cached request state for {req_id}; "
                    "adapter must observe the request's first scheduled step"
                )
            state = self._requests[req_id]
            requests.append(
                RequestWorkload(
                    request_id=req_id,
                    num_computed_tokens=state.num_computed_tokens,
                    num_scheduled_tokens=int(num_tokens),
                    speculative_tokens=len(spec.get(req_id, ())),
                )
            )

        return LogicalWorkload(tuple(requests))
