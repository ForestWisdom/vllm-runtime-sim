from __future__ import annotations

from dataclasses import dataclass

from .ir import AttentionOp, CollectiveOp, GemmOp, PhysicalDAG, PointwiseOp


@dataclass(frozen=True)
class HardwareProfile:
    peak_tflops: float = 1000.0
    memory_bandwidth_gbps: float = 3000.0
    collective_bandwidth_gbps: float = 450.0
    collective_latency_us: float = 2.0
    kernel_launch_us: float = 3.0


class AnalyticalBackend:
    """Simple sanity-check backend, not intended as the final predictor.

    The interfaces are deliberately compatible with swapping in profiled
    Vidur/NeuSight-style compute models and SimAI-style communication models.
    """

    def __init__(self, profile: HardwareProfile | None = None):
        self.profile = profile or HardwareProfile()

    def estimate_us(self, op) -> float:
        p = self.profile
        if isinstance(op, GemmOp):
            flops = 2 * op.m * op.n * op.k
            return flops / (p.peak_tflops * 1e12) * 1e6 + p.kernel_launch_us
        if isinstance(op, CollectiveOp):
            # Ring-like bandwidth approximation; a real backend should model
            # algorithm/topology or delegate the collective to SimAI.
            factor = 2 * (op.world_size - 1) / op.world_size
            bytes_on_wire = factor * op.payload_bytes
            return (
                p.collective_latency_us
                + bytes_on_wire / (p.collective_bandwidth_gbps * 1e9) * 1e6
            )
        if isinstance(op, AttentionOp):
            qk_pairs = 0
            for q, past in zip(op.query_lens, op.past_kv_lens):
                # causal chunk: sum(past + 1 ... past + q)
                qk_pairs += q * past + q * (q + 1) // 2
            flops = 4 * qk_pairs * op.num_q_heads * op.head_dim
            return flops / (p.peak_tflops * 1e12) * 1e6 + p.kernel_launch_us
        if isinstance(op, PointwiseOp):
            bytes_touched = op.elements * 4  # coarse read+write BF16 proxy
            return bytes_touched / (p.memory_bandwidth_gbps * 1e9) * 1e6 + p.kernel_launch_us
        raise TypeError(type(op))

    def estimate_dag_us(self, dag: PhysicalDAG) -> float:
        # v0.1 executes the linearized dependency chain. A later timeline
        # engine will replace this with critical-path and overlap scheduling.
        return sum(self.estimate_us(op) for op in dag.operations)
