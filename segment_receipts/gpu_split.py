from __future__ import annotations

from segment_receipts.graph import GraphModel
from segment_receipts.models import GpuCopyPoint, Segment
from segment_receipts.rules import SegmentRules


def plan_gpu_splits(
    graph: GraphModel,
    segments: list[Segment],
    rules: SegmentRules,
) -> list[GpuCopyPoint]:
    """
    Multi-GPU pipeline split visualization (FTL Fig 7).

    Inserts cross-gpu-copy points after configured nodes or at GPU assignment boundaries.
    """
    copies: list[GpuCopyPoint] = []
    node_to_segment = {
        name: seg.id for seg in segments for name in seg.node_names
    }

    for node_name in rules.cross_gpu_after_nodes:
        if node_name not in graph.node_by_name:
            continue
        seg_id = node_to_segment.get(node_name)
        if seg_id is None:
            continue
        node = graph.node_by_name[node_name]
        tensor = node.outputs[0] if node.outputs else node_name
        from_gpu = rules.gpu_assignments.get(node_name, 0)
        to_gpu = from_gpu + 1
        copies.append(
            GpuCopyPoint(
                after_segment_id=seg_id,
                from_gpu=from_gpu,
                to_gpu=to_gpu,
                tensor_name=tensor,
            )
        )

    # Auto-detect GPU boundary crossings from assignments
    prev_gpu: int | None = None
    for seg in segments:
        seg_gpus = {
            rules.gpu_assignments.get(n, 0) for n in seg.node_names
        }
        seg_gpu = min(seg_gpus) if seg_gpus else 0
        if prev_gpu is not None and seg_gpu != prev_gpu:
            last_node = seg.node_names[-1]
            node = graph.node_by_name[last_node]
            tensor = node.outputs[0] if node.outputs else last_node
            copies.append(
                GpuCopyPoint(
                    after_segment_id=seg.id,
                    from_gpu=prev_gpu,
                    to_gpu=seg_gpu,
                    tensor_name=tensor,
                )
            )
        prev_gpu = seg_gpu

    return copies
