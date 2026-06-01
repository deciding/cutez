"""Canonical example kernel and host helper for ``cutez_trace``.

This example records four warp-owned trace segments in one block. Each warp
emits one outer scope and repeated inner add-scope records so the per-warp ring
buffer wraps once `iters` is large enough.
"""

from __future__ import annotations

from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from .core import SharedStorage, clock_record, finanlize_clock, init_clock
from .session import CutezTraceSession

THREADS = 128
WARPS_PER_BLOCK = THREADS // 32
SEGMENT_BYTES = 64
REGION_NAMES = {1: "outer", 2: "add"}


@cute.kernel
def sample_trace_kernel(out: cute.Tensor, iters: Int32):
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    clock_ptr = storage.clock_buf.data_ptr()
    out_ptr = out.iterator

    wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    seg_addr, out_addr, is_leader = init_clock(
        clock_ptr, out_ptr, seg_idx=wid, segment_size=SEGMENT_BYTES
    )

    outer_scope = Int32(1)
    add_scope = Int32(2)
    clock_idx = Int32(0)

    clock_record(True, outer_scope, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)
    clock_idx += 1
    for _ in cutlass.range(iters):
        clock_record(True, add_scope, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)
        clock_idx += 1
        clock_record(False, add_scope, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)
        clock_idx += 1
    clock_record(False, outer_scope, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)

    cute.arch.sync_threads()
    finanlize_clock(seg_addr, out_addr, SEGMENT_BYTES)


@cute.jit
def launch_sample_trace(out: cute.Tensor, iters: Int32):
    sample_trace_kernel(out, iters).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))


def run_sample_trace(trace_path: str | Path, *, iters: int = 4):
    """Run the 4-warp trace example and write a Chrome trace JSON artifact."""

    session = CutezTraceSession(
        blocks=1,
        warps_per_block=WARPS_PER_BLOCK,
        segment_bytes=SEGMENT_BYTES,
    )
    out = session.allocate_buffer()
    out_cute = from_dlpack(out, assumed_align=8)
    compiled = cute.compile(launch_sample_trace, out_cute, Int32(iters))
    compiled(out_cute, Int32(iters))
    torch.cuda.synchronize()

    counts = {(0, warp): 2 + 2 * iters for warp in (0, 1, 2, 3)}
    session.write_trace_json(trace_path, out, counts=counts, region_names=REGION_NAMES)
    return {
        "trace_path": str(trace_path),
        "counts": counts,
        "buffer": out,
        "session": session,
    }


if __name__ == "__main__":
    run_sample_trace("trace.json", iters=2)
