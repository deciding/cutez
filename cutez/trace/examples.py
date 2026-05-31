"""Example kernels and launch helpers for the ``cutez_trace`` tool.

These examples intentionally keep warp filtering and event indexing in user
code so they mirror the intended manual ``wid``/segment/``clock_idx`` workflow.
"""

from __future__ import annotations

from pathlib import Path

import cutlass.cute as cute
import torch
import cutlass
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from .core import SharedStorage, clock_record, finanlize_clock, init_clock
from .session import CutezTraceSession

THREADS = 128
WARPS_PER_BLOCK = THREADS // 32
SEGMENT_BYTES = 32
REGION_NAMES = {1: "outer", 2: "inner"}


@cute.kernel
def sample_trace_kernel_warp0(out: cute.Tensor, iters: Int32):
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    clock_ptr = storage.clock_buf.data_ptr()
    out_ptr = out.iterator

    wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    seg_addr, out_addr, is_leader = init_clock(clock_ptr, out_ptr, wid, SEGMENT_BYTES)

    if wid == 0:
        clock_idx = Int32(0)
        clock_record(True, 1, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)
        clock_idx += 1
        for _ in cutlass.range(iters):
            clock_record(True, 10, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)
            clock_idx += 1
            clock_record(False, 10, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)
            clock_idx += 1
        clock_record(False, 1, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)

    cute.arch.sync_threads()
    if wid == 0:
        finanlize_clock(seg_addr, out_addr, SEGMENT_BYTES)


@cute.jit
def launch_sample_trace_warp0(out: cute.Tensor, iters: Int32):
    sample_trace_kernel_warp0(out, iters).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))


@cute.kernel
def sample_trace_kernel_warp01(out: cute.Tensor, iters: Int32):
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    clock_ptr = storage.clock_buf.data_ptr()
    out_ptr = out.iterator

    wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    seg_addr, out_addr, is_leader = init_clock(clock_ptr, out_ptr, wid, SEGMENT_BYTES)

    if wid == 0:
        clock_idx = Int32(0)
        clock_record(True, 1, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)
        clock_idx += 1
        for _ in cutlass.range(iters):
            clock_record(True, 10, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)
            clock_idx += 1
            clock_record(False, 10, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)
            clock_idx += 1
        clock_record(False, 1, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)

    if wid == 1:
        clock_idx = Int32(0)
        clock_record(True, 2, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)
        clock_idx += 1
        for _ in cutlass.range(iters):
            clock_record(True, 11, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)
            clock_idx += 1
            clock_record(False, 11, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)
            clock_idx += 1
        clock_record(False, 2, clock_idx, seg_addr, is_leader, SEGMENT_BYTES)

    cute.arch.sync_threads()
    if wid == 0 or wid == 1:
        finanlize_clock(seg_addr, out_addr, SEGMENT_BYTES)


@cute.jit
def launch_sample_trace_warp01(out: cute.Tensor, iters: Int32):
    sample_trace_kernel_warp01(out, iters).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))


def run_sample_trace(
    trace_path: str | Path, *, iters: int = 1, active_warps: tuple[int, ...] = (0,)
):
    """Run a sample instrumented kernel and write a trace JSON artifact.

    The caller passes the output path, iteration count, and selected recording
    warps. This helper follows the manual instrumentation contract directly:
    each recording warp uses `wid` as its segment id, maintains its own
    `clock_idx`, and only `(0,)` or `(0, 1)` are currently implemented.
    Returns a small result dict with the trace path, per-warp counts, buffer,
    and session.
    """
    if active_warps not in {(0,), (0, 1)}:
        raise ValueError("run_sample_trace only supports active_warps=(0,) or (0, 1)")

    session = CutezTraceSession(
        blocks=1,
        warps_per_block=WARPS_PER_BLOCK,
        segment_bytes=SEGMENT_BYTES,
    )
    out = session.allocate_buffer()
    out_cute = from_dlpack(out, assumed_align=8)
    launcher = launch_sample_trace_warp0 if active_warps == (0,) else launch_sample_trace_warp01
    compiled = cute.compile(launcher, out_cute, Int32(iters))
    compiled(out_cute, Int32(iters))
    torch.cuda.synchronize()

    counts = {(0, warp): 2 + 2 * iters for warp in active_warps}
    session.write_trace_json(trace_path, out, counts=counts, region_names=REGION_NAMES)
    return {
        "trace_path": str(trace_path),
        "counts": counts,
        "buffer": out,
        "session": session,
    }

if __name__ == '__main__':
    res = run_sample_trace('trace.json', iters=1, active_warps=(0, 1))
