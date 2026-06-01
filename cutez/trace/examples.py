"""Canonical example kernels and host helpers for ``cutez_trace`` comparison."""

from __future__ import annotations

from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32

from .core import CutezTracer, SharedStorage
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
    tidx, _, _ = cute.arch.thread_idx()

    wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tracer = CutezTracer.create(
        clock_ptr, out_ptr, seg_idx=wid, segment_size=SEGMENT_BYTES
    )

    outer_scope = Int32(1)
    add_scope = Int32(2)
    acc = Int32(wid)

    tracer.enter_scope(outer_scope)
    for i in cutlass.range(iters):
        tracer.enter_scope(add_scope)

        # Do some real integer work so the traced region is not empty.
        step = Int32(i + wid + 1)
        for j in cutlass.range(64):
            delta = step + Int32(j)
            acc = acc + delta
            #acc = acc * Int32(3)
            #acc = acc * Int32(5)
            acc = acc + delta

        tracer.exit_scope(add_scope)
    tracer.exit_scope(outer_scope)

    # Keep the arithmetic live without changing the trace structure.
    if tidx == 0:
        cute.printf(acc)

    cute.arch.sync_threads()
    tracer.flush()


@cute.jit
def launch_sample_trace(out: cute.Tensor, iters: Int32):
    sample_trace_kernel(out, iters).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))


def run_sample_trace(trace_path: str | Path, *, iters: int = 4):
    """Run the 4-warp cutez trace example and write a Chrome trace JSON artifact."""

    session = CutezTraceSession(
        blocks_per_sm=1,
        warps_per_block=WARPS_PER_BLOCK,
        segment_bytes=SEGMENT_BYTES,
        trace_path=trace_path,
        region_names=REGION_NAMES,
    )
    compiled = cute.compile(launch_sample_trace, session.buffer, Int32(iters))
    compiled(session.buffer, Int32(iters))
    torch.cuda.synchronize()

    session.write_trace_json()
    return {
        "trace_path": str(session.trace_path),
        "buffer": session.buffer_tensor,
        "session": session,
    }


def run_quack_trace(trace_path: str | Path, *, iters: int = 4):
    """Run a QuACK-based trace with the same 4-warp loop shape for comparison."""

    from cutlass.cutlass_dsl import Int64
    from quack.trace import TraceContext, TraceSession

    @cute.kernel
    def sample_quack_trace_kernel(trace_ptr: None, inner_iters: Int32):
        ctx = TraceContext.create(trace_ptr)
        wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        acc = Int32(wid)
        tidx, _, _ = cute.arch.thread_idx()

        ctx.b("outer")
        # for i in cutlass.range(inner_iters):
        #    ctx.b("add")

        #    step = Int32(i + wid + 1)
        #    for j in cutlass.range(64):
        #        delta = step + Int32(j)
        #        acc = acc + delta
        #        acc = acc * Int32(3)
        #        acc = acc * Int32(5)
        #        acc = acc + delta

        #    ctx.e("add")
        ctx.b("add0")

        step = Int32(wid + 1)
        for j in cutlass.range(64):
            delta = step + Int32(j)
            acc = acc + delta
            acc = acc * Int32(3)
            acc = acc * Int32(5)
            acc = acc + delta

        ctx.e("add0")
        ctx.b("add1")

        step = Int32(wid + 2)
        for j in cutlass.range(64):
            delta = step + Int32(j)
            acc = acc + delta
            acc = acc * Int32(3)
            acc = acc * Int32(5)
            acc = acc + delta

        ctx.e("add1")

        ctx.e("outer")

        # Keep the arithmetic live without changing the trace structure.
        if tidx == 0:
            cute.printf(acc)
        ctx.flush()

    @cute.jit
    def launch_quack_trace(trace_ptr: None, inner_iters: Int32):
        sample_quack_trace_kernel(trace_ptr, inner_iters).launch(
            grid=(1, 1, 1), block=(THREADS, 1, 1)
        )

    trace_path = str(trace_path)
    with TraceSession(trace_path, grid_size=1, block_size=THREADS) as session:
        launch_quack_trace(session.ptr, Int32(iters))

    return {
        "trace_path": trace_path,
        "trace_enabled": session.ptr is not None,
        "session": session,
    }


if __name__ == "__main__":
    cutez_res = run_sample_trace("trace_cutez.json", iters=4)
    print(cutez_res["trace_path"])
    # quack_res = run_quack_trace("trace_quack.json", iters=2)
    # print(quack_res["trace_path"])
