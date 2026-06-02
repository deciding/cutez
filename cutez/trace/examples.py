"""Canonical example kernels and host helpers for ``cutez_trace`` comparison."""

from __future__ import annotations

from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32

from .core import CutezTracer, TraceConfig, debug_smem_usage, get_smem_cap
from .session import CutezTraceSession

THREADS = 128
WARPS_PER_BLOCK = THREADS // 32
BLOCKS_PER_SM = 1
TOTAL_BLOCKS = 16
SM_SMEM_AVAILABLE_BYTES = 100352


@cute.kernel
def sample_trace_kernel(out: cute.Tensor, iters: Int32, trace_cfg: TraceConfig):
    tidx, _, _ = cute.arch.thread_idx()

    wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tracer = CutezTracer.create(out, seg_idx=wid, cfg=trace_cfg)

    acc = Int32(wid)

    tracer.enter_scope("outer")
    for i in cutlass.range(iters):
        tracer.enter_scope("add")

        # Do some real integer work so the traced region is not empty.
        step = Int32(i + wid + 1)
        for j in cutlass.range(64):
            delta = step + Int32(j)
            acc = acc + delta
            # acc = acc * Int32(3)
            # acc = acc * Int32(5)
            acc = acc + delta

        tracer.exit_scope("add")
    tracer.exit_scope("outer")

    # Keep the arithmetic live without changing the trace structure.
    if tidx == 0:
        cute.printf(acc)
        # debug_smem_usage(101376)

    cute.arch.sync_threads()
    tracer.flush()


@cute.jit
def launch_sample_trace(out: cute.Tensor, iters: Int32, trace_cfg: TraceConfig):
    sample_trace_kernel(out, iters, trace_cfg).launch(
        grid=(TOTAL_BLOCKS, 1, 1), block=(THREADS, 1, 1)
    )


def run_sample_trace(trace_path: str | Path, *, iters: int = 4):
    """Run the 4-warp cutez trace example and write a Chrome trace JSON artifact."""

    session = CutezTraceSession(
        sm_smem_available_bytes=SM_SMEM_AVAILABLE_BYTES,
        total_blocks=TOTAL_BLOCKS,
        warps_per_block=WARPS_PER_BLOCK,
        trace_path=trace_path,
        dummy=False,
    )
    get_smem_cap()
    compiled = cute.compile(
        launch_sample_trace, session.buffer, Int32(iters), session.trace_config
    )
    compiled(session.buffer, Int32(iters), session.trace_config)
    torch.cuda.synchronize()

    # Uncomment for raw per-block/warp clock diagnostics before JSON normalization.
    # print(session.debug_dump_segments())

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
            grid=(TOTAL_BLOCKS, 1, 1), block=(THREADS, 1, 1)
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
