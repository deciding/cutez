"""Intra-kernel trace of a CuTe-DSL element-wise multiply with G-Watch.

Demonstrates:
  1. Embedding scope_start / scope_end sentinels via CuTe DSL inline asm.
  2. Running do_trace() to instrument the JIT-compiled cubin and collect
     per-thread trace records at PTX level.

Usage:
    python3 trace_cute_matmul.py
"""
import os

# Must be set BEFORE importing cutlass so its env manager picks up the PTX
# dump directory and keep-PTX flag during module init. do_trace falls back
# to this directory when CuTe's cubin bypasses CUPTI's module-load capture.
os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")
os.environ.setdefault(
    "CUTE_DSL_DUMP_DIR", os.path.dirname(os.path.abspath(__file__))
)

# Order matters. Import gwatch.cuda.trace.cute FIRST — it transitively
# creates the GWCapsule — THEN call init_cupti_hooks() to actually install
# the CUPTI driver-API callbacks (no-op if the capsule doesn't exist yet).
# This must happen BEFORE cutlass is imported so CUPTI is listening by
# the time @cute.jit triggers cuLibraryLoadData / cuModuleLoadData.
import gwatch.libpygwatch as pygwatch
import gwatch.cuda.trace.cute as gw_trace
from gwatch.cuda.trace import do_trace
from gwatch.common.format import File
from gwatch.cuda.trace.format import Section_IntraKernelTrace
assert pygwatch.init_cupti_hooks(), "failed to install CUPTI hooks"

import argparse
from collections import Counter

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack


BLOCK_SIZE = 1024


def dump_report(result, stem, report_path=None):
    """Dump a trace report to ``report_path`` (format from extension), or to
    ``results/<stem>.html`` by default. Use a ``.json`` path for the
    machine-readable archive that ``gwatch show`` reads. (YAML is not
    supported — JSON is far faster for large traces.)"""
    if not result or not result.get("trace_results"):
        print(f"[{stem}] no trace records; skipping report")
        return
    paths = [report_path] if report_path else [os.path.join("results", stem + ".html")]
    section = Section_IntraKernelTrace()
    section.add_run(result)
    report = File(title=f"Intra-kernel trace — {stem}")
    report.add_section(section)
    for p in paths:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        report.render(p)
        print(f"Report written to: {os.path.abspath(p)}")


@cute.kernel
def mul_kernel(
    x: cute.Tensor,
    y: cute.Tensor,
    output: cute.Tensor,
    n_elements: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    offset = bidx * BLOCK_SIZE + tidx

    if offset < n_elements:
        # One scope per phase so the trace timeline shows several distinct
        # regions (two loads, a multiply, a compute-heavy poly, the store).
        gw_trace.scope_start(1)        # load_x
        val_x = x[offset]
        gw_trace.scope_end(1)

        gw_trace.scope_start(2)        # load_y
        val_y = y[offset]
        gw_trace.scope_end(2)

        gw_trace.scope_start(3)        # mul
        p = val_x * val_y
        gw_trace.scope_end(3)

        gw_trace.scope_start(4)        # poly — a few fused multiply-adds
        for _ in range(8):
            p = p * 0.999 + val_x
        gw_trace.scope_end(4)

        gw_trace.scope_start(5)        # store
        output[offset] = p
        gw_trace.scope_end(5)


@cute.jit
def launch_mul(x, y, output, n_elements, stream):
    grid_x = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    mul_kernel(x, y, output, Int32(n_elements)).launch(
        grid=[grid_x, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


def run_mul(x, y, output, n_elements):
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    launch_mul(
        from_dlpack(x, assumed_align=16),
        from_dlpack(y, assumed_align=16),
        from_dlpack(output, assumed_align=16),
        n_elements,
        stream,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CuTe-DSL mul-kernel trace demo.")
    parser.add_argument(
        "--report", type=str, default=None,
        help="Output report path; format from extension (.html interactive / "
             ".json machine-readable). Default: results/<script>.html."
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    size = 98432
    x = torch.randn(size, device="cuda")
    y = torch.randn(size, device="cuda")
    n_elements = x.numel()
    output_cute = torch.empty_like(x)

    scope_name_map = {1: "load_x", 2: "load_y", 3: "mul", 4: "poly", 5: "store"}

    result = do_trace(
        fn=lambda: run_mul(x, y, output_cute, n_elements),
        kernel_name_pattern=r".*mul_kernel.*",
        dsl="cute",
        scope_name_map=scope_name_map,
        instrumentation_tier="ptx",
    )

    print(f"kernel_prototype: {result.get('kernel_prototype')}")
    if result.get("compile_results"):
        print(f"compile_results:  {result['compile_results']}")
    records = result.get("trace_results") or []
    print(f"trace_results:    {len(records)} records")
    counts = Counter((r["scope_label"], r["type_str"]) for r in records)
    print("  per-scope record counts:")
    for (label, ttype) in sorted(counts):
        print(f"    {label:<12s} {ttype:<12s} {counts[(label, ttype)]}")

    ref = x * y
    for _ in range(8):
        ref = ref * 0.999 + x
    assert torch.allclose(output_cute, ref, atol=1e-3, rtol=1e-3), \
        "CuTe output does not match PyTorch output"
    print("PASS: CuTe output matches PyTorch output")

    dump_report(result, os.path.splitext(os.path.basename(__file__))[0], args.report)
