#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.experimental import iket

ITERS = 8
BLOCK_THREADS = 64
GRID = 2


@cute.kernel
def _trace_kernel(iters: Int32) -> None:
    iket.range_push("kernel_total")
    for _ in cutlass.range(iters):
        iket.range_push("loop_body")
        iket.mark("loop_mark")
        iket.range_pop()
    iket.range_pop()


@cute.jit
def _launch(iters: Int32) -> None:
    _trace_kernel(iters).launch(
        grid=(GRID, 1, 1),
        block=(BLOCK_THREADS, 1, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(dump_dir))

    torch.cuda.init()
    _launch(ITERS)
    torch.cuda.synchronize()
    print(f"ran IKET trace workload: grid={GRID}, block={BLOCK_THREADS}, iters={ITERS}")
    print(f"dump_dir={dump_dir}")


if __name__ == "__main__":
    main()
