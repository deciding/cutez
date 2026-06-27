import argparse
import os
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--seqlen-q", type=int, default=256)
    parser.add_argument("--seqlen-k", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    os.environ["USE_SIMPLE_FA4"] = "1"
    os.environ["USE_IKET_FA4"] = "1"
    os.environ["USE_TRACE_FA4"] = "1"
    os.environ["TRACE_FA4_PATH"] = str(dump_dir / "fa4_trace.json")
    os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(dump_dir))
    os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")
    os.environ.setdefault("CUTE_DSL_KEEP_CUBIN", "1")
    os.environ.setdefault("CUTE_DSL_LINEINFO", "1")

    sys.path.insert(0, "/workspace/fa4")
    sys.path.insert(0, "/workspace")

    from flash_attn_local.cute import interface as interface_local

    dtype = torch.bfloat16
    q = torch.randn(
        args.batch_size,
        args.seqlen_q,
        args.nheads,
        args.head_dim,
        dtype=dtype,
        device="cuda",
    )
    k = torch.randn(
        args.batch_size,
        args.seqlen_k,
        args.nheads,
        args.head_dim,
        dtype=dtype,
        device="cuda",
    )
    v = torch.randn(
        args.batch_size,
        args.seqlen_k,
        args.nheads,
        args.head_dim,
        dtype=dtype,
        device="cuda",
    )

    interface_local.flash_attn_func(q, k, v, causal=False)
    torch.cuda.synchronize()
    print(f"IKET FA4 run complete. dump_dir={dump_dir}")


if __name__ == "__main__":
    main()
