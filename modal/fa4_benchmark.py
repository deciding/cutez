"""
Modal script to run FA4 (CuTeDSL) sm100 (Blackwell) benchmark on B200 GPU.
Compares local (mounted) version against PyTorch reference implementation.
"""

from modal import Image, App, Volume
import pathlib

root_dir = pathlib.Path(__file__).parent
GPU_model = "B200"

app = App(name="fa4-sm100-benchmark")

VOLUME_NAME = "fa4-dump"
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

fa4_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("wget", "curl", "gnupg", "git")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
    )
    .apt_install("cuda-toolkit-13-2")
    .workdir("/workspace")
)

fa4_image = (
    fa4_image.pip_install("torch", "pytest", "einops")
    #.pip_install("nvidia-cutlass-dsl>=4.4.1")
    .pip_install("nvidia-cutlass-dsl==4.6.0.dev0")
    .pip_install("apache-tvm-ffi>=0.1.5,<0.2")
    .pip_install("torch-c-dlpack-ext")
    .pip_install("triton==3.5.1")
    #.pip_install("quack-kernels>=0.2.10")
    #.pip_install("flash-attn-4==4.0.0b4")
    .pip_install("flash-attn-4>=4.0.0b16")
    .pip_install("quack-kernels>=0.5.0")
    .pip_install("teraxlang==3.5.1.dev4")
    .add_local_dir(root_dir / "fa4", remote_path="/workspace/fa4")
    .add_local_dir(
        root_dir.parent / "cutez",
        remote_path="/workspace/cutez",
    )
)


@app.function(
    gpu=GPU_model,
    image=fa4_image,
    timeout=600,
    volumes={"/workspace/dump": volume},
)
def run_fa4_benchmark(
    use_simple: bool = False,
    use_trace: bool = False,
    use_iket: bool = False,
):
    import torch
    import sys
    import math
    from typing import NamedTuple
    from triton.testing import do_bench
    import os
    import subprocess

    # result = subprocess.run(
    #    ["find", "/", "-name", "cuobjdump"],
    #    capture_output=True,
    #    text=True,
    #    check=True,
    # )
    # output = result.stdout
    # print(output)
    cuobjdump_path = "/usr/local/cuda-13.2/bin/cuobjdump"
    from pathlib import Path

    def run_cuobjdump(DUMP_DIR):
        dump_path = Path(DUMP_DIR)

        # 1. Search for the cubin file containing 'localcute'
        files = list(dump_path.glob("*localcute*.cubin"))

        if not files:
            print(f"Error: No .cubin file with 'localcute' found in {DUMP_DIR}")
            return

        target_file = files[0]
        log_file = dump_path / "cuobjdump.log"

        print(f"Processing: {target_file.name}...")

        try:
            # 2. Run cuobjdump -res-usage to check for LMEM/Register stats
            # -res-usage is the most direct way to see if 'LMEM' > 0
            result = subprocess.run(
                # [cuobjdump_path, "-res-usage", str(target_file)],
                [cuobjdump_path, "--dump-sass", str(target_file)],
                capture_output=True,
                text=True,
                check=True,
            )

            # 3. Write output to log file
            with open(log_file, "w") as f:
                f.write(result.stdout)

            print(f"Success! Output written to: {log_file}")

            # Quick check for spills in the console
            if "LMEM" in result.stdout:
                print("\nQuick Summary:")
                print(result.stdout.strip())

        except subprocess.CalledProcessError as e:
            print(f"Command failed. Error output:\n{e.stderr}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    from datetime import datetime

    # Set environment variable BEFORE importing flash_attn
    if use_trace:
        use_simple = True
        os.environ["USE_TRACE_FA4"] = "1"
        os.environ["TRACE_FA4_PATH"] = "/workspace/dump/fa4_trace.json"
    if use_iket:
        use_simple = True
        os.environ["USE_IKET_FA4"] = "1"
        print("USE_TRACE_FA4")
    if use_simple:
        os.environ["USE_SIMPLE_FA4"] = "1"
        print("USE_SIMPLE_FA4")
    else:
        os.environ["USE_SIMPLE_FA4"] = "0"

    dump_name = "fa4" + "".join(str(datetime.now()).replace(":", ".").split())
    DUMP_DIR = "/workspace/dump/" + dump_name
    os.makedirs(DUMP_DIR, exist_ok=True)
    os.environ["CUTE_DSL_DUMP_DIR"] = DUMP_DIR
    os.environ["CUTE_DSL_KEEP_PTX"] = "1"
    os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
    os.environ["CUTE_DSL_LINEINFO"] = "1"

    class Timing(NamedTuple):
        mean: float

    def time_fwd(func, *args, repeats=30, **kwargs):
        return Timing(
            do_bench(lambda: func(*args, **kwargs), warmup=5, rep=repeats) * 1e-3
        )

    def calc_tflops(
        time_ms, batch_size, nheads, seqlen_q, seqlen_k, head_dim, causal=False
    ):
        avg_seqlen = seqlen_k if not causal else (seqlen_k - seqlen_q + seqlen_k) // 2
        flops = batch_size * nheads * 2 * seqlen_q * avg_seqlen * (head_dim + head_dim)
        return flops / time_ms / 1e12

    def attention_ref(q, k, v, causal=False, upcast=True):
        """PyTorch reference implementation of attention."""
        dtype_og = q.dtype
        if upcast:
            q, k, v = q.float(), k.float(), v.float()

        seqlen_q, seqlen_k = q.shape[1], k.shape[1]
        d = q.shape[-1]
        softmax_scale = 1.0 / math.sqrt(d)

        scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)

        if causal:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1).to(
            dtype_og.dtype if not upcast else torch.float32
        )
        output = torch.einsum("bhts,bshd->bthd", attention, v)

        if upcast:
            output = output.to(dtype_og)

        # Compute LSE for comparison
        lse = scores.logsumexp(dim=-1)

        return output, lse

    print("=" * 60)
    print("FA4 sm100 (Blackwell) Benchmark - Local vs PyTorch Reference")
    print("=" * 60)

    # Check GPU
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    # Config: seq_len=8k, head_dim=128, batch=4 (matching official benchmark)
    batch_size = 4
    nheads = 16
    seqlen_q = 8192
    seqlen_k = 8192
    head_dim = 128
    dtype = torch.bfloat16
    causal = False
    repeats = 30

    print(
        f"\nConfig: batch={batch_size}, heads={nheads}, seq_len={seqlen_q}, head_dim={head_dim}, causal={causal}"
    )
    print(f"Dtype: {dtype}, Repeats: {repeats}")

    # Create inputs
    q = torch.randn(batch_size, seqlen_q, nheads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, seqlen_k, nheads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, seqlen_k, nheads, head_dim, dtype=dtype, device="cuda")

    # ===== Import Local (Mounted) Version =====
    print("\n" + "=" * 60)
    print("=== Local (Mounted) FA4 ===")
    print("=" * 60)

    sys.path.insert(0, "/workspace/fa4")
    sys.path.insert(0, "/workspace/cutez")
    from flash_attn_local.cute import interface as interface_local

    flash_attn_func_local = interface_local.flash_attn_func

    # Run local version once (with warmup if not tracing)
    if not use_trace:
        warmup_iters = 5
        for i in range(warmup_iters):
            _ = flash_attn_func_local(q, k, v, causal=causal)
            torch.cuda.synchronize()

    o_local, lse_local = flash_attn_func_local(q, k, v, causal=causal)
    torch.cuda.synchronize()

    if not use_trace:
        # Benchmark timing
        m_local = time_fwd(
            flash_attn_func_local, q, k, v, causal=causal, repeats=repeats
        )
        tflops_local = calc_tflops(
            m_local.mean, batch_size, nheads, seqlen_q, seqlen_k, head_dim, causal
        )

        print(f"Mean time: {m_local.mean * 1e3:.3f} ms")
        print(f"TFLOPS: {tflops_local:.2f}")

    # ===== Compute PyTorch Reference =====
    print("\n" + "=" * 60)
    print("=== PyTorch Reference ===")
    print("=" * 60)

    # Use FP32 for reference to minimize numerical error
    o_ref, lse_ref = attention_ref(q, k, v, causal=causal, upcast=True)
    torch.cuda.synchronize()
    print(f"Reference output computed (upcast to FP32)")

    # ===== Comparison =====
    print("\n" + "=" * 60)
    print("=== Output Comparison ===")
    print("=" * 60)

    # Print sample values from outputs
    print(f"\nLocal output sample values (first 5 elements):")
    print(f"  {o_local[0, 0, 0, :5].tolist()}")
    print(f"Local output shape: {o_local.shape}, dtype: {o_local.dtype}")
    if lse_local is not None:
        print(f"Local LSE sample values (first 5):")
        print(f"  {lse_local[0, 0, :5].tolist()}")
    else:
        print(f"Local LSE: None")

    print(f"\nReference output sample values (first 5 elements):")
    print(f"  {o_ref[0, 0, 0, :5].tolist()}")
    print(f"Reference output shape: {o_ref.shape}, dtype: {o_ref.dtype}")
    if lse_ref is not None:
        print(f"Reference LSE sample values (first 5):")
        print(f"  {lse_ref[0, 0, :5].tolist()}")

    # Compare outputs against reference
    diff = o_local.float() - o_ref.float()
    abs_diff = torch.abs(diff)
    rel_diff = abs_diff / (torch.abs(o_ref.float()) + 1e-8)

    print(f"\nOutput difference stats (vs Reference):")
    print(f"  Max absolute diff: {abs_diff.max().item():.6e}")
    print(f"  Mean absolute diff: {abs_diff.mean().item():.6e}")
    print(f"  Max relative diff: {rel_diff.max().item():.6e}")
    print(f"  Mean relative diff: {rel_diff.mean().item():.6e}")

    # Find where the max difference occurs
    max_diff_idx = abs_diff.argmax()
    max_diff_flat_idx = max_diff_idx.item()
    # Convert flat index to multi-dimensional indices
    b_idx = max_diff_flat_idx // (
        o_local.shape[1] * o_local.shape[2] * o_local.shape[3]
    )
    h_idx = (
        max_diff_flat_idx % (o_local.shape[1] * o_local.shape[2] * o_local.shape[3])
    ) // (o_local.shape[2] * o_local.shape[3])
    s_idx = (
        max_diff_flat_idx % (o_local.shape[2] * o_local.shape[3])
    ) // o_local.shape[3]
    d_idx = max_diff_flat_idx % o_local.shape[3]
    print(f"  Max diff location: batch={b_idx}, head={h_idx}, seq={s_idx}, dim={d_idx}")
    print(
        f"  Local value at max diff: {o_local[b_idx, h_idx, s_idx, d_idx].item():.6e}"
    )
    print(
        f"  Reference value at max diff: {o_ref[b_idx, h_idx, s_idx, d_idx].item():.6e}"
    )

    # ===== Position-Based Error Analysis =====
    print(
        f"\n=== Position-Based Error Analysis ==="
    )  # Analyze error distribution across sequence positions
    # Shape: (batch, seq_q, nheads, head_dim)
    seq_len = o_local.shape[1]
    if seq_len >= 4:
        # Split sequence into quadrants
        mid_seq = seq_len // 2
        mid_dim = o_local.shape[3] // 2

        # Top-left quadrant (early sequence, early dims)
        top_left = abs_diff[:, :mid_seq, :, :mid_dim]
        # Top-right quadrant (early sequence, late dims)
        top_right = abs_diff[:, :mid_seq, :, mid_dim:]
        # Bottom-left quadrant (late sequence, early dims)
        bottom_left = abs_diff[:, mid_seq:, :, :mid_dim]
        # Bottom-right quadrant (late sequence, late dims)
        bottom_right = abs_diff[:, mid_seq:, :, mid_dim:]

        print(f"\n  Quadrant Analysis (sequence x head_dim):")
        print(
            f"    Top-Left    (seq 0-{mid_seq - 1}, dim0-{mid_dim - 1}): mean={top_left.mean().item():.6e}, max={top_left.max().item():.6e}"
        )
        print(
            f"    Top-Right   (seq 0-{mid_seq - 1}, dim{mid_dim}-{o_local.shape[3] - 1}): mean={top_right.mean().item():.6e}, max={top_right.max().item():.6e}"
        )
        print(
            f"    Bottom-Left (seq {mid_seq}-{seq_len - 1}, dim 0-{mid_dim - 1}): mean={bottom_left.mean().item():.6e}, max={bottom_left.max().item():.6e}"
        )
        print(
            f"    Bottom-Right (seq {mid_seq}-{seq_len - 1}, dim {mid_dim}-{o_local.shape[3] - 1}): mean={bottom_right.mean().item():.6e}, max={bottom_right.max().item():.6e}"
        )

        # Per-sequence-position error (average over batch, heads, dims)
        seq_errors = abs_diff.mean(dim=(0, 2, 3))  # shape: (seq_len,)
        print(f"\n  Sequence Position Error Profile:")
        print(f"    First 5 positions: {seq_errors[:5].tolist()}")
        print(f"    Last 5 positions: {seq_errors[-5:].tolist()}")
        print(f"    First half avg: {seq_errors[:mid_seq].mean().item():.6e}")
        print(f"    Second half avg: {seq_errors[mid_seq:].mean().item():.6e}")

        # Per-head-dimension error (average over batch, seq, heads)
        dim_errors = abs_diff.mean(dim=(0, 1, 2))  # shape: (head_dim,)
        print(f"\n  Head Dimension Error Profile:")
        print(f"    First 8 dims: {dim_errors[:8].tolist()}")
        print(f"    Last 8 dims: {dim_errors[-8:].tolist()}")
        print(f"    First half avg: {dim_errors[:mid_dim].mean().item():.6e}")
        print(f"    Second half avg: {dim_errors[mid_dim:].mean().item():.6e}")

    # ===== Percentage Not Close Analysis =====
    print(f"\n=== Percentage Not Close Analysis ===")
    tolerance_levels = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    total_elements = abs_diff.numel()

    print(f"  Total elements: {total_elements:,}")
    for tol in tolerance_levels:
        not_close = (abs_diff > tol).sum().item()
        pct_not_close = 100.0 * not_close / total_elements
        print(
            f"    > {tol:.0e}: {not_close:,} elements ({pct_not_close:.2f}% not close)"
        )

    # Histogram of errors
    print(f"\n  Error Histogram (absolute difference):")
    hist_bins = [0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, float("inf")]
    hist_labels = [
        "[0, 1e-4)",
        "[1e-4, 1e-3)",
        "[1e-3, 1e-2)",
        "[1e-2, 5e-2)",
        "[5e-2, 1e-1)",
        "[1e-1, 5e-1)",
        "[5e-1, 1.0)",
        "[1.0, inf)",
    ]
    for i in range(len(hist_bins) - 1):
        if i == len(hist_bins) - 2:
            count = (abs_diff >= hist_bins[i]).sum().item()
        else:
            count = (
                ((abs_diff >= hist_bins[i]) & (abs_diff < hist_bins[i + 1]))
                .sum()
                .item()
            )
        pct = 100.0 * count / total_elements
        bar = "=" * int(pct / 2)
        print(f"    {hist_labels[i]}: {count:,} ({pct:.1f}%) {bar}")

    # Check numerical closeness
    atol = 1e-2  # Absolute tolerance
    rtol = 1e-2  # Relative tolerance
    is_close = torch.allclose(o_local.float(), o_ref.float(), atol=atol, rtol=rtol)
    print(f"\n  All close vs reference (atol={atol}, rtol={rtol}): {is_close}")

    # Check for NaN/Inf
    print(f"  Local has NaN: {torch.isnan(o_local).any().item()}")
    print(f"  Local has Inf: {torch.isinf(o_local).any().item()}")
    print(f"  Reference has NaN: {torch.isnan(o_ref).any().item()}")
    print(f"  Reference has Inf: {torch.isinf(o_ref).any().item()}")

    # Compare LSE
    if lse_local is not None and lse_ref is not None:
        lse_diff = torch.abs(lse_local.float() - lse_ref.float())
        print(f"\nLSE difference stats (vs Reference):")
        print(f"  Max LSE diff: {lse_diff.max().item():.6e}")
        print(f"  Mean LSE diff: {lse_diff.mean().item():.6e}")

    # ===== Optional: Compare with Pip Version =====
    if not use_trace:
        print("\n" + "=" * 60)
        print("=== Optional: Pip FA4 Comparison ===")
        print("=" * 60)

        try:
            # Remove local path temporarily to import pip version
            sys.path.remove("/workspace/fa4")

            from flash_attn.cute.interface import flash_attn_func as flash_attn_func_pip

            # Warmup
            for _ in range(5):
                _ = flash_attn_func_pip(q, k, v, causal=causal)
            torch.cuda.synchronize()

            # Run pip version and collect output
            o_pip, lse_pip = flash_attn_func_pip(q, k, v, causal=causal)
            torch.cuda.synchronize()

            # Benchmark timing
            m_pip = time_fwd(
                flash_attn_func_pip, q, k, v, causal=causal, repeats=repeats
            )
            tflops_pip = calc_tflops(
                m_pip.mean, batch_size, nheads, seqlen_q, seqlen_k, head_dim, causal
            )

            print(f"Pip output sample values (first 5 elements):")
            print(f"  {o_pip[0, 0, 0, :5].tolist()}")
            print(f"Pip TFLOPS: {tflops_pip:.2f}")

            # Compare pip vs reference
            diff_pip = o_pip.float() - o_ref.float()
            abs_diff_pip = torch.abs(diff_pip)
            print(f"Pip vs Reference max diff: {abs_diff_pip.max().item():.6e}")

            # Compare local vs pip
            diff_local_pip = o_local.float() - o_pip.float()
            abs_diff_local_pip = torch.abs(diff_local_pip)
            print(f"Local vs Pip max diff: {abs_diff_local_pip.max().item():.6e}")

            print("\n=== Performance Comparison ===")
            print(f"Local (Mounted): {tflops_local:.2f} TFLOPS")
            print(f"Pip (Official):  {tflops_pip:.2f} TFLOPS")
            perf_diff = tflops_local - tflops_pip
            perf_diff_pct = (perf_diff / tflops_pip) * 100 if tflops_pip != 0 else 0
            print(f"Difference:      {perf_diff:+.2f} TFLOPS ({perf_diff_pct:+.2f}%)")

        except Exception as e:
            print(f"Pip comparison skipped: {e}")

    # Save results
    results_file = "/workspace/dump/fa4_benchmark_results.txt"
    with open(results_file, "w") as f:
        f.write(f"FA4 sm100 Benchmark - Local vs PyTorch Reference\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(
            f"Config: batch={batch_size}, heads={nheads}, seq_len={seqlen_q}, head_dim={head_dim}, causal={causal}\n"
        )
        f.write(f"\n")
        f.write(f"=== Performance ===\n")
        if not use_trace:
            f.write(f"Local (Mounted): {tflops_local:.2f} TFLOPS\n")
        f.write(f"\n")
        f.write(f"=== Output Sample Values ===\n")
        f.write(f"Local output[0,0,0,:5]: {o_local[0, 0, 0, :5].tolist()}\n")
        f.write(f"Reference output[0,0,0,:5]:  {o_ref[0, 0, 0, :5].tolist()}\n")
        if lse_local is not None:
            f.write(f"Local LSE[0,0,:5]: {lse_local[0, 0, :5].tolist()}\n")
        else:
            f.write(f"Local LSE: None\n")
        if lse_ref is not None:
            f.write(f"Reference LSE[0,0,:5]:  {lse_ref[0, 0, :5].tolist()}\n")
        f.write(f"\n")
        f.write(f"=== Output Difference (vs Reference) ===\n")
        f.write(f"Max absolute diff: {abs_diff.max().item():.6e}\n")
        f.write(f"Mean absolute diff: {abs_diff.mean().item():.6e}\n")
        f.write(f"Max relative diff: {rel_diff.max().item():.6e}\n")
        f.write(f"All close (atol={atol}, rtol={rtol}): {is_close}\n")
        f.write(f"Local has NaN: {torch.isnan(o_local).any().item()}\n")
        f.write(f"Local has Inf: {torch.isinf(o_local).any().item()}\n")
        f.write(f"Reference has NaN: {torch.isnan(o_ref).any().item()}\n")
        f.write(f"Reference has Inf: {torch.isinf(o_ref).any().item()}\n")
        f.write(f"\n")
        f.write(f"=== Percentage Not Close ===\n")
        for tol in tolerance_levels:
            not_close = (abs_diff > tol).sum().item()
            pct_not_close = 100.0 * not_close / total_elements
            f.write(
                f"  > {tol:.0e}: {not_close:,} elements ({pct_not_close:.2f}% not close)\n"
            )
        f.write(f"\n")
        if seq_len >= 4:
            f.write(f"=== Position-Based Error ===\n")
            f.write(f"Top-Left quadrant mean: {top_left.mean().item():.6e}\n")
            f.write(f"Top-Right quadrant mean: {top_right.mean().item():.6e}\n")
            f.write(f"Bottom-Left quadrant mean: {bottom_left.mean().item():.6e}\n")
            f.write(f"Bottom-Right quadrant mean: {bottom_right.mean().item():.6e}\n")
            f.write(
                f"First half sequence avg: {seq_errors[:mid_seq].mean().item():.6e}\n"
            )
            f.write(
                f"Second half sequence avg: {seq_errors[mid_seq:].mean().item():.6e}\n"
            )
            f.write(
                f"First half head_dim avg: {dim_errors[:mid_dim].mean().item():.6e}\n"
            )
            f.write(
                f"Second half head_dim avg: {dim_errors[mid_dim:].mean().item():.6e}\n"
            )

    print(f"\nResults saved to {results_file}")

    run_cuobjdump(DUMP_DIR)

    # result = subprocess.run(
    #    ["find", "/", "-name", "cuobjdump"],
    #    capture_output=True,
    #    text=True,
    #    check=True,
    # )
    # output = result.stdout
    # print(output)
    # Generate HTML viewers for PTX files
    from teraxlang.tools import generate_htmls

    print("\nGenerating HTML viewers for PTX files...")
    #generate_htmls(DUMP_DIR, "/workspace/fa4/flash_attn_local/cute/flash_fwd_sm100.py")
    generate_htmls(DUMP_DIR, "/workspace/fa4/flash_attn_local/cute/flash_fwd_sm100_trace.py")
    print("HTML generation complete!")

    if use_trace:
        print(f"Download trace with: modal volume get {VOLUME_NAME} fa4_trace.json")
    print("Done!")
    print(f"to download and view: modal volume get {VOLUME_NAME} {dump_name}")


@app.local_entrypoint()
def main(use_simple: bool = True, use_trace: bool = True, use_iket: bool = True):
    run_fa4_benchmark.remote(use_simple=use_simple, use_trace=use_trace, use_iket=use_iket)
