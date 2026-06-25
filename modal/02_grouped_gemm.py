"""
Modal script to benchmark CUTLASS CuTeDSL grouped GEMM on Blackwell B200 GPU.
Tests the official Blackwell grouped GEMM kernel with correctness and TFLOPS measurement.
"""

RUN_TESTS = [
    "grouped_gemm",
    # "grouped_gemm_small",
]

from datetime import datetime
from modal import Image, App, Volume
import pathlib

root_dir = pathlib.Path(__file__).parent
GPU_model = "B200"

app = App(name="cutlass-grouped-gemm")

VOLUME_NAME = "cutlass-dump"
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

cutlass_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("wget", "curl", "gnupg", "git")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
    )
    .apt_install("cuda-toolkit-13-1")
    .workdir("/workspace")
)

cutlass_image = (
    cutlass_image.pip_install("torch", "pytest")
    .pip_install("nvidia-cutlass-dsl>=4.5.2")
    .pip_install("triton==3.5.1")
    .add_local_dir(
        root_dir / "blackwell",
        remote_path="/workspace/cuteDSL/blackwell",
        ignore="cutlass",
    )
)


@app.function(
    gpu=GPU_model,
    image=cutlass_image,
    timeout=600,
    volumes={"/workspace/dump": volume},
)
def run_grouped_gemm():
    import torch
    import os
    from datetime import datetime

    dump_name = "grouped_gemm" + "".join(str(datetime.now()).replace(":", ".").split())
    DUMP_DIR = "/workspace/dump/" + dump_name
    os.makedirs(DUMP_DIR, exist_ok=True)
    os.environ["CUTE_DSL_DUMP_DIR"] = DUMP_DIR
    os.environ["CUTE_DSL_KEEP_PTX"] = "1"
    os.environ["CUTE_DSL_LINEINFO"] = "1"
    os.environ["CUDA_PTXAS_FLAGS"] = "-v"
    os.environ["CUTE_LOG_LEVEL"] = "DEBUG"

    DEVICE = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"RUN_TESTS: {RUN_TESTS}")

    import cutlass
    import cutlass.utils as cutlass_utils

    warmup = 10
    repeats = 100

    # 1. grouped_gemm.py (MoE-style problem sizes)
    if "grouped_gemm" in RUN_TESTS:
        print("\n=== 1. grouped_gemm.py Benchmark ===")
        from cuteDSL.blackwell.grouped_gemm import run

        problem_sizes = (
            (8192, 1280, 32, 1),
            (16, 384, 1536, 1),
            (640, 1280, 16, 1),
            (640, 160, 16, 1),
        )
        total_flops = 2 * sum(M * N * K for (M, N, K, _) in problem_sizes)
        print(f"Problem sizes: {problem_sizes}")
        print(f"Total FLOPs per iteration: {total_flops}")

        us = run(
            num_groups=len(problem_sizes),
            problem_sizes_mnkl=problem_sizes,
            host_problem_shape_available=True,
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
            tensormap_update_mode=cutlass_utils.TensorMapUpdateMode.SMEM,
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
        )
        time_ms = us / 1000
        tflops = total_flops / time_ms / 1e9
        print(f"\n=== Summary ===")
        print(f"grouped_gemm: {time_ms:.4f} ms, {tflops:.2f} TFLOPS")

    # 2. Small correctness-only test
    if "grouped_gemm_small" in RUN_TESTS:
        print("\n=== 2. grouped_gemm.py Small Correctness Test ===")
        from cuteDSL.blackwell.grouped_gemm import run

        problem_sizes = (
            (128, 128, 128, 1),
            (256, 128, 64, 1),
        )

        us = run(
            num_groups=len(problem_sizes),
            problem_sizes_mnkl=problem_sizes,
            host_problem_shape_available=True,
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
            tensormap_update_mode=cutlass_utils.TensorMapUpdateMode.SMEM,
            tolerance=0.1,
            warmup_iterations=0,
            iterations=1,
            skip_ref_check=False,
        )
        print(f"Small test PASSED, execution time: {us / 1000:.4f} ms")

    print(f"\nDone! Results saved to: {DUMP_DIR}")


@app.local_entrypoint()
def main():
    run_grouped_gemm.remote()
