"""
Modal script to benchmark CUTLASS CuTeDSL dense GEMM on Blackwell B200 GPU.
Tests torch.matmul and dense_gemm with same settings.
"""

# Which tests to run - modify this list to choose which tests to execute
# Available options: "torch", "dense_gemm", "dense_gemm_1", "dense_gemm_2", "dense_gemm_3", "dense_gemm_4", "dense_gemm_5", "dense_gemm_6", "dense_gemm_7", "dense_gemm_7min"
RUN_TESTS = [
    # "torch",
    # "dense_gemm",
    # "dense_gemm_1",
    # "dense_gemm_2",
    # "dense_gemm_3",
    # "dense_gemm_4",
    # "dense_gemm_5",
    # "dense_gemm_6",
    # "dense_gemm_7",
    "dense_gemm_7min",
    # "dense_gemm_8_tracer",
    # "persistent",
    # "prefetch",
    # "software_pipeline",
    # "2sm",
    # "cute_pipeline",
]

# Input initialization mode for benchmark tensors.
# Options: "randint" or "gaussian"
INIT_MODE = "randint"
NORMAL_MEAN = 0.0
NORMAL_STD = 1.0


from datetime import datetime
from modal import Image, App, Volume
import pathlib

root_dir = pathlib.Path(__file__).parent
GPU_model = "B200"

app = App(name="cutlass-dense-gemm")

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
    # .apt_install("cuda-toolkit-12-6")
    .apt_install("cuda-toolkit-13-1")
    .workdir("/workspace")
)

cutlass_image = (
    cutlass_image.pip_install("torch", "pytest")
    .pip_install("quack-kernels==0.5.0")
    .pip_install("nvidia-cutlass-dsl>=4.4.1")
    ## cutez 2: local build
    # .add_local_dir(
    #    root_dir.parent / "dist",
    #    remote_path="/workspace/dist",
    #    copy=True,
    # )
    # .run_commands("python -m pip install /workspace/dist/cutez-0.1.1-py3-none-any.whl")
    ## cutez 3: pypi
    # .pip_install("cutez==0.1.0")
    .pip_install("triton==3.5.1")
    .pip_install("teraxlang==3.5.1.dev4")
    .add_local_dir(
        root_dir / "blackwell",
        remote_path="/workspace/cuteDSL/blackwell",
        ignore="cutlass",
    )
    # cutez 1: local dir
    .add_local_dir(
        root_dir.parent / "cutez",
        remote_path="/workspace/cutez",
        # copy=True,
    )
)


@app.function(
    gpu=GPU_model,
    image=cutlass_image,
    timeout=600,
    volumes={"/workspace/dump": volume},
)
def run_dense_gemm():
    import torch
    import os
    import subprocess
    from datetime import datetime

    dump_name = "dense_gemm" + "".join(str(datetime.now()).replace(":", ".").split())
    DUMP_DIR = "/workspace/dump/" + dump_name
    os.makedirs(DUMP_DIR, exist_ok=True)
    os.environ["CUTE_DSL_DUMP_DIR"] = DUMP_DIR
    os.environ["CUTE_DSL_KEEP_PTX"] = "1"
    os.environ["CUTE_DSL_LINEINFO"] = "1"
    os.environ["CUDA_PTXAS_FLAGS"] = "-v"
    os.environ["CUTE_LOG_LEVEL"] = "DEBUG"

    DEVICE = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    import cutlass

    blackwell_dst = "/workspace/cuteDSL/blackwell"

    M = 8192
    N = 8192
    K = 4096
    warmup = 10
    repeats = 100

    print(f"\n=== Benchmark: {M}x{N}x{K} ===")
    print(f"Warmup: {warmup}, Iterations: {repeats}")
    print(f"RUN_TESTS: {RUN_TESTS}")
    print(f"INIT_MODE: {INIT_MODE}")
    if INIT_MODE == "gaussian":
        print(f"NORMAL_MEAN/NORMAL_STD: {NORMAL_MEAN}/{NORMAL_STD}")

    import torch.utils.benchmark as benchmark

    flops = 2.0 * M * N * K

    # 1. torch.matmul (uses cuBLAS under the hood)
    if "torch" in RUN_TESTS:
        print("\n=== 1. torch.matmul Benchmark ===")
        torch.manual_seed(1111)
        if INIT_MODE == "randint":
            a = torch.empty(M, K, dtype=torch.float16).random_(-2, 3).to("cuda")
            b = torch.empty(N, K, dtype=torch.float16).random_(-2, 3).to("cuda")
        elif INIT_MODE == "gaussian":
            a = (
                torch.empty(M, K, dtype=torch.float16)
                .normal_(NORMAL_MEAN, NORMAL_STD)
                .to("cuda")
            )
            b = (
                torch.empty(N, K, dtype=torch.float16)
                .normal_(NORMAL_MEAN, NORMAL_STD)
                .to("cuda")
            )
        else:
            raise ValueError(f"Unsupported INIT_MODE: {INIT_MODE}")

        timer = benchmark.Timer(
            stmt="torch.matmul(a, b.T)",
            globals={"a": a, "b": b},
        )
        result = timer.blocked_autorange(min_run_time=1.0)
        avg_time_ms = result.mean * 1e3
        tflops = flops / (avg_time_ms * 1e-3) / 1e12

        print(f"torch.matmul: {avg_time_ms:.4f} ms, {tflops:.2f} TFLOPS")
        print(f"  (median: {result.median * 1e3:.4f} ms)")

    # 2. dense_gemm.py
    if "dense_gemm" in RUN_TESTS:
        print("\n=== 2. dense_gemm.py Benchmark ===")

        TERMINAL = False
        if TERMINAL:
            result = subprocess.run(
                [
                    "python",
                    "/workspace/cuteDSL/blackwell/dense_gemm.py",
                    "--mnkl",
                    f"{M},{N},{K},1",
                    "--ab_dtype",
                    "Float16",
                    "--acc_dtype",
                    "Float32",
                    "--c_dtype",
                    "Float16",
                    "--mma_tiler_mn",
                    "256,256",
                    "--cluster_shape_mn",
                    "2,1",
                    "--use_2cta_instrs",
                    "--use_tma_store",
                    "--warmup_iterations",
                    str(warmup),
                    "--iterations",
                    str(repeats),
                    "--skip_ref_check",
                    "--init_mode",
                    INIT_MODE,
                    "--normal_mean",
                    str(NORMAL_MEAN),
                    "--normal_std",
                    str(NORMAL_STD),
                ],
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

        else:
            from cuteDSL.blackwell.dense_gemm import run

            us = run(
                (M, N, K, 1),
                ab_dtype=cutlass.Float16,
                c_dtype=cutlass.Float16,
                acc_dtype=cutlass.Float32,
                a_major="k",
                b_major="k",
                c_major="n",
                mma_tiler_mn=(256, 256),
                cluster_shape_mn=(2, 1),
                use_2cta_instrs=True,
                use_tma_store=True,
                tolerance=0.1,
                warmup_iterations=warmup,
                iterations=repeats,
                skip_ref_check=False,
                use_cold_l2=False,
                init_mode=INIT_MODE,
                normal_mean=NORMAL_MEAN,
                normal_std=NORMAL_STD,
            )
            time_ms = us / 1000
            tflops = flops / time_ms / 1e9
            print(f"dense_gemm: {time_ms:.4f} ms, {tflops:.2f} TFLOPS")

    # 3. dense_gemm_1.py (low-level mbarrier API)
    if "dense_gemm_1" in RUN_TESTS:
        print("\n=== 3. dense_gemm_1.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_1 import run_dense_gemm as run_dense_gemm_1

        us = run_dense_gemm_1(
            (M, N, K),
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            init_mode=INIT_MODE,
            normal_mean=NORMAL_MEAN,
            normal_std=NORMAL_STD,
        )
        time_ms = us / 1000
        tflops1 = flops / time_ms / 1e9
        print(f"dense_gemm_1: {time_ms:.4f} ms, {tflops1:.2f} TFLOPS")

    # 4. dense_gemm_2.py (Pipeline API)
    if "dense_gemm_2" in RUN_TESTS:
        print("\n=== 4. dense_gemm_2.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_2 import run_dense_gemm as run_dense_gemm_2

        us = run_dense_gemm_2(
            (M, N, K),
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            init_mode=INIT_MODE,
            normal_mean=NORMAL_MEAN,
            normal_std=NORMAL_STD,
        )
        time_ms = us / 1000
        tflops2 = flops / time_ms / 1e9
        print(f"dense_gemm_2: {time_ms:.4f} ms, {tflops2:.2f} TFLOPS")

    # 5. dense_gemm_3.py (with cluster support)
    if "dense_gemm_3" in RUN_TESTS:
        print("\n=== 5. dense_gemm_3.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_3 import run_dense_gemm as run_dense_gemm_3

        us = run_dense_gemm_3(
            (M, N, K),
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            init_mode=INIT_MODE,
            normal_mean=NORMAL_MEAN,
            normal_std=NORMAL_STD,
        )
        time_ms = us / 1000
        tflops3 = flops / time_ms / 1e9
        print(f"dense_gemm_3: {time_ms:.4f} ms, {tflops3:.2f} TFLOPS")

    # 6. dense_gemm_4.py (with pair-UMMA / CtaGroup.TWO)
    if "dense_gemm_4" in RUN_TESTS:
        print("\n=== 6. dense_gemm_4.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_4 import run_dense_gemm as run_dense_gemm_4

        us = run_dense_gemm_4(
            (M, N, K),
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            init_mode=INIT_MODE,
            normal_mean=NORMAL_MEAN,
            normal_std=NORMAL_STD,
        )
        time_ms = us / 1000
        tflops4 = flops / time_ms / 1e9
        print(f"dense_gemm_4: {time_ms:.4f} ms, {tflops4:.2f} TFLOPS")

    # 7. dense_gemm_5.py (with pair-UMMA + TMA Store)
    if "dense_gemm_5" in RUN_TESTS:
        print("\n=== 7. dense_gemm_5.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_5 import run_dense_gemm as run_dense_gemm_5

        us = run_dense_gemm_5(
            (M, N, K),
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            init_mode=INIT_MODE,
            normal_mean=NORMAL_MEAN,
            normal_std=NORMAL_STD,
        )
        time_ms = us / 1000
        tflops5 = flops / time_ms / 1e9
        print(f"dense_gemm_5: {time_ms:.4f} ms, {tflops5:.2f} TFLOPS")

        # Generate HTML viewers for IR files
        print(f"\n=== Generating HTML viewers for IR files ===")
        print(f"IR dump directory: {DUMP_DIR}")

        # Python source file is mounted at /workspace/tutorials/vector_add.py
        py_file = "/workspace/cuteDSL/blackwell/dense_gemm_5.py"

        # Generate HTML viewers for all IR files in the dump directory
        from teraxlang.tools.build_binding_view import generate_htmls

        generate_htmls(DUMP_DIR, py_file, verbose=True)

        print(f"\nHTML viewers generated!")
        print(f"to download and view: modal volume get {VOLUME_NAME} {dump_name}")

    # 8. dense_gemm_persistent.py
    if "persistent" in RUN_TESTS:
        print("\n=== 8. dense_gemm_persistent.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_persistent import run as run_persistent

        us = run_persistent(
            (M, N, K, 1),
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
            use_tma_store=True,
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            use_cold_l2=False,
            benchmark=True,
        )
        time_ms = us / 1000
        tflops = flops / time_ms / 1e9
        print(f"persistent: {time_ms:.4f} ms, {tflops:.2f} TFLOPS")

    # 9. dense_gemm_persistent_prefetch.py
    if "prefetch" in RUN_TESTS:
        print("\n=== 9. dense_gemm_persistent_prefetch.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_persistent_prefetch import run as run_prefetch

        us = run_prefetch(
            (M, N, K, 1),
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
            use_tma_store=True,
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            use_cold_l2=False,
            benchmark=True,
        )
        time_ms = us / 1000
        tflops = flops / time_ms / 1e9
        print(f"prefetch: {time_ms:.4f} ms, {tflops:.2f} TFLOPS")

    # 10. dense_gemm_software_pipeline.py
    if "software_pipeline" in RUN_TESTS:
        print("\n=== 10. dense_gemm_software_pipeline.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_software_pipeline import run as run_swp

        us = run_swp(
            (M, N, K, 1),
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
            use_tma_store=True,
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            use_cold_l2=False,
            benchmark=True,
        )
        time_ms = us / 1000
        tflops = flops / time_ms / 1e9
        print(f"software_pipeline: {time_ms:.4f} ms, {tflops:.2f} TFLOPS")

    # 11. dense_gemm_2sm.py
    if "2sm" in RUN_TESTS:
        print("\n=== 11. dense_gemm_2sm.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_2sm import run as run_2sm

        us = run_2sm(
            (M, N, K, 1),
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
            use_tma_store=True,
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            use_cold_l2=False,
            benchmark=True,
        )
        time_ms = us / 1000
        tflops = flops / time_ms / 1e9
        print(f"2sm: {time_ms:.4f} ms, {tflops:.2f} TFLOPS")

    # 12. dense_gemm_cute_pipeline.py
    if "cute_pipeline" in RUN_TESTS:
        print("\n=== 12. dense_gemm_cute_pipeline.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_cute_pipeline import run as run_cute_p

        us = run_cute_p(
            (M, N, K, 1),
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="k",
            c_major="n",
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            use_2cta_instrs=True,
            use_tma_store=True,
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            use_cold_l2=False,
            benchmark=True,
        )
        time_ms = us / 1000
        tflops = flops / time_ms / 1e9
        print(f"cute_pipeline: {time_ms:.4f} ms, {tflops:.2f} TFLOPS")

    # 13. dense_gemm_6.py (dense_gemm_5 + Persistent)
    if "dense_gemm_6" in RUN_TESTS:
        print("\n=== 13. dense_gemm_6.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_6 import run_dense_gemm as run_dense_gemm_6

        us = run_dense_gemm_6(
            (M, N, K),
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            init_mode=INIT_MODE,
            normal_mean=NORMAL_MEAN,
            normal_std=NORMAL_STD,
        )
        time_ms = us / 1000
        tflops6 = flops / time_ms / 1e9
        print(f"dense_gemm_6: {time_ms:.4f} ms, {tflops6:.2f} TFLOPS")

    # 14. dense_gemm_7.py
    if "dense_gemm_7" in RUN_TESTS:
        print("\n=== 14. dense_gemm_7.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_7 import run_dense_gemm as run_dense_gemm_7

        us = run_dense_gemm_7(
            (M, N, K),
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            init_mode=INIT_MODE,
            normal_mean=NORMAL_MEAN,
            normal_std=NORMAL_STD,
        )
        time_ms = us / 1000
        tflops7 = flops / time_ms / 1e9
        print(f"dense_gemm_7: {time_ms:.4f} ms, {tflops7:.2f} TFLOPS")

    # 15. dense_gemm_7min.py
    if "dense_gemm_7min" in RUN_TESTS:
        print("\n=== 15. dense_gemm_7min.py Benchmark ===")
        from cuteDSL.blackwell.dense_gemm_7min import (
            run_dense_gemm as run_dense_gemm_7min,
        )

        us = run_dense_gemm_7min(
            (M, N, K),
            tolerance=0.1,
            warmup_iterations=warmup,
            iterations=repeats,
            skip_ref_check=False,
            init_mode=INIT_MODE,
            normal_mean=NORMAL_MEAN,
            normal_std=NORMAL_STD,
        )
        time_ms = us / 1000
        tflops7min = flops / time_ms / 1e9
        print(f"dense_gemm_7min: {time_ms:.4f} ms, {tflops7min:.2f} TFLOPS")
        # Generate HTML viewers for all IR files in the dump directory
        from teraxlang.tools.build_binding_view import generate_htmls

        py_file = "/workspace/cuteDSL/blackwell/dense_gemm_7min.py"
        generate_htmls(DUMP_DIR, py_file, verbose=True)

        print(f"\nHTML viewers generated!")
        print(f"to download and view: modal volume get {VOLUME_NAME} {dump_name}")

    # 16. dense_gemm_8_tracer.py (trace test)
    if "dense_gemm_8_tracer" in RUN_TESTS:
        print("\n=== 16. dense_gemm_8_tracer.py Trace Test ===")
        from cuteDSL.blackwell.dense_gemm_8_tracer import (
            run_dense_gemm,
        )

        os.environ["QUACK_TRACE"] = "0"

        run_dense_gemm(
            (M, N, K),
            tolerance=0.1,
            warmup_iterations=1,
            iterations=1,
            skip_ref_check=False,
            init_mode=INIT_MODE,
            normal_mean=NORMAL_MEAN,
            normal_std=NORMAL_STD,
            trace_path=os.path.join(DUMP_DIR, "trace_dense_gemm_8.json"),
            # trace_path=os.path.join(DUMP_DIR, "gmem_trace_dense_gemm_8.json"),
            quack_trace_path=os.path.join(DUMP_DIR, "quack_trace_dense_gemm_8.json"),
        )
        print(
            f"Download trace with: modal volume get {VOLUME_NAME} {dump_name}/trace_dense_gemm_8.json"
        )

    print(f"\nDone! Results saved to: {DUMP_DIR}")


@app.local_entrypoint()
def main():
    run_dense_gemm.remote()
