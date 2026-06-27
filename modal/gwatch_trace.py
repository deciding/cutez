from modal import App, Image, Volume
import pathlib

root_dir = pathlib.Path(__file__).parent

app = App(name="gwatch-trace")

VOLUME_NAME = "gwatch-dump"
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("wget", "curl", "gnupg")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
    )
    .apt_install("cuda-toolkit-12-9")
    .workdir("/workspace")
    .pip_install("torch")
    .pip_install("nvidia-cutlass-dsl==4.6.0.dev0")
    .pip_install("apache-tvm-ffi>=0.1.5,<0.2")
    .pip_install("gwatch")
    .add_local_dir(root_dir, remote_path="/workspace/modal")
)


@app.function(
    gpu="B200",
    image=image,
    timeout=600,
    volumes={"/workspace/dump": volume},
)
def run_gwatch_trace():
    import os
    import subprocess
    from datetime import datetime
    from pathlib import Path

    dump_name = "gwatch_" + "".join(str(datetime.now()).replace(":", ".").split())
    dump_dir = Path("/workspace/dump") / dump_name
    dump_dir.mkdir(parents=True, exist_ok=True)

    report_path = dump_dir / "gwatch_cute_matmul.json"
    log_path = dump_dir / "gwatch_trace.log"

    env = os.environ.copy()
    env["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
    env["CUTE_DSL_KEEP_PTX"] = "1"

    cmd = [
        "python",
        "/workspace/modal/tmp/gwatch_cute_matmul.py",
        "--report",
        str(report_path),
    ]

    print("Running G-Watch trace command:")
    print(" ".join(cmd))
    print(f"Dump directory: {dump_dir}")
    print(f"JSON report path: {report_path}")
    print(f"Log path: {log_path}")
    with open(log_path, "w") as log_file:
        subprocess.run(
            cmd,
            check=True,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    print(f"All dump files are under: {dump_dir}")
    print(f"to download and view: modal volume get {VOLUME_NAME} {dump_name}")


@app.local_entrypoint()
def main():
    run_gwatch_trace.remote()
