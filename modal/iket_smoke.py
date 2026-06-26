from modal import App, Image, Volume
import pathlib

root_dir = pathlib.Path(__file__).parent

app = App(name="iket-smoke")

VOLUME_NAME = "iket-test"
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("wget", "curl", "gnupg")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
    )
    .apt_install("cuda-toolkit-13-2")
    .workdir("/workspace")
    .pip_install("torch")
    .pip_install("nvidia-cutlass-dsl==4.6.0.dev0")
    .pip_install("apache-tvm-ffi>=0.1.5,<0.2")
    .pip_install("teraxlang==3.5.1.dev4")
    .add_local_dir(root_dir, remote_path="/workspace/modal")
)


@app.function(
    gpu="B200",
    image=image,
    timeout=600,
    volumes={"/workspace/dump": volume},
)
def run_iket_smoke():
    import os
    import subprocess
    from datetime import datetime
    from pathlib import Path

    dump_name = "iket_smoke_" + "".join(str(datetime.now()).replace(":", ".").split())
    dump_dir = Path("/workspace/dump") / dump_name
    dump_dir.mkdir(parents=True, exist_ok=True)

    iket_output_dir = dump_dir / "iket"
    iket_log_path = dump_dir / "iket_profile.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace"
    cmd = [
        "python",
        "-m",
        "iket.cli.main",
        "--log-level",
        "info",
        "--output-dir",
        str(iket_output_dir),
        "--clobber",
        "profile",
        "--postprocess",
        "perfetto",
        "--",
        "env",
        "PYTHONPATH=/workspace",
        "python",
        "/workspace/modal/iket_smoke_runner.py",
        "--dump-dir",
        str(dump_dir),
    ]

    print("Running IKET smoke command:")
    print(" ".join(cmd))
    print(f"Saving IKET log to: {iket_log_path}")
    with open(iket_log_path, "w") as iket_log:
        subprocess.run(
            cmd,
            check=True,
            env=env,
            stdout=iket_log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    print(f"IKET output directory: {iket_output_dir}")
    print(f"IKET log path: {iket_log_path}")
    print(f"to download and view: modal volume get {VOLUME_NAME} {dump_name}")


@app.local_entrypoint()
def main():
    run_iket_smoke.remote()
