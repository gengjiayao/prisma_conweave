"""Run a simulator without hiding failures or rebuilding between paired trials."""

import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import secrets


def create_run_directory(output_root):
    """Atomically reserve a numeric output ID for concurrent simulators."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        path = root / str(secrets.randbelow(1000000000))
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def binary_identity(binary):
    if not binary:
        return None
    path = Path(binary).resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("Simulator is not an executable file: {}".format(path))
    modules = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(path.parent.parent.glob('libns3*.so'))}
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ns3_shared_modules": modules}


def execute_simulator(config_path, output_log, binary=None, cwd=None):
    cwd = Path(cwd or os.getcwd()).resolve()
    env = os.environ.copy()
    if binary:
        identity = binary_identity(binary)
        executable = Path(identity["path"])
        command = [str(executable), str(Path(config_path).resolve())]
        # ns-3 links its modules from the build directory beside scratch/.
        build_dir = str(executable.parent.parent)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            part for part in [build_dir, env.get("LD_LIBRARY_PATH")] if part
        )
    else:
        command = ["python2", "./waf", "--run",
                   "scratch/network-load-balance {}".format(shlex.quote(str(config_path)))]
    print("Simulator command: {}".format(shlex.join(command)), flush=True)
    with open(output_log, "w") as log:
        subprocess.run(command, cwd=str(cwd), env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return command
