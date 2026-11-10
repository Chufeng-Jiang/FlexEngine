"""Reproduction metadata collection without unintended CUDA initialization."""
from __future__ import annotations

import platform
import subprocess

import torch


def _command_output(command: list[str]) -> str | None:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def collect_environment(device: str = "cuda:0") -> dict:
    use_cuda = str(device).startswith("cuda")
    data = {
        "system": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "configured_device": str(device),
        "gpu": None,
        "gpu_total_bytes": None,
        "driver": None,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
    }
    try:
        import transformers

        data["transformers"] = transformers.__version__
    except ImportError:
        data["transformers"] = None
    if use_cuda and torch.cuda.is_available():
        data["gpu"] = torch.cuda.get_device_name(0)
        data["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        output = _command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"]
        )
        data["driver"] = output.splitlines()[0].strip() if output else None
    return data


def validate_run_metadata(data: dict) -> list[str]:
    required = (
        "timestamp_utc",
        "torch",
        "cuda",
        "transformers",
        "driver",
        "gpu",
        "gpu_total_bytes",
        "model",
        "model_revision",
        "trace_identity",
        "config.dtype",
        "config.scheduler_policy",
        "config.attention_backend",
        "config.random_seed",
    )
    missing = []
    for path in required:
        value = data
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is None or value == "":
            missing.append(path)
    return missing
