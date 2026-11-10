"""Shared process and artifact utilities for reproducible experiments."""
from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
HEALTH_URL = "http://127.0.0.1:8000/healthz"


def wait_ready(
    process: subprocess.Popen,
    *,
    timeout: float,
    health_url: str = HEALTH_URL,
) -> None:
    """Wait until the child server reports healthy or fails."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode}")
        try:
            response = httpx.get(health_url, timeout=1.0)
            if response.status_code == 200 and response.json().get("status") == "ok":
                return
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"server did not become ready within {timeout:g}s")


def stop_process(process: subprocess.Popen, *, timeout: float = 20.0) -> None:
    """Terminate a child process and escalate to kill if it does not exit."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@contextmanager
def running_server(
    command: Sequence[str],
    *,
    log_path: Path,
    timeout: float,
    env: dict[str, str] | None = None,
    cwd: Path = ROOT,
) -> Iterator[subprocess.Popen]:
    """Run a benchmark server with consistent logging, readiness and cleanup."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env or os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(process, timeout=timeout)
            yield process
        finally:
            stop_process(process)


def runs_complete(destination: Path, repetitions: int) -> bool:
    """Return whether every expected repetition has a summary artifact."""
    return all(
        (destination / f"run-{index}" / "summary.json").exists()
        for index in range(1, repetitions + 1)
    )


def add_workload_trace(
    command: list[str],
    *,
    trace_path: Path,
) -> list[str]:
    """Attach a saved trace or request creation of a deterministic new trace."""
    if trace_path.exists():
        return [*command, "--workload-trace", str(trace_path)]
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    return [*command, "--save-workload-trace", str(trace_path)]
