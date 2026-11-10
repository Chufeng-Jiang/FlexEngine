import torch

from flexengine.operations.observability.environment import collect_environment, validate_run_metadata


def test_collect_environment_does_not_probe_cuda_for_cpu(monkeypatch):
    def fail_probe():
        raise AssertionError("CPU metadata collection must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", fail_probe)

    data = collect_environment(device="cpu")

    assert data["torch"] == torch.__version__
    assert data["configured_device"] == "cpu"
    assert data["gpu"] is None
    assert data["gpu_total_bytes"] is None


def test_validate_run_metadata_reports_reproduction_fields():
    data = {
        "timestamp_utc": "2026-11-08T12:00:00+00:00",
        "torch": "2.15.0+cu132",
        "cuda": "13.2",
        "transformers": "5.22.2",
        "driver": "610.60",
        "gpu": "NVIDIA GeForce RTX 3080 Ti Laptop GPU",
        "gpu_total_bytes": 17179869184,
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "revision-sha",
        "trace_identity": "w1-materialized-c2-r1",
        "config": {
            "dtype": "float16",
            "scheduler_policy": "decode_first",
            "attention_backend": "materialized",
            "random_seed": 42,
        },
    }

    assert validate_run_metadata(data) == []

    del data["trace_identity"]
    del data["config"]["random_seed"]
    assert validate_run_metadata(data) == ["trace_identity", "config.random_seed"]
