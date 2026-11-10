"""Deterministic Proposal experiment matrix and resumable run markers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    name: str
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    workload: str = "w1"
    concurrency: int = 4
    policy: str = "decode_first"
    attention: str = "materialized"
    token_budget: int = 128
    block_size: int = 16
    prefix_cache: bool = False
    prefix_ratio: int = 0
    cuda_graph: bool = False


def _name_value(key: str, value) -> str:
    if key == "model":
        return "1.5b" if "1.5B" in value else "0.5b"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value).replace("_", "-")


def load_matrix(path: Path) -> list[RunSpec]:
    document = json.loads(path.read_text(encoding="utf8"))
    specs = []
    for experiment, definition in document["experiments"].items():
        fixed = dict(definition.get("fixed", {}))
        dimensions = definition.get("matrix", {})
        keys = list(dimensions)
        combinations = product(*(dimensions[key] for key in keys)) if keys else [()]
        for values in combinations:
            item = {**fixed, **dict(zip(keys, values))}
            name_values = {key: _name_value(key, value) for key, value in item.items()}
            item["experiment"] = experiment
            item["name"] = definition["name"].format(**name_values)
            specs.append(RunSpec(**item))
    return specs


def run_id(spec: RunSpec, repeat: int) -> str:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    return f"{spec.name}-r{repeat}"


def mark_complete(run_dir: Path, identity: str, summary: dict) -> None:
    payload = {"run_id": identity, "spec": summary.pop("spec", None), **summary}
    temporary = run_dir / "COMPLETE.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf8")
    temporary.replace(run_dir / "COMPLETE.json")


def is_complete(run_dir: Path, identity: str) -> bool:
    marker = run_dir / "COMPLETE.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf8")).get("run_id") == identity
    except (OSError, json.JSONDecodeError):
        return False


def spec_dict(spec: RunSpec) -> dict:
    return asdict(spec)
