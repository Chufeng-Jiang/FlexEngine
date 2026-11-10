"""JSONL event recording and NVTX tagging (no-op outside CUDA)."""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

import torch

from flexengine.observability.environment import collect_environment


@contextmanager
def nvtx(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


class TraceWriter:
    def __init__(self, directory: str):
        self.directory = directory
        self.lock = threading.Lock()
        if directory:
            os.makedirs(directory, exist_ok=True)

    def write(self, name: str, item: dict) -> None:
        if not self.directory:
            return
        with self.lock, open(os.path.join(self.directory, f"{name}.jsonl"), "a", encoding="utf8") as f:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    def metadata(self, config, model=None) -> None:
        if not self.directory:
            return
        data = collect_environment(config.device)
        data.update({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": vars(config),
            "model": str(model) if model is not None else config.model,
            "model_revision": config.model_revision,
            "trace_identity": os.getenv("FLEX_TRACE_IDENTITY")
            or os.path.basename(os.path.abspath(self.directory)),
        })
        with open(os.path.join(self.directory, "run_metadata.json"), "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
