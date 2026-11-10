"""Backfill immutable local HF snapshot SHAs into already captured metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REVISIONS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/proposal_20261108")
    args = parser.parse_args()
    changed = 0
    for path in Path(args.input).glob("E*/**/engine-trace/run_metadata.json"):
        data = json.loads(path.read_text(encoding="utf8"))
        revision = REVISIONS[data["model"]]
        data["model_revision"] = revision
        data["config"]["model_revision"] = revision
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf8")
        changed += 1
    print(json.dumps({"updated_metadata_files": changed}, indent=2))


if __name__ == "__main__":
    main()
