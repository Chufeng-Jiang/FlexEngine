"""Export both reproducible Python environments and host/GPU metadata."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from flexengine.observability.environment import collect_environment

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "data" / "environment"


def command(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def freeze(python: Path, destination: Path) -> int:
    text = command(str(python), "-m", "pip", "freeze")
    destination.write_text(text + "\n", encoding="utf8")
    return len(text.splitlines())
def check_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(collect_environment(args.device), indent=2, default=str))




def export_main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    main_count = freeze(ROOT / ".venv" / "bin" / "python", OUTPUT / "requirements-flexengine.txt")
    vllm_count = freeze(ROOT / ".venv-vllm" / "bin" / "python", OUTPUT / "requirements-vllm.txt")
    torch_data = json.loads(
        command(
            str(ROOT / ".venv" / "bin" / "python"),
            "-c",
            "import json,torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}))",
        )
    )
    gpu = command(
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    )
    data = {
        "exported_date": "2026-11-09",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu,
        **torch_data,
        "triton": command(str(ROOT / ".venv" / "bin" / "python"), "-c", "import triton; print(triton.__version__)"),
        "transformers": command(str(ROOT / ".venv" / "bin" / "python"), "-c", "import transformers; print(transformers.__version__)"),
        "vllm": command(str(ROOT / ".venv-vllm" / "bin" / "python"), "-c", "import vllm; print(vllm.__version__)"),
        "package_counts": {"flexengine": main_count, "vllm": vllm_count},
        "model_revisions": {
            "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
            "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        },
    }
    (OUTPUT / "system.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    export_main()
