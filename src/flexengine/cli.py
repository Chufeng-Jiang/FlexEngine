"""Unified command-line entry point for FlexEngine operations."""
from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

COMMANDS = {
    "serve": ("scripts.run_server", "Start the OpenAI-compatible inference server"),
    "smoke": ("scripts.smoke_client", "Run the service smoke client"),
    "benchmark": ("benchmarks.replay", "Replay an HTTP benchmark workload"),
    "run-proposal": ("experiments.run_proposal", "Run the Proposal experiment matrix"),
    "analyze-proposal": ("experiments.analyze_proposal", "Aggregate Proposal results"),
    "run-upgrade": ("experiments.run_upgrade", "Run the warm upgrade matrix"),
    "analyze-upgrade": ("experiments.analyze_upgrade", "Aggregate warm upgrade results"),
    "check-env": ("scripts.check_environment", "Validate the CUDA environment"),
    "export-env": ("scripts.export_environment", "Export reproducible environments"),
}


def usage() -> str:
    lines = ["usage: flexengine <command> [options]", "", "commands:"]
    width = max(map(len, COMMANDS))
    lines.extend(f"  {name:<{width}}  {description}" for name, (_, description) in COMMANDS.items())
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(usage())
        return
    command = arguments.pop(0)
    if command not in COMMANDS:
        raise SystemExit(f"unknown command: {command}\n\n{usage()}")
    module_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)
    sys.argv = [f"flexengine {command}", *arguments]
    module.main()


if __name__ == "__main__":
    main()
