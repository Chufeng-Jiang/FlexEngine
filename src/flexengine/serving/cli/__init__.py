"""Unified command-line entry point for FlexEngine operations."""
from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

COMMANDS = {
    "serve": ("flexengine.serving.cli.server", "main", "Start the OpenAI-compatible inference server"),
    "benchmark": ("flexengine.research.benchmark.runner", "main", "Replay an HTTP benchmark workload"),
    "run-proposal": ("flexengine.research.experiments.runner", "main", "Run the Proposal experiment matrix"),
    "analyze-proposal": ("flexengine.research.analysis.results", "main", "Aggregate Proposal results"),
    "run-warm-benchmarks": ("flexengine.research.experiments.runner", "warm_main", "Run the warm benchmark matrix"),
    "analyze-warm-benchmarks": ("flexengine.research.analysis.results", "warm_main", "Aggregate warm benchmark results"),
    "check-env": ("flexengine.operations.tooling.environment", "check_main", "Validate the CUDA environment"),
    "export-env": ("flexengine.operations.tooling.environment", "export_main", "Export reproducible environments"),
}


def usage() -> str:
    lines = ["usage: flexengine <command> [options]", "", "commands:"]
    width = max(map(len, COMMANDS))
    lines.extend(f"  {name:<{width}}  {description}" for name, (_, _, description) in COMMANDS.items())
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(usage())
        return
    command = arguments.pop(0)
    if command not in COMMANDS:
        raise SystemExit(f"unknown command: {command}\n\n{usage()}")
    module_name, function_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)
    sys.argv = [f"flexengine {command}", *arguments]
    getattr(module, function_name)()


if __name__ == "__main__":
    main()
