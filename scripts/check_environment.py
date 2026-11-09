"""Print the environment metadata used by FlexEngine run artifacts."""
import argparse
import json

from flexengine.observability.environment import collect_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(collect_environment(args.device), indent=2, default=str))


if __name__ == "__main__":
    main()
