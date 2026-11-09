"""Stream a request, then demonstrate explicit cancellation endpoint."""
import argparse
import json

import httpx


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--prompt", default="Explain continuous batching in two sentences.")
    p.add_argument("--max-tokens", type=int, default=48)
    args=p.parse_args()
    with httpx.Client(timeout=180.0) as c:
        print("Health", c.get(args.base_url+"/healthz").json())
        with c.stream("POST",args.base_url+"/v1/completions",
                      json={"model":args.model,"prompt":args.prompt,"max_tokens":args.max_tokens,"stream":True}) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: "):
                    raw=line[6:]
                    if raw=="[DONE]":
                        break
                    data=json.loads(raw)
                    if "choices" in data:
                        print(data["choices"][0]["text"],end="",flush=True)
                    else:
                        print("\nERROR",data)
        print("\n")


if __name__ == "__main__":
    main()
