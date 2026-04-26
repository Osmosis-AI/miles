"""Test: csgmv vs triton end-to-end log prob comparison.

Launches SGLang twice (once per backend), sends the same prompts with
the same LoRA adapters, and compares the returned log probs.

Usage:
    python examples/multi_lora/test_csgmv_vs_triton_e2e.py \
        --model /root/Qwen3-4B \
        --lora-dir examples/multi_lora/adapters \
        --port-csgmv 30000 --port-triton 30001
"""

import argparse
import json
import time
from pathlib import Path

import requests


def launch_server(model, lora_paths, port, backend, extra_args=None):
    """Launch SGLang server and wait for it to be ready."""
    import subprocess

    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--lora-paths", *lora_paths,
        "--max-loras-per-batch", str(len(lora_paths)),
        "--lora-backend", backend,
        "--disable-radix-cache",
        "--mem-fraction-static", "0.45",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Launching {backend} server on port {port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for _ in range(120):
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                print(f"  {backend} server ready")
                return proc
        except Exception:
            pass
        time.sleep(2)

    proc.kill()
    raise TimeoutError(f"{backend} server didn't start in 240s")


def get_logprobs(port, prompt, lora_name, max_tokens=1):
    """Get log probs from SGLang server."""
    resp = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={
            "model": lora_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "logprobs": 1,
            "echo": True,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["logprobs"]["token_logprobs"]


def compare_backends(port_a, port_b, prompt, lora_name):
    """Compare log probs between two servers."""
    lp_a = get_logprobs(port_a, prompt, lora_name)
    lp_b = get_logprobs(port_b, prompt, lora_name)

    min_len = min(len(lp_a), len(lp_b))
    diffs = []
    for i in range(min_len):
        if lp_a[i] is not None and lp_b[i] is not None:
            diffs.append(abs(lp_a[i] - lp_b[i]))

    if not diffs:
        return 0.0, 0.0, 0

    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    return max_diff, mean_diff, len(diffs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--port-csgmv", type=int, default=30000)
    parser.add_argument("--port-triton", type=int, default=30001)
    parser.add_argument("--extra-args", nargs="*", default=[])
    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    adapter_names = []
    for d in sorted(lora_dir.iterdir()):
        if (d / "adapter.yaml").exists():
            adapter_names.append(d.name)

    if not adapter_names:
        print(f"No adapters found in {lora_dir}")
        return

    print(f"Adapters: {adapter_names}")

    # For SGLang, lora_paths need to be actual model paths.
    # In the tensor-loaded multi-LoRA case we can't easily do this.
    # Instead, this test is for file-based adapters.
    # If your adapters are tensor-loaded, skip this and use the training
    # metrics comparison instead.

    prompts = [
        "What is 2+2?",
        "Explain the theory of relativity in simple terms.",
        "Write a haiku about machine learning.",
        "The capital of France is",
        "def fibonacci(n):",
    ]

    # Launch both servers
    lora_paths = [str(lora_dir / name) for name in adapter_names]
    proc_csgmv = launch_server(args.model, lora_paths, args.port_csgmv, "csgmv", args.extra_args)
    proc_triton = launch_server(args.model, lora_paths, args.port_triton, "triton", args.extra_args)

    try:
        print(f"\n{'='*60}")
        print("Comparing csgmv vs triton log probs")
        print(f"{'='*60}\n")

        for adapter in adapter_names:
            print(f"Adapter: {adapter}")
            for prompt in prompts:
                max_d, mean_d, n = compare_backends(
                    args.port_csgmv, args.port_triton,
                    prompt, adapter,
                )
                status = "OK" if max_d < 1e-4 else "DIFF"
                print(f"  [{status}] max={max_d:.2e} mean={mean_d:.2e} n={n}  prompt={prompt[:40]}...")

            print()
    finally:
        proc_csgmv.kill()
        proc_triton.kill()
        proc_csgmv.wait()
        proc_triton.wait()


if __name__ == "__main__":
    main()
