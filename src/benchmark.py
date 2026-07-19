"""Benchmark tokens/sec and first-token latency across local models and NIM.

For local Ollama models this uses the native ``/api/chat`` endpoint, which
reports exact ``eval_count`` and ``eval_duration`` values, so the tokens/sec
figure is measured by the runtime rather than guessed. For NVIDIA NIM it streams
over the OpenAI API and times the wall clock, counting tokens from the usage
block when the server returns one.

Usage:
    python -m src.benchmark                       # every installed local model
    python -m src.benchmark --models llama3.1:8b qwen2.5:7b
    python -m src.benchmark --nim                 # add cloud NIM to the comparison
    python -m src.benchmark --num-predict 256 --prompt "Explain B-trees."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import requests
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import LLMClient, ollama_host, resolve_model  # noqa: E402
from model_manager import check_server, list_models  # noqa: E402

console = Console()

DEFAULT_PROMPT = "Write a concise paragraph explaining how a CPU cache improves performance."
DEFAULT_NUM_PREDICT = 200

# RTX 5070 has 12 GB of VRAM. This note is printed with the results so the numbers
# have context: a model that fits entirely in VRAM will report far higher
# tokens/sec than one that spills layers into system RAM.
GPU_NOTE = (
    "RTX 5070 / 12 GB: a Q4 8B model (~5 GB) fits in VRAM with room for a large "
    "context and should run fastest. A 14B Q4 model (~9 GB) fits at a reduced "
    "context window; push num_ctx too high and layers offload to system RAM, "
    "which drops tokens/sec sharply. See docs/gpu-notes.md."
)


@dataclass
class Result:
    backend: str
    model: str
    ttft_s: float | None       # time to first token, seconds
    tokens: int                # generated tokens counted
    gen_time_s: float          # generation wall time (excludes load)
    tokens_per_sec: float
    load_s: float | None       # model load time when reported
    error: str | None = None


def bench_ollama(model: str, prompt: str, num_predict: int) -> Result:
    """Benchmark one local model via Ollama's native streaming chat API."""
    url = f"{ollama_host()}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"num_predict": num_predict},
    }
    start = time.perf_counter()
    ttft: float | None = None
    final: dict = {}
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=None)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            if ttft is None and event.get("message", {}).get("content"):
                ttft = time.perf_counter() - start
            if event.get("done"):
                final = event
    except requests.RequestException as exc:
        return Result("ollama", model, None, 0, 0.0, 0.0, None, error=str(exc))

    # Durations from Ollama are in nanoseconds.
    eval_count = int(final.get("eval_count", 0))
    eval_ns = int(final.get("eval_duration", 0))
    load_ns = int(final.get("load_duration", 0))
    gen_time = eval_ns / 1e9 if eval_ns else max(time.perf_counter() - start, 1e-9)
    tps = eval_count / gen_time if gen_time > 0 else 0.0
    return Result(
        backend="ollama",
        model=model,
        ttft_s=ttft,
        tokens=eval_count,
        gen_time_s=gen_time,
        tokens_per_sec=tps,
        load_s=load_ns / 1e9 if load_ns else None,
    )


def bench_nim(prompt: str, num_predict: int) -> Result:
    """Benchmark the cloud NIM chat model via the OpenAI streaming API."""
    model = resolve_model("chat", "nim")
    try:
        client = LLMClient.create("nim")
    except Exception as exc:  # MissingAPIKey or config error
        return Result("nim", model, None, 0, 0.0, 0.0, None, error=str(exc))

    messages = [{"role": "user", "content": prompt}]
    start = time.perf_counter()
    ttft: float | None = None
    tokens = 0
    usage_tokens = None
    try:
        stream = client.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=num_predict,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage is not None:
                usage_tokens = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - start
                tokens += 1  # fallback count: one streamed delta ~= one token
    except Exception as exc:
        return Result("nim", model, None, 0, 0.0, 0.0, None, error=str(exc))

    gen_time = max(time.perf_counter() - start, 1e-9)
    counted = usage_tokens if usage_tokens else tokens
    tps = counted / gen_time if gen_time > 0 else 0.0
    return Result("nim", model, ttft, counted, gen_time, tps, load_s=None)


def render(results: list[Result]) -> None:
    """Print a comparison table of the benchmark results."""
    table = Table(title="tokens/sec benchmark")
    table.add_column("Backend")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("First token", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Gen time", justify="right")
    table.add_column("Tokens/sec", justify="right", style="bold green")
    table.add_column("Load", justify="right")

    for r in sorted(results, key=lambda x: x.tokens_per_sec, reverse=True):
        if r.error:
            table.add_row(r.backend, r.model, "-", "-", "-", "[red]error[/red]", "-")
            continue
        table.add_row(
            r.backend,
            r.model,
            f"{r.ttft_s:.2f} s" if r.ttft_s is not None else "-",
            str(r.tokens),
            f"{r.gen_time_s:.2f} s",
            f"{r.tokens_per_sec:.1f}",
            f"{r.load_s:.2f} s" if r.load_s is not None else "-",
        )
    console.print(table)

    errored = [r for r in results if r.error]
    for r in errored:
        console.print(f"[red]{r.model}:[/red] {r.error}")

    console.print(f"\n[dim]{GPU_NOTE}[/dim]")


def select_local_models(explicit: list[str] | None) -> list[str]:
    """Return the local models to benchmark: explicit list or all installed chat models."""
    if explicit:
        return explicit
    installed = list_models()
    # Skip pure embedding models; they cannot answer a chat prompt.
    return [
        m["name"]
        for m in installed
        if "embed" not in m.get("name", "").lower()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark", description="Measure tokens/sec and first-token latency.")
    parser.add_argument("--models", nargs="*", help="Specific local models to test (default: all installed)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send")
    parser.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT, help="Max tokens to generate")
    parser.add_argument("--nim", action="store_true", help="Also benchmark cloud NVIDIA NIM (needs NVIDIA_API_KEY)")
    parser.add_argument("--no-local", action="store_true", help="Skip local models (use with --nim)")
    args = parser.parse_args(argv)

    results: list[Result] = []

    if not args.no_local:
        if not check_server():
            return 1
        models = select_local_models(args.models)
        if not models:
            console.print("No local chat models installed. Try [bold]python -m src.model_manager recommend[/bold].")
        for model in models:
            console.print(f"Benchmarking [cyan]{model}[/cyan] (warming up + generating) ...")
            results.append(bench_ollama(model, args.prompt, args.num_predict))

    if args.nim:
        console.print("Benchmarking [cyan]NVIDIA NIM[/cyan] ...")
        results.append(bench_nim(args.prompt, args.num_predict))

    if not results:
        console.print("Nothing to benchmark.")
        return 1

    render(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
