"""Benchmark tokens/sec and first-token latency across local models and NIM.

For local Ollama models this uses the native ``/api/chat`` endpoint, which
reports exact ``eval_count`` and ``eval_duration`` values, so the decode rate is
measured by the runtime rather than guessed. For NVIDIA NIM it streams over the
OpenAI API and times each chunk on the client.

Two rates are reported so local and cloud numbers can sit in one table:

* **decode tok/s**: generation speed once tokens are flowing. Local: Ollama's
  ``eval_count / eval_duration``. NIM: tokens after the first one divided by the
  time between the first and the last streamed chunk. Neither includes model
  load, prompt processing or network latency before the first token.
* **end-to-end tok/s**: generated tokens divided by the whole request's wall
  time, first-token latency (and, locally, model load) included.

Usage:
    python -m src.benchmark                       # every installed local chat model
    python -m src.benchmark --models llama3.1:8b qwen2.5:7b
    python -m src.benchmark --nim                 # add cloud NIM to the comparison
    python -m src.benchmark --num-predict 256 --prompt "Explain B-trees."
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import requests
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .client import LLMClient, MissingAPIKey, ollama_host, resolve_model
from .model_manager import OllamaError, check_server, list_models, show_model

console = Console()

DEFAULT_PROMPT = "Write a concise paragraph explaining how a CPU cache improves performance."
DEFAULT_NUM_PREDICT = 200

# Used only when the server is too old to report model capabilities in /api/show.
EMBEDDING_NAME_HINTS = (
    "embed", "bge-", "bge:", "minilm", "e5-", "gte-", "paraphrase-", "nomic-bert", "snowflake-arctic",
)


@dataclass
class Result:
    backend: str
    model: str
    ttft_s: float | None = None        # client-measured time to first token, seconds
    tokens: int = 0                    # generated tokens
    decode_s: float = 0.0              # time spent generating (see module docstring)
    decode_tps: float = 0.0            # generation speed, comparable across backends
    total_s: float = 0.0               # wall time of the whole request
    e2e_tps: float = 0.0               # tokens / total_s
    load_s: float | None = None        # model load time when reported (local only)
    error: str | None = None

    @property
    def tokens_per_sec(self) -> float:
        """Backwards-compatible alias for :attr:`decode_tps`."""
        return self.decode_tps


def _error_text(resp: requests.Response) -> str:
    try:
        err = resp.json().get("error")
    except ValueError:
        err = None
    if isinstance(err, dict):
        err = err.get("message")
    return str(err) if err else f"HTTP {resp.status_code} {resp.reason}"


def bench_ollama(model: str, prompt: str, num_predict: int) -> Result:
    """Benchmark one local model via Ollama's native streaming chat API.

    Ollama reports failures in two ways and both become an error row: a non-2xx
    response with a JSON ``error`` body, or an ``{"error": ...}`` event inside an
    HTTP 200 stream (for example when the model cannot be loaded into memory).
    """
    url = f"{ollama_host()}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"num_predict": num_predict},
    }

    def failed(message: str) -> Result:
        return Result("ollama", model, error=message)

    start = time.perf_counter()
    ttft: float | None = None
    final: dict | None = None
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=None)
        if not resp.ok:
            return failed(_error_text(resp))
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return failed(f"malformed line in Ollama's response stream: {line[:80]!r}")
            if not isinstance(event, dict):
                return failed(f"unexpected event in Ollama's response stream: {line[:80]!r}")
            if event.get("error"):
                return failed(str(event["error"]))
            if ttft is None and (event.get("message") or {}).get("content"):
                ttft = time.perf_counter() - start
            if event.get("done"):
                final = event
    except requests.RequestException as exc:
        return failed(str(exc))
    total = time.perf_counter() - start

    if final is None:
        return failed("the response stream ended without a final 'done' event")

    # Durations from Ollama are in nanoseconds.
    eval_count = int(final.get("eval_count") or 0)
    eval_ns = int(final.get("eval_duration") or 0)
    load_ns = int(final.get("load_duration") or 0)
    if eval_count == 0:
        return failed("the model generated no tokens")
    decode_s = eval_ns / 1e9
    return Result(
        backend="ollama",
        model=model,
        ttft_s=ttft,
        tokens=eval_count,
        decode_s=decode_s,
        decode_tps=eval_count / decode_s if decode_s > 0 else 0.0,
        total_s=total,
        e2e_tps=eval_count / total if total > 0 else 0.0,
        load_s=load_ns / 1e9 if load_ns else None,
    )


def stream_stats(start: float, arrivals: list[float], end: float, usage_tokens: int | None) -> dict:
    """Timing figures for a client-side timed token stream.

    ``arrivals`` are the ``perf_counter`` timestamps of every chunk that carried
    text. The first chunk is assumed to hold one token, so the decode rate is
    ``(tokens - 1) / (last - first)``: time to first token is excluded, exactly
    like Ollama's ``eval_duration`` excludes load and prompt processing.
    """
    tokens = usage_tokens if usage_tokens else len(arrivals)
    total = max(end - start, 1e-9)
    if not arrivals:
        return {"ttft_s": None, "tokens": tokens, "decode_s": 0.0, "decode_tps": 0.0,
                "total_s": total, "e2e_tps": tokens / total}
    decode_s = arrivals[-1] - arrivals[0]
    decode_tps = (tokens - 1) / decode_s if tokens > 1 and decode_s > 0 else 0.0
    return {"ttft_s": arrivals[0] - start, "tokens": tokens, "decode_s": decode_s,
            "decode_tps": decode_tps, "total_s": total, "e2e_tps": tokens / total}


def bench_nim(prompt: str, num_predict: int, model: str | None = None) -> Result:
    """Benchmark the cloud NIM chat model via the OpenAI streaming API."""
    model = model or resolve_model("chat", "nim")
    try:
        client = LLMClient.create("nim")
    except (MissingAPIKey, ValueError) as exc:
        return Result("nim", model, error=str(exc))

    messages = [{"role": "user", "content": prompt}]
    arrivals: list[float] = []
    usage_tokens = None
    start = time.perf_counter()
    try:
        stream = client.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=num_predict,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage is not None and chunk.usage.completion_tokens:
                usage_tokens = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                arrivals.append(time.perf_counter())
    except Exception as exc:  # network, auth, rate limit: report as an error row
        return Result("nim", model, error=str(exc))
    stats = stream_stats(start, arrivals, time.perf_counter(), usage_tokens)
    if not stats["tokens"]:
        return Result("nim", model, error="the model generated no tokens")
    return Result("nim", model, **stats)


def render(results: list[Result]) -> None:
    """Print a comparison table of the benchmark results."""
    table = Table(title="tokens/sec benchmark")
    table.add_column("Backend")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("First token", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Decode tok/s", justify="right", style="bold green")
    table.add_column("End-to-end tok/s", justify="right")
    table.add_column("Load", justify="right")

    for r in sorted(results, key=lambda x: (x.error is not None, -x.decode_tps)):
        if r.error:
            table.add_row(r.backend, escape(r.model), "-", "-", "[red]error[/red]", "-", "-")
            continue
        table.add_row(
            r.backend,
            escape(r.model),
            f"{r.ttft_s:.2f} s" if r.ttft_s is not None else "-",
            str(r.tokens),
            f"{r.decode_tps:.1f}",
            f"{r.e2e_tps:.1f}",
            f"{r.load_s:.2f} s" if r.load_s is not None else "-",
        )
    console.print(table)

    for r in results:
        if r.error:
            console.print(f"[red]{escape(r.model)}:[/red] {escape(r.error)}")


def looks_like_embedding_model(name: str, family: str = "") -> bool:
    """Name/family heuristic, used only when /api/show reports no capabilities."""
    text = f"{name} {family}".lower()
    return any(hint in text for hint in EMBEDDING_NAME_HINTS) or family.lower() in ("bert", "nomic-bert")


def select_local_models(explicit: list[str] | None) -> tuple[list[str], list[tuple[str, str]]]:
    """Return ``(models to benchmark, [(skipped model, reason), ...])``.

    An explicit list is used as-is. Otherwise every installed model that can
    generate text is selected: ``/api/show`` capabilities decide (models without
    ``completion``, i.e. embedding-only models such as ``bge-m3``, are skipped),
    with a name heuristic for servers too old to report capabilities. Tags that
    share a manifest digest are the same weights, so only the first is kept.
    """
    if explicit:
        return list(explicit), []
    selected: list[str] = []
    skipped: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for m in sorted(list_models(), key=lambda x: x.get("name", "")):
        name = m.get("name", "")
        digest = m.get("digest")
        if digest and digest in seen:
            skipped.append((name, f"same weights as {seen[digest]}"))
            continue
        try:
            caps = show_model(name).get("capabilities")
        except (OllamaError, requests.RequestException):
            caps = None
        if caps:
            chat_capable = "completion" in caps
        else:
            chat_capable = not looks_like_embedding_model(name, (m.get("details") or {}).get("family", ""))
        if not chat_capable:
            skipped.append((name, "cannot generate text (embedding model)"))
            continue
        if digest:
            seen[digest] = name
        selected.append(name)
    return selected, skipped


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
        models, skipped = select_local_models(args.models)
        for name, reason in skipped:
            console.print(f"[dim]Skipping {escape(name)}: {escape(reason)}[/dim]")
        if not models:
            console.print("No local chat models installed. Try [bold]python -m src.model_manager recommend[/bold].")
        for model in models:
            console.print(f"Benchmarking [cyan]{escape(model)}[/cyan] ...")
            results.append(bench_ollama(model, args.prompt, args.num_predict))

    if args.nim:
        console.print("Benchmarking [cyan]NVIDIA NIM[/cyan] ...")
        results.append(bench_nim(args.prompt, args.num_predict))

    if not results:
        console.print("Nothing to benchmark.")
        return 1

    render(results)
    return 1 if all(r.error for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
