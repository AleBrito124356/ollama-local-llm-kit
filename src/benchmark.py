"""Benchmark tokens/sec and first-token latency across local models and NIM.

For local Ollama models this uses the native ``/api/chat`` endpoint, which
reports exact ``prompt_eval_*`` and ``eval_*`` counters, so the rates are
measured by the runtime rather than guessed. For NVIDIA NIM it streams over the
OpenAI API and times each chunk on the client.

Every model gets one warm-up request (loads the model; its load time is shown as
"cold load" and it is excluded from the statistics) and then ``--runs`` measured
requests. The table shows medians, plus the min-max decode range. Each run's
prompt starts with a distinct ``[run N]`` tag so Ollama's prompt cache cannot
skip prompt processing and inflate the prefill rate.

Rates reported:

* **prefill tok/s**: prompt processing speed (``prompt_eval_count /
  prompt_eval_duration``). Local only; NIM does not report it.
* **decode tok/s**: generation speed once tokens are flowing. Local: Ollama's
  ``eval_count / eval_duration``. NIM: tokens after the first one divided by the
  time between the first and the last streamed chunk. Neither includes model
  load, prompt processing or network latency before the first token.
* **end-to-end tok/s**: generated tokens divided by the whole request's wall
  time, first-token latency included.

After a model's runs, ``/api/ps`` tells how much of it Ollama kept in VRAM; a
share under 100% means layers were offloaded to the CPU, the usual reason for a
low decode rate on consumer GPUs.

Usage:
    python -m src.benchmark                                  # every installed local chat model
    python -m src.benchmark --models llama3.1:8b qwen2.5:14b --runs 5
    python -m src.benchmark --models qwen2.5:14b --num-ctx 16384 --num-gpu 40
    python -m src.benchmark --nim                            # add cloud NIM to the comparison
    python -m src.benchmark --json results.json --csv results.csv --markdown results.md
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
import time
from dataclasses import dataclass, field

import requests
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import hardware as hw
from .client import LLMClient, MissingAPIKey, ollama_host, resolve_model
from .model_manager import (
    OllamaError,
    check_server,
    gpu_share,
    list_models,
    running_models,
    same_model,
    show_model,
)

console = Console()

DEFAULT_PROMPT = "Write a concise paragraph explaining how a CPU cache improves performance."
DEFAULT_NUM_PREDICT = 200
DEFAULT_RUNS = 3

# Used only when the server is too old to report model capabilities in /api/show.
EMBEDDING_NAME_HINTS = (
    "embed", "bge-", "bge:", "minilm", "e5-", "gte-", "paraphrase-", "nomic-bert", "snowflake-arctic",
)


@dataclass
class Result:
    """One request's measurements."""

    backend: str
    model: str
    ttft_s: float | None = None        # client-measured time to first token, seconds
    tokens: int = 0                    # generated tokens
    decode_s: float = 0.0              # time spent generating (see module docstring)
    decode_tps: float = 0.0            # generation speed, comparable across backends
    total_s: float = 0.0               # wall time of the whole request
    e2e_tps: float = 0.0               # tokens / total_s
    load_s: float | None = None        # model load time when reported (local only)
    prompt_tokens: int | None = None   # prompt tokens processed (local only)
    prefill_s: float | None = None
    prefill_tps: float | None = None
    error: str | None = None

    @property
    def tokens_per_sec(self) -> float:
        """Backwards-compatible alias for :attr:`decode_tps`."""
        return self.decode_tps


def _median(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


@dataclass
class Summary:
    """A model's warm-up plus measured runs."""

    backend: str
    model: str
    runs: list[Result] = field(default_factory=list)
    warmup: Result | None = None
    gpu_percent: float | None = None
    error: str | None = None

    @property
    def ok_runs(self) -> list[Result]:
        return [r for r in self.runs if r.error is None]

    def stat(self, attr: str) -> float | None:
        return _median([getattr(r, attr) for r in self.ok_runs])

    @property
    def decode_range(self) -> tuple[float, float] | None:
        rates = [r.decode_tps for r in self.ok_runs]
        return (min(rates), max(rates)) if rates else None

    def row(self) -> dict:
        """Flat record used by the JSON / CSV / Markdown exports."""
        rng = self.decode_range

        def rnd(value, digits=2):
            return None if value is None else round(value, digits)

        return {
            "backend": self.backend,
            "model": self.model,
            "runs": len(self.ok_runs),
            "tokens": rnd(self.stat("tokens"), 1),
            "ttft_s": rnd(self.stat("ttft_s"), 3),
            "prompt_tokens": rnd(self.stat("prompt_tokens"), 1),
            "prefill_tps": rnd(self.stat("prefill_tps"), 1),
            "decode_tps": rnd(self.stat("decode_tps"), 1),
            "decode_tps_min": rnd(rng[0], 1) if rng else None,
            "decode_tps_max": rnd(rng[1], 1) if rng else None,
            "e2e_tps": rnd(self.stat("e2e_tps"), 1),
            "cold_load_s": rnd(self.warmup.load_s if self.warmup else None, 3),
            "gpu_percent": rnd(self.gpu_percent, 1),
            "error": self.error,
        }


def _error_text(resp: requests.Response) -> str:
    try:
        err = resp.json().get("error")
    except ValueError:
        err = None
    if isinstance(err, dict):
        err = err.get("message")
    return str(err) if err else f"HTTP {resp.status_code} {resp.reason}"


def parse_keep_alive(value: str | None):
    """Ollama accepts a duration string ("5m", "1h") or a number of seconds (0 unloads)."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def bench_ollama(
    model: str,
    prompt: str,
    num_predict: int,
    options: dict | None = None,
    keep_alive=None,
) -> Result:
    """Run one request against a local model via Ollama's native streaming chat API.

    Ollama reports failures in two ways and both become an error result: a non-2xx
    response with a JSON ``error`` body, or an ``{"error": ...}`` event inside an
    HTTP 200 stream (for example when the model cannot be loaded into memory).
    """
    url = f"{ollama_host()}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {**(options or {}), "num_predict": num_predict},
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

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
    prompt_count = final.get("prompt_eval_count")
    prompt_ns = int(final.get("prompt_eval_duration") or 0)
    if eval_count == 0:
        return failed("the model generated no tokens")
    decode_s = eval_ns / 1e9
    prefill_s = prompt_ns / 1e9 if prompt_ns else None
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
        prompt_tokens=int(prompt_count) if prompt_count is not None else None,
        prefill_s=prefill_s,
        prefill_tps=(int(prompt_count) / prefill_s) if prefill_s and prompt_count else None,
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
    """Run one request against the cloud NIM chat model via the OpenAI streaming API."""
    model = model or resolve_model("chat", "nim")
    try:
        client = LLMClient.create("nim")
    except (MissingAPIKey, ValueError) as exc:
        return Result("nim", model, error=str(exc))

    messages = [{"role": "user", "content": prompt}]
    arrivals: list[float] = []
    usage = None
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
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                arrivals.append(time.perf_counter())
    except Exception as exc:  # network, auth, rate limit: report as an error result
        return Result("nim", model, error=str(exc))
    stats = stream_stats(start, arrivals, time.perf_counter(), usage.completion_tokens if usage else None)
    if not stats["tokens"]:
        return Result("nim", model, error="the model generated no tokens")
    return Result("nim", model, prompt_tokens=usage.prompt_tokens if usage else None, **stats)


def run_prompt(prompt: str, run: int) -> str:
    """Tag each run's prompt so Ollama's prompt (KV) cache cannot skip prefill."""
    return f"[run {run}] {prompt}"


def benchmark_model(
    backend: str,
    model: str,
    prompt: str,
    num_predict: int,
    runs: int = DEFAULT_RUNS,
    warmup: bool = True,
    options: dict | None = None,
    keep_alive=None,
) -> Summary:
    """One excluded warm-up request, then ``runs`` measured requests."""

    def once(i: int) -> Result:
        if backend == "ollama":
            return bench_ollama(model, run_prompt(prompt, i), num_predict, options, keep_alive)
        return bench_nim(run_prompt(prompt, i), num_predict, model)

    summary = Summary(backend, model)
    if warmup:
        summary.warmup = once(0)
        if summary.warmup.error:
            summary.error = summary.warmup.error
            return summary
    summary.runs = [once(i) for i in range(1, runs + 1)]
    errors = [r.error for r in summary.runs if r.error]
    if errors and not summary.ok_runs:
        summary.error = errors[0]
    elif errors:
        summary.error = f"{len(errors)} of {runs} runs failed: {errors[0]}"
    if backend == "ollama" and summary.ok_runs:
        try:
            entry = next((m for m in running_models() if same_model(m.get("name", ""), model)), None)
        except requests.RequestException:
            entry = None
        summary.gpu_percent = gpu_share(entry) if entry else None
    return summary


# -- output ---------------------------------------------------------------------------
COLUMNS = ["backend", "model", "runs", "tokens", "ttft_s", "prompt_tokens", "prefill_tps", "decode_tps",
           "decode_tps_min", "decode_tps_max", "e2e_tps", "cold_load_s", "gpu_percent", "error"]


def _fmt(value, suffix: str = "", digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}{suffix}"


def render(summaries: list[Summary], out: Console | None = None) -> None:
    """Print a comparison table of the benchmark summaries."""
    out = out or console
    table = Table(title="tokens/sec benchmark (medians)")
    table.add_column("Backend")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Runs", justify="right")
    table.add_column("First token", justify="right")
    table.add_column("Prefill tok/s", justify="right")
    table.add_column("Decode tok/s", justify="right", style="bold green")
    table.add_column("min-max", justify="right")
    table.add_column("End-to-end tok/s", justify="right")
    table.add_column("Cold load", justify="right")
    table.add_column("On GPU", justify="right")

    def key(s: Summary):
        return (not s.ok_runs, -(s.stat("decode_tps") or 0.0))

    for s in sorted(summaries, key=key):
        if not s.ok_runs:
            table.add_row(s.backend, escape(s.model), "0", "-", "-", "[red]error[/red]", "-", "-", "-", "-")
            continue
        rng = s.decode_range
        table.add_row(
            s.backend,
            escape(s.model),
            str(len(s.ok_runs)),
            _fmt(s.stat("ttft_s"), " s", 2),
            _fmt(s.stat("prefill_tps")),
            _fmt(s.stat("decode_tps")),
            f"{rng[0]:.1f}-{rng[1]:.1f}" if rng else "-",
            _fmt(s.stat("e2e_tps")),
            _fmt(s.warmup.load_s if s.warmup else None, " s", 2),
            _fmt(s.gpu_percent, "%", 0),
        )
    out.print(table)

    for s in summaries:
        if s.error:
            out.print(f"[red]{escape(s.model)}:[/red] {escape(s.error)}")
    for s in summaries:
        if s.gpu_percent is not None and s.gpu_percent < 99.5:
            out.print(
                f"[yellow]{escape(s.model)}: only {s.gpu_percent:.0f}% of the model is in VRAM; the rest runs on "
                "the CPU and caps decode speed.[/yellow] [dim]Lower --num-ctx or try a smaller model; plan it with "
                "python -m src.model_manager fit[/dim]"
            )


def to_markdown(rows: list[dict]) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join("" if r[c] is None else str(r[c]).replace("|", "\\|") for c in COLUMNS) + " |"
            for r in rows]
    return "\n".join([header, sep, *body]) + "\n"


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def export(path: str, text: str) -> None:
    if path == "-":
        sys.stdout.write(text)
        sys.stdout.flush()
    else:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark", description="Measure prefill/decode tokens/sec and latency.")
    parser.add_argument("--models", nargs="*", help="Specific local models to test (default: all installed chat models)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send")
    parser.add_argument("--prompt-file", default=None, help="Read the prompt from a file (a long prompt gives a steadier prefill rate)")
    parser.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT, help="Max tokens to generate")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"Measured runs per model (default {DEFAULT_RUNS})")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the excluded warm-up request")
    parser.add_argument("--num-ctx", type=int, default=None, help="Ollama option num_ctx (context window)")
    parser.add_argument("--num-gpu", type=int, default=None, help="Ollama option num_gpu (layers on the GPU; 0 = CPU only)")
    parser.add_argument("--keep-alive", default=None, help="How long Ollama keeps the model loaded (e.g. 5m, 0)")
    parser.add_argument("--nim", action="store_true", help="Also benchmark cloud NVIDIA NIM (needs NVIDIA_API_KEY)")
    parser.add_argument("--nim-model", default=None, help="NIM model id (default: the chat role's NIM model)")
    parser.add_argument("--no-local", action="store_true", help="Skip local models (use with --nim)")
    parser.add_argument("--json", metavar="PATH", help="Write results as JSON ('-' for stdout)")
    parser.add_argument("--csv", metavar="PATH", help="Write results as CSV ('-' for stdout)")
    parser.add_argument("--markdown", metavar="PATH", help="Write results as a Markdown table ('-' for stdout)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    to_stdout = "-" in (args.json, args.csv, args.markdown)
    out = Console(stderr=True) if to_stdout else console
    if args.runs < 1:
        out.print("[red]--runs must be at least 1[/red]")
        return 2
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as fh:
            prompt = fh.read().strip()

    options = {k: v for k, v in (("num_ctx", args.num_ctx), ("num_gpu", args.num_gpu)) if v is not None}
    keep_alive = parse_keep_alive(args.keep_alive)
    hardware = hw.detect()
    out.print(f"[dim]{escape(hardware.describe())}[/dim]")
    if options or keep_alive is not None:
        out.print(f"[dim]Ollama options: {escape(json.dumps({**options, 'keep_alive': keep_alive}))}[/dim]")

    summaries: list[Summary] = []
    if not args.no_local:
        if not check_server():
            return 1
        models, skipped = select_local_models(args.models)
        for name, reason in skipped:
            out.print(f"[dim]Skipping {escape(name)}: {escape(reason)}[/dim]")
        if not models:
            out.print("No local chat models installed. Try [bold]python -m src.model_manager recommend[/bold].")
        for model in models:
            out.print(f"Benchmarking [cyan]{escape(model)}[/cyan]: "
                      f"{'1 warm-up + ' if not args.no_warmup else ''}{args.runs} run(s) ...")
            summaries.append(benchmark_model("ollama", model, prompt, args.num_predict, args.runs,
                                             not args.no_warmup, options, keep_alive))

    if args.nim:
        nim_model = args.nim_model or resolve_model("chat", "nim")
        out.print(f"Benchmarking [cyan]NVIDIA NIM {escape(nim_model)}[/cyan] ...")
        summaries.append(benchmark_model("nim", nim_model, prompt, args.num_predict, args.runs, not args.no_warmup))

    if not summaries:
        out.print("Nothing to benchmark.")
        return 1

    render(summaries, out)
    rows = [s.row() for s in summaries]
    if args.json:
        export(args.json, json.dumps({
            "hardware": hardware.as_dict(),
            "settings": {"prompt": prompt, "num_predict": args.num_predict, "runs": args.runs,
                         "warmup": not args.no_warmup, "options": options, "keep_alive": keep_alive},
            "results": rows,
        }, indent=2) + "\n")
    if args.csv:
        export(args.csv, to_csv(rows))
    if args.markdown:
        export(args.markdown, to_markdown(rows))
    return 1 if all(not s.ok_runs for s in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
