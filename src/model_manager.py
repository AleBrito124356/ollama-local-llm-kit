"""Manage local Ollama models and plan them against your hardware.

Wraps the Ollama native REST API (``/api/tags``, ``/api/pull``, ``/api/show``,
``/api/delete``, ``/api/ps``) so you can list, pull with a live progress bar,
inspect, remove and total up disk usage without leaving Python. ``recommend``
prints the curated ``models.yaml`` catalog marked against your GPU, ``fit``
estimates VRAM use per model and context size, and ``doctor`` checks the whole
setup (server, models, key, GPU, current CPU/GPU split).

Usage:
    python -m src.model_manager list
    python -m src.model_manager recommend
    python -m src.model_manager fit --ctx 8192             # detected GPU, or --vram 12
    python -m src.model_manager fit qwen2.5:14b --vram 12 --ctx 32768
    python -m src.model_manager doctor
    python -m src.model_manager pull llama3.1:8b
    python -m src.model_manager show llama3.1:8b
    python -m src.model_manager remove llava:7b
    python -m src.model_manager du
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests
import yaml
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from . import hardware as hw
from .client import (
    mask_key,
    nim_base_url,
    nim_is_hosted,
    nim_key_problem,
    ollama_host,
    require_nim_key,
    resolve_backend,
    resolve_model,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_YAML = os.path.join(REPO_ROOT, "models.yaml")

console = Console()


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


def _api(path: str) -> str:
    return f"{ollama_host()}{path}"


def check_server() -> bool:
    """Return True if the Ollama server answers, else print a hint and return False."""
    try:
        requests.get(_api("/api/version"), timeout=3).raise_for_status()
        return True
    except requests.RequestException:
        console.print(
            f"[red]Cannot reach Ollama at {ollama_host()}.[/red]\n"
            "Start it with [bold]ollama serve[/bold] (or install it — see docs/install-ollama.md)."
        )
        return False


def human_size(num_bytes: float) -> str:
    """Format a byte count with decimal units, the same way ``ollama list`` does."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1000
    return f"{size:.1f} TB"


def _error_message(resp: requests.Response) -> str:
    """The ``error`` text Ollama puts in a failed response body, or the HTTP reason."""
    try:
        err = resp.json().get("error")
    except ValueError:
        err = None
    if isinstance(err, dict):
        err = err.get("message")
    return str(err) if err else f"HTTP {resp.status_code} {resp.reason}"


# -- Ollama API calls -----------------------------------------------------
def list_models() -> list[dict]:
    """Return the locally installed models from ``/api/tags``."""
    resp = requests.get(_api("/api/tags"), timeout=10)
    resp.raise_for_status()
    return resp.json().get("models", [])


def show_model(name: str) -> dict:
    """Return model metadata from ``/api/show``."""
    resp = requests.post(_api("/api/show"), json={"model": name, "name": name}, timeout=30)
    if resp.status_code == 404:
        raise OllamaError(f"Model {name!r} is not installed locally.")
    if not resp.ok:
        raise OllamaError(_error_message(resp))
    return resp.json()


def delete_model(name: str) -> None:
    """Remove a model with ``/api/delete``."""
    resp = requests.delete(_api("/api/delete"), json={"model": name, "name": name}, timeout=30)
    if resp.status_code == 404:
        raise OllamaError(f"Model {name!r} is not installed locally.")
    if not resp.ok:
        raise OllamaError(_error_message(resp))


def pull_model(name: str) -> None:
    """Pull a model, streaming download progress to a live Rich bar.

    Ollama streams newline-delimited JSON objects, one per progress update of
    each layer (``digest``, ``total``, ``completed``). The bar shows the sum over
    all layers, so it moves forward once instead of restarting for every layer.
    """
    resp = requests.post(
        _api("/api/pull"), json={"model": name, "name": name, "stream": True}, stream=True, timeout=None
    )
    if not resp.ok:
        raise OllamaError(_error_message(resp))

    layers: dict[str, tuple[int, int]] = {}
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = None
        last_status = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "error" in event:
                raise OllamaError(event["error"])

            status = event.get("status", "")
            total = event.get("total")

            if total:
                layers[event.get("digest") or status] = (int(total), int(event.get("completed") or 0))
                grand_total = sum(t for t, _c in layers.values())
                done = sum(c for _t, c in layers.values())
                if task_id is None:
                    task_id = progress.add_task("downloading", total=grand_total)
                progress.update(task_id, description=f"downloading {len(layers)} layer(s)",
                                total=grand_total, completed=done)
            elif status and status != last_status:
                # Non-download phases: verifying, writing manifest, success.
                console.print(f"  [dim]{escape(status)}[/dim]")
            last_status = status

    console.print(f"[green]Done.[/green] {escape(name)} is ready.")


# -- curated recommendations ---------------------------------------------
def load_catalog() -> dict:
    """Load the curated ``models.yaml`` catalog."""
    with open(MODELS_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# -- command handlers -----------------------------------------------------
def cmd_list(_args) -> int:
    if not check_server():
        return 1
    models = list_models()
    if not models:
        console.print("No models installed yet. Try [bold]python -m src.model_manager recommend[/bold].")
        return 0

    table = Table(title="Installed Ollama models")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right")
    table.add_column("Family")
    table.add_column("Params")
    table.add_column("Quant")
    for m in sorted(models, key=lambda x: x.get("name", "")):
        details = m.get("details", {}) or {}
        table.add_row(
            escape(m.get("name", "?")),
            human_size(m.get("size", 0)),
            details.get("family", "-"),
            details.get("parameter_size", "-"),
            details.get("quantization_level", "-"),
        )
    console.print(table)
    return 0


def running_models() -> list[dict]:
    """Models currently loaded in memory, from ``/api/ps`` (with ``size_vram``)."""
    resp = requests.get(_api("/api/ps"), timeout=10)
    resp.raise_for_status()
    return resp.json().get("models", [])


def gpu_share(entry: dict) -> float | None:
    """Percent of a loaded model that sits in VRAM, from an ``/api/ps`` entry."""
    size, vram = entry.get("size") or 0, entry.get("size_vram")
    if not size or vram is None:
        return None
    return 100.0 * min(vram, size) / size


def same_model(a: str, b: str) -> bool:
    """True if two model references name the same tag (``llama3.1`` == ``llama3.1:latest``)."""
    norm = lambda n: n if ":" in n else f"{n}:latest"  # noqa: E731
    return norm(a) == norm(b)


# -- hardware planning --------------------------------------------------------------
def planning_budget(vram: float | None, ram: float | None) -> tuple[float, float | None, hw.Hardware | None, str]:
    """Resolve ``(vram_gib, ram_gib, detected_hardware, description)`` from flags or detection."""
    detected = None
    if vram is None or ram is None:
        detected = hw.detect()
    vram_gib = vram if vram is not None else detected.vram_gib
    ram_gib = ram if ram is not None else detected.ram_total_gib
    if vram is not None:
        v = f"{vram_gib:.1f} GiB VRAM (--vram)"
    elif detected.gpus:
        v = f"{vram_gib:.1f} GiB VRAM ({', '.join(g.name for g in detected.gpus)})"
    else:
        v = f"no GPU detected ({detected.gpu_note}); pass --vram GB to plan for one"
    if ram is not None:
        r = f"{ram_gib:.1f} GiB RAM (--ram)"
    elif ram_gib is not None:
        r = f"{ram_gib:.1f} GiB RAM (detected)"
    else:
        r = "RAM unknown (pass --ram GB)"
    return vram_gib, ram_gib, detected, f"{v}  |  {r}"


def catalog_specs(sections=("chat", "vision")) -> list[hw.ModelSpec]:
    catalog = load_catalog()
    return [hw.ModelSpec.from_catalog(e) for s in sections for e in catalog.get(s, [])]


def installed_spec(name: str) -> hw.ModelSpec | None:
    """ModelSpec for an installed model, from its /api/tags size and /api/show GGUF metadata."""
    match = next((m for m in list_models() if same_model(m.get("name", ""), name)), None)
    if match is None:
        return None
    info = show_model(match["name"])
    return hw.ModelSpec.from_ollama(match["name"], int(match.get("size") or 0), info.get("model_info") or {})


def resolve_specs(names: list[str], installed: bool) -> tuple[list[hw.ModelSpec], list[str]]:
    """Specs for the requested names (catalog first, then installed models); unknown names returned."""
    by_name = {s.name: s for s in catalog_specs(("chat", "vision", "embed"))}
    if not names and not installed:
        return catalog_specs(), []
    specs, unknown = [], []
    reachable = None
    if installed:
        for m in sorted(list_models(), key=lambda x: x.get("name", "")):
            info = show_model(m["name"])
            caps = info.get("capabilities") or []
            if caps and "completion" not in caps:
                continue
            specs.append(hw.ModelSpec.from_ollama(m["name"], int(m.get("size") or 0), info.get("model_info") or {}))
    for name in names:
        if name in by_name:
            specs.append(by_name[name])
            continue
        if reachable is None:
            try:
                requests.get(_api("/api/version"), timeout=3).raise_for_status()
                reachable = True
            except requests.RequestException:
                reachable = False
        spec = None
        if reachable:
            try:
                spec = installed_spec(name)
            except (OllamaError, requests.RequestException):
                spec = None
        if spec is None:
            unknown.append(name)
        else:
            specs.append(spec)
    return specs, unknown


def _fit_row(result: hw.FitResult) -> dict:
    return {
        "model": result.model,
        "status": result.status,
        "verdict": result.label,
        "num_ctx": result.num_ctx,
        "weights_gib": round(result.weights_gib, 2),
        "kv_cache_gib": round(result.kv_gib, 2),
        "overhead_gib": round(result.overhead_gib, 2),
        "total_gib": round(result.total_gib, 2),
        "gpu_layers": result.gpu_layers,
        "n_layers": result.n_layers,
        "gpu_percent": round(result.gpu_percent, 1),
        "max_ctx_on_gpu": result.max_ctx_on_gpu,
        "note": result.note,
    }


_VERDICT_STYLE = {"gpu": "green", "partial": "yellow", "cpu": "yellow", "too_large": "red"}


def cmd_fit(args) -> int:
    try:
        specs, unknown = resolve_specs(args.models or [], args.installed)
    except requests.RequestException:
        console.print(f"[red]Cannot reach Ollama at {escape(ollama_host())} to read installed models.[/red]")
        return 1
    vram_gib, ram_gib, _detected, budget = planning_budget(args.vram, args.ram)
    results = [hw.plan_fit(s, vram_gib, ram_gib, args.ctx, args.kv_cache, args.overhead) for s in specs]

    if args.json:
        print(json.dumps({
            "vram_gib": round(vram_gib, 2), "ram_gib": None if ram_gib is None else round(ram_gib, 2),
            "num_ctx": args.ctx, "kv_cache": args.kv_cache, "overhead_gib": args.overhead,
            "models": [_fit_row(r) for r in results], "unknown": unknown,
        }, indent=2))
        return 1 if unknown else 0

    console.print(f"[bold]Planning for[/bold] {escape(budget)}")
    console.print(f"[dim]num_ctx {args.ctx}, KV cache {args.kv_cache}, overhead {args.overhead:.1f} GiB[/dim]")
    table = Table(title=f"Will it fit? (num_ctx = {args.ctx})")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Weights", justify="right")
    table.add_column("KV cache", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Verdict")
    table.add_column("GPU layers", justify="right")
    table.add_column("Max ctx fully on GPU", justify="right")
    for r in results:
        layers = f"{r.gpu_layers}/{r.n_layers}" if r.gpu_layers is not None and r.n_layers else "-"
        table.add_row(
            escape(r.model),
            f"{r.weights_gib:.1f} GiB",
            f"{r.kv_gib:.1f} GiB" if r.n_layers else "?",
            f"{r.total_gib:.1f} GiB",
            f"[{_VERDICT_STYLE[r.status]}]{escape(r.label)}[/{_VERDICT_STYLE[r.status]}]",
            layers,
            f"{r.max_ctx_on_gpu:,}" if r.max_ctx_on_gpu else "-",
        )
    console.print(table)
    for r in results:
        if r.note:
            console.print(f"[dim]{escape(r.model)}: {escape(r.note)}[/dim]")
    for name in unknown:
        console.print(f"[red]Unknown model {escape(name)}:[/red] not in models.yaml and not installed in Ollama.")
    console.print(
        "[dim]Estimates: weights + KV cache (2 x layers x kv_heads x head_dim x bytes x num_ctx) + fixed overhead.\n"
        "Try a setting for real: python -m src.benchmark --models <model> --num-ctx <N>; "
        "check the actual split with: python -m src.model_manager doctor[/dim]"
    )
    return 1 if unknown else 0


def cmd_recommend(args) -> int:
    catalog = load_catalog()
    vram_gib, ram_gib, _detected, budget = planning_budget(args.vram, args.ram)
    console.print(f"[dim]Marked against {escape(budget)}, num_ctx {args.ctx}[/dim]")
    for section in ("chat", "embed", "vision"):
        entries = catalog.get(section, [])
        if not entries:
            continue
        table = Table(title=f"Recommended {section} models")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Params", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Min VRAM", justify="right")
        table.add_column("On this machine")
        table.add_column("Tools", justify="center")
        table.add_column("Notes")
        for e in entries:
            fit = hw.plan_fit(hw.ModelSpec.from_catalog(e), vram_gib, ram_gib, args.ctx)
            style = _VERDICT_STYLE[fit.status]
            table.add_row(
                e["name"],
                str(e.get("params", "-")),
                f"{e.get('size_gb', 0):.1f} GB",
                f"{e.get('min_vram_gb', '-')} GB",
                f"[{style}]{escape(fit.label)}[/{style}]",
                "yes" if e.get("tools") else "-",
                e.get("note", ""),
            )
        console.print(table)
    console.print("\nPull one with: [bold]python -m src.model_manager pull <model>[/bold]"
                  "   Size a context window with: [bold]python -m src.model_manager fit --ctx 8192[/bold]")
    return 0


# -- doctor ---------------------------------------------------------------------------
_STATUS_STYLE = {"ok": "[green]ok[/green]", "warn": "[yellow]warn[/yellow]",
                 "fail": "[red]FAIL[/red]", "info": "[dim]info[/dim]"}


def run_checks(online: bool = False, hardware: hw.Hardware | None = None) -> list[tuple[str, str, str]]:
    """Return ``(status, check, detail)`` rows; status is ok / warn / fail / info."""
    rows: list[tuple[str, str, str]] = []
    add = lambda status, name, detail: rows.append((status, name, detail))  # noqa: E731

    py = sys.version_info
    add("ok" if py >= (3, 10) else "fail", "Python", f"{py.major}.{py.minor}.{py.micro} ({hw.platform_summary()})")

    try:
        backend = resolve_backend()
        add("ok", "LLM_BACKEND", backend)
    except ValueError as exc:
        backend = None
        add("fail", "LLM_BACKEND", str(exc))
    local_needed = backend in (None, "ollama")

    # Ollama server and models
    version = None
    try:
        resp = requests.get(_api("/api/version"), timeout=3)
        resp.raise_for_status()
        version = resp.json().get("version", "?")
        add("ok", "Ollama server", f"v{version} at {ollama_host()}")
    except (requests.RequestException, ValueError):
        add("fail" if local_needed else "warn", "Ollama server",
            f"not reachable at {ollama_host()}: start it with `ollama serve` (see docs/install-ollama.md)")

    if version is not None:
        try:
            names = [m.get("name", "") for m in list_models()]
        except requests.RequestException as exc:
            names = []
            add("warn", "Installed models", f"could not list: {exc}")
        wanted = [
            ("chat", "Chat model", "fail" if local_needed else "warn", "used by chat and the examples"),
            ("embed", "Embedding model", "warn", "needed for RAG (rag_local, chat --rag)"),
            ("vision", "Vision model", "info", "only for examples/vision.py"),
        ]
        for role, label, missing_status, why in wanted:
            model = resolve_model(role, "ollama")
            if any(same_model(n, model) for n in names):
                add("ok", label, f"{model} installed")
            else:
                add(missing_status, label,
                    f"{model} not installed ({why}): python -m src.model_manager pull {model}")
        try:
            loaded = running_models()
        except requests.RequestException:
            loaded = []
        if not loaded:
            add("info", "Loaded now", "no model is loaded in memory right now")
        for entry in loaded:
            share = gpu_share(entry)
            name = entry.get("name", "?")
            if share is None:
                add("info", "Loaded now", f"{name}: GPU share not reported")
            elif share >= 99.5:
                add("ok", "Loaded now", f"{name}: 100% on GPU")
            elif share <= 0.5:
                add("warn", "Loaded now", f"{name}: 100% on CPU (slow). No GPU in use for this model")
            else:
                add("warn", "Loaded now",
                    f"{name}: {share:.0f}% GPU / {100 - share:.0f}% CPU, partially offloaded; tokens/sec will "
                    "drop. Lower num_ctx or pick a smaller model (python -m src.model_manager fit)")

    # NVIDIA NIM
    key = os.getenv("NVIDIA_API_KEY", "")
    problem = nim_key_problem(key)
    if not nim_is_hosted():
        add("ok", "NIM endpoint", f"self-hosted at {nim_base_url()} (no NVIDIA key required)")
    elif problem:
        add("fail" if backend == "nim" else "info", "NVIDIA_API_KEY",
            f"{problem} Only needed for LLM_BACKEND=nim; free key at https://build.nvidia.com")
    else:
        add("ok", "NVIDIA_API_KEY", f"{mask_key(key)} (format looks valid; not verified online)")
    if online and (not problem or not nim_is_hosted()):
        try:
            resp = requests.get(f"{nim_base_url()}/models", timeout=15,
                                headers={"Authorization": f"Bearer {require_nim_key()}"})
            if resp.ok:
                add("ok", "NIM online check", f"{len(resp.json().get('data', []))} models listed at {nim_base_url()}")
            else:
                add("fail", "NIM online check", f"HTTP {resp.status_code} from {nim_base_url()}/models")
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            add("fail", "NIM online check", str(exc))

    # Hardware
    hardware = hardware or hw.detect()
    if hardware.gpus:
        for g in hardware.gpus:
            used = f", {g.used_gib:.1f} GiB in use" if g.used_gib is not None else ""
            add("ok", f"GPU {g.index}", f"{g.name}: {g.total_gib:.1f} GiB VRAM{used}")
    else:
        add("warn", "GPU", f"no NVIDIA GPU detected ({hardware.gpu_note}). Models run on the CPU unless "
                           "Ollama finds another GPU; plan with: python -m src.model_manager fit --vram <GB>")
    if hardware.ram_total_gib is not None:
        free = f", {hardware.ram_available_gib:.1f} GiB free" if hardware.ram_available_gib is not None else ""
        add("info", "System RAM", f"{hardware.ram_total_gib:.1f} GiB{free}")

    # Does the default chat model fit this machine?
    chat_model = resolve_model("chat", "ollama")
    spec = next((s for s in catalog_specs() if s.name == chat_model), None)
    if spec is not None:
        fit = hw.plan_fit(spec, hardware.vram_gib, hardware.ram_total_gib, hw.DEFAULT_NUM_CTX)
        status = {"gpu": "ok", "partial": "warn", "cpu": "warn", "too_large": "fail"}[fit.status]
        add(status, "Chat model fit", f"{chat_model} at num_ctx {hw.DEFAULT_NUM_CTX}: {fit.label} (estimate)")
    return rows


def cmd_doctor(args) -> int:
    rows = run_checks(online=args.online)
    if args.json:
        print(json.dumps([{"status": s, "check": c, "detail": d} for s, c, d in rows], indent=2))
    else:
        table = Table(title="ollama-local-llm-kit doctor")
        table.add_column("Status", no_wrap=True)
        table.add_column("Check", style="bold", no_wrap=True)
        table.add_column("Detail")
        for status, check, detail in rows:
            table.add_row(_STATUS_STYLE[status], escape(check), escape(detail))
        console.print(table)
        fails = sum(1 for s, _c, _d in rows if s == "fail")
        warns = sum(1 for s, _c, _d in rows if s == "warn")
        verdict = "[green]All good.[/green]" if not fails else f"[red]{fails} problem(s) to fix.[/red]"
        console.print(f"{verdict} [dim]{warns} warning(s).[/dim]")
    return 1 if any(s == "fail" for s, _c, _d in rows) else 0


def cmd_pull(args) -> int:
    if not check_server():
        return 1
    console.print(f"Pulling [cyan]{escape(args.name)}[/cyan] ...")
    try:
        pull_model(args.name)
    except OllamaError as exc:
        console.print(f"[red]Pull failed:[/red] {escape(str(exc))}")
        return 1
    return 0


def cmd_show(args) -> int:
    if not check_server():
        return 1
    try:
        info = show_model(args.name)
    except OllamaError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return 1
    details = info.get("details", {}) or {}
    table = Table(title=escape(args.name), show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Family", details.get("family", "-"))
    table.add_row("Parameters", details.get("parameter_size", "-"))
    table.add_row("Quantization", details.get("quantization_level", "-"))
    table.add_row("Format", details.get("format", "-"))
    caps = info.get("capabilities")
    if caps:
        table.add_row("Capabilities", ", ".join(caps))
    console.print(table)

    params = info.get("parameters")
    if params:
        console.print("\n[bold]Default parameters[/bold]")
        console.print(params.strip(), markup=False, highlight=False)
    return 0


def cmd_remove(args) -> int:
    if not check_server():
        return 1
    try:
        delete_model(args.name)
    except OllamaError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return 1
    console.print(f"[green]Removed[/green] {escape(args.name)}.")
    return 0


def disk_usage(models: list[dict]) -> tuple[list[dict], int]:
    """Group installed tags by manifest digest and total the bytes on disk.

    Ollama stores content-addressed blobs, so two tags with the same digest
    (``llama3.1:latest`` and ``llama3.1:8b``, say) are one copy on disk. Summing
    ``size`` per tag double-counts them. Returns one row per digest, largest
    first (``{"names": [...], "size": int, "digest": str}``), and the real total.
    """
    groups: dict[str, dict] = {}
    for m in models:
        key = m.get("digest") or f"name:{m.get('name')}"
        group = groups.setdefault(key, {"names": [], "size": int(m.get("size", 0) or 0), "digest": m.get("digest", "")})
        group["names"].append(m.get("name", "?"))
    rows = sorted(groups.values(), key=lambda g: (-g["size"], g["names"][0]))
    for row in rows:
        row["names"].sort()
    return rows, sum(g["size"] for g in rows)


def cmd_du(_args) -> int:
    if not check_server():
        return 1
    models = list_models()
    if not models:
        console.print("No models installed, so nothing on disk.")
        return 0
    rows, total = disk_usage(models)
    table = Table(title="Disk usage by model")
    table.add_column("Model", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Shared with", style="dim")
    for row in rows:
        first, *aliases = row["names"]
        table.add_row(escape(first), human_size(row["size"]), escape(", ".join(aliases)) if aliases else "")
    table.add_section()
    table.add_row("[bold]Total on disk[/bold]", f"[bold]{human_size(total)}[/bold]", "")
    console.print(table)
    shared = sum(len(r["names"]) - 1 for r in rows)
    if shared:
        console.print(
            f"[dim]{shared} tag(s) share the same manifest digest as another tag and are counted once.[/dim]"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model_manager",
        description="Manage local Ollama models and plan them against your GPU.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def budget_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--vram", type=float, default=None, metavar="GB",
                       help="GPU memory to plan for in GiB (default: detected with nvidia-smi; 0 = CPU only)")
        p.add_argument("--ram", type=float, default=None, metavar="GB",
                       help="System RAM in GiB (default: detected)")
        p.add_argument("--ctx", type=int, default=hw.DEFAULT_NUM_CTX, metavar="N",
                       help=f"Context window num_ctx to plan for (default {hw.DEFAULT_NUM_CTX}, Ollama's default)")

    sub.add_parser("list", help="List installed models").set_defaults(func=cmd_list)
    p_rec = sub.add_parser("recommend", help="Print the curated catalog, marked against your hardware")
    budget_args(p_rec)
    p_rec.set_defaults(func=cmd_recommend)
    sub.add_parser("du", help="Show disk usage by model").set_defaults(func=cmd_du)

    p_fit = sub.add_parser("fit", help="Estimate VRAM use, CPU offload and the largest context that fits")
    p_fit.add_argument("models", nargs="*", help="Catalog or installed model names (default: the chat+vision catalog)")
    budget_args(p_fit)
    p_fit.add_argument("--installed", action="store_true", help="Plan every installed chat model (reads /api/show)")
    p_fit.add_argument("--kv-cache", default="f16", choices=sorted(hw.KV_BYTES_PER_ELEMENT),
                       help="KV cache precision, as set by OLLAMA_KV_CACHE_TYPE (default f16)")
    p_fit.add_argument("--overhead", type=float, default=hw.DEFAULT_OVERHEAD_GIB, metavar="GB",
                       help=f"Fixed allowance for CUDA context and buffers (default {hw.DEFAULT_OVERHEAD_GIB} GiB)")
    p_fit.add_argument("--json", action="store_true", help="Print the plan as JSON")
    p_fit.set_defaults(func=cmd_fit)

    p_doc = sub.add_parser("doctor", help="Check server, models, NIM key, GPU and the current CPU/GPU split")
    p_doc.add_argument("--online", action="store_true",
                       help="Also call the NIM endpoint's /models with your key (the only network call it makes)")
    p_doc.add_argument("--json", action="store_true", help="Print the checks as JSON")
    p_doc.set_defaults(func=cmd_doctor)

    p_pull = sub.add_parser("pull", help="Download a model with a progress bar")
    p_pull.add_argument("name", help="Model tag, e.g. llama3.1:8b")
    p_pull.set_defaults(func=cmd_pull)

    p_show = sub.add_parser("show", help="Show model metadata")
    p_show.add_argument("name", help="Installed model tag")
    p_show.set_defaults(func=cmd_show)

    p_remove = sub.add_parser("remove", help="Delete an installed model")
    p_remove.add_argument("name", help="Installed model tag")
    p_remove.set_defaults(func=cmd_remove)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except requests.RequestException as exc:
        console.print(f"[red]Ollama request failed:[/red] {escape(str(exc))}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
