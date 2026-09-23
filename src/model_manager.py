"""Manage local Ollama models from the command line.

Wraps the Ollama native REST API (``/api/tags``, ``/api/pull``, ``/api/show``,
``/api/delete``) so you can list, pull with a live progress bar, inspect, remove
and total up disk usage without leaving Python. The ``recommend`` command prints
the curated ``models.yaml`` table with a one-line note per model.

Usage:
    python -m src.model_manager list
    python -m src.model_manager recommend
    python -m src.model_manager pull llama3.1:8b
    python -m src.model_manager show llama3.1:8b
    python -m src.model_manager remove llava:7b
    python -m src.model_manager du
"""

from __future__ import annotations

import argparse
import json
import os

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

from .client import ollama_host

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


def cmd_recommend(_args) -> int:
    catalog = load_catalog()
    for section in ("chat", "embed", "vision"):
        entries = catalog.get(section, [])
        if not entries:
            continue
        table = Table(title=f"Recommended {section} models")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Params", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Min VRAM", justify="right")
        table.add_column("Tools", justify="center")
        table.add_column("Notes")
        for e in entries:
            table.add_row(
                e["name"],
                str(e.get("params", "-")),
                f"{e.get('size_gb', 0):.1f} GB",
                f"{e.get('min_vram_gb', '-')} GB",
                "yes" if e.get("tools") else "-",
                e.get("note", ""),
            )
        console.print(table)
    console.print("\nPull one with: [bold]python -m src.model_manager pull <model>[/bold]")
    return 0


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
        description="Manage local Ollama models: list, pull, show, remove, disk usage.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List installed models").set_defaults(func=cmd_list)
    sub.add_parser("recommend", help="Print the curated model catalog").set_defaults(func=cmd_recommend)
    sub.add_parser("du", help="Show disk usage by model").set_defaults(func=cmd_du)

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
