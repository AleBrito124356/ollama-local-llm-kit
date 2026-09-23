"""Streaming chat REPL with live backend, model and RAG switching.

A small terminal chat loop that streams tokens as they arrive and lets you change
everything mid-conversation:

    /backend nim         switch to cloud NVIDIA NIM  (or  /backend ollama)
    /model qwen2.5:7b    switch the chat model on the current backend
    /rag on | off        ground answers in sample_docs/ via the local RAG store
    /models              list installed local models
    /system <text>       set the system prompt and reset history
    /reset               clear the conversation
    /help                show this help
    /exit                quit

Start it with:
    python -m src.chat
    python -m src.chat --backend nim
    python -m src.chat --rag
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.markup import escape

from .client import LLMClient, MissingAPIKey, resolve_backend

console = Console()

HELP_TEXT = """\
[bold]Commands[/bold]
  /backend <ollama|nim>   switch backend (cloud NIM needs NVIDIA_API_KEY)
  /model <name>           switch chat model on the current backend
  /rag <on|off>           toggle local retrieval over sample_docs/
  /models                 list installed local models
  /system <text>          set system prompt and reset the conversation
  /reset                  clear conversation history
  /help                   show this help
  /exit                   quit
"""


def print_plain(text: str, end: str = "\n") -> None:
    """Print model or user text literally.

    Model output routinely contains square brackets (``[INST]``, ``[/INST]``,
    ``[1]``, ``[bold]``). Rich would parse them as markup: unknown tags vanish
    and an unmatched closing tag raises ``MarkupError``, which used to crash the
    REPL mid-answer. Markup, emoji codes and auto-highlighting are off here.
    """
    console.print(text, end=end, markup=False, highlight=False, emoji=False)


def print_error(prefix: str, exc: BaseException | str) -> None:
    """Print an error whose message may contain brackets from the server or model."""
    console.print(f"[red]{prefix}[/red]{escape(str(exc))}" if prefix else f"[red]{escape(str(exc))}[/red]")


class ChatSession:
    """Holds conversation state and the active client, and renders turns."""

    def __init__(self, backend: str, system: str, rag: bool):
        self.system = system
        self.rag = rag
        self.history: list[dict] = []
        self.rag_store = None       # lazily built VectorStore
        self.rag_embed_client = None
        self.client = LLMClient.create(backend)
        self.model_override: str | None = None

    # -- state helpers ----------------------------------------------------
    @property
    def backend(self) -> str:
        return self.client.backend

    @property
    def model(self) -> str:
        return self.model_override or self.client.model_for("chat")

    def reset(self) -> None:
        self.history = []

    def _messages(self, user_text: str) -> list[dict]:
        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})
        return messages

    # -- RAG --------------------------------------------------------------
    def _ensure_rag(self):
        """Build the local vector store the first time RAG is used."""
        if self.rag_store is not None:
            return
        # Imported lazily so plain chat has no NumPy import cost.
        from .rag_local import VectorStore

        self.rag_embed_client = LLMClient.create("ollama")
        console.print("[dim]Building local RAG index ...[/dim]")
        self.rag_store = VectorStore(self.rag_embed_client).build(force=False)

    def _augment_with_rag(self, user_text: str) -> tuple[str, list]:
        """Return a context-grounded user message and the retrieved hits."""
        from .rag_local import format_context

        self._ensure_rag()
        hits = self.rag_store.search(user_text, k=4)
        context = format_context(hits)
        grounded = (
            "Answer using ONLY the numbered context below and cite sources like [1].\n\n"
            f"Context:\n{context}\n\nQuestion: {user_text}"
        )
        return grounded, hits

    # -- turn -------------------------------------------------------------
    def send(self, user_text: str) -> None:
        """Stream one assistant reply and record it in history."""
        hits = []
        if self.rag:
            try:
                prompt_text, hits = self._augment_with_rag(user_text)
            except Exception as exc:
                print_error("RAG unavailable: ", exc)
                prompt_text = user_text
        else:
            prompt_text = user_text

        messages = self._messages(prompt_text)
        console.print("[bold green]assistant[/bold green] ", end="")
        parts: list[str] = []
        try:
            for token in self.client.stream(messages, model=self.model_override):
                parts.append(token)
                print_plain(token, end="")
        except MissingAPIKey as exc:
            console.print()
            print_error("", exc)
            return
        except Exception as exc:
            console.print()
            print_error("Request failed: ", exc)
            return
        console.print()

        reply = "".join(parts)
        # Store the original user text, not the RAG-augmented prompt, so history stays clean.
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})

        if hits:
            names = ", ".join(f"[{i}] {c.doc}" for i, (c, _s) in enumerate(hits, start=1))
            console.print(f"[dim]sources: {escape(names)}[/dim]")


def handle_command(session: ChatSession, line: str) -> bool:
    """Process a /command. Return False to signal exit, True to continue."""
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        return False

    if cmd == "/help":
        console.print(HELP_TEXT)
    elif cmd == "/reset":
        session.reset()
        console.print("[dim]Conversation cleared.[/dim]")
    elif cmd == "/system":
        session.system = arg
        session.reset()
        console.print("[dim]System prompt set; conversation cleared.[/dim]")
    elif cmd == "/rag":
        if arg.lower() in ("on", "off"):
            session.rag = arg.lower() == "on"
            console.print(f"[dim]RAG {'enabled' if session.rag else 'disabled'}.[/dim]")
        else:
            console.print("Usage: /rag on | off")
    elif cmd == "/model":
        if arg:
            session.model_override = arg
            console.print(f"[dim]Chat model set to {escape(arg)}.[/dim]")
        else:
            print_plain(f"Current model: {session.model}")
    elif cmd == "/backend":
        if arg.lower() in ("ollama", "nim"):
            try:
                session.client = LLMClient.create(arg.lower())
                session.model_override = None
                console.print(
                    f"[dim]Backend switched to {session.backend}; model reset to {escape(session.model)}.[/dim]"
                )
            except MissingAPIKey as exc:
                print_error("", exc)
        else:
            console.print("Usage: /backend ollama | nim")
    elif cmd == "/models":
        _print_models()
    else:
        print_plain(f"Unknown command {cmd!r}. Try /help.")
    return True


def _print_models() -> None:
    try:
        from .model_manager import check_server, list_models

        if not check_server():
            return
        models = list_models()
        if not models:
            console.print("No local models installed.")
            return
        for m in sorted(models, key=lambda x: x.get("name", "")):
            print_plain(f"  {m.get('name')}")
    except Exception as exc:
        print_error("Could not list models: ", exc)


def banner(session: ChatSession) -> None:
    console.print("[bold]ollama-local-llm-kit[/bold] chat REPL. Type /help for commands, /exit to quit.")
    console.print(
        f"[dim]backend={session.backend}  model={escape(session.model)}  "
        f"rag={'on' if session.rag else 'off'}[/dim]\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chat", description="Streaming chat REPL with live switching.")
    parser.add_argument("--backend", default=None, help="ollama (default) or nim")
    parser.add_argument("--system", default="You are a helpful, concise assistant.", help="Initial system prompt")
    parser.add_argument("--rag", action="store_true", help="Start with local RAG enabled")
    args = parser.parse_args(argv)

    try:
        backend = resolve_backend(args.backend)
    except ValueError as exc:
        print_error("", exc)
        return 1

    try:
        session = ChatSession(backend=backend, system=args.system, rag=args.rag)
    except MissingAPIKey as exc:
        print_error("", exc)
        return 1

    banner(session)

    while True:
        try:
            line = console.input("[bold blue]you[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            return 0

        if not line.strip():
            continue
        if line.startswith("/"):
            if not handle_command(session, line):
                console.print("[dim]Bye.[/dim]")
                return 0
            continue

        session.send(line)


if __name__ == "__main__":
    raise SystemExit(main())
