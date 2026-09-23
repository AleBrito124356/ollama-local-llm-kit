"""Fully local Retrieval-Augmented Generation over your own documents.

Everything here runs on your machine: ``nomic-embed-text`` embeds the documents
via Ollama, a plain NumPy matrix stores the vectors and does cosine similarity,
and a local chat model answers using the retrieved passages. Zero cloud calls,
zero cost, and answers carry ``[n]`` citations that resolve to exact line ranges
(``gpu_offloading.md:L9-L16``).

Run it:
    python -m src.rag_local build                          # sample_docs/
    python -m src.rag_local build --docs ~/notes           # your own folder (recursive)
    python -m src.rag_local ask "How do I offload layers to the GPU?"
    python -m src.rag_local ask "What is our retry policy?" --docs ./docs --min-score 0.35
    python -m src.rag_local ask "..." --json               # machine-readable answer + sources
    python -m src.rag_local search "num_ctx" --k 3         # retrieval only, no chat model

Each corpus gets its own cache under ``.rag_cache/`` (or ``RAG_CACHE_DIR``). The
cache stores one vector per chunk, keyed by a hash of the chunk's text, so a
rebuild only embeds chunks that are new or changed and drops the rows of chunks
that disappeared. Changing the embedding model or backend re-embeds everything.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import openai
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from .client import LLMClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "sample_docs")

console = Console()

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
TOP_K = 4
DEFAULT_BATCH = 32
DEFAULT_GLOBS = ("*.md", "*.txt", "*.rst")
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__", ".rag_cache",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build", "site-packages",
})
CACHE_VERSION = 2
NOT_FOUND = "Not found in the documents."


# -- documents and chunks ---------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    doc: str          # path relative to the corpus root, with forward slashes
    index: int        # position of the chunk within its document
    text: str
    start_line: int   # 1-based, inclusive
    end_line: int     # 1-based, inclusive

    @property
    def cite(self) -> str:
        """Citation that points at the exact lines, e.g. ``gpu_offloading.md:L9-L16``."""
        return f"{self.doc}:L{self.start_line}-L{self.end_line}"

    @property
    def key(self) -> str:
        """Content hash; identical text is embedded once and reused across builds."""
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()


def cache_dir() -> str:
    return os.getenv("RAG_CACHE_DIR") or os.path.join(REPO_ROOT, ".rag_cache")


def corpus_cache_path(root: str) -> str:
    """One cache file per corpus, named after a hash of its absolute path."""
    norm = os.path.normcase(os.path.abspath(root))
    return os.path.join(cache_dir(), hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16] + ".npz")


def parse_globs(values: list[str] | None) -> tuple[str, ...]:
    """``--glob '*.md,*.txt' --glob '*.rst'`` -> ('*.md', '*.txt', '*.rst')."""
    if not values:
        return DEFAULT_GLOBS
    globs = tuple(g.strip() for value in values for g in value.split(",") if g.strip())
    return globs or DEFAULT_GLOBS


def find_documents(root: str, globs: tuple[str, ...] = DEFAULT_GLOBS) -> list[str]:
    """Absolute paths of the files under ``root`` whose name or relative path matches a glob.

    ``root`` may also be a single file. Hidden directories and tool/vendor folders
    (``.git``, ``node_modules``, ``.venv``, ...) are skipped.
    """
    root = os.path.abspath(os.path.expanduser(root))
    if os.path.isfile(root):
        return [root]
    if not os.path.isdir(root):
        raise RuntimeError(f"Document path not found: {root}")
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
            if any(fnmatch.fnmatch(name.lower(), g.lower()) or fnmatch.fnmatch(rel.lower(), g.lower())
                   for g in globs):
                found.append(os.path.join(dirpath, name))
    return found


def _read_text(path: str) -> str | None:
    with open(path, "rb") as fh:
        raw = fh.read()
    if b"\x00" in raw[:8192]:
        return None  # binary file that happens to match the glob
    return raw.decode("utf-8", errors="replace")


def chunk_document(text: str, max_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int, str]]:
    """Split a document into overlapping chunks of whole lines.

    Returns ``(start_line, end_line, text)`` with 1-based inclusive line numbers,
    so every chunk can be cited as an exact line range. A chunk grows line by
    line up to ``max_chars`` and prefers to end before a blank line or a Markdown
    heading once it is at least half full. The next chunk starts up to
    ``overlap`` characters of whole lines earlier, so a sentence cut at a
    boundary still appears whole in one chunk. A single line longer than
    ``max_chars`` is split by characters and cited as that one line.
    """
    lines = text.splitlines()
    n = len(lines)
    pieces: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        if len(lines[i]) > max_chars:
            line = lines[i]
            step = max(max_chars - overlap, 1)
            for start in range(0, len(line), step):
                part = line[start:start + max_chars].strip()
                if part:
                    pieces.append((i + 1, i + 1, part))
                if start + max_chars >= len(line):
                    break
            i += 1
            continue

        j, size, best_break = i, 0, None
        while j < n and len(lines[j]) <= max_chars:
            grown = size + (1 if j > i else 0) + len(lines[j])
            if grown > max_chars:
                break
            size = grown
            j += 1
            nxt = lines[j] if j < n else ""
            if j < n and (not nxt.strip() or nxt.lstrip().startswith("#")) and size >= max_chars // 2:
                best_break = j
        if j < n and best_break is not None:
            j = best_break
        end = j
        while end > i and not lines[end - 1].strip():
            end -= 1
        pieces.append((i + 1, end, "\n".join(lines[i:end]).strip()))
        if j >= n:
            break
        back, carried = j, 0
        while back - 1 > i and carried + len(lines[back - 1]) + 1 <= overlap:
            back -= 1
            carried += len(lines[back]) + 1
        i = back if back > i else j
    return pieces


def build_chunks(root: str = DOCS_DIR, globs: tuple[str, ...] = DEFAULT_GLOBS) -> list[Chunk]:
    """Read and chunk every matching document under ``root``."""
    root = os.path.abspath(os.path.expanduser(root))
    base = os.path.dirname(root) if os.path.isfile(root) else root
    chunks: list[Chunk] = []
    for path in find_documents(root, globs):
        text = _read_text(path)
        if not text:
            continue
        rel = os.path.relpath(path, base).replace(os.sep, "/")
        for index, (start, end, piece) in enumerate(chunk_document(text)):
            chunks.append(Chunk(doc=rel, index=index, text=piece, start_line=start, end_line=end))
    return chunks


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# -- vector store -----------------------------------------------------------------
@dataclass
class BuildReport:
    files: int
    chunks: int
    embedded: int     # texts sent to the embedding model in this build
    reused: int       # chunks whose vector came from the cache
    pruned: int       # cached vectors dropped (changed or deleted text, or a new model)
    cache_path: str


class VectorStore:
    """An in-memory NumPy vector store with cosine search and an incremental on-disk cache."""

    def __init__(
        self,
        client: LLMClient,
        docs: str | None = None,
        globs: tuple[str, ...] | list[str] | None = None,
        batch_size: int = DEFAULT_BATCH,
        out: Console | None = None,
    ):
        self.client = client
        self.docs_root = os.path.abspath(os.path.expanduser(docs or DOCS_DIR))
        self.globs = tuple(globs or DEFAULT_GLOBS)
        self.batch_size = max(1, int(batch_size))
        self.embed_model = client.model_for("embed")
        self.out = out or console
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None  # shape (n_chunks, dim), L2-normalized
        self.report: BuildReport | None = None

    @property
    def cache_path(self) -> str:
        return corpus_cache_path(self.docs_root)

    def _meta(self) -> dict:
        return {
            "version": CACHE_VERSION,
            "corpus": self.docs_root,
            "embed_model": self.embed_model,
            "embed_backend": self.client.backend,
            "input_type": "passage",
        }

    def build(self, force: bool = False) -> "VectorStore":
        """Embed the corpus, reusing every cached vector whose chunk text is unchanged."""
        self.chunks = build_chunks(self.docs_root, self.globs)
        if not self.chunks:
            raise RuntimeError(
                f"No documents matching {', '.join(self.globs)} found in {self.docs_root}."
            )
        keys = [c.key for c in self.chunks]
        cached, stale = ({}, 0) if force else self._load_cache()

        todo: dict[str, str] = {}
        for chunk, key in zip(self.chunks, keys):
            if key not in cached and key not in todo:
                todo[key] = chunk.text
        fresh = self._embed_batches(list(todo.items())) if todo else {}

        vectors = {**cached, **fresh}
        self.matrix = _normalize(np.array([vectors[k] for k in keys], dtype=np.float32))
        live = set(keys)
        self.report = BuildReport(
            files=len({c.doc for c in self.chunks}),
            chunks=len(self.chunks),
            embedded=len(fresh),
            reused=sum(1 for k in keys if k in cached),
            pruned=stale + sum(1 for k in cached if k not in live),
            cache_path=self.cache_path,
        )
        if fresh or self.report.pruned or not os.path.exists(self.cache_path):
            self._save_cache(keys)
        r = self.report
        self.out.print(
            f"[dim]{r.files} file(s), {r.chunks} chunks: embedded {r.embedded}, reused {r.reused}, "
            f"pruned {r.pruned}  ({escape(self.embed_model)} via {self.client.backend})[/dim]"
        )
        return self

    def _embed_batches(self, items: list[tuple[str, str]]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        with Progress(
            TextColumn("[bold blue]embedding"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("chunks"),
            console=self.out,
            transient=True,
        ) as progress:
            task = progress.add_task("embed", total=len(items))
            for batch in batches:
                vectors = self.client.embed([text for _k, text in batch], input_type="passage")
                if len(vectors) != len(batch):
                    raise RuntimeError(f"embedding server returned {len(vectors)} vectors for {len(batch)} inputs")
                for (key, _text), vec in zip(batch, vectors):
                    out[key] = np.asarray(vec, dtype=np.float32)
                progress.advance(task, len(batch))
        return out

    def _load_cache(self) -> tuple[dict[str, np.ndarray], int]:
        """``({chunk key: vector}, stale rows)`` from this corpus's cache.

        Vectors made by another embedding model or backend live in a different
        space, so none of them are reused; they are reported as stale (pruned).
        """
        path = self.cache_path
        if not os.path.exists(path):
            return {}, 0
        try:
            # Only numeric and unicode arrays are stored, so pickled objects are
            # never needed; refusing them keeps a tampered file from running code.
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                keys = [str(k) for k in data["keys"]]
                matrix = np.asarray(data["matrix"], dtype=np.float32)
        except (OSError, ValueError, KeyError, TypeError):
            return {}, 0
        if not isinstance(meta, dict) or meta.get("version") != CACHE_VERSION:
            return {}, 0
        if matrix.ndim != 2 or matrix.shape[0] != len(keys):
            return {}, 0
        if (meta.get("embed_model"), meta.get("embed_backend")) != (self.embed_model, self.client.backend):
            return {}, len(keys)
        return dict(zip(keys, matrix)), 0

    def _save_cache(self, keys: list[str]) -> None:
        path = self.cache_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, meta=np.array(json.dumps(self._meta())), keys=np.array(keys), matrix=self.matrix)
        os.replace(tmp, path)

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks and their cosine scores for a query (one embedding call)."""
        if self.matrix is None:
            raise RuntimeError("Store not built. Call build() first.")
        q_vec = np.array(self.client.embed([query], input_type="query")[0], dtype=np.float32)
        q_vec /= np.linalg.norm(q_vec) or 1.0
        scores = self.matrix @ q_vec
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


# -- answering --------------------------------------------------------------------
PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question using ONLY the numbered context
below. Cite the sources you use with bracketed numbers like [1] or [2]. If the
context does not contain the answer, say you do not know.

Context:
{context}

Question: {question}

Answer:"""


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks as a numbered, citable context block."""
    lines = []
    for n, (chunk, _score) in enumerate(hits, start=1):
        snippet = " ".join(chunk.text.split())
        lines.append(f"[{n}] ({chunk.cite}) {snippet}")
    return "\n\n".join(lines)


@dataclass
class RagAnswer:
    question: str
    text: str
    hits: list[tuple[Chunk, float]]
    found: bool

    def sources(self) -> list[dict]:
        return [
            {"n": n, "file": c.doc, "lines": [c.start_line, c.end_line], "cite": c.cite, "score": round(s, 4)}
            for n, (c, s) in enumerate(self.hits, start=1)
        ]


def ask(
    store: VectorStore,
    chat_client: LLMClient,
    question: str,
    k: int = TOP_K,
    min_score: float | None = None,
    stream: bool = True,
    out: Console | None = None,
    hits: list[tuple[Chunk, float]] | None = None,
) -> RagAnswer:
    """Retrieve, then answer from the retrieved passages.

    If ``min_score`` is set and even the best passage scores below it, the
    documents do not cover the question: :data:`NOT_FOUND` is returned without
    calling the chat model at all. With ``out=None`` nothing is printed.
    """
    if hits is None:
        hits = store.search(question, k=k)
    if not hits or (min_score is not None and hits[0][1] < min_score):
        if out is not None:
            out.print()
            out.print(NOT_FOUND, markup=False, highlight=False)
        return RagAnswer(question, NOT_FOUND, hits, found=False)

    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(context=format_context(hits), question=question)}]
    parts: list[str] = []
    if out is not None:
        out.print()
    # Model text is printed literally: brackets such as [1] or [/INST] are not Rich markup.
    if stream:
        for token in chat_client.stream(messages, temperature=0.2):
            parts.append(token)
            if out is not None:
                out.print(token, end="", markup=False, highlight=False, emoji=False)
        if out is not None:
            out.print()
    else:
        parts.append(chat_client.chat(messages, temperature=0.2))
        if out is not None:
            out.print(parts[-1], markup=False, highlight=False, emoji=False)
    return RagAnswer(question, "".join(parts), hits, found=True)


def print_sources(hits: list[tuple[Chunk, float]], out: Console | None = None) -> None:
    out = out or console
    out.print("\n[dim]Sources:[/dim]")
    for n, (chunk, score) in enumerate(hits, start=1):
        out.print(f"  [dim]{escape(f'[{n}] {chunk.cite}')}  score {score:.3f}[/dim]")


def answer(
    store: VectorStore,
    chat_client: LLMClient,
    question: str,
    k: int = TOP_K,
    stream: bool = True,
    hits: list[tuple[Chunk, float]] | None = None,
) -> str:
    """Retrieve, print a grounded answer and its sources, and return the answer text."""
    result = ask(store, chat_client, question, k=k, stream=stream, out=console, hits=hits)
    print_sources(result.hits)
    return result.text


# -- CLI ------------------------------------------------------------------------------
def _store_from_args(args, out: Console) -> VectorStore:
    embed_client = LLMClient.create(args.embed_backend)
    return VectorStore(embed_client, docs=args.docs, globs=parse_globs(args.glob), batch_size=args.batch, out=out)


def cmd_build(args) -> int:
    store = _store_from_args(args, console).build(force=args.force)
    console.print(f"[green]Index ready.[/green] [dim]{escape(store.cache_path)}[/dim]")
    return 0


def cmd_search(args) -> int:
    out = Console(stderr=True) if args.json else console
    store = _store_from_args(args, out).build()
    hits = store.search(args.query, k=args.k)
    if args.json:
        print(json.dumps({"query": args.query, "results": RagAnswer(args.query, "", hits, True).sources()}, indent=2))
        return 0
    for n, (chunk, score) in enumerate(hits, start=1):
        console.print(f"[bold]{escape(f'[{n}] {chunk.cite}')}[/bold]  score {score:.3f}")
        console.print(chunk.text, markup=False, highlight=False)
        console.print()
    return 0


def cmd_ask(args) -> int:
    # Embeddings default to local Ollama; the chat model may be local or cloud.
    out = Console(stderr=True) if args.json else console
    store = _store_from_args(args, out).build()
    chat_client = LLMClient.create(args.backend)
    hits = store.search(args.question, k=args.k)  # the question is embedded exactly once
    if args.show_context and not args.json:
        console.print("[bold]Retrieved context[/bold]")
        console.print(format_context(hits), markup=False, highlight=False)
    result = ask(
        store, chat_client, args.question, k=args.k, min_score=args.min_score,
        stream=not (args.no_stream or args.json), out=None if args.json else console, hits=hits,
    )
    if args.json:
        print(json.dumps({
            "question": result.question,
            "answer": result.text,
            "found": result.found,
            "backend": chat_client.backend,
            "model": chat_client.model_for("chat"),
            "embed_model": store.embed_model,
            "sources": result.sources(),
        }, indent=2))
        return 0
    print_sources(result.hits)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag_local", description="Local, cited RAG over sample_docs/ or your own docs.")
    sub = parser.add_subparsers(dest="command", required=True)

    def corpus_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--docs", default=None, help="Folder (searched recursively) or file to index (default: sample_docs/)")
        p.add_argument("--glob", action="append", help="File patterns, comma-separated or repeated (default: *.md,*.txt,*.rst)")
        p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Chunks per embedding request (default 32)")
        p.add_argument("--embed-backend", default="ollama", choices=("ollama", "nim"),
                       help="Backend for embeddings (default: ollama, fully local)")

    p_build = sub.add_parser("build", help="Embed the corpus into its vector cache (incremental)")
    corpus_args(p_build)
    p_build.add_argument("--force", action="store_true", help="Re-embed everything, ignoring the cache")
    p_build.set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask", help="Ask a question against the corpus")
    p_ask.add_argument("question", help="Your question")
    corpus_args(p_ask)
    p_ask.add_argument("--k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    p_ask.add_argument("--backend", default="ollama", help="Chat backend for the answer (default: ollama)")
    p_ask.add_argument("--min-score", type=float, default=None,
                       help="If the best passage scores below this, answer 'not found' without calling the chat model")
    p_ask.add_argument("--show-context", action="store_true", help="Print the retrieved passages first")
    p_ask.add_argument("--no-stream", action="store_true", help="Wait for the full answer instead of streaming")
    p_ask.add_argument("--json", action="store_true", help="Print {answer, sources} as JSON (status goes to stderr)")
    p_ask.set_defaults(func=cmd_ask)

    p_search = sub.add_parser("search", help="Show the best-matching passages without calling a chat model")
    p_search.add_argument("query", help="Search text")
    corpus_args(p_search)
    p_search.add_argument("--k", type=int, default=TOP_K, help="Number of chunks to show")
    p_search.add_argument("--json", action="store_true", help="Print results as JSON")
    p_search.set_defaults(func=cmd_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, ValueError, openai.APIError) as exc:
        # RuntimeError covers MissingAPIKey; APIError covers "Ollama is not running".
        Console(stderr=True).print(f"[red]{escape(str(exc))}[/red]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
