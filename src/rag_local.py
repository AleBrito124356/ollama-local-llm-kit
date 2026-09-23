"""Fully local Retrieval-Augmented Generation.

Everything here runs on your machine: ``nomic-embed-text`` embeds the documents
via Ollama, a plain NumPy matrix stores the vectors and does cosine similarity,
and a local chat model answers using the retrieved passages. Zero cloud calls,
zero cost, and answers carry inline ``[n]`` citations back to the source files.

Run it:
    python -m src.rag_local build          # embed sample_docs/ into a cache
    python -m src.rag_local ask "How do I offload layers to the GPU?"
    python -m src.rag_local ask "What quantization should I use?" --show-context

The vector cache lives in ``.rag_cache.npz`` next to the docs and is rebuilt
automatically when a source file changes.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
from dataclasses import dataclass

import numpy as np
import openai
from rich.console import Console
from rich.markup import escape

from .client import LLMClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "sample_docs")
CACHE_PATH = os.path.join(REPO_ROOT, ".rag_cache.npz")


def cache_path() -> str:
    """Where the vector cache lives (``RAG_CACHE_DIR`` overrides the repo root)."""
    override = os.getenv("RAG_CACHE_DIR")
    return os.path.join(override, ".rag_cache.npz") if override else CACHE_PATH

console = Console()

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
TOP_K = 4


@dataclass
class Chunk:
    doc: str
    index: int
    text: str


def _read_docs() -> list[tuple[str, str]]:
    """Return a list of (filename, full_text) for every doc in the corpus."""
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")) + glob.glob(os.path.join(DOCS_DIR, "*.txt")))
    docs = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            docs.append((os.path.basename(path), fh.read()))
    return docs


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping character windows, preferring paragraph breaks."""
    text = text.strip()
    if len(text) <= CHUNK_CHARS:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        window = text[start:end]
        # Prefer to break on the last paragraph or sentence boundary in the window.
        if end < len(text):
            for sep in ("\n\n", "\n", ". "):
                cut = window.rfind(sep)
                if cut > CHUNK_CHARS // 2:
                    end = start + cut + len(sep)
                    window = text[start:end]
                    break
        cleaned = window.strip()
        if cleaned:
            chunks.append(cleaned)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def build_chunks() -> list[Chunk]:
    """Read and chunk every document in the corpus."""
    chunks: list[Chunk] = []
    for name, text in _read_docs():
        for i, piece in enumerate(_chunk_text(text)):
            chunks.append(Chunk(doc=name, index=i, text=piece))
    return chunks


def _corpus_fingerprint(chunks: list[Chunk], model: str) -> str:
    """Stable hash of the chunk texts plus embedding model, for cache validation."""
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))
    hasher.update(b"input_type=passage")  # documents are embedded as passages
    for c in chunks:
        hasher.update(c.doc.encode("utf-8"))
        hasher.update(c.text.encode("utf-8"))
    return hasher.hexdigest()


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorStore:
    """An in-memory NumPy vector store with cosine search and on-disk caching."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.embed_model = client.model_for("embed")
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None  # shape (n_chunks, dim), L2-normalized
        self.fingerprint: str = ""

    def build(self, force: bool = False) -> "VectorStore":
        """Embed the corpus, loading from cache when the fingerprint matches."""
        self.chunks = build_chunks()
        if not self.chunks:
            raise RuntimeError(f"No documents found in {DOCS_DIR}. Add some .md or .txt files.")
        self.fingerprint = _corpus_fingerprint(self.chunks, self.embed_model)

        if not force and self._load_cache():
            return self

        console.print(f"Embedding {len(self.chunks)} chunks with [cyan]{escape(self.embed_model)}[/cyan] ...")
        vectors = self.client.embed([c.text for c in self.chunks], input_type="passage")
        self.matrix = _normalize(np.array(vectors, dtype=np.float32))
        self._save_cache()
        return self

    def _load_cache(self) -> bool:
        path = cache_path()
        if not os.path.exists(path):
            return False
        try:
            # The cache only holds a float matrix and a unicode string, so pickled
            # objects are never needed; refusing them keeps a tampered cache file
            # from executing code.
            with np.load(path, allow_pickle=False) as data:
                if str(data["fingerprint"]) != self.fingerprint:
                    return False
                matrix = data["matrix"]
        except (OSError, ValueError, KeyError):
            return False
        if matrix.shape[0] != len(self.chunks):
            return False
        self.matrix = matrix
        console.print(f"[dim]Loaded {self.matrix.shape[0]} cached embeddings.[/dim]")
        return True

    def _save_cache(self) -> None:
        path = cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            np.savez(fh, matrix=self.matrix, fingerprint=np.array(self.fingerprint))

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks and their cosine scores for a query."""
        if self.matrix is None:
            raise RuntimeError("Store not built. Call build() first.")
        q_vec = np.array(self.client.embed([query], input_type="query")[0], dtype=np.float32)
        q_vec /= np.linalg.norm(q_vec) or 1.0
        scores = self.matrix @ q_vec
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


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
        lines.append(f"[{n}] ({chunk.doc}) {snippet}")
    return "\n\n".join(lines)


def answer(
    store: VectorStore,
    chat_client: LLMClient,
    question: str,
    k: int = TOP_K,
    stream: bool = True,
    hits: list[tuple[Chunk, float]] | None = None,
) -> str:
    """Retrieve (unless ``hits`` are given), build a cited prompt, and print a grounded answer."""
    if hits is None:
        hits = store.search(question, k=k)
    context = format_context(hits)
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(context=context, question=question)}]

    console.print()
    parts: list[str] = []
    # Model text is printed literally: brackets such as [1] or [/INST] are not Rich markup.
    if stream:
        for token in chat_client.stream(messages, temperature=0.2):
            parts.append(token)
            console.print(token, end="", markup=False, highlight=False, emoji=False)
        console.print("\n")
    else:
        text = chat_client.chat(messages, temperature=0.2)
        parts.append(text)
        console.print(text + "\n", markup=False, highlight=False, emoji=False)

    console.print("[dim]Sources:[/dim]")
    for n, (chunk, score) in enumerate(hits, start=1):
        console.print(f"  [dim]{escape(f'[{n}] {chunk.doc}')}  (chunk {chunk.index}, score {score:.3f})[/dim]")
    return "".join(parts)


# -- CLI ------------------------------------------------------------------
def cmd_build(args) -> int:
    embed_client = LLMClient.create("ollama")
    VectorStore(embed_client).build(force=args.force)
    console.print("[green]Index ready.[/green]")
    return 0


def cmd_ask(args) -> int:
    # Embeddings are always local; the chat model may be local or, if you insist,
    # cloud — but the RAG demo defaults to fully local for zero cost and privacy.
    embed_client = LLMClient.create("ollama")
    chat_client = LLMClient.create(args.backend)
    store = VectorStore(embed_client).build(force=False)
    hits = store.search(args.question, k=args.k)  # the question is embedded exactly once
    if args.show_context:
        console.print("[bold]Retrieved context[/bold]")
        console.print(format_context(hits), markup=False, highlight=False)
    answer(store, chat_client, args.question, k=args.k, stream=not args.no_stream, hits=hits)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag_local", description="Fully local, cited RAG over sample_docs/.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Embed the corpus into the vector cache")
    p_build.add_argument("--force", action="store_true", help="Rebuild even if the cache is current")
    p_build.set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask", help="Ask a question against the corpus")
    p_ask.add_argument("question", help="Your question")
    p_ask.add_argument("--k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    p_ask.add_argument("--backend", default="ollama", help="Chat backend for the answer (default: ollama)")
    p_ask.add_argument("--show-context", action="store_true", help="Print the retrieved passages first")
    p_ask.add_argument("--no-stream", action="store_true", help="Wait for the full answer instead of streaming")
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, ValueError, openai.APIError) as exc:
        # RuntimeError covers MissingAPIKey; APIError covers "Ollama is not running".
        console.print(f"[red]{escape(str(exc))}[/red]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
