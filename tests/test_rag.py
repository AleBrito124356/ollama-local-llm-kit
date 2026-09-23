"""Local RAG over the fake server: retrieval quality, caching and safe output."""

from __future__ import annotations

import os
import pickle

import numpy as np

from src import rag_local
from src.client import LLMClient


class Boom:
    """Unpickling this would create a marker file: proof that pickle ran."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def test_num_gpu_question_ranks_gpu_offloading_first(fake):
    store = rag_local.VectorStore(LLMClient.create("ollama")).build()
    hits = store.search("How do I offload layers to the GPU with num_gpu?", k=4)
    assert hits[0][0].doc == "gpu_offloading.md"
    assert [s for _c, s in hits] == sorted((s for _c, s in hits), reverse=True)


def test_ask_embeds_the_question_once(fake, capsys):
    """Regression: `ask` searched twice, embedding the question twice per question."""
    assert rag_local.main(["build"]) == 0
    before = len(fake.requests("/v1/embeddings"))
    assert rag_local.main(["ask", "How do I offload layers to the GPU?", "--show-context"]) == 0
    assert len(fake.requests("/v1/embeddings")) - before == 1
    assert len(fake.requests("/v1/chat/completions")) == 1
    out = capsys.readouterr().out
    assert "Retrieved context" in out and "Sources:" in out


def test_documents_and_queries_use_their_input_type(fake):
    store = rag_local.VectorStore(LLMClient.create("ollama")).build()
    store.search("quantization", k=1)
    inputs = [r["body"]["input"] for r in fake.requests("/v1/embeddings")]
    assert all(t.startswith("search_document: ") for t in inputs[0])
    assert inputs[-1] == ["search_query: quantization"]


def test_cache_is_reused_and_never_unpickled(fake, tmp_path):
    client = LLMClient.create("ollama")
    rag_local.VectorStore(client).build()
    n = len(fake.requests("/v1/embeddings"))
    rag_local.VectorStore(client).build()
    assert len(fake.requests("/v1/embeddings")) == n, "second build must come from the cache"

    marker = tmp_path / "pwned.txt"
    cache_files = [os.path.join(root, f) for root, _d, files in os.walk(os.environ["RAG_CACHE_DIR"])
                   for f in files if f.endswith(".npz")]
    assert cache_files
    for path in cache_files:
        with open(path, "wb") as fh:
            np.savez(fh, matrix=np.array([Boom(str(marker))], dtype=object),
                     fingerprint=np.array([Boom(str(marker))], dtype=object),
                     meta=np.array([Boom(str(marker))], dtype=object))
    assert b"pwned" in pickle.dumps(Boom(str(marker)))  # sanity: the payload is really there
    store = rag_local.VectorStore(client).build()
    assert not marker.exists(), "a tampered cache must not execute pickled code"
    assert store.matrix.dtype == np.float32  # it rebuilt from the documents instead


def test_answer_prints_model_text_literally(fake, capsys):
    class Scripted:
        def stream(self, messages, **_kw):
            yield from ["Use num_gpu ", "[/INST]", " [bold]now[/bold] [2]"]

    store = rag_local.VectorStore(LLMClient.create("ollama")).build()
    text = rag_local.answer(store, Scripted(), "offload layers?")
    assert text == "Use num_gpu [/INST] [bold]now[/bold] [2]"
    assert "Use num_gpu [/INST] [bold]now[/bold] [2]" in capsys.readouterr().out


def test_ask_with_ollama_down_is_a_clean_error(capsys):
    assert rag_local.main(["ask", "anything"]) == 1
    assert "Traceback" not in capsys.readouterr().out
