"""Local RAG over the fake server: retrieval quality, caching, citations and safe output."""

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pytest

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
                     keys=np.array([Boom(str(marker))], dtype=object),
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
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err


# -- bring-your-own-docs, incremental cache, line citations ---------------------------------
def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def paragraph(topic: str, n: int = 12) -> str:
    return "\n".join(f"Line {i} about {topic} and how {topic} behaves under load number {i}." for i in range(n))


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    write(root / "alpha.md", "# Alpha\n\n" + paragraph("alpha caching") + "\n\n" + paragraph("alpha retries"))
    write(root / "nested" / "beta.txt", paragraph("beta sockets"))
    write(root / "nested" / "deeper" / "gamma.rst", paragraph("gamma quotas"))
    write(root / "ignored.py", "print('not a doc')\n")
    write(root / ".hidden" / "secret.md", "hidden folder, never indexed\n")
    write(root / "node_modules" / "pkg" / "README.md", "vendored, never indexed\n")
    return root


def embedded_inputs(fake) -> list[str]:
    return [t for r in fake.requests("/v1/embeddings") for t in r["body"]["input"]]


def test_chunks_cover_every_line_with_exact_ranges():
    text = "\n".join(f"line {i} " + "x" * (i % 70) for i in range(1, 200))
    lines = text.splitlines()
    pieces = rag_local.chunk_document(text, max_chars=300, overlap=60)
    covered = set()
    for start, end, chunk_text in pieces:
        assert 1 <= start <= end <= len(lines)
        assert chunk_text == "\n".join(lines[start - 1:end]).strip()
        assert len(chunk_text) <= 300
        covered.update(range(start, end + 1))
    assert covered == set(range(1, len(lines) + 1))
    # Consecutive chunks overlap (or at least touch) so no sentence is lost at a boundary.
    for (s1, e1, _), (s2, _e2, _) in zip(pieces, pieces[1:]):
        assert s1 < s2 <= e1 + 1


def test_chunker_prefers_paragraph_breaks_and_splits_giant_lines():
    para = "\n".join(f"sentence number {i} in a paragraph of text." for i in range(12))
    pieces = rag_local.chunk_document(para + "\n\n" + para, max_chars=600, overlap=0)
    assert pieces[0][1] == 12  # ends exactly at the end of the first paragraph
    giant = "y" * 2000
    split = rag_local.chunk_document("intro\n" + giant + "\noutro", max_chars=900, overlap=100)
    assert [p[:2] for p in split] == [(1, 1), (2, 2), (2, 2), (2, 2), (3, 3)]
    assert all(p[2] in ("intro", "outro") or set(p[2]) == {"y"} for p in split)
    assert rag_local.chunk_document("\n\n  \n") == []


def test_find_documents_walks_recursively_and_skips_vendor_dirs(corpus):
    found = [os.path.relpath(p, corpus).replace(os.sep, "/") for p in rag_local.find_documents(str(corpus))]
    assert found == ["alpha.md", "nested/beta.txt", "nested/deeper/gamma.rst"]
    only_md = rag_local.find_documents(str(corpus), rag_local.parse_globs(["*.md"]))
    assert [os.path.basename(p) for p in only_md] == ["alpha.md"]
    assert rag_local.parse_globs(["*.md,*.txt", "*.rst"]) == ("*.md", "*.txt", "*.rst")
    with pytest.raises(RuntimeError, match="not found"):
        rag_local.find_documents(str(corpus / "missing"))


def test_incremental_rebuild_embeds_only_what_changed(fake, corpus):
    client = LLMClient.create("ollama")
    store = rag_local.VectorStore(client, docs=str(corpus)).build()
    n = store.report.chunks
    assert store.report.embedded == n == len(embedded_inputs(fake)), "first build embeds every chunk"
    assert store.report.files == 3

    # Unchanged corpus: nothing is embedded again.
    before = len(embedded_inputs(fake))
    again = rag_local.VectorStore(client, docs=str(corpus)).build()
    assert again.report.embedded == 0 and again.report.reused == n
    assert len(embedded_inputs(fake)) == before

    # Edit one file: only that file's chunks go to the embedding model.
    beta = corpus / "nested" / "beta.txt"
    beta.write_text(paragraph("beta sockets") + "\nA brand new closing line about beta.", encoding="utf-8")
    edited = rag_local.VectorStore(client, docs=str(corpus)).build()
    new_inputs = embedded_inputs(fake)[before:]
    beta_texts = {c.text for c in edited.chunks if c.doc == "nested/beta.txt"}
    assert new_inputs and all(t.removeprefix("search_document: ") in beta_texts for t in new_inputs)
    assert edited.report.embedded == len(new_inputs) <= len(beta_texts)
    assert edited.report.pruned >= 1

    # Delete a file: its rows are pruned from the cache and never searched again.
    gamma_chunks = sum(1 for c in edited.chunks if c.doc == "nested/deeper/gamma.rst")
    (corpus / "nested" / "deeper" / "gamma.rst").unlink()
    pruned = rag_local.VectorStore(client, docs=str(corpus)).build()
    assert pruned.report.pruned == gamma_chunks and pruned.report.embedded == 0
    assert pruned.matrix.shape[0] == pruned.report.chunks == edited.report.chunks - gamma_chunks
    assert all(c.doc != "nested/deeper/gamma.rst" for c in pruned.chunks)


def test_each_corpus_has_its_own_cache(fake, corpus):
    client = LLMClient.create("ollama")
    a = rag_local.VectorStore(client, docs=str(corpus)).build()
    b = rag_local.VectorStore(client).build()  # sample_docs
    assert a.cache_path != b.cache_path
    assert os.path.exists(a.cache_path) and os.path.exists(b.cache_path)


def test_changing_embedding_model_reembeds_everything(fake, corpus, monkeypatch):
    client = LLMClient.create("ollama")
    first = rag_local.VectorStore(client, docs=str(corpus)).build()
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "bge-m3")
    second = rag_local.VectorStore(client, docs=str(corpus)).build()
    assert second.embed_model == "bge-m3"
    assert second.report.embedded == first.report.chunks and second.report.reused == 0
    assert second.report.pruned == first.report.chunks
    assert second.matrix.shape[1] == 1024


def test_batches_respect_batch_size(fake, corpus):
    store = rag_local.VectorStore(LLMClient.create("ollama"), docs=str(corpus), batch_size=2).build()
    sizes = [len(r["body"]["input"]) for r in fake.requests("/v1/embeddings")]
    assert max(sizes) == 2 and sum(sizes) == store.report.chunks
    assert len(sizes) == -(-store.report.chunks // 2)


def test_citations_point_at_the_right_lines(fake):
    store = rag_local.VectorStore(LLMClient.create("ollama")).build()
    for chunk, _score in store.search("num_gpu layers offload", k=4):
        lines = (Path(rag_local.DOCS_DIR) / chunk.doc).read_text(encoding="utf-8").splitlines()
        assert chunk.text == "\n".join(lines[chunk.start_line - 1:chunk.end_line]).strip()
        assert chunk.cite == f"{chunk.doc}:L{chunk.start_line}-L{chunk.end_line}"
    top = store.search("How do I offload layers to the GPU with num_gpu?", k=1)[0][0]
    assert top.doc == "gpu_offloading.md" and "num_gpu" in top.text


def test_min_score_skips_the_chat_model(fake, capsys):
    code = rag_local.main(["ask", "zebra migration patterns in the serengeti", "--min-score", "0.3"])
    assert code == 0
    assert rag_local.NOT_FOUND in capsys.readouterr().out
    assert fake.requests("/v1/chat/completions") == []


def test_ask_json_schema(fake, capsys):
    code = rag_local.main(["ask", "How do I offload layers to the GPU with num_gpu?", "--json", "--k", "3"])
    captured = capsys.readouterr()
    assert code == 0
    data = json.loads(captured.out)  # stdout is pure JSON; status lines went to stderr
    assert set(data) == {"question", "answer", "found", "backend", "model", "embed_model", "sources"}
    assert data["found"] is True and data["backend"] == "ollama" and data["model"] == "llama3.1:8b"
    assert len(data["sources"]) == 3
    src = data["sources"][0]
    assert set(src) == {"n", "file", "lines", "cite", "score"}
    assert src["file"] == "gpu_offloading.md"
    assert src["cite"] == f"gpu_offloading.md:L{src['lines'][0]}-L{src['lines'][1]}"
    assert re.search(r"\[\d\]", data["answer"])


def test_ask_json_not_found(fake, capsys):
    rag_local.main(["ask", "zebra migration", "--json", "--min-score", "0.9"])
    data = json.loads(capsys.readouterr().out)
    assert data["found"] is False and data["answer"] == rag_local.NOT_FOUND


def test_search_command(fake, corpus, capsys):
    code = rag_local.main(["search", "gamma quotas", "--docs", str(corpus), "--k", "1", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and data["results"][0]["file"] == "nested/deeper/gamma.rst"
    assert fake.requests("/v1/chat/completions") == []


def test_build_cli_on_your_own_folder(fake, corpus, run_cli):
    out = run_cli("-m", "src.rag_local", "build", "--docs", str(corpus), "--glob", "*.md,*.txt")
    assert out.returncode == 0, out.stderr
    assert "2 file(s)" in out.stdout and "Index ready" in out.stdout


def test_empty_corpus_is_a_clean_error(fake, tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert rag_local.main(["build", "--docs", str(tmp_path / "empty")]) == 1
    assert "No documents matching" in capsys.readouterr().err


def test_chat_rag_over_custom_folder(fake, corpus, capsys):
    from src.chat import ChatSession, handle_command

    s = ChatSession("ollama", system="", rag=False)
    handle_command(s, f"/rag on {corpus}")
    assert s.rag and s.rag_docs == str(corpus)
    s.send("Tell me about gamma quotas")
    out = capsys.readouterr().out
    assert "sources: [1] nested/deeper/gamma.rst:L" in out
