"""Benchmark: honest numbers, errors reported as errors, comparable local vs NIM rates."""

from __future__ import annotations

import pytest

from src import benchmark as bm


def test_local_metrics_come_from_ollama(fake):
    r = bm.bench_ollama("llama3.1:8b", "hi", 64)
    assert r.error is None
    assert r.tokens == 64
    assert r.decode_tps == pytest.approx(52.5, rel=1e-6)  # the fake's canned eval rate
    assert r.load_s == pytest.approx(2.1)
    assert r.ttft_s is not None and r.total_s >= r.ttft_s
    assert r.tokens_per_sec == r.decode_tps  # old attribute name still works


def test_streamed_error_event_is_an_error_row(fake):
    """Regression: an {"error": ...} event inside an HTTP 200 stream was reported as 0 tok/s success."""
    r = bm.bench_ollama("oom:qwen2.5:14b", "hi", 50)
    assert r.error == bm_oom()
    assert r.tokens == 0


def bm_oom() -> str:
    from src.fake_ollama import OOM_MESSAGE

    return OOM_MESSAGE


def test_http_error_body_is_surfaced(fake):
    r = bm.bench_ollama("http500:llama3.1:8b", "hi", 50)
    assert r.error == bm_oom()
    r = bm.bench_ollama("ghost:1b", "hi", 50)
    assert "not found" in r.error


def test_malformed_ndjson_is_an_error_not_a_crash(fake):
    """Regression: a bad NDJSON line raised an uncaught JSONDecodeError."""
    r = bm.bench_ollama("garbage:llama3.1:8b", "hi", 50)
    assert r.error.startswith("malformed line")


def test_nim_decode_rate_excludes_time_to_first_token():
    """Regression: NIM tok/s divided by wall time including TTFT (39 tok/s instead of ~200)."""
    arrivals = [2.0 + 0.005 * i for i in range(100)]  # 2 s TTFT, then 100 tokens 5 ms apart
    stats = bm.stream_stats(0.0, arrivals, end=arrivals[-1], usage_tokens=100)
    assert stats["ttft_s"] == pytest.approx(2.0)
    assert stats["decode_tps"] == pytest.approx(200.0)
    assert stats["e2e_tps"] == pytest.approx(100 / 2.495)
    # Without a usage block, each text chunk counts as one token.
    assert bm.stream_stats(0.0, arrivals, arrivals[-1], None)["tokens"] == 100
    assert bm.stream_stats(0.0, [], 1.0, None)["decode_tps"] == 0.0


def test_nim_over_http_with_real_delays(fake_nim):
    fake_nim.state.first_token_delay_s = 0.4
    fake_nim.state.token_delay_s = 0.002
    r = bm.bench_nim("hi", 60)
    assert r.error is None
    assert r.tokens == 60
    assert r.ttft_s >= 0.4
    assert r.decode_tps > 3 * r.e2e_tps  # the TTFT no longer drags the decode rate down
    body = fake_nim.requests("/v1/chat/completions")[-1]["body"]
    assert body["stream_options"] == {"include_usage": True} and body["max_tokens"] == 60


def test_nim_without_key_is_an_error_row():
    r = bm.bench_nim("hi", 10)
    assert r.backend == "nim" and "build.nvidia.com" in r.error


def test_auto_selection_skips_embedding_models_and_aliases(fake):
    """Regression: bge-m3 was benchmarked with a chat prompt because only 'embed' names were skipped."""
    models, skipped = bm.select_local_models(None)
    assert models == ["llama3.1:8b", "llama3.2:3b", "llava:7b", "qwen2.5:14b"]
    reasons = dict(skipped)
    assert "embedding" in reasons["bge-m3:latest"]
    assert "embedding" in reasons["nomic-embed-text:latest"]
    assert reasons["llama3.1:latest"] == "same weights as llama3.1:8b"


def test_selection_falls_back_to_name_heuristic(fake, monkeypatch):
    monkeypatch.setattr(bm, "show_model", lambda name: {"capabilities": None})
    models, skipped = bm.select_local_models(None)
    assert "bge-m3:latest" in dict(skipped) and "nomic-embed-text:latest" in dict(skipped)
    assert "llama3.2:3b" in models


def test_explicit_models_are_used_verbatim(fake):
    assert bm.select_local_models(["a", "b"]) == (["a", "b"], [])


def test_cli_end_to_end(fake, capsys):
    code = bm.main(["--num-predict", "20", "--models", "llama3.2:3b", "oom:llama3.1:8b"])
    out = capsys.readouterr().out
    assert code == 0
    assert "llama3.2:3b" in out and "95.0" in out  # canned decode rate of the fake
    assert "oom:llama3.1:8b:" in out and "more system memory" in out


def test_cli_unreachable(capsys):
    assert bm.main([]) == 1
    assert "Cannot reach Ollama" in capsys.readouterr().out
