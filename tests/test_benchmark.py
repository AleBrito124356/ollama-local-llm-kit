"""Benchmark: honest numbers, errors reported as errors, comparable local vs NIM rates."""

from __future__ import annotations

import csv
import io
import json

import pytest

from src import benchmark as bm
from src import hardware as hw
from src.fake_ollama import prompt_token_count

RTX_5070 = hw.Hardware(gpus=[hw.GPU(0, "NVIDIA GeForce RTX 5070", 12 * 1024, 900)], ram_total_gib=32.0)


@pytest.fixture(autouse=True)
def canned_hardware(monkeypatch):
    monkeypatch.setattr(bm.hw, "detect", lambda *a, **k: RTX_5070)


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


# -- benchmark v2: warm-up, runs, prefill vs decode, options, exports ------------------------
def test_warmup_is_excluded_and_runs_are_measured(fake):
    s = bm.benchmark_model("ollama", "llama3.1:8b", "hello there", 32, runs=3)
    chats = fake.requests("/api/chat")
    assert len(chats) == 4  # 1 warm-up + 3 runs
    prompts = [c["body"]["messages"][0]["content"] for c in chats]
    assert prompts == [f"[run {i}] hello there" for i in range(4)]  # distinct: no prompt-cache hits
    assert s.warmup.load_s == pytest.approx(2.1)  # cold load happens in the warm-up...
    assert all(r.load_s == pytest.approx(0.015) for r in s.runs)  # ...not in the measured runs
    assert len(s.ok_runs) == 3 and s.error is None


def test_prefill_and_decode_rates_are_exact(fake):
    s = bm.benchmark_model("ollama", "llama3.1:8b", "Explain B-trees.", 64, runs=2)
    r = s.runs[0]
    expected_prompt = prompt_token_count([{"role": "user", "content": "[run 1] Explain B-trees."}])
    assert r.prompt_tokens == expected_prompt
    assert r.prefill_tps == pytest.approx(1180.0, rel=1e-6)  # the fake's canned prompt_eval rate
    assert r.decode_tps == pytest.approx(52.5, rel=1e-6)     # the fake's canned eval rate
    row = s.row()
    assert row["prefill_tps"] == 1180.0 and row["decode_tps"] == 52.5 and row["runs"] == 2


def test_median_and_range_over_runs(monkeypatch):
    rates = iter([99.0, 10.0, 30.0, 20.0])  # warm-up, then three runs

    def scripted(model, prompt, num_predict, options=None, keep_alive=None):
        return bm.Result("ollama", model, ttft_s=0.1, tokens=50, decode_tps=next(rates), e2e_tps=5.0)

    monkeypatch.setattr(bm, "bench_ollama", scripted)
    monkeypatch.setattr(bm, "running_models", lambda: [])
    s = bm.benchmark_model("ollama", "m", "p", 50, runs=3)
    assert s.stat("decode_tps") == 20.0 and s.decode_range == (10.0, 30.0)


def test_partial_failures_are_reported(monkeypatch):
    calls = iter([None, None, "boom", None])

    def flaky(model, prompt, num_predict, options=None, keep_alive=None):
        err = next(calls)
        if err:
            return bm.Result("ollama", model, error=err)
        return bm.Result("ollama", model, tokens=10, decode_tps=10.0)

    monkeypatch.setattr(bm, "bench_ollama", flaky)
    monkeypatch.setattr(bm, "running_models", lambda: [])
    s = bm.benchmark_model("ollama", "m", "p", 10, runs=3)
    assert len(s.ok_runs) == 2 and s.error == "1 of 3 runs failed: boom"


def test_warmup_failure_skips_the_runs(fake):
    s = bm.benchmark_model("ollama", "oom:qwen2.5:14b", "p", 10, runs=3)
    assert s.error and not s.runs and len(fake.requests("/api/chat")) == 1


def test_ollama_options_reach_the_server(fake):
    bm.benchmark_model("ollama", "llama3.1:8b", "p", 16, runs=1, options={"num_ctx": 16384, "num_gpu": 20},
                       keep_alive="10m")
    body = fake.requests("/api/chat")[-1]["body"]
    assert body["options"] == {"num_ctx": 16384, "num_gpu": 20, "num_predict": 16}
    assert body["keep_alive"] == "10m"


def test_gpu_share_comes_from_api_ps(fake):
    assert bm.benchmark_model("ollama", "llama3.1:8b", "p", 8, runs=1).gpu_percent == 100.0
    assert bm.benchmark_model("ollama", "qwen2.5:14b", "p", 8, runs=1).gpu_percent == pytest.approx(72.0)
    cpu = bm.benchmark_model("ollama", "llama3.2:3b", "p", 8, runs=1, options={"num_gpu": 0})
    assert cpu.gpu_percent == 0.0 and cpu.stat("decode_tps") == pytest.approx(95.0 * 0.2)
    unloaded = bm.benchmark_model("ollama", "llava:7b", "p", 8, runs=1, keep_alive=0)
    assert unloaded.gpu_percent is None
    assert bm.parse_keep_alive("0") == 0 and bm.parse_keep_alive("5m") == "5m"


def test_cli_header_offload_warning_and_exports(fake, tmp_path, capsys):
    paths = {ext: tmp_path / f"results.{ext}" for ext in ("json", "csv", "md")}
    code = bm.main(["--models", "llama3.1:8b", "qwen2.5:14b", "--runs", "2", "--num-predict", "24",
                    "--num-ctx", "8192", "--json", str(paths["json"]), "--csv", str(paths["csv"]),
                    "--markdown", str(paths["md"])])
    out = capsys.readouterr().out
    assert code == 0
    assert "NVIDIA GeForce RTX 5070 (12.0 GiB VRAM" in out  # detected-hardware header, not a canned note
    assert "RTX 5070 / 12 GB: a Q4 8B model" not in out
    assert "qwen2.5:14b: only 72% of the model is in VRAM" in out
    assert '"num_ctx": 8192' in out

    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["hardware"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 5070"
    assert data["settings"]["runs"] == 2 and data["settings"]["options"] == {"num_ctx": 8192}
    rows = {r["model"]: r for r in data["results"]}
    assert rows["llama3.1:8b"]["decode_tps"] == 52.5 and rows["llama3.1:8b"]["gpu_percent"] == 100.0
    assert rows["qwen2.5:14b"]["decode_tps"] == 14.2 and rows["qwen2.5:14b"]["gpu_percent"] == 72.0

    csv_rows = list(csv.DictReader(io.StringIO(paths["csv"].read_text(encoding="utf-8"))))
    assert [r["model"] for r in csv_rows] == ["llama3.1:8b", "qwen2.5:14b"]
    assert list(csv_rows[0]) == bm.COLUMNS and csv_rows[0]["prefill_tps"] == "1180.0"

    md = paths["md"].read_text(encoding="utf-8").splitlines()
    assert md[0].startswith("| backend | model |") and md[1].startswith("|---|") and len(md) == 4


def test_json_to_stdout_keeps_stdout_clean(fake, capsys):
    code = bm.main(["--models", "llama3.2:3b", "--runs", "1", "--num-predict", "8", "--json", "-"])
    captured = capsys.readouterr()
    assert code == 0
    data = json.loads(captured.out)  # nothing but JSON on stdout
    assert data["results"][0]["model"] == "llama3.2:3b"
    assert "tokens/sec benchmark" in captured.err


def test_nim_in_the_same_table(fake, fake_nim, capsys):
    code = bm.main(["--models", "llama3.2:3b", "--nim", "--runs", "2", "--num-predict", "20"])
    out = capsys.readouterr().out
    assert code == 0
    assert "meta/llama-3.3-70b-instruct" in out and "llama3.2:3b" in out
    assert len(fake_nim.requests("/v1/chat/completions")) == 3  # warm-up + 2 runs


def test_runs_must_be_positive(capsys):
    assert bm.main(["--runs", "0"]) == 2
