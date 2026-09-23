"""Hardware detection, the fit planner, recommend and doctor."""

from __future__ import annotations

import json
import subprocess

import pytest
import requests

from src import hardware as hw
from src import model_manager as mm

GiB = 1024 ** 3

SMI_TWO_GPUS = (
    "0, NVIDIA GeForce RTX 5070, 12227, 1024\n"
    "1, NVIDIA RTX A2000 12GB, 12282, [N/A]\n"
)


def canned(stdout="", returncode=0, stderr=""):
    def runner(cmd, **_kw):
        assert cmd[1].startswith("--query-gpu=index,name,memory.total,memory.used")
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)
    return runner


RTX_5070 = hw.Hardware(gpus=[hw.GPU(0, "NVIDIA GeForce RTX 5070", 12 * 1024, 900)], ram_total_gib=32.0,
                       ram_available_gib=20.0)
NO_GPU = hw.Hardware(gpus=[], ram_total_gib=16.0, ram_available_gib=8.0, gpu_note="nvidia-smi not found")


# -- detection --------------------------------------------------------------------------
def test_parse_nvidia_smi():
    gpus = hw.parse_nvidia_smi(SMI_TWO_GPUS + "\ngarbage line\n")
    assert [(g.index, g.name, g.total_mib, g.used_mib) for g in gpus] == [
        (0, "NVIDIA GeForce RTX 5070", 12227, 1024),
        (1, "NVIDIA RTX A2000 12GB", 12282, None),
    ]
    assert gpus[0].total_gib == pytest.approx(12227 / 1024)


def test_detect_gpus_with_canned_output():
    gpus, note = hw.detect_gpus(canned(SMI_TWO_GPUS), exe="nvidia-smi")
    assert len(gpus) == 2 and note == ""
    hwinfo = hw.Hardware(gpus=gpus, ram_total_gib=64.0)
    assert hwinfo.vram_gib == pytest.approx((12227 + 12282) / 1024)
    assert "NVIDIA GeForce RTX 5070 (11.9 GiB VRAM, 1.0 GiB in use)" in hwinfo.describe()


def test_missing_or_broken_nvidia_smi_is_graceful(monkeypatch):
    monkeypatch.setattr(hw, "find_nvidia_smi", lambda: None)
    assert hw.detect_gpus() == ([], "nvidia-smi not found: no NVIDIA driver, or a non-NVIDIA GPU")

    def boom(cmd, **_kw):
        raise FileNotFoundError(2, "No such file")
    gpus, note = hw.detect_gpus(boom, exe="nvidia-smi")
    assert gpus == [] and "could not run" in note

    gpus, note = hw.detect_gpus(canned(returncode=9, stderr="NVIDIA-SMI has failed because it couldn't "
                                                            "communicate with the NVIDIA driver."), exe="x")
    assert gpus == [] and note.startswith("nvidia-smi failed: NVIDIA-SMI has failed")
    assert hw.detect_gpus(canned("No devices were found\n"), exe="x") == ([], "nvidia-smi reported no GPUs")


def test_system_memory_is_detected_on_this_machine():
    total, available = hw.system_memory()
    assert total is not None and total > 0.5
    assert available is None or 0 < available <= total


def test_detect_on_this_machine_never_raises():
    info = hw.detect()
    assert isinstance(info.describe(), str) and info.vram_gib >= 0
    json.dumps(info.as_dict())


# -- fit math -------------------------------------------------------------------------------
def spec(name):
    return next(s for s in mm.catalog_specs(("chat", "vision")) if s.name == name)


def test_kv_cache_formula():
    llama = spec("llama3.1:8b")
    assert llama.kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2 == 131072  # 128 KiB per token
    assert llama.kv_bytes_per_token("q8_0") == pytest.approx(131072 * 34 / 64)
    phi = spec("phi3.5:3.8b")  # no grouped-query attention: 32 KV heads of 96
    assert phi.kv_bytes_per_token() == 2 * 32 * 32 * 96 * 2


def test_llama31_8b_at_8k_fits_in_12gb():
    r = hw.plan_fit(spec("llama3.1:8b"), vram_gib=12, ram_gib=32, num_ctx=8192)
    assert r.status == "gpu" and r.gpu_layers == 32 and r.gpu_percent == 100
    assert r.kv_gib == pytest.approx(1.0)  # 131072 B x 8192 = exactly 1 GiB
    assert r.weights_gib == pytest.approx(4.9e9 / GiB)
    # Hand calculation of the largest context that stays on the GPU:
    free = 12 * GiB - 4.9e9 - 1.0 * GiB          # VRAM - weights - overhead = 6,911,160,064 B
    expected = int(free // 131072)                # 52,728 tokens
    expected -= expected % 1024                   # 52,224
    assert expected == 52224 and r.max_ctx_on_gpu == expected


def test_qwen25_14b_at_32k_is_partial_offload_on_12gb():
    r = hw.plan_fit(spec("qwen2.5:14b"), vram_gib=12, ram_gib=32, num_ctx=32768)
    assert r.status == "partial"
    assert r.kv_gib == pytest.approx(6.0)  # 2*48*8*128*2 B x 32768 = 6 GiB
    per_layer = (9.0e9 / GiB + 6.0) / 48
    assert r.gpu_layers == int(11.0 // per_layer) == 36
    assert r.gpu_percent == pytest.approx(75.0)
    assert r.label == "partial offload (~75% on GPU)"
    # ...but it fits entirely at a smaller context:
    assert r.max_ctx_on_gpu == 13312
    assert hw.plan_fit(spec("qwen2.5:14b"), 12, 32, num_ctx=13312).status == "gpu"
    assert hw.plan_fit(spec("qwen2.5:14b"), 12, 32, num_ctx=14336).status == "partial"


def test_quantized_kv_cache_buys_context():
    f16 = hw.max_ctx_on_gpu(spec("qwen2.5:14b"), 12, "f16")
    q8 = hw.max_ctx_on_gpu(spec("qwen2.5:14b"), 12, "q8_0")
    assert q8 > 1.8 * f16
    assert hw.max_ctx_on_gpu(spec("qwen2.5:7b"), 24) == 32768  # capped at the trained context


def test_cpu_only_and_too_large():
    assert hw.plan_fit(spec("llama3.1:8b"), vram_gib=0, ram_gib=16, num_ctx=4096).status == "cpu"
    big = hw.plan_fit(spec("qwen2.5:14b"), vram_gib=0, ram_gib=8, num_ctx=32768)
    assert big.status == "too_large" and "RAM" in big.note
    assert hw.plan_fit(spec("qwen2.5:14b"), vram_gib=0, ram_gib=None, num_ctx=4096).status == "cpu"
    assert hw.plan_fit(spec("llama3.1:8b"), 4, 16, num_ctx=200_000).note.startswith("num_ctx is above")
    with pytest.raises(ValueError):
        hw.plan_fit(spec("llama3.1:8b"), 12, 16, kv_type="fp8")


def test_spec_from_ollama_metadata():
    info = {"general.architecture": "qwen2", "qwen2.block_count": 48, "qwen2.attention.head_count": 40,
            "qwen2.attention.head_count_kv": 8, "qwen2.embedding_length": 5120, "qwen2.context_length": 32768}
    s = hw.ModelSpec.from_ollama("qwen2.5:14b", 8_988_124_069, info)
    assert (s.n_layers, s.n_kv_heads, s.head_dim, s.context_length) == (48, 8, 128, 32768)
    assert s.size_gb == pytest.approx(8.988, abs=1e-3)


# -- fit / recommend / doctor CLIs -----------------------------------------------------------
def run(capsys, *argv):
    code = mm.main(list(argv))
    return code, capsys.readouterr().out


def test_fit_cli_with_explicit_budget(capsys, monkeypatch):
    monkeypatch.setattr(hw, "detect", lambda *a, **k: pytest.fail("must not detect when --vram/--ram are given"))
    code, out = run(capsys, "fit", "--vram", "12", "--ram", "32", "--ctx", "8192")
    assert code == 0
    assert "12.0 GiB VRAM (--vram)" in out and "32.0 GiB RAM (--ram)" in out
    assert "llama3.1:8b" in out and "52,224" in out and "fits in VRAM" in out


def test_fit_cli_json(capsys):
    code, out = run(capsys, "fit", "qwen2.5:14b", "--vram", "12", "--ram", "32", "--ctx", "32768", "--json")
    data = json.loads(out)
    assert code == 0 and data["unknown"] == []
    row = data["models"][0]
    assert row["status"] == "partial" and row["gpu_layers"] == 36 and row["max_ctx_on_gpu"] == 13312


def test_fit_uses_detected_hardware(capsys, monkeypatch):
    monkeypatch.setattr(hw, "detect", lambda *a, **k: NO_GPU)
    code, out = run(capsys, "fit", "llama3.1:8b")
    assert code == 0 and "no GPU detected" in out and "CPU only" in out


def test_fit_installed_models_from_ollama_metadata(fake, capsys):
    # llama3.1:latest is not in models.yaml: its shape comes from /api/show model_info.
    code, out = run(capsys, "fit", "llama3.1:latest", "ghost:1b", "--vram", "12", "--ram", "32",
                    "--ctx", "8192", "--json")
    data = json.loads(out)
    assert code == 1 and data["unknown"] == ["ghost:1b"]
    row = data["models"][0]
    assert row["model"] == "llama3.1:latest" and row["kv_cache_gib"] == 1.0 and row["status"] == "gpu"
    code, out = run(capsys, "fit", "--installed", "--vram", "8", "--ram", "32", "--json")
    names = [m["model"] for m in json.loads(out)["models"]]
    assert "qwen2.5:14b" in names and "llava:7b" in names
    assert not any("embed" in n or "bge" in n for n in names)


def test_recommend_marks_models_for_the_budget(capsys):
    code, out = run(capsys, "recommend", "--vram", "6", "--ram", "32", "--ctx", "4096")
    assert code == 0
    assert "On this machine" in out
    assert "fits in VRAM" in out and "partial offload" in out  # 3B fits in 6 GiB, 14B does not


def test_doctor_all_good(fake, monkeypatch):
    rows = mm.run_checks(hardware=RTX_5070)
    status = {check: s for s, check, _d in rows}
    assert status["Ollama server"] == "ok"
    assert status["Chat model"] == "ok" and status["Embedding model"] == "ok" and status["Vision model"] == "ok"
    assert status["NVIDIA_API_KEY"] == "info"
    assert status["GPU 0"] == "ok" and status["Chat model fit"] == "ok"
    assert not any(s == "fail" for s, _c, _d in rows)


def test_doctor_reports_partial_offload_from_api_ps(fake):
    requests.post(fake.url + "/api/chat", json={"model": "qwen2.5:14b", "messages": [], "stream": False})
    rows = mm.run_checks(hardware=RTX_5070)
    loaded = [d for s, c, d in rows if c == "Loaded now"]
    assert loaded == ["qwen2.5:14b: 72% GPU / 28% CPU, partially offloaded; tokens/sec will drop. Lower num_ctx "
                      "or pick a smaller model (python -m src.model_manager fit)"]


def test_doctor_missing_models_and_bad_key(fake, monkeypatch):
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
    monkeypatch.setenv("LLM_BACKEND", "nim")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-XXXXXXXXXXXXXXXXXXXXXXXX")
    rows = {c: (s, d) for s, c, d in mm.run_checks(hardware=NO_GPU)}
    assert rows["Embedding model"][0] == "warn" and "pull mxbai-embed-large" in rows["Embedding model"][1]
    assert rows["NVIDIA_API_KEY"][0] == "fail" and "placeholder" in rows["NVIDIA_API_KEY"][1]
    assert rows["GPU"][0] == "warn"
    assert rows["Chat model fit"] == ("warn", "llama3.1:8b at num_ctx 4096: CPU only (estimate)")


def test_doctor_masks_a_valid_key(fake, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-abcdefghijklmnopqrstuvwxyz0123456789")
    rows = {c: (s, d) for s, c, d in mm.run_checks(hardware=RTX_5070)}
    status, detail = rows["NVIDIA_API_KEY"]
    assert status == "ok" and "nvapi-...6789" in detail and "abcdefghijklmnop" not in detail


def test_doctor_online_check_against_self_hosted_nim(fake, fake_nim):
    rows = {c: (s, d) for s, c, d in mm.run_checks(online=True, hardware=RTX_5070)}
    assert rows["NIM endpoint"][0] == "ok"
    assert rows["NIM online check"] == ("ok", f"4 models listed at {fake_nim.v1}")


def test_doctor_cli_exit_codes(fake, capsys, monkeypatch):
    monkeypatch.setattr(hw, "detect", lambda *a, **k: RTX_5070)
    code, out = run(capsys, "doctor")
    assert code == 0 and "All good." in out
    code, out = run(capsys, "doctor", "--json")
    assert code == 0 and {r["check"] for r in json.loads(out)} >= {"Ollama server", "GPU 0"}


def test_doctor_without_ollama_fails_cleanly(capsys, monkeypatch):
    monkeypatch.setattr(hw, "detect", lambda *a, **k: NO_GPU)
    code, out = run(capsys, "doctor")
    assert code == 1
    assert "not reachable" in out and "ollama serve" in out and "1 problem(s) to fix." in out


def test_doctor_and_fit_run_for_real_on_this_machine(run_cli):
    """No mocks: real nvidia-smi lookup and RAM detection, Ollama pointed at a dead port."""
    out = run_cli("-m", "src.model_manager", "fit", "--vram", "12", "--ctx", "8192")
    assert out.returncode == 0, out.stderr
    assert "Planning for 12.0 GiB VRAM (--vram)" in out.stdout and "RAM (detected)" in out.stdout
    doc = run_cli("-m", "src.model_manager", "doctor")
    assert doc.returncode == 1 and "Traceback" not in doc.stderr
    assert "Ollama server" in doc.stdout and "System RAM" in doc.stdout
