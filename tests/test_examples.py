"""The three examples: the safe calculator, and each script end to end against the fake server."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def load_example(name: str):
    spec = importlib.util.spec_from_file_location(f"example_{name}", REPO / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fc = load_example("function_calling")


@pytest.mark.parametrize(
    ("expr", "result"),
    [("128 * 47", 6016), ("(1 + 2) * 3 / 4", 2.25), ("10 // 3", 3), ("-3**2", -9), ("2 ** 0.5", 2 ** 0.5),
     ("7 % 4", 3), ("1.5e3 + 1", 1501), ("2**1000", 2**1000)],
)
def test_calculator_arithmetic(expr, result):
    assert json.loads(fc.calculate(expr))["result"] == pytest.approx(result)


@pytest.mark.parametrize(
    ("expr", "error"),
    [
        ("9**9**9**9", "too large"),          # used to hang forever under eval
        ("9**9999", "too large"),             # used to crash json.dumps (int -> str limit)
        ("0.5 ** -5000", "too large"),
        ("1e308 * 10", "too large"),
        ("1/0", "division by zero"),
        ("5 % 0", "division by zero"),
        ("__import__('os').system('echo hi')", "unsupported syntax"),
        ("a + 1", "unsupported syntax: Name"),
        ("(-8) ** 0.5", "complex"),
        ("2 +", "not a valid arithmetic expression"),
        ("", "empty"),
        ("1+" * 150 + "1", "longer than"),
    ],
)
def test_calculator_rejects_safely(expr, error):
    start = time.perf_counter()
    out = json.loads(fc.calculate(expr))
    assert time.perf_counter() - start < 1.0
    assert error in out["error"]


def test_run_tool_handles_model_mistakes():
    assert "unknown tool" in json.loads(fc.run_tool("rm_rf", "{}"))["error"]
    assert "not valid JSON" in json.loads(fc.run_tool("calculate", "{oops"))["error"]
    assert "bad arguments" in json.loads(fc.run_tool("calculate", '{"expr": "1+1"}'))["error"]
    assert json.loads(fc.run_tool("calculate", '{"expression": "2+2"}'))["result"] == 4


def test_function_calling_script(fake, run_cli):
    out = run_cli("examples/function_calling.py")
    assert out.returncode == 0, out.stderr
    assert '-> calculate({"expression": "128 * 47"}) = {"expression": "128 * 47", "result": 6016}' in out.stdout
    assert '-> get_weather({"city": "Panama City"})' in out.stdout
    assert "thunderstorms" in out.stdout


def test_function_calling_script_on_nim(fake_nim, run_cli):
    out = run_cli("examples/function_calling.py", LLM_BACKEND="nim")
    assert out.returncode == 0, out.stderr
    assert "[backend=nim  model=meta/llama-3.3-70b-instruct]" in out.stdout
    assert '"result": 6016' in out.stdout


def test_structured_output_script(fake, run_cli):
    out = run_cli("examples/structured_output.py")
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout.split("Validated Person object:", 1)[1])
    assert data == {"name": "Grace Hopper", "age": 79, "city": "New York",
                    "known_for": "popularized the term 'debugging'"}


def test_vision_script(fake, run_cli):
    pytest.importorskip("PIL")
    out = run_cli("examples/vision.py")
    assert out.returncode == 0, out.stderr
    assert "[backend=ollama  model=llava:7b]" in out.stdout
    assert "480x320 PNG image" in out.stdout
    body = fake.requests("/v1/chat/completions")[-1]["body"]
    assert body["model"] == "llava:7b"
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
