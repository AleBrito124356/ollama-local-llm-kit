"""Shared fixtures: an in-process fake Ollama / NIM server and a network guard.

Everything runs offline. Non-loopback connections and DNS lookups are blocked in
this process (and in every subprocess the tests start), no ``.env`` file is
read, and the environment variables the kit reads are cleared per test.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NETGUARD = Path(__file__).resolve().parent / "netguard"

# Must happen before any src module is imported (client.py loads .env at import,
# Rich consoles read COLUMNS when they are created).
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["COLUMNS"] = "200"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(proxy_var, None)

sys.path.insert(0, str(REPO))

# Load tests/netguard/sitecustomize.py under its own name (a site-wide
# ``sitecustomize`` may already be imported); executing it installs the guard.
_spec = importlib.util.spec_from_file_location("kit_netguard", NETGUARD / "sitecustomize.py")
netguard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(netguard)

from src.fake_ollama import FakeOllamaServer  # noqa: E402

KIT_ENV_VARS = (
    "LLM_BACKEND", "NVIDIA_API_KEY", "NIM_BASE_URL", "NIM_MODEL", "NIM_CHAT_SMALL_MODEL",
    "NIM_VISION_MODEL", "NIM_EMBED_MODEL", "OLLAMA_HOST", "OLLAMA_CHAT_MODEL",
    "OLLAMA_CHAT_SMALL_MODEL", "OLLAMA_EMBED_MODEL", "OLLAMA_VISION_MODEL", "RAG_CACHE_DIR",
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
)

# Nothing listens on port 9 (discard) of the loopback interface, so connecting is refused.
# (On Windows a refused loopback connect takes ~2 s because of SYN retries.)
REFUSED_OLLAMA = "http://127.0.0.1:9"
TEST_NIM_KEY = "nvapi-test-key-0123456789abcdefghij"


class HangUpServer:
    """Accepts TCP connections and closes them at once: an 'unreachable' Ollama that fails fast."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(64)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            conn.close()


_HANG_UP = HangUpServer()
DEAD_OLLAMA = _HANG_UP.url


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Start every test from a known environment that cannot reach anything real."""
    for var in KIT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OLLAMA_HOST", DEAD_OLLAMA)
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path / "rag_cache"))


@pytest.fixture(scope="session")
def _ollama_server():
    with FakeOllamaServer() as server:
        yield server


@pytest.fixture(scope="session")
def _nim_server():
    with FakeOllamaServer(nim=True) as server:
        yield server


@pytest.fixture
def fake(_ollama_server, monkeypatch):
    """A fresh fake Ollama server, with OLLAMA_HOST pointing at it."""
    _ollama_server.reset()
    monkeypatch.setenv("OLLAMA_HOST", _ollama_server.url)
    return _ollama_server


@pytest.fixture
def fake_nim(_nim_server, monkeypatch):
    """A fresh strict NIM-like server, with NIM_BASE_URL and a test key configured."""
    _nim_server.reset()
    monkeypatch.setenv("NIM_BASE_URL", _nim_server.v1)
    monkeypatch.setenv("NVIDIA_API_KEY", TEST_NIM_KEY)
    return _nim_server


def cli_env(**overrides) -> dict:
    """Environment for a subprocess: guarded network, no .env, current kit vars."""
    env = {k: v for k, v in os.environ.items() if k not in KIT_ENV_VARS}
    for var in KIT_ENV_VARS:
        if var in os.environ:
            env[var] = os.environ[var]
    env["PYTHONPATH"] = os.pathsep.join([str(NETGUARD), str(REPO)])
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["COLUMNS"] = "200"
    env.update({k: str(v) for k, v in overrides.items()})
    return env


@pytest.fixture
def run_cli():
    """Run ``python <args>`` from the repo root with the guarded environment."""

    def _run(*args: str, input: str | None = None, timeout: float = 60, cwd: Path | None = None, **env):
        return subprocess.run(
            [sys.executable, *args],
            cwd=str(cwd or REPO),
            env=cli_env(**env),
            input=input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )

    return _run
