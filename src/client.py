"""Unified LLM client for local Ollama and cloud NVIDIA NIM.

The whole point of this kit: write your application against one OpenAI-compatible
client, then switch between a fully local Ollama model and free cloud NVIDIA NIM
by flipping a single environment variable, ``LLM_BACKEND``.

Both backends speak the OpenAI wire protocol, so the same ``chat`` / ``stream`` /
``embed`` calls run unchanged against either one. The only differences are the
base URL, the API key, and which concrete model id a logical role maps to.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BACKEND = "ollama"
VALID_BACKENDS = ("ollama", "nim")

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def ollama_host() -> str:
    """Base host for the local Ollama server, without the ``/v1`` suffix."""
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def ollama_base_url() -> str:
    """OpenAI-compatible base URL exposed by Ollama."""
    return f"{ollama_host()}/v1"


# A logical role (what the code asks for) maps to a concrete model id per backend.
# Override any of these with the matching environment variable.
MODEL_MAP: dict[str, dict[str, str]] = {
    "chat": {
        "ollama": os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b"),
        "nim": os.getenv("NIM_MODEL", "meta/llama-3.3-70b-instruct"),
    },
    "chat_small": {
        "ollama": os.getenv("OLLAMA_CHAT_SMALL_MODEL", "llama3.2:3b"),
        "nim": os.getenv("NIM_MODEL", "meta/llama-3.3-70b-instruct"),
    },
    "vision": {
        "ollama": os.getenv("OLLAMA_VISION_MODEL", "llava:7b"),
        "nim": os.getenv("NIM_VISION_MODEL", "meta/llama-3.2-90b-vision-instruct"),
    },
    "embed": {
        "ollama": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        "nim": os.getenv("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5"),
    },
}


class MissingAPIKey(RuntimeError):
    """Raised when the selected backend needs a key that is not configured."""


def resolve_backend(override: str | None = None) -> str:
    """Return the active backend, validating the value.

    Priority: explicit ``override`` argument, then ``LLM_BACKEND`` env, then the
    local default. An unknown value is a hard error rather than a silent fallback.
    """
    backend = (override or os.getenv("LLM_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown LLM_BACKEND {backend!r}. Use one of: {', '.join(VALID_BACKENDS)}."
        )
    return backend


def resolve_model(role: str, backend: str) -> str:
    """Map a logical role to the concrete model id for a backend."""
    try:
        return MODEL_MAP[role][backend]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_MAP))
        raise KeyError(f"No model mapping for role {role!r} (known roles: {known}).") from exc


_NIM_SIGNUP_HINT = (
    "NVIDIA NIM needs a free API key and none was found.\n"
    "  1. Open https://build.nvidia.com and sign in (no credit card, ~2 minutes).\n"
    "  2. Pick any model, click 'Get API Key', copy the key that starts with 'nvapi-'.\n"
    "  3. Put it in your .env file:  NVIDIA_API_KEY=nvapi-...\n"
    "Or stay fully local with LLM_BACKEND=ollama."
)


def require_nim_key() -> str:
    """Return the NVIDIA key or raise a friendly, actionable error."""
    key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not key:
        raise MissingAPIKey(_NIM_SIGNUP_HINT)
    return key


def build_openai_client(backend: str) -> OpenAI:
    """Construct a configured OpenAI SDK client for the given backend."""
    if backend == "ollama":
        # Ollama ignores the key but the SDK requires a non-empty string.
        return OpenAI(base_url=ollama_base_url(), api_key="ollama")
    if backend == "nim":
        return OpenAI(base_url=NIM_BASE_URL, api_key=require_nim_key())
    raise ValueError(f"Unknown backend {backend!r}.")


@dataclass
class LLMClient:
    """Thin, backend-agnostic wrapper over the OpenAI SDK.

    Create one with :meth:`create` and call :meth:`chat`, :meth:`stream` or
    :meth:`embed`. Model selection is by logical role so the identical call runs
    on whichever backend is active.
    """

    backend: str
    client: OpenAI

    @classmethod
    def create(cls, backend: str | None = None) -> "LLMClient":
        resolved = resolve_backend(backend)
        return cls(backend=resolved, client=build_openai_client(resolved))

    # -- model resolution -------------------------------------------------
    def model_for(self, role: str) -> str:
        return resolve_model(role, self.backend)

    def _pick_model(self, role: str, model: str | None) -> str:
        return model or self.model_for(role)

    # -- completions ------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        role: str = "chat",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        """Return a single completed assistant message as text."""
        resp = self.client.chat.completions.create(
            model=self._pick_model(role, model),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def stream(
        self,
        messages: list[dict],
        role: str = "chat",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """Yield assistant text deltas as they arrive."""
        stream = self.client.chat.completions.create(
            model=self._pick_model(role, model),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def raw_chat(self, messages: list[dict], role: str = "chat", model: str | None = None, **kwargs):
        """Escape hatch returning the full OpenAI response object.

        Use this when you need tool calls, usage stats or finish reasons rather
        than just the assistant text.
        """
        return self.client.chat.completions.create(
            model=self._pick_model(role, model),
            messages=messages,
            **kwargs,
        )

    # -- embeddings -------------------------------------------------------
    def embed(
        self,
        texts: Iterable[str],
        role: str = "embed",
        model: str | None = None,
        **kwargs,
    ) -> list[list[float]]:
        """Embed a batch of strings, returning one vector per input."""
        items = list(texts)
        if not items:
            return []
        resp = self.client.embeddings.create(
            model=self._pick_model(role, model),
            input=items,
            **kwargs,
        )
        return [row.embedding for row in resp.data]


def describe_active() -> str:
    """One-line human summary of the current backend and chat model."""
    backend = resolve_backend()
    base = ollama_base_url() if backend == "ollama" else NIM_BASE_URL
    return f"backend={backend}  chat_model={resolve_model('chat', backend)}  endpoint={base}"


if __name__ == "__main__":
    # Quick sanity check without sending any request.
    try:
        print(describe_active())
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
