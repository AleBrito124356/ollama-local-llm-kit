"""Unified LLM client for local Ollama and cloud NVIDIA NIM.

The whole point of this kit: write your application against one OpenAI-compatible
client, then switch between a fully local Ollama model and free cloud NVIDIA NIM
by flipping a single environment variable, ``LLM_BACKEND``.

Both backends speak the OpenAI wire protocol, so the same ``chat`` / ``stream`` /
``embed`` calls run unchanged against either one. The differences that do exist
are handled here rather than in your code:

* the base URL and API key (``OLLAMA_HOST``, ``NIM_BASE_URL``, ``NVIDIA_API_KEY``);
* which concrete model id a logical role maps to (``MODEL_MAP`` plus env overrides);
* how an embedding request says whether the text is a *query* or a *passage*.
  NIM's asymmetric retrieval models require an ``input_type`` field, and local
  models such as ``nomic-embed-text`` expect a task prefix instead.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env_file() -> None:
    """Load ``.env`` from the working directory, else from the kit's root.

    ``python-dotenv``'s default search walks up to the filesystem root, which can
    silently pick up an unrelated ``.env`` from a parent folder. Only these two
    well-defined locations are read. ``PYTHON_DOTENV_DISABLED=1`` turns it off.
    """
    for candidate in (os.path.join(os.getcwd(), ".env"), os.path.join(REPO_ROOT, ".env")):
        if os.path.isfile(candidate):
            load_dotenv(candidate)
            return


_load_env_file()

DEFAULT_BACKEND = "ollama"
VALID_BACKENDS = ("ollama", "nim")

HOSTED_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Kept for backwards compatibility; prefer nim_base_url(), which honours NIM_BASE_URL.
NIM_BASE_URL = HOSTED_NIM_BASE_URL

DEFAULT_OLLAMA_PORT = 11434


def ollama_host() -> str:
    """Base URL of the local Ollama server, without the ``/v1`` suffix.

    Accepts the same forms Ollama itself accepts in ``OLLAMA_HOST``: a full URL,
    ``host:port`` without a scheme, or a bind address such as ``0.0.0.0:11500``
    (rewritten to ``127.0.0.1``, because connecting to 0.0.0.0 does not work on
    every platform).
    """
    raw = (os.getenv("OLLAMA_HOST") or "").strip() or f"http://localhost:{DEFAULT_OLLAMA_PORT}"
    if "://" not in raw:
        raw = "http://" + raw
    parts = urlsplit(raw)
    host = parts.hostname or "localhost"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    port = parts.port or DEFAULT_OLLAMA_PORT
    if ":" in host:  # bare IPv6 literal
        host = f"[{host}]"
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme or "http", f"{host}:{port}", path, "", ""))


def ollama_base_url() -> str:
    """OpenAI-compatible base URL exposed by Ollama."""
    return f"{ollama_host()}/v1"


def nim_base_url() -> str:
    """Base URL for the ``nim`` backend.

    Defaults to NVIDIA's hosted endpoint. Set ``NIM_BASE_URL`` to point the same
    backend at a self-hosted NIM container (e.g. ``http://localhost:8000/v1``).
    """
    return (os.getenv("NIM_BASE_URL") or HOSTED_NIM_BASE_URL).strip().rstrip("/")


def nim_is_hosted() -> bool:
    """True when the ``nim`` backend points at NVIDIA's hosted API."""
    return nim_base_url() == HOSTED_NIM_BASE_URL


# A logical role (what the code asks for) maps to a concrete model id per backend.
# These are the defaults. The environment variable listed in MODEL_ENV for a
# role/backend overrides the default at call time, so both editing this dict and
# changing the environment take effect immediately, for every module of the kit.
MODEL_MAP: dict[str, dict[str, str]] = {
    "chat": {"ollama": "llama3.1:8b", "nim": "meta/llama-3.3-70b-instruct"},
    "chat_small": {"ollama": "llama3.2:3b", "nim": "meta/llama-3.1-8b-instruct"},
    "vision": {"ollama": "llava:7b", "nim": "meta/llama-3.2-90b-vision-instruct"},
    "embed": {"ollama": "nomic-embed-text", "nim": "nvidia/nv-embedqa-e5-v5"},
}

MODEL_ENV: dict[tuple[str, str], str] = {
    ("chat", "ollama"): "OLLAMA_CHAT_MODEL",
    ("chat", "nim"): "NIM_MODEL",
    ("chat_small", "ollama"): "OLLAMA_CHAT_SMALL_MODEL",
    ("chat_small", "nim"): "NIM_CHAT_SMALL_MODEL",
    ("vision", "ollama"): "OLLAMA_VISION_MODEL",
    ("vision", "nim"): "NIM_VISION_MODEL",
    ("embed", "ollama"): "OLLAMA_EMBED_MODEL",
    ("embed", "nim"): "NIM_EMBED_MODEL",
}

# Local embedding models trained with task prefixes. The prefix plays the same
# role as NIM's ``input_type``: it tells an asymmetric model whether the text is
# a search query or a document passage.
LOCAL_EMBED_PREFIXES: dict[str, dict[str, str]] = {
    "nomic-embed-text": {"query": "search_query: ", "passage": "search_document: "},
    "mxbai-embed-large": {
        "query": "Represent this sentence for searching relevant passages: ",
        "passage": "",
    },
}

INPUT_TYPES = ("query", "passage")


class MissingAPIKey(RuntimeError):
    """Raised when the selected backend needs a key that is missing or unusable."""


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
    """Map a logical role to the concrete model id for a backend.

    The environment variable for that role/backend (see ``MODEL_ENV``) wins over
    the default in ``MODEL_MAP``.
    """
    try:
        default = MODEL_MAP[role][backend]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_MAP))
        raise KeyError(f"No model mapping for role {role!r} (known roles: {known}).") from exc
    env_name = MODEL_ENV.get((role, backend))
    if env_name:
        value = (os.getenv(env_name) or "").strip()
        if value:
            return value
    return default


_NIM_SIGNUP_HINT = (
    "  1. Open https://build.nvidia.com and sign in (no credit card, ~2 minutes).\n"
    "  2. Pick any model, click 'Get API Key', copy the key that starts with 'nvapi-'.\n"
    "  3. Put it in your .env file:  NVIDIA_API_KEY=nvapi-...\n"
    "Or stay fully local with LLM_BACKEND=ollama."
)

# nvapi-XXXXXXXX..., nvapi-..., <your key>, your-api-key-here, changeme
_PLACEHOLDER_RE = re.compile(r"^(nvapi-)?(x+|\.{3}.*|<.*>|your.*|changeme)$", re.IGNORECASE)


def nim_key_problem(key: str | None) -> str | None:
    """Explain why ``key`` cannot be a real NVIDIA API key, or return None if it looks valid.

    This is a local, offline sanity check (it never contacts NVIDIA). It catches
    the common setup mistakes: no key, the ``.env.example`` placeholder copied
    verbatim, or a value that is not an ``nvapi-`` key at all.
    """
    key = (key or "").strip()
    if not key:
        return "NVIDIA_API_KEY is not set."
    if _PLACEHOLDER_RE.match(key):
        return "NVIDIA_API_KEY still holds a placeholder value (e.g. the one from .env.example)."
    if not key.startswith("nvapi-"):
        return "NVIDIA_API_KEY does not start with 'nvapi-', so it is not an NVIDIA API key."
    if any(ch.isspace() for ch in key) or len(key) < 20:
        return "NVIDIA_API_KEY is malformed (too short or contains whitespace)."
    return None


def mask_key(key: str | None) -> str:
    """Return a key with everything but its prefix and last 4 characters hidden."""
    key = (key or "").strip()
    if not key:
        return "(not set)"
    if len(key) <= 12:
        return key[:2] + "..."
    return f"{key[:6]}...{key[-4:]}"


def require_nim_key() -> str:
    """Return the key to send to the ``nim`` backend, or raise a friendly error.

    For NVIDIA's hosted API the key must look like a real ``nvapi-`` key. A
    self-hosted NIM container (``NIM_BASE_URL`` set to anything else) does not
    check keys by default, so an unset or placeholder key is replaced by a dummy
    value there, and any other key is passed through unchanged.
    """
    key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    problem = nim_key_problem(key)
    if not nim_is_hosted():
        if not key or _PLACEHOLDER_RE.match(key):
            return "not-needed-for-self-hosted-nim"
        return key
    if problem:
        raise MissingAPIKey(f"NVIDIA NIM needs a free API key. {problem}\n{_NIM_SIGNUP_HINT}")
    return key


def build_openai_client(backend: str, **client_kwargs) -> OpenAI:
    """Construct a configured OpenAI SDK client for the given backend.

    Extra keyword arguments (``timeout``, ``max_retries``, ...) go to ``OpenAI``.
    A local server that refuses a connection will not start answering after an
    exponential back-off, so the Ollama client does not retry by default.
    """
    if backend == "ollama":
        client_kwargs.setdefault("max_retries", 0)
        # Ollama ignores the key but the SDK requires a non-empty string.
        return OpenAI(base_url=ollama_base_url(), api_key="ollama", **client_kwargs)
    if backend == "nim":
        return OpenAI(base_url=nim_base_url(), api_key=require_nim_key(), **client_kwargs)
    raise ValueError(f"Unknown backend {backend!r}.")


def local_embed_prefix(model: str, input_type: str | None) -> str:
    """Task prefix a local embedding model expects for ``input_type`` (or "")."""
    if not input_type:
        return ""
    base = model.split(":", 1)[0].rsplit("/", 1)[-1].lower()
    prefixes = LOCAL_EMBED_PREFIXES.get(base)
    return prefixes.get(input_type, "") if prefixes else ""


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
    def create(cls, backend: str | None = None, **client_kwargs) -> "LLMClient":
        resolved = resolve_backend(backend)
        return cls(backend=resolved, client=build_openai_client(resolved, **client_kwargs))

    @property
    def base_url(self) -> str:
        return str(self.client.base_url).rstrip("/")

    # -- model resolution -------------------------------------------------
    def model_for(self, role: str) -> str:
        return resolve_model(role, self.backend)

    def _pick_model(self, role: str, model: str | None) -> str:
        return model or self.model_for(role)

    @staticmethod
    def _sampling(temperature: float | None, max_tokens: int | None, kwargs: dict) -> dict:
        """Only send sampling fields that were actually set (never ``null``)."""
        params = dict(kwargs)
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return params

    # -- completions ------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        role: str = "chat",
        model: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        """Return a single completed assistant message as text."""
        resp = self.client.chat.completions.create(
            model=self._pick_model(role, model),
            messages=messages,
            stream=False,
            **self._sampling(temperature, max_tokens, kwargs),
        )
        return resp.choices[0].message.content or ""

    def stream(
        self,
        messages: list[dict],
        role: str = "chat",
        model: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """Yield assistant text deltas as they arrive."""
        stream = self.client.chat.completions.create(
            model=self._pick_model(role, model),
            messages=messages,
            stream=True,
            **self._sampling(temperature, max_tokens, kwargs),
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
        input_type: str | None = None,
        **kwargs,
    ) -> list[list[float]]:
        """Embed a batch of strings, returning one vector per input.

        ``input_type`` is ``"query"`` for search queries or ``"passage"`` for
        documents being indexed. On NIM it is sent as the ``input_type`` field that
        asymmetric retrieval models such as ``nvidia/nv-embedqa-e5-v5`` require
        (``"query"`` when not given). On Ollama it becomes the task prefix the
        model was trained with, when it has one (``nomic-embed-text``,
        ``mxbai-embed-large``); other local models get the text unchanged.
        """
        if input_type is not None and input_type not in INPUT_TYPES:
            raise ValueError(f"input_type must be one of {INPUT_TYPES}, got {input_type!r}.")
        items = list(texts)
        if not items:
            return []
        model_id = self._pick_model(role, model)
        if self.backend == "nim":
            extra_body = dict(kwargs.pop("extra_body", None) or {})
            extra_body.setdefault("input_type", input_type or "query")
            kwargs["extra_body"] = extra_body
        else:
            prefix = local_embed_prefix(model_id, input_type)
            if prefix:
                items = [prefix + text for text in items]
        resp = self.client.embeddings.create(model=model_id, input=items, **kwargs)
        rows = sorted(resp.data, key=lambda row: row.index)
        return [row.embedding for row in rows]


def describe_active() -> str:
    """One-line human summary of the current backend and chat model."""
    backend = resolve_backend()
    base = ollama_base_url() if backend == "ollama" else nim_base_url()
    return f"backend={backend}  chat_model={resolve_model('chat', backend)}  endpoint={base}"


if __name__ == "__main__":
    # Quick sanity check without sending any request.
    try:
        print(describe_active())
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
