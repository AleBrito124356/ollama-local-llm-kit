"""A deterministic fake Ollama / NVIDIA NIM server for offline demos and tests.

It speaks the same wire protocols as the real servers, so the kit's unchanged
code path (``requests`` for Ollama's native API, the official OpenAI SDK for the
``/v1`` API) runs end to end with no Ollama install, no GPU and no API key:

    python -m src.fake_ollama --port 11555
    OLLAMA_HOST=http://127.0.0.1:11555 python -m src.model_manager list

    python -m src.fake_ollama --port 11556 --nim     # strict NIM-like server
    NIM_BASE_URL=http://127.0.0.1:11556/v1 NVIDIA_API_KEY=nvapi-fake-key-for-local-tests \\
        LLM_BACKEND=nim python -m src.chat

What it is NOT: a language model. Replies come from small deterministic rules
(echo, extractive answers over a RAG prompt, heuristic tool calls and JSON
extraction), embeddings are feature-hashed bags of words, and every timing and
memory figure it reports is canned. It exists to exercise the plumbing.

Routes
    Native Ollama:  GET /  /api/version  /api/tags  /api/ps
                    POST /api/show  /api/pull  /api/chat     DELETE /api/delete
    OpenAI-style:   GET /v1/models   POST /v1/chat/completions  /v1/embeddings

Fault injection (prefix any model name):
    oom:<model>      /api/chat streams an ``{"error": ...}`` event after HTTP 200;
                     /v1 answers HTTP 500 with the same message
    http500:<model>  HTTP 500 with a JSON error body on every route
    garbage:<model>  /api/chat emits a malformed NDJSON line

``--nim`` switches to a strict NIM-like mode: only NIM model ids, no ``/api``
routes, ``Authorization: Bearer nvapi-...`` required, and ``input_type`` required
on the asymmetric embedding model.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

FAKE_VERSION = "0.12.3-fake"

STOPWORDS = frozenset(
    """a about an and any are as at be been but by can could do does did for from had has have
    how i if in into is it its just me more most my no not of on or our so some such than that
    the their them then there these they this those to too up us use used using was we were
    what when where which while who why will with would you your""".split()
)

_WORD_RE = re.compile(r"[a-z0-9_]+")
_TOKEN_RE = re.compile(r"\s*\S+")
_TASK_PREFIX_RE = re.compile(r"^(search_query|search_document|query|passage):\s*", re.IGNORECASE)
_MXBAI_QUERY = "Represent this sentence for searching relevant passages: "


def _digest(name: str) -> str:
    return "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()


def stem(word: str) -> str:
    """A tiny suffix stripper so 'layers'/'layer' and 'offloading'/'offload' match."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> list[str]:
    """Lower-cased, stop-word-free, stemmed words of ``text``."""
    return [stem(w) for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


def hashed_embedding(text: str, dim: int) -> list[float]:
    """Deterministic feature-hashed bag-of-words vector, L2-normalised.

    Texts that share content words get a high cosine similarity, so retrieval
    over it is lexically meaningful (a crude BM25-without-IDF stand-in).
    """
    text = _TASK_PREFIX_RE.sub("", text)
    if text.startswith(_MXBAI_QUERY):
        text = text[len(_MXBAI_QUERY):]
    counts: dict[str, int] = {}
    for word in content_words(text):
        counts[word] = counts.get(word, 0) + 1
    vec = [0.0] * dim
    for word, tf in counts.items():
        h = int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "big")
        sign = 1.0 if (h >> 40) & 1 else -1.0
        vec[h % dim] += sign * (1.0 + math.log(tf))
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


def split_tokens(text: str) -> list[str]:
    """Split a reply into stream pieces: one word with its leading whitespace each."""
    return _TOKEN_RE.findall(text)


@dataclass
class FakeModel:
    name: str
    family: str
    parameter_size: str
    quantization: str
    size: int
    capabilities: tuple[str, ...]
    digest: str = ""
    decode_tps: float = 50.0          # canned tokens/s reported in eval_duration
    prefill_tps: float = 1000.0       # canned tokens/s reported in prompt_eval_duration
    load_s: float = 2.0               # canned cold-load time
    gpu_fraction: float = 1.0         # share of the model /api/ps reports on the GPU
    embed_dim: int = 0
    arch: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.digest:
            self.digest = _digest(self.name)

    def tag_entry(self) -> dict:
        return {
            "name": self.name,
            "model": self.name,
            "modified_at": "2026-09-01T12:00:00Z",
            "size": self.size,
            "digest": self.digest,
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": self.family,
                "families": [self.family],
                "parameter_size": self.parameter_size,
                "quantization_level": self.quantization,
            },
        }


def _llama_arch(layers: int, heads: int, kv_heads: int, embd: int, ctx: int, arch: str = "llama") -> dict:
    return {
        "general.architecture": arch,
        f"{arch}.block_count": layers,
        f"{arch}.attention.head_count": heads,
        f"{arch}.attention.head_count_kv": kv_heads,
        f"{arch}.embedding_length": embd,
        f"{arch}.context_length": ctx,
    }


def default_ollama_models() -> list[FakeModel]:
    shared = _digest("llama3.1:8b")
    return [
        FakeModel("llama3.1:8b", "llama", "8.0B", "Q4_K_M", 4_920_753_328, ("completion", "tools"),
                  digest=shared, decode_tps=52.5, prefill_tps=1180.0, load_s=2.1,
                  arch=_llama_arch(32, 32, 8, 4096, 131072)),
        # Same manifest digest as llama3.1:8b: an alias tag that shares every blob.
        FakeModel("llama3.1:latest", "llama", "8.0B", "Q4_K_M", 4_920_753_328, ("completion", "tools"),
                  digest=shared, decode_tps=52.5, prefill_tps=1180.0, load_s=2.1,
                  arch=_llama_arch(32, 32, 8, 4096, 131072)),
        FakeModel("llama3.2:3b", "llama", "3.2B", "Q4_K_M", 2_019_393_189, ("completion", "tools"),
                  decode_tps=95.0, prefill_tps=2400.0, load_s=1.2,
                  arch=_llama_arch(28, 24, 8, 3072, 131072)),
        FakeModel("qwen2.5:14b", "qwen2", "14.8B", "Q4_K_M", 8_988_124_069, ("completion", "tools"),
                  decode_tps=14.2, prefill_tps=310.0, load_s=5.8, gpu_fraction=0.72,
                  arch=_llama_arch(48, 40, 8, 5120, 32768, arch="qwen2")),
        FakeModel("llava:7b", "llama", "7B", "Q4_0", 4_733_363_377, ("completion", "vision"),
                  decode_tps=48.0, prefill_tps=900.0, load_s=2.4,
                  arch=_llama_arch(32, 32, 8, 4096, 32768)),
        FakeModel("nomic-embed-text:latest", "nomic-bert", "137M", "F16", 274_302_450, ("embedding",),
                  embed_dim=768, load_s=0.4),
        FakeModel("bge-m3:latest", "bert", "567M", "F16", 1_157_672_605, ("embedding",),
                  embed_dim=1024, load_s=0.6),
    ]


def default_nim_models() -> list[FakeModel]:
    return [
        FakeModel("meta/llama-3.3-70b-instruct", "llama", "70B", "-", 0, ("completion", "tools")),
        FakeModel("meta/llama-3.1-8b-instruct", "llama", "8B", "-", 0, ("completion", "tools")),
        FakeModel("meta/llama-3.2-90b-vision-instruct", "llama", "90B", "-", 0, ("completion", "vision")),
        FakeModel("nvidia/nv-embedqa-e5-v5", "e5", "335M", "-", 0, ("embedding", "asymmetric"),
                  embed_dim=1024),
    ]


# Pullable models the fake "registry" knows about (anything else fails to pull).
PULLABLE = {
    "qwen2.5:7b": FakeModel("qwen2.5:7b", "qwen2", "7.6B", "Q4_K_M", 4_683_087_332, ("completion", "tools"),
                            decode_tps=55.0, prefill_tps=1300.0,
                            arch=_llama_arch(28, 28, 4, 3584, 32768, arch="qwen2")),
    "phi3.5:3.8b": FakeModel("phi3.5:3.8b", "phi3", "3.8B", "Q4_0", 2_176_178_913, ("completion",),
                             decode_tps=80.0, arch=_llama_arch(32, 32, 32, 3072, 131072, arch="phi3")),
    "mxbai-embed-large:latest": FakeModel("mxbai-embed-large:latest", "bert", "335M", "F16", 669_615_493,
                                          ("embedding",), embed_dim=1024),
}


class FakeState:
    """Mutable server state plus the request log tests read."""

    def __init__(self, nim: bool = False, first_token_delay_s: float = 0.0, token_delay_s: float = 0.0):
        self.nim = nim
        self.first_token_delay_s = first_token_delay_s
        self.token_delay_s = token_delay_s
        self.lock = threading.Lock()
        self.models: dict[str, FakeModel] = {}
        self.loaded: dict[str, dict] = {}
        self.log: list[dict] = []
        self.reset()

    def reset(self) -> None:
        with self.lock:
            models = default_nim_models() if self.nim else default_ollama_models()
            self.models = {m.name: m for m in models}
            self.loaded = {}
            self.log = []

    # -- lookups ----------------------------------------------------------
    def find(self, name: str) -> FakeModel | None:
        if not name:
            return None
        if name in self.models:
            return self.models[name]
        if ":" not in name and f"{name}:latest" in self.models:
            return self.models[f"{name}:latest"]
        return None

    def record(self, entry: dict) -> None:
        with self.lock:
            self.log.append(entry)

    def requests(self, path: str | None = None, method: str | None = None) -> list[dict]:
        with self.lock:
            return [
                e for e in self.log
                if (path is None or e["path"] == path) and (method is None or e["method"] == method)
            ]


def split_fault(name: str) -> tuple[str | None, str]:
    for fault in ("oom", "http500", "garbage"):
        if name.startswith(fault + ":"):
            return fault, name[len(fault) + 1:]
    return None, name


OOM_MESSAGE = "model requires more system memory (11.2 GiB) than is available (7.9 GiB)"


# -- reply generation ------------------------------------------------------------
_CONTEXT_RE = re.compile(r"^\[(\d+)\] \(([^)]*)\) (.*)$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_ARITH_RE = re.compile(r"[-+(]*\d[\d.]*(?:\s*(?:\*\*|//|[-+*/%])\s*[-+(]*\d[\d.]*\)*)+")
_PLACE_RE = re.compile(r"\b(?:in|for|at|from)\s+((?:[A-Z][\w'.-]*)(?:\s+[A-Z][\w'.-]*)*)")
_NAME_RE = re.compile(r"^\s*((?:[A-Z][\w'.-]*)(?:\s+[A-Z][\w'.-]*)*)")
_INT_RE = re.compile(r"(?<![\w.])(\d{1,3})(?![\w.])")

FILLER = (
    "A CPU cache keeps recently used data close to the core so repeated reads skip the slow trip "
    "to main memory. Small fast levels sit in front of larger slower ones, and hits in the nearest "
    "level cost only a few cycles."
)


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _images_of(messages: list[dict]) -> list[bytes]:
    images: list[bytes] = []
    for msg in messages:
        for img in msg.get("images") or []:  # native API: base64 strings
            try:
                images.append(base64.b64decode(img))
            except (ValueError, TypeError):
                pass
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:") and "," in url:
                        try:
                            images.append(base64.b64decode(url.split(",", 1)[1]))
                        except (ValueError, TypeError):
                            pass
    return images


def _describe_image(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return f"a {width}x{height} PNG image ({len(data)} bytes)"
    if data[:3] == b"\xff\xd8\xff":
        return f"a JPEG image ({len(data)} bytes)"
    return f"an image of {len(data)} bytes"


def rag_answer(prompt: str) -> str | None:
    """Extractive answer for a numbered-context RAG prompt, with [n] citations."""
    if "Context:" not in prompt or "Question:" not in prompt:
        return None
    question = prompt.rsplit("Question:", 1)[1].split("\n", 1)[0].strip()
    passages = []
    for block in prompt.split("\n\n"):
        m = _CONTEXT_RE.match(block.strip())
        if m:
            passages.append((int(m.group(1)), m.group(3)))
    if not passages:
        return None
    q_words = set(content_words(question))
    scored = []
    for order, (n, text) in enumerate(passages):
        for pos, sentence in enumerate(_SENTENCE_RE.split(text)):
            overlap = len(q_words & set(content_words(sentence)))
            if overlap:
                scored.append((-overlap, order, pos, n, sentence.strip()))
    if not scored:
        return "I do not know: the provided context does not answer the question."
    scored.sort()
    picked, seen = [], set()
    for _neg, _order, _pos, n, sentence in scored:
        if sentence in seen:
            continue
        seen.add(sentence)
        picked.append(f"{sentence.rstrip('.')} [{n}].")
        if len(picked) == 2:
            break
    return " ".join(picked)


def _schema_in(messages: list[dict]) -> dict | None:
    decoder = json.JSONDecoder()
    for msg in messages:
        text = _text_of(msg.get("content"))
        start = text.find("{")
        while start != -1:
            try:
                obj, _end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                start = text.find("{", start + 1)
                continue
            if isinstance(obj, dict) and isinstance(obj.get("properties"), dict):
                return obj
            start = text.find("{", start + 1)
    return None


def _allows(prop: dict, kind: str) -> bool:
    if prop.get("type") == kind:
        return True
    return any(opt.get("type") == kind for opt in prop.get("anyOf", []) if isinstance(opt, dict))


def json_answer(messages: list[dict]) -> str:
    """Heuristic JSON extraction guided by a JSON schema found in the prompt."""
    user = _text_of(messages[-1].get("content")) if messages else ""
    text = user.split(":", 1)[1].strip() if user.lower().startswith("extract") and ":" in user else user
    schema = _schema_in(messages)
    if not schema:
        return json.dumps({"text": text})
    clauses = [c.strip().rstrip(".") for c in text.split(",") if c.strip()]
    used: set[int] = set()
    out: dict = {}
    props = schema["properties"]
    for key, prop in props.items():
        lname = key.lower()
        if _allows(prop, "integer") or _allows(prop, "number"):
            for i, clause in enumerate(clauses):
                m = _INT_RE.search(clause)
                if m and i not in used:
                    out[key] = int(m.group(1))
                    used.add(i)
                    break
        elif "name" in lname:
            m = _NAME_RE.match(text)
            if m:
                out[key] = m.group(1)
                used.update(i for i, c in enumerate(clauses) if c == m.group(1))
        elif any(w in lname for w in ("city", "location", "place", "country")):
            m = _PLACE_RE.search(text)
            if m:
                out[key] = m.group(1).rstrip(".")
                used.update(i for i, c in enumerate(clauses) if m.group(1) in c)
    for key, prop in props.items():
        if key in out or not _allows(prop, "string"):
            continue
        rest = [c for i, c in enumerate(clauses) if i not in used]
        if rest:
            out[key] = rest[-1]
            used.add(clauses.index(rest[-1]))
    for key in schema.get("required", []):
        out.setdefault(key, None)
    return json.dumps(out)


def tool_calls_for(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Decide which of the offered tools the (fake) model calls for the last user turn."""
    user = _text_of(messages[-1].get("content"))
    calls = []
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        props = (fn.get("parameters") or {}).get("properties", {})
        required = (fn.get("parameters") or {}).get("required", list(props))
        args: dict = {}
        if "expression" in props:
            m = max(_ARITH_RE.findall(user), key=len, default="")
            if m:
                args["expression"] = m.strip()
        keywords = [w for w in name.lower().split("_") if w not in ("get", "set", "do", "run")]
        mentioned = any(k and k in user.lower() for k in keywords)
        for key in props:
            if key in ("city", "location", "place") and mentioned:
                m = _PLACE_RE.search(user)
                if m:
                    args[key] = m.group(1).rstrip("?.!,")
        if args and all(r in args for r in required):
            call_id = "call_" + hashlib.sha1(f"{name}{args}".encode()).hexdigest()[:12]
            calls.append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            })
    return calls


def tool_summary(messages: list[dict]) -> str:
    parts = []
    names = {}
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            names[tc.get("id")] = tc.get("function", {}).get("name", "tool")
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        name = msg.get("name") or names.get(msg.get("tool_call_id"), "tool")
        parts.append(f"{name} returned {msg.get('content')}")
    return "Here is what the tools returned: " + "; ".join(parts) + "."


def compose_reply(model: FakeModel, messages: list[dict], limit: int | None,
                  tools: list[dict] | None = None, json_mode: bool = False) -> tuple[str, list[dict]]:
    """Return (text, tool_calls) for a chat request."""
    last = messages[-1] if messages else {"role": "user", "content": ""}
    if tools and last.get("role") == "user":
        calls = tool_calls_for(messages, tools)
        if calls:
            return "", calls
    if last.get("role") == "tool":
        return tool_summary(messages), []
    if json_mode:
        return json_answer(messages), []
    images = _images_of(messages)
    user_text = _text_of(last.get("content"))
    if images:
        return (
            f"I received {_describe_image(images[-1])}. This offline fake cannot see pixels; "
            f"a real vision model such as {model.name} would describe the picture here."
        ), []
    rag = rag_answer(user_text)
    if rag is not None:
        return rag, []
    text = f"Echo from fake {model.name}: {user_text.strip()}"
    if limit:
        # A max-token budget was given (a benchmark): keep generating until it is used up.
        pieces = split_tokens(text)
        filler = split_tokens(" " + FILLER)
        i = 0
        while len(pieces) < limit:
            pieces.append(filler[i % len(filler)])
            i += 1
        text = "".join(pieces)
    return text, []


def prompt_token_count(messages: list[dict]) -> int:
    return sum(len(_text_of(m.get("content")).split()) + 4 for m in messages) + 1


# -- HTTP layer ------------------------------------------------------------------
class FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FakeOllama/" + FAKE_VERSION

    @property
    def state(self) -> FakeState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt, *args) -> None:  # silence the default stderr access log
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    # -- plumbing ---------------------------------------------------------
    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw.decode("utf-8", "replace")}

    def _send_json(self, status: int, obj) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, status: int, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _start_stream(self, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")
        self.wfile.flush()

    def _end_stream(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _api_error(self, status: int, message: str) -> None:
        if self.path.startswith("/v1"):
            self._send_json(status, {"error": {"message": message, "type": "api_error", "param": None, "code": None}})
        else:
            self._send_json(status, {"error": message})

    def _dispatch(self, method: str) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        body = self._read_body() if method in ("POST", "DELETE") else None
        self.state.record({
            "method": method,
            "path": path,
            "body": body,
            "auth": self.headers.get("Authorization"),
            "time": time.time(),
        })
        routes = {
            ("GET", "/"): self.h_root,
            ("HEAD", "/"): self.h_root,
            ("GET", "/api/version"): self.h_version,
            ("GET", "/api/tags"): self.h_tags,
            ("GET", "/api/ps"): self.h_ps,
            ("POST", "/api/show"): self.h_show,
            ("POST", "/api/pull"): self.h_pull,
            ("DELETE", "/api/delete"): self.h_delete,
            ("POST", "/api/chat"): self.h_chat,
            ("GET", "/v1/models"): self.h_v1_models,
            ("POST", "/v1/chat/completions"): self.h_v1_chat,
            ("POST", "/v1/embeddings"): self.h_v1_embeddings,
        }
        handler = routes.get((method, path))
        if handler is None:
            self._api_error(404, f"404 page not found: {method} {path}")
            return
        if self.state.nim and path.startswith("/api"):
            self._api_error(404, "404 page not found (NIM has no native Ollama API)")
            return
        if self.state.nim and path.startswith("/v1") and not self._nim_authorized():
            self._send_json(401, {"status": 401, "title": "Unauthorized", "detail": "Authentication failed"})
            return
        try:
            handler(body or {})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _nim_authorized(self) -> bool:
        auth = self.headers.get("Authorization") or ""
        token = auth[7:] if auth.startswith("Bearer ") else ""
        return token.startswith("nvapi-") and "XXXX" not in token

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("HEAD")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    # -- native routes ----------------------------------------------------
    def h_root(self, _body) -> None:
        self._send_text(200, "Ollama is running")

    def h_version(self, _body) -> None:
        self._send_json(200, {"version": FAKE_VERSION})

    def h_tags(self, _body) -> None:
        with self.state.lock:
            models = [m.tag_entry() for m in self.state.models.values()]
        self._send_json(200, {"models": models})

    def h_ps(self, _body) -> None:
        with self.state.lock:
            running = []
            for name, info in self.state.loaded.items():
                m = self.state.models.get(name)
                if not m:
                    continue
                entry = m.tag_entry()
                entry.update({
                    "expires_at": "2026-09-01T12:05:00Z",
                    "size_vram": int(m.size * info["gpu_fraction"]),
                    "context_length": info["num_ctx"],
                })
                running.append(entry)
        self._send_json(200, {"models": running})

    def h_show(self, body) -> None:
        name = body.get("model") or body.get("name") or ""
        _fault, name = split_fault(name)
        m = self.state.find(name)
        if not m:
            self._api_error(404, f"model '{name}' not found")
            return
        info = {"general.parameter_count": 0, **m.arch}
        self._send_json(200, {
            "license": "fake",
            "modelfile": f"FROM {m.name}\n",
            "parameters": "stop \"<|eot_id|>\"\ntemperature 0.7" if "completion" in m.capabilities else "",
            "template": "{{ .Prompt }}",
            "details": m.tag_entry()["details"],
            "model_info": info,
            "capabilities": list(m.capabilities),
            "modified_at": "2026-09-01T12:00:00Z",
        })

    def h_pull(self, body) -> None:
        name = body.get("model") or body.get("name") or ""
        full = name if ":" in name else f"{name}:latest"
        known = PULLABLE.get(name) or PULLABLE.get(full) or self.state.find(name)
        stream = body.get("stream", True)
        if known is None:
            if stream:
                self._start_stream("application/x-ndjson")
                self._chunk(json.dumps({"status": "pulling manifest"}).encode() + b"\n")
                self._chunk(json.dumps({"error": "pull model manifest: file does not exist"}).encode() + b"\n")
                self._end_stream()
            else:
                self._api_error(500, "pull model manifest: file does not exist")
            return
        events = [{"status": "pulling manifest"}]
        total = known.size
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            events.append({"status": f"pulling {known.digest[7:19]}", "digest": known.digest,
                           "total": total, "completed": int(total * frac)})
        events += [{"status": "verifying sha256 digest"}, {"status": "writing manifest"}, {"status": "success"}]
        with self.state.lock:
            self.state.models.setdefault(known.name, known)
        if not stream:
            self._send_json(200, {"status": "success"})
            return
        self._start_stream("application/x-ndjson")
        for event in events:
            self._chunk(json.dumps(event).encode() + b"\n")
        self._end_stream()

    def h_delete(self, body) -> None:
        name = body.get("model") or body.get("name") or ""
        with self.state.lock:
            m = self.state.find(name)
            if m:
                self.state.models.pop(m.name, None)
                self.state.loaded.pop(m.name, None)
        if not m:
            self._api_error(404, f"model '{name}' not found")
            return
        self._send_json(200, {})

    def _load(self, m: FakeModel, options: dict, keep_alive) -> float:
        """Mark a model loaded; return the canned load time in seconds."""
        num_gpu = options.get("num_gpu")
        gpu_fraction = 0.0 if num_gpu == 0 else m.gpu_fraction
        with self.state.lock:
            cold = m.name not in self.state.loaded
            if keep_alive in (0, "0", "0s", "0m"):
                self.state.loaded.pop(m.name, None)
            else:
                self.state.loaded[m.name] = {
                    "gpu_fraction": gpu_fraction,
                    "num_ctx": int(options.get("num_ctx") or 4096),
                }
        return m.load_s if cold else 0.015

    def h_chat(self, body) -> None:
        fault, name = split_fault(body.get("model", ""))
        m = self.state.find(name)
        if not m:
            self._api_error(404, f'model "{name}" not found, try pulling it first')
            return
        if "completion" not in m.capabilities:
            self._api_error(400, f'"{m.name}" does not support chat')
            return
        if fault == "http500":
            self._api_error(500, OOM_MESSAGE)
            return
        options = body.get("options") or {}
        messages = body.get("messages") or []
        limit = options.get("num_predict")
        limit = int(limit) if isinstance(limit, (int, float)) and limit > 0 else None
        load_s = self._load(m, options, body.get("keep_alive"))
        json_mode = body.get("format") in ("json",) or isinstance(body.get("format"), dict)
        text, calls = compose_reply(m, messages, limit, tools=body.get("tools"), json_mode=json_mode)
        pieces = split_tokens(text)
        done_reason = "stop"
        if limit and len(pieces) >= limit:
            pieces, done_reason = pieces[:limit], "length"
        cpu_slowdown = 0.2 if options.get("num_gpu") == 0 else 1.0
        eval_count = max(len(pieces), 1)
        prompt_count = prompt_token_count(messages)
        final = {
            "model": m.name,
            "created_at": "2026-09-01T12:00:00Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": done_reason,
            "total_duration": 0,
            "load_duration": int(load_s * 1e9),
            "prompt_eval_count": prompt_count,
            "prompt_eval_duration": int(round(prompt_count / m.prefill_tps * 1e9)),
            "eval_count": eval_count,
            "eval_duration": int(round(eval_count / (m.decode_tps * cpu_slowdown) * 1e9)),
        }
        final["total_duration"] = final["load_duration"] + final["prompt_eval_duration"] + final["eval_duration"]
        if not body.get("stream", True):
            final["message"] = {"role": "assistant", "content": "".join(pieces)}
            if calls:
                final["message"]["tool_calls"] = [
                    {"function": {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}}
                    for c in calls
                ]
            self._send_json(200, final)
            return
        self._start_stream("application/x-ndjson")
        if fault == "oom":
            self._chunk(json.dumps({"error": OOM_MESSAGE}).encode() + b"\n")
            self._end_stream()
            return
        time.sleep(self.state.first_token_delay_s)
        for i, piece in enumerate(pieces):
            if i:
                time.sleep(self.state.token_delay_s)
            event = {"model": m.name, "created_at": "2026-09-01T12:00:00Z",
                     "message": {"role": "assistant", "content": piece}, "done": False}
            self._chunk(json.dumps(event).encode() + b"\n")
            if fault == "garbage" and i == 0:
                self._chunk(b"{this is not json\n")
        self._chunk(json.dumps(final).encode() + b"\n")
        self._end_stream()

    # -- OpenAI-compatible routes ------------------------------------------
    def h_v1_models(self, _body) -> None:
        with self.state.lock:
            data = [{"id": m.name, "object": "model", "created": 1767225600,
                     "owned_by": "nvidia" if self.state.nim else "library"}
                    for m in self.state.models.values()]
        self._send_json(200, {"object": "list", "data": data})

    def h_v1_chat(self, body) -> None:
        fault, name = split_fault(body.get("model", ""))
        m = self.state.find(name)
        if not m:
            self._api_error(404, f'model "{name}" not found, try pulling it first')
            return
        if "completion" not in m.capabilities:
            self._api_error(400, f'"{m.name}" does not support chat')
            return
        if fault in ("oom", "http500"):
            self._api_error(500, OOM_MESSAGE)
            return
        messages = body.get("messages") or []
        limit = body.get("max_tokens") or body.get("max_completion_tokens")
        limit = int(limit) if isinstance(limit, (int, float)) and limit > 0 else None
        if not self.state.nim:
            self._load(m, {}, body.get("keep_alive"))
        json_mode = (body.get("response_format") or {}).get("type") in ("json_object", "json_schema")
        text, calls = compose_reply(m, messages, limit, tools=body.get("tools"), json_mode=json_mode)
        pieces = split_tokens(text)
        finish = "stop"
        if limit and len(pieces) >= limit:
            pieces, finish = pieces[:limit], "length"
        if calls:
            finish = "tool_calls"
        prompt_count = prompt_token_count(messages)
        usage = {"prompt_tokens": prompt_count, "completion_tokens": len(pieces),
                 "total_tokens": prompt_count + len(pieces)}
        created = int(time.time())
        cid = "chatcmpl-" + hashlib.sha1(f"{created}{text}".encode()).hexdigest()[:16]
        if not body.get("stream"):
            message = {"role": "assistant", "content": "".join(pieces)}
            if calls:
                message["tool_calls"] = calls
            self._send_json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": m.name,
                "system_fingerprint": "fp_fake",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": usage,
            })
            return

        def chunk(delta: dict, finish_reason=None, choices=True, extra=None) -> bytes:
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": m.name,
                   "system_fingerprint": "fp_fake",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}] if choices else []}
            if extra:
                obj.update(extra)
            return b"data: " + json.dumps(obj).encode() + b"\n\n"

        self._start_stream("text/event-stream")
        time.sleep(self.state.first_token_delay_s)
        for i, piece in enumerate(pieces):
            if i:
                time.sleep(self.state.token_delay_s)
            delta = {"content": piece}
            if i == 0:
                delta["role"] = "assistant"
            self._chunk(chunk(delta))
        self._chunk(chunk({}, finish_reason=finish))
        if (body.get("stream_options") or {}).get("include_usage"):
            self._chunk(chunk({}, choices=False, extra={"usage": usage}))
        self._chunk(b"data: [DONE]\n\n")
        self._end_stream()

    def h_v1_embeddings(self, body) -> None:
        fault, name = split_fault(body.get("model", ""))
        m = self.state.find(name)
        if not m:
            self._api_error(404, f'model "{name}" not found, try pulling it first')
            return
        if fault:
            self._api_error(500, OOM_MESSAGE)
            return
        raw = body.get("input")
        inputs = [raw] if isinstance(raw, str) else list(raw or [])
        if not inputs or not all(isinstance(t, str) for t in inputs):
            self._api_error(400, "invalid input: expected a string or a list of strings")
            return
        if self.state.nim and "asymmetric" in m.capabilities and body.get("input_type") not in ("query", "passage"):
            self._send_json(400, {"object": "error", "type": "invalid_request_error", "detail":
                                  "'input_type' parameter is required for asymmetric models"})
            return
        dim = m.embed_dim or 384
        b64 = body.get("encoding_format") == "base64"
        data = []
        for i, text in enumerate(inputs):
            vec = hashed_embedding(text, dim)
            emb = base64.b64encode(struct.pack(f"<{dim}f", *vec)).decode("ascii") if b64 else vec
            data.append({"object": "embedding", "index": i, "embedding": emb})
        tokens = sum(len(t.split()) for t in inputs)
        self._send_json(200, {"object": "list", "data": data, "model": m.name,
                              "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        """Clients hanging up on a keep-alive connection is normal, not an error."""
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class FakeOllamaServer:
    """Run the fake in a background thread: ``with FakeOllamaServer() as fake: fake.url``."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, nim: bool = False,
                 first_token_delay_s: float = 0.0, token_delay_s: float = 0.0, verbose: bool = False):
        self.state = FakeState(nim=nim, first_token_delay_s=first_token_delay_s, token_delay_s=token_delay_s)
        self.httpd = _QuietThreadingHTTPServer((host, port), FakeHandler)
        self.httpd.state = self.state  # type: ignore[attr-defined]
        self.httpd.verbose = verbose  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def v1(self) -> str:
        return f"{self.url}/v1"

    def start(self) -> "FakeOllamaServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="fake-ollama", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def requests(self, path: str | None = None, method: str | None = None) -> list[dict]:
        return self.state.requests(path, method)

    def reset(self) -> None:
        self.state.reset()
        self.state.first_token_delay_s = 0.0
        self.state.token_delay_s = 0.0

    def __enter__(self) -> "FakeOllamaServer":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fake_ollama", description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=11555, help="Port (default 11555; 0 = any free port)")
    parser.add_argument("--nim", action="store_true", help="Strict NVIDIA NIM-like mode")
    parser.add_argument("--ttft-ms", type=float, default=0.0, help="Delay before the first streamed token")
    parser.add_argument("--token-ms", type=float, default=0.0, help="Delay between streamed tokens")
    parser.add_argument("--verbose", action="store_true", help="Print an access log line per request")
    args = parser.parse_args(argv)

    server = FakeOllamaServer(args.host, args.port, nim=args.nim, first_token_delay_s=args.ttft_ms / 1000,
                              token_delay_s=args.token_ms / 1000, verbose=args.verbose)
    kind = "NIM-like" if args.nim else "Ollama"
    print(f"Fake {kind} server listening on {server.url}  (Ctrl+C to stop)", flush=True)
    if args.nim:
        print(f"  NIM_BASE_URL={server.v1}  NVIDIA_API_KEY=nvapi-<anything>  LLM_BACKEND=nim", flush=True)
    else:
        print(f"  OLLAMA_HOST={server.url}", flush=True)
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
