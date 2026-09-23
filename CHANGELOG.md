# Changelog

## 0.2.0

### Added

- **Offline mode for everything.** `python -m src.fake_ollama` is a fake Ollama / NVIDIA NIM server (native `/api` NDJSON streams with real-looking metrics, OpenAI-compatible `/v1` with SSE, usage, tool calls, JSON mode and embeddings, fault injection, a strict `--nim` mode and a `--realtime` mode). Every tool and example runs against it with no Ollama, GPU or key.
- **Test suite.** `python -m pytest` runs offline with non-loopback networking blocked, including end-to-end runs of every CLI and example. `pyproject.toml` with `[examples]` and `[dev]` extras.
- **`model_manager fit`**: estimates weights + KV cache + overhead per model and context size, the number of layers Ollama can keep on the GPU, and the largest `num_ctx` that stays fully in VRAM (`--vram`, `--ram`, `--ctx`, `--kv-cache f16|q8_0|q4_0`, `--installed`, `--json`). Installed models outside the catalog are planned from their `/api/show` GGUF metadata.
- **`model_manager doctor`**: checks Python, backend, Ollama server and version, the chat/embedding/vision models, the CPU/GPU split of loaded models (`/api/ps`), the NIM key (masked, offline format check; `--online` calls `/models`), GPU and RAM, and whether the chat model fits. Exits 1 when something must be fixed.
- **`src/hardware.py`**: NVIDIA GPU detection through `nvidia-smi`, system RAM on Windows, Linux and macOS.
- **`recommend`** marks each catalog model as fits / partial offload / CPU only / too large for the detected (or given) hardware.
- **RAG over your own documents**: `--docs PATH` (recursive, `--glob`), one incremental cache per corpus in `.rag_cache/`, batched embeddings with a progress bar (`--batch`), `file:Lstart-Lend` citations, `--min-score` (answers "Not found in the documents." without calling the chat model), `ask --json`, a `search` command, and `--embed-backend nim`.
- **Chat**: `--rag-docs PATH` and `/rag on [PATH]`; `/rag` shows the current state.
- **Benchmark v2**: one excluded warm-up plus `--runs N` (medians and min-max), separate prefill and decode tok/s, end-to-end tok/s, the model's VRAM share from `/api/ps` with an offload warning, `--num-ctx`, `--num-gpu`, `--keep-alive`, `--prompt-file`, `--nim-model`, and `--json` / `--csv` / `--markdown` exports (`-` writes to stdout).
- **Client**: `embed(..., input_type="query"|"passage")`; `NIM_BASE_URL` for self-hosted NIM; `NIM_CHAT_SMALL_MODEL`; `mask_key`, `nim_key_problem`, `nim_base_url`; `LLMClient.create(backend, **openai_kwargs)`.
- `models.yaml`: `n_layers`, `n_kv_heads`, `head_dim` and `context_length` for every chat and vision model.

### Fixed

- The chat REPL crashed when a model's output contained text such as `[/INST]` (it was parsed as Rich markup, and the error handler re-raised); tags such as `[bold]` silently disappeared. Model text is now printed literally and error messages are escaped, in the REPL and in RAG answers.
- Every module imported the client twice (`client` and `src.client`), so exceptions, `MODEL_MAP` edits and `.env` loading were duplicated. All modules now use package-relative imports.
- NIM embeddings failed with the default `nvidia/nv-embedqa-e5-v5` model because `input_type` was never sent.
- The `.env.example` placeholder key was accepted as a real key, so the sign-up hint never appeared.
- `chat_small` on NIM resolved to the same 70B model as `chat`.
- Benchmark: streamed `{"error": ...}` events (e.g. out of memory) and HTTP error bodies were reported as successful 0 tok/s runs, and a malformed NDJSON line crashed the run. NIM tok/s included time-to-first-token, so it could not be compared with local decode rates. Embedding-only models such as `bge-m3` were benchmarked with a chat prompt. It announced a warm-up it never ran.
- `du` double-counted tags that share a manifest digest.
- `rag_local ask` embedded the question twice, and the cache was loaded with `allow_pickle=True`.
- `examples/function_calling.py`: the `eval`-based calculator hung on `9**9**9**9` and crashed on `9**9999`; it is now an AST evaluator with size limits that always returns JSON. Bad tool arguments from the model become JSON errors.
- The README sample outputs did not match what the code prints; they are now real outputs, labelled as produced against the fake server.

### Changed (behaviour)

- `MODEL_MAP` now holds the defaults, and the `OLLAMA_*_MODEL` / `NIM_*_MODEL` environment variables override them when a request is made rather than at import time.
- `.env` is read from the current directory or the repo root only, no longer from any parent directory.
- `OLLAMA_HOST` accepts Ollama's own forms (`host:port`, `0.0.0.0:port`).
- The Ollama client no longer retries failed requests (a local server that refuses a connection will not recover during a back-off); NIM keeps the SDK's retries.
- Unset `max_tokens` is omitted from requests instead of being sent as `null`.
- Sizes are shown in decimal units (GB = 10^9 bytes), like `ollama list`.
- `du` shows one row per set of shared blobs with a "Shared with" column.
- Benchmark `Result.gen_time_s` became `decode_s`; `tokens_per_sec` is kept as an alias of `decode_tps`. The hard-coded RTX 5070 note is replaced by the detected hardware.
- RAG caches moved from `.rag_cache.npz` to `.rag_cache/<hash>.npz` (the old file is ignored and can be deleted). `Chunk` gained `start_line` / `end_line`, and the context block and sources cite `file:Lstart-Lend`.
- `models.yaml`: `llama3.1:8b` size updated to 4.9 GB, matching the Q4_K_M tag Ollama serves.

## 0.1.0

- Initial release: unified Ollama / NIM client, model manager, local RAG over `sample_docs/`, streaming chat REPL, tokens/sec benchmark and three examples.
