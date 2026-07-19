# Ollama basics

Ollama runs open-weight large language models on your own machine. It downloads a
model once, keeps it in a local store, and serves it over a small HTTP API on port
11434. Nothing you type leaves your computer.

## Pulling and running models

You pull a model by its tag, which combines a name and a size or quantization:

    ollama pull llama3.1:8b
    ollama run llama3.1:8b

The `run` command starts an interactive chat. The first token takes a moment
because the model has to be loaded into memory; after that, generation is fast.
Ollama keeps a model resident for five minutes by default so follow-up prompts do
not pay the load cost again. You can change how long a model stays loaded with the
`keep_alive` parameter on an API request.

## The HTTP API

Two API surfaces exist. The native API lives under `/api` and exposes endpoints
like `/api/generate`, `/api/chat`, `/api/tags` for listing installed models,
`/api/pull` for downloading, and `/api/show` for metadata. The native chat and
generate responses include timing fields such as `eval_count` and
`eval_duration`, which let you compute an exact tokens-per-second figure.

Ollama also exposes an OpenAI-compatible API under `/v1`. Point any OpenAI client
at `http://localhost:11434/v1`, use any non-empty string as the API key, and the
familiar `chat.completions` and `embeddings` calls work unchanged. This is what
makes it possible to write code once and run it against either a local model or a
cloud provider by changing only the base URL and key.

## Model storage and disk usage

Models are stored as content-addressed layers, so two tags that share a base
model do not duplicate the weights on disk. Use `ollama list` to see installed
models and their sizes. A quantized 8B model is roughly 4 to 5 GB; a 14B model is
around 9 GB. Removing a model with `ollama rm` frees its unshared layers.

## Modelfiles

A Modelfile customizes a model: it can set a system prompt, adjust default
parameters like temperature and context length, and point at a base model. You
build a named model from a Modelfile with `ollama create mymodel -f Modelfile`.
This is the local equivalent of saving a configured assistant.
