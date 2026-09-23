# GPU notes: fitting models on consumer hardware

This kit targets consumer GPUs with roughly 8 to 16 GB of VRAM. The goal is always
the same: keep the whole model plus its context cache in VRAM, and avoid spilling
layers into system RAM. You do not have to do the arithmetic below by hand; the kit
ships three tools that do it for your machine:

```bash
python -m src.model_manager doctor              # what is installed, what is loaded, and how much of it is on the GPU
python -m src.model_manager fit --ctx 8192      # will each model fit at this context? (detected GPU, or --vram 12)
python -m src.benchmark --models qwen2.5:14b --num-ctx 16384   # measure it for real
```

## The three things that use VRAM

1. **Model weights.** Size depends on parameter count and quantization. A rough
   estimate is `bits_per_weight * params / 8` bytes. An 8B model at Q4 is about
   4 to 5 GB; at Q8 about 8 to 9 GB. `fit` uses the download size from
   `models.yaml` (or `/api/tags` for any installed model).
2. **KV cache (the context window).** Grows linearly with `num_ctx`:

   ```text
   KV bytes = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_element x num_ctx
   ```

   For `llama3.1:8b` (32 layers, 8 KV heads of 128) that is 128 KiB per token at
   f16, so 8,192 tokens cost exactly 1 GiB. Models without grouped-query attention
   pay much more: `phi3.5:3.8b` (32 KV heads of 96) needs 384 KiB per token. The
   layer and head counts come from `models.yaml`, or from the GGUF metadata that
   `/api/show` returns for any installed model.
3. **Overhead.** The CUDA context, compute buffers and, for vision models, the
   image encoder. `fit` budgets a fixed 1 GiB (`--overhead` to change it).

If the sum exceeds VRAM, Ollama offloads whole layers to the CPU and generation
slows down a lot. `fit` estimates how many layers stay on the GPU; `doctor` and the
benchmark read the real split from Ollama's `/api/ps` (`size_vram` / `size`).

## num_gpu: controlling offload

`num_gpu` is the number of model layers placed on the GPU. Ollama sets it
automatically, but you can override it per request or in a Modelfile.

- Increase `num_gpu` to force more layers onto the GPU. If you exceed VRAM,
  loading fails or falls back to CPU, so raise it in steps and watch VRAM.
- Set `num_gpu` to 0 to run entirely on the CPU. Slow, but a useful fallback.

Try it without editing code: `python -m src.benchmark --models llama3.1:8b --num-gpu 20`.
The benchmark prints how much of the model ended up in VRAM next to the tokens/sec.

## Context window versus memory

`num_ctx` sets the context length. The KV cache scales with it, so a large context
can push weights off the GPU even when the weights alone fit. If tokens-per-second
falls off a cliff after you raise `num_ctx`, that is the cause. Lower `num_ctx`, or
pick a smaller model or heavier quantization to make room.

`fit` prints the largest `num_ctx` that keeps each model entirely on your GPU. Two
more levers are worth knowing:

- **KV cache quantization.** Setting `OLLAMA_KV_CACHE_TYPE=q8_0` on the Ollama
  server (it needs flash attention, `OLLAMA_FLASH_ATTENTION=1`) roughly halves the
  cache; `q4_0` quarters it. Plan with `fit --kv-cache q8_0`.
- **Models with fewer KV heads.** `qwen2.5:7b` has only 4 KV heads, so its long
  contexts are cheap.

## Quantization levels quick reference

| Level   | Bits/weight | 8B size | Quality        | Use when                          |
|---------|-------------|---------|----------------|-----------------------------------|
| Q4_K_M  | ~4          | ~4.7 GB | very good      | default; best size-to-quality     |
| Q5_K_M  | ~5          | ~5.7 GB | slightly better| spare VRAM, want sharper answers  |
| Q8_0    | ~8          | ~8.5 GB | near-lossless  | plenty of VRAM, quality critical  |

Below 4 bits, quality drops noticeably; prefer a smaller model at Q4 over a larger
model squeezed under 4 bits.

## Sizing table for a 12 GB GPU

Generated with `python -m src.model_manager fit --vram 12 --ram 32 --ctx <N>`
(f16 KV cache, 1 GiB overhead). These are estimates from the formula above, not
measurements; run the benchmark to confirm on your card.

| Model               | Weights  | Total at 8k ctx | At 8k ctx    | At 32k ctx                 | Largest ctx fully on GPU |
|---------------------|----------|-----------------|--------------|----------------------------|--------------------------|
| llama3.2:3b         | 1.9 GiB  | 3.7 GiB         | fits in VRAM | fits in VRAM               | 84,992                   |
| llama3.1:8b         | 4.6 GiB  | 6.6 GiB         | fits in VRAM | fits in VRAM               | 52,224                   |
| qwen2.5:7b          | 4.4 GiB  | 5.8 GiB         | fits in VRAM | fits in VRAM               | 32,768 (model maximum)   |
| qwen2.5:14b         | 8.4 GiB  | 10.9 GiB        | fits in VRAM | partial offload (~75% GPU) | 13,312                   |
| phi3.5:3.8b         | 2.0 GiB  | 6.0 GiB         | fits in VRAM | partial offload (~78% GPU) | 23,552                   |
| llava:7b            | 4.4 GiB  | 6.4 GiB         | fits in VRAM | fits in VRAM               | 32,768 (model maximum)   |
| llama3.2-vision:11b | 7.4 GiB  | 9.6 GiB         | fits in VRAM | partial offload (~88% GPU) | 23,552                   |

On an 8 GB card (`--vram 8`) at 8k context, `qwen2.5:14b` and `llama3.2-vision:11b`
no longer fit (about 71% and 80% of their layers on the GPU), and `llama3.1:8b` fits
up to about 19k tokens.

## When local is not enough

If a local Q4 model is not accurate enough, try the cheap levers first: move to
Q5 or Q8 if VRAM allows, or step up a size class (8B to 14B). If your hardware
cannot hold a big enough model, switch the backend to free cloud NVIDIA NIM. With
the OpenAI-compatible client in this kit that is a single environment variable
change; the rest of your code is identical. See the local-versus-cloud decision
table in the README.

## Measuring your own numbers

Do not guess. Run the benchmark against your installed models:

```bash
python -m src.benchmark
python -m src.benchmark --models llama3.1:8b qwen2.5:14b --num-predict 256 --runs 5
python -m src.benchmark --models qwen2.5:14b --num-ctx 32768 --markdown results.md
```

Each model gets one warm-up request (its load time is reported as "cold load" and
excluded) and then `--runs` measured requests; the table shows medians and the
min-max decode range. For local models the prefill and decode rates come straight
from Ollama's own `prompt_eval_*` and `eval_*` counters, so they reflect what your
GPU actually did, and the "On GPU" column shows how much of the model Ollama kept
in VRAM during the run.
