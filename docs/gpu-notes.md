# GPU notes: fitting models on consumer hardware

This kit is tuned for a consumer GPU with roughly 8 to 16 GB of VRAM. The examples
below use an RTX 5070 with 12 GB, but the reasoning transfers to any card. The
goal is always the same: keep the whole model plus its context cache in VRAM, and
avoid spilling layers into system RAM.

## The three things that use VRAM

1. **Model weights.** Size depends on parameter count and quantization. A rough
   estimate is `bits_per_weight * params / 8` bytes. An 8B model at Q4 is about
   4 to 5 GB; at Q8 about 8 to 9 GB.
2. **KV cache (the context window).** Grows with `num_ctx`. Longer context means
   more cache, which competes with the weights for VRAM.
3. **Overhead.** The CUDA context, activations, and for vision models the image
   encoder. Budget a few hundred MB to a couple of GB.

If the sum exceeds VRAM, Ollama offloads some layers to the CPU and generation
slows down a lot.

## num_gpu: controlling offload

`num_gpu` is the number of model layers placed on the GPU. Ollama sets it
automatically, but you can override it per request or in a Modelfile.

- Increase `num_gpu` to force more layers onto the GPU. If you exceed VRAM,
  loading fails or falls back to CPU, so raise it in steps and watch VRAM.
- Set `num_gpu` to 0 to run entirely on the CPU. Slow, but a useful fallback.

Watch the load logs: Ollama prints how many layers were offloaded to the GPU. If
it is fewer than the model's total, the model did not fully fit.

## Context window versus memory

`num_ctx` sets the context length. The KV cache scales with it, so a large context
can push weights off the GPU even when the weights alone fit. If tokens-per-second
falls off a cliff after you raise `num_ctx`, that is the cause. Lower `num_ctx`, or
pick a smaller model or heavier quantization to make room.

## Quantization levels quick reference

| Level   | Bits/weight | 8B size | Quality        | Use when                          |
|---------|-------------|---------|----------------|-----------------------------------|
| Q4_K_M  | ~4          | ~4.7 GB | very good      | default; best size-to-quality     |
| Q5_K_M  | ~5          | ~5.7 GB | slightly better| spare VRAM, want sharper answers  |
| Q8_0    | ~8          | ~8.5 GB | near-lossless  | plenty of VRAM, quality critical  |

Below 4 bits, quality drops noticeably; prefer a smaller model at Q4 over a larger
model squeezed under 4 bits.

## Sizing table for a 12 GB GPU (RTX 5070)

| Model            | Quant  | Weights | Fits fully? | Suggested num_ctx |
|------------------|--------|---------|-------------|-------------------|
| llama3.2:3b      | Q4_K_M | ~2 GB   | yes         | 8k or more        |
| llama3.1:8b      | Q4_K_M | ~4.7 GB | yes         | 8k comfortably    |
| qwen2.5:14b      | Q4_K_M | ~9 GB   | yes, tight  | ~4k               |
| llava:7b         | Q4_0   | ~4.7 GB | yes         | 4k plus image     |
| llama3.2-vision  | Q4_K_M | ~7.9 GB | yes, tight  | small context     |

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
python -m src.benchmark --models llama3.1:8b qwen2.5:14b --num-predict 256
```

For local models the tokens-per-second figure comes straight from Ollama's own
`eval_count` and `eval_duration`, so it reflects what your GPU actually did.
