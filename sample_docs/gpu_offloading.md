# GPU offloading and memory

A language model runs fastest when its weights sit entirely in GPU memory, or
VRAM. When a model is larger than the available VRAM, the runtime keeps some
layers on the GPU and the rest in system RAM, computing the CPU-resident layers
much more slowly. This split is called offloading.

## The num_gpu setting

Ollama decides automatically how many layers to place on the GPU based on how much
VRAM is free. You can override this with the `num_gpu` option, which sets the
number of model layers to offload to the GPU. Setting `num_gpu` to a high value
forces more layers onto the GPU; if that exceeds VRAM, loading fails or falls back
to CPU, so raise it gradually. Setting `num_gpu` to 0 runs entirely on the CPU,
which is slow but always works.

## Context window versus memory

The context window is how many tokens the model can attend to at once. A larger
context, set with `num_ctx`, costs memory: the key-value cache grows with the
context length, and that cache competes with the model weights for VRAM. On a
12 GB GPU an 8B model leaves plenty of room for a large context, but a 14B model
fills most of the VRAM with weights, so you must shrink `num_ctx` to avoid
spilling into system RAM. If tokens-per-second drops sharply when you increase the
context, the KV cache has pushed layers off the GPU.

## Practical guidance for a 12 GB GPU

A quantized 8B model uses about 5 GB of VRAM and runs comfortably with a context
of 8k tokens or more. A quantized 14B model uses about 9 GB and fits with a
context around 4k tokens. Vision models need extra memory for the image encoder,
so budget conservatively. Watch VRAM usage while a model loads: if the runtime
reports offloading only part of the layers to the GPU, either pick a smaller model
or a heavier quantization to reclaim memory.

## Keeping the GPU busy

Batch size and parallelism also affect throughput. Serving several requests at
once shares the loaded weights and raises total tokens-per-second, at the cost of
higher latency per request. For a single interactive chat, the default settings
are usually best.
