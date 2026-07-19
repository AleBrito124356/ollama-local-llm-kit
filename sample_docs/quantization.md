# Quantization levels

Quantization shrinks a model by storing its weights at lower numerical precision.
A model trained in 16-bit floating point can be converted to 8, 5, or 4 bits per
weight. Fewer bits means a smaller file, less memory, and faster inference, at a
gradual cost to answer quality.

## Reading a quant tag

Ollama tags use names like `Q4_K_M`, `Q5_K_M`, and `Q8_0`. The number is the
approximate bits per weight. The `K` denotes the k-quant family, which allocates
precision unevenly across a tensor to preserve the most important weights. The
trailing letter is the size within that family: `S` for small, `M` for medium,
`L` for large. `Q4_K_M` is the most common default because it balances size and
quality well.

## Choosing a level

- Q4_K_M is the recommended default. It roughly quarters the memory of the 16-bit
  model with only a small quality loss, and it is what most Ollama tags pull by
  default. Use it when you want the best size-to-quality tradeoff.
- Q5_K_M keeps a little more quality for a modest increase in size. Choose it when
  you have spare VRAM and want the answers slightly sharper.
- Q8_0 is near-lossless relative to the original weights but roughly doubles the
  memory of a Q4 model. Use it only when you have plenty of VRAM and quality
  matters more than speed.
- Below 4 bits, quality degrades noticeably. Prefer a smaller model at Q4 over a
  larger model squeezed to 3 bits or less.

## Memory rule of thumb

Estimate memory as bits-per-weight times the parameter count, divided by eight, to
get bytes. An 8B model at Q4 is about 4 to 5 GB; the same model at Q8 is about 8 to
9 GB. Add memory for the context window's key-value cache on top of the weights.

## When quality matters more than size

If a local Q4 model is not accurate enough for a task, you have two levers before
reaching for the cloud: move to a higher quantization like Q5 or Q8 if VRAM
allows, or move up a size class, for example from an 8B to a 14B model. If neither
fits your hardware, switching the backend to a large cloud model is the escape
hatch, and with an OpenAI-compatible client that switch is a one-line change.
