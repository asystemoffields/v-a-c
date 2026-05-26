# llama.cpp Integration Plan

## The Core Idea

Ship a compressed model (~3 GB download, ~4-5 GB runtime) that runs in
llama.cpp using the existing LoRA code path with a small optimization patch.

**Key insight:** llama.cpp already supports LoRA adapters (A and B matrices).
A VAC factorized model IS a LoRA with no base weight. If W=0 and scale=1.0:

    output = 0 + 1.0 * B * (A * x) = B*A*x

## Architecture

```
User downloads:

  stub-base.gguf    (~1.3 GB)
  |-- embed_tokens, lm_head        (REAL weights)
  |-- layer norms                  (REAL weights)
  |-- linear layers                (STUBS, zero-filled)

  vac-adapter.gguf  (~2 GB at Q8, ~1 GB at Q4)
  |-- All 224 factorized A/B matrix pairs
  |-- lora_alpha = 1.0 (B pre-scaled by rank)

  Total: ~2.3-3.3 GB
```

## Size Comparison

| Format | Size |
|--------|------|
| Original F16 | 14.6 GB |
| Original Q8 | 7.3 GB |
| Original Q4 | 4.1 GB |
| **VAC Q8** | **3.5 GB** |
| **VAC Q4** | **2.2 GB** |

VAC Q8 (3.5 GB) is smaller than the original at Q4 (4.1 GB) while being
Q8 quality (near-lossless quantization of the factors).

## Per-Layer Rank Variation

VAC uses different ranks for attention vs MLP. llama.cpp LoRA uses a global
alpha, but A/B dimensions encode rank implicitly. The solution: pre-scale
each B matrix so the global alpha works correctly.

```python
# For each layer with rank r:
# llama.cpp computes: output = (alpha/rank) * B * (A * x)
# Set alpha = 1.0, store B_scaled = B * rank
# Then: output = (1/rank) * (B*rank) * (A*x) = B*A*x
```

## Phase 1: Works Today (No Patch)

The stub + LoRA approach works with stock llama.cpp. The only waste is
~500 MB of memory for the zero stubs and multiply-by-zero computation.

```bash
./llama-server \
    -m OLMo-3-3.5B-Think-VAC-stub.gguf \
    --lora OLMo-3-3.5B-Think-VAC-adapter.gguf \
    --host 0.0.0.0 --port 8080
```

## Phase 2: Optimized (5-Line Patch)

A small patch to `build_lora_mm()` in llama.cpp skips the base matmul
when the weight is flagged as a stub:

```cpp
if (lw != nullptr && (w->flags & GGML_TENSOR_FLAG_STUB)) {
    // Skip base matmul -- LoRA IS the entire computation
    res = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur));
    // ...
}
```

This saves ~500 MB runtime memory and eliminates wasted multiply-by-zero ops.

## Runtime Memory

| Component | Memory |
|-----------|--------|
| Embed + lm_head + norms | ~800 MB |
| LoRA A/B matrices (Q8) | ~2.7 GB |
| KV cache (2048 context) | ~512 MB |
| **Total** | **~4 GB** |

A 7B-quality thinking model in 4 GB of memory.
