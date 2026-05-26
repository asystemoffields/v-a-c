# VAC: Variable Allocation Compression

**Compress a LLM with minimal capability loss.**

VAC is a structured compression toolkit that uses Fisher-informed low-rank factorization with evolutionary strategy search to find the optimal compression for each weight matrix in a transformer model. Unlike uniform quantization, VAC allocates compression budgets *per-matrix* using a multiple-choice knapsack solver, achieving dramatically better quality at the same storage cost.

> **Note:** VAC produces factorized models that run via HuggingFace Transformers with `trust_remote_code=True`. GGUF/llama.cpp/Ollama support is not yet available — the factorized format requires a custom inference path. See [Limitations](#limitations) for details.

## Key Results

| Method | Pre-KD PPL | Compression | Notes |
|--------|-----------|-------------|-------|
| Naive SVD (uniform 2x) | 9,739 | 2.0x | Model destroyed |
| Sequential Fisher (v1) | 144 | 2.0x | 67x better than naive |
| **VAC evolved (v2)** | **90.54** | **1.8x** | **39% better than v1** |
| After full recovery | ~27 | 1.8x | Within 6 PPL of teacher |

The evolved strategy discovered three key insights:
- **Middle-out compression order** (+21% over front-to-back): compress the easy middle layers first so hard layers get accurate Fisher on a barely-distorted model
- **Cube-root Fisher scaling** (+18% over sqrt): gentler weighting avoids over-trusting the diagonal Fisher approximation
- **Attention-heavy allocation**: attention absorbs 4x compression with minimal damage; MLP is the sensitive component

## Installation

```bash
pip install vac-compress
```

Or from source:

```bash
git clone https://github.com/asystemoffields/v-a-c.git
cd v-a-c
pip install -e .
```

## Quick Start

```python
import torch
from vac import compress_model

# Compress any HuggingFace model
model, metadata = compress_model(
    "allenai/OLMo-3-7B-Think",
    target_ratio=2.0,
    device="cuda",
)
# model is now a factorized transformer with ~50% fewer stored parameters
# and ~2x faster inference (fewer FLOPs per layer)
```

## Loading a Compressed Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the VAC-compressed model (requires trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "asystemoffields/OLMo-3-7B-Think-VAC",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-3-7B-Think")

# For smaller GPUs (8+ GB), use INT8 quantization on load:
# model = AutoModelForCausalLM.from_pretrained(
#     "asystemoffields/OLMo-3-7B-Think-VAC",
#     trust_remote_code=True,
#     load_in_8bit=True,
# )

messages = [{"role": "user", "content": "What is the capital of France?"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
output = model.generate(inputs.to(model.device), max_new_tokens=512, temperature=0.6, top_p=0.95, do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## How It Works

1. **Profile** each weight matrix with diagonal Fisher information (sensitivity analysis)
2. **Allocate** per-matrix rank budgets via MCKP knapsack (spend bits where they matter)
3. **Compress sequentially** with Fisher recomputed per layer (accounts for error propagation). The evolved strategy uses middle-out order and cube-root Fisher scaling.
4. **Recover** via knowledge distillation on the model's original training data

The sequential compression is the key breakthrough: each layer is optimized for the *actual distorted activations* it will see at inference, not the pristine activations from the original model. This single change gives 67x better perplexity than naive SVD at the same compression ratio.

## Comparison

| Format | Download | VRAM | Quality | Inference Speed |
|--------|----------|------|---------|-----------------|
| Original (bf16) | 14.6 GB | 14.6 GB | Baseline (PPL 21) | 1.0x |
| Original GPTQ Q4 | 4.1 GB | ~5 GB | Good (PPL ~23) | ~1.0x |
| **VAC 1.8x (bf16)** | **8.9 GB** | **8.9 GB** | **PPL 27** | **~1.8x faster** |
| **VAC 1.8x (INT8)** | **8.9 GB** | **~4.5 GB** | **PPL 27.3** | **~1.8x faster** |

VAC provides both smaller storage AND faster inference. Quantization-only methods (GPTQ, AWQ) reduce storage but perform the same number of FLOPs. VAC's factorized layers (`x @ B.T @ A.T` instead of `x @ W.T`) have physically fewer multiply-accumulate operations.

## Limitations

- **No GGUF support.** VAC models cannot currently be run in llama.cpp, Ollama, or LM Studio. The factorized layer format (two smaller matrices per layer) is not supported by these inference engines. A [plan exists](docs/llama_cpp.md) for integration via llama.cpp's LoRA mechanism, but it requires ecosystem work.
- **Requires `trust_remote_code=True`** for loading via HuggingFace Transformers.
- **Loading requires ~16 GB system RAM** (the model loads to CPU first, then moves to GPU). GPU VRAM requirement is only 8.9 GB (bf16) or ~4.5 GB (INT8).
- **Not lossless.** There is a ~6 PPL gap from the teacher model. For most interactive use this is imperceptible, but on precise benchmarks (math, code) there may be measurable differences.

## Pipeline Overview

```
Profile (Fisher)  -->  Allocate (MCKP)  -->  Compress (Sequential Fisher SVD)
     30 min               5 min                    1 hour

  -->  KD Recovery (DOLMA)  -->  Post-training (SFT/DPO)  -->  Package
         4-6 hours                  4-6 hours                  10 min
```

Total: ~12-20 H100-hours for a complete 7B compression run.

## Advanced Usage

### Evolutionary Strategy Search

Find the optimal compression strategy for your model:

```python
from vac.evolve import run_evolution

best_genome, fitness = run_evolution(
    model_name="your-model",
    target_ratio=2.0,
    population_size=30,
    n_generations=50,
)
```

### Custom Rank Allocation

```python
from vac.allocate import solve_allocation
from vac.profile import profile_model

# Profile all matrices
results = profile_model("your-model", n_calib=64, seq_len=4096)

# Solve optimal allocation
allocation = solve_allocation(results, target_ratio=2.0, n_layers=32)
```

### Knowledge Distillation

```python
from vac.kd import train_kd

train_kd(
    student=compressed_model,
    teacher_name="original-model",
    dataset="allenai/dolma",
    n_steps=5000,
    seq_len=4096,
)
```

## Mathematical Foundation

VAC uses a **separable Fisher scaling approximation** for weighted SVD:

1. Compute diagonal Fisher: `F_ij = E[(dL/dW_ij)^2]`
2. Extract row/column marginals: `s_row = sqrt(mean(F, dim=1))`, `s_col = sqrt(mean(F, dim=0))`
3. Scale the weight: `W_scaled = diag(s_row) @ W @ diag(s_col)`
4. Standard SVD on the scaled matrix: `W_scaled = U S V^T`
5. Truncate to rank r and un-scale back to original coordinates

This makes the SVD preferentially discard directions that are *functionally unimportant* (low Fisher), preserving the model's behavior even at aggressive compression ratios.

See [docs/math.md](docs/math.md) for the complete mathematical specification.

## Prior Art & References

VAC builds on and extends prior work in neural network compression:

- **GPTQ** (Frantar et al., 2022) — Layer-wise sequential quantization with Hessian information. VAC's sequential compression approach (processing layers front-to-back with updated statistics) draws from this insight.
- **ASVD** (Activation-aware SVD) — Using activation/gradient information to weight the SVD decomposition. VAC's Fisher-scaled SVD is in this family.
- **Optimal Brain Damage** (LeCun et al., 1990) / **Optimal Brain Surgeon** (Hasler & Stork, 1993) — Using second-order (Fisher/Hessian) information to identify removable parameters. The foundational insight behind Fisher-guided compression.
- **Knowledge Distillation** (Hinton et al., 2015) — Training a student to match a teacher's soft predictions. Used in VAC's recovery phase.
- **QuaRot / SpinQuant** (Ashkboos et al., 2024) — Hadamard rotations to improve quantization. Related to VAC's exploration of rotation-before-compression.
- **PMRA** (asystemoffields) — Per-matrix rate allocation for quantization. VAC extends this framework to structural (rank) allocation.

## Citation

If you use VAC in your research, please cite:

```bibtex
@software{vac2026,
  title={VAC: Variable Allocation Compression},
  author={asystemoffields},
  year={2026},
  url={https://github.com/asystemoffields/v-a-c},
}
```

## Acknowledgments

- **Allen AI** for publishing OLMo with full training data, post-training datasets, and evaluation infrastructure. Their radical openness made this research possible.
- Built on [PyTorch](https://pytorch.org/) and [HuggingFace Transformers](https://huggingface.co/transformers/).

## License

Apache 2.0. See [LICENSE](LICENSE).
