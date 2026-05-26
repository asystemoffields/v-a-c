# VAC: Variable Allocation Compression

**Compress a LLM with minimal capability loss.**

VAC is a structured compression toolkit that uses Fisher-informed low-rank factorization with evolutionary strategy search to find the optimal compression for each weight matrix in a transformer model. Unlike uniform quantization, VAC allocates compression budgets *per-matrix* using a multiple-choice knapsack solver, achieving dramatically better quality at the same storage cost.

## Key Results

| Method | Pre-KD PPL | Compression | Notes |
|--------|-----------|-------------|-------|
| Naive SVD (uniform 2x) | 9,739 | 2.0x | Model destroyed |
| Sequential Fisher (v1) | 144 | 2.0x | 67x better than naive |
| **VAC evolved (v2)** | **90.54** | **1.8x** | **39% better than v1** |
| After KD recovery | ~28 | 1.8x | Within 7 PPL of teacher |

The evolved strategy discovered three key insights:
- **Middle-out compression order** (+21% over front-to-back): compress the easy middle layers first so hard layers get accurate Fisher
- **Cube-root Fisher scaling** (+18% over sqrt): gentler weighting avoids over-trusting the diagonal Fisher approximation
- **Attention-heavy allocation**: attention absorbs 4x compression with minimal damage; MLP can't

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

# Compress any HuggingFace model to ~2x smaller
model, metadata = compress_model(
    "allenai/OLMo-3-7B-Think",
    target_ratio=2.0,
    device="cuda",
)
# model is now a factorized transformer with ~50% fewer parameters
```

## Loading a Compressed Model

```python
from vac.modeling import VACModel

model = VACModel.from_pretrained(
    "asystemoffields/OLMo-3-3.5B-Think-VAC",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-3-7B-Think")
output = model.generate(tokenizer("Hello", return_tensors="pt")["input_ids"].cuda())
```

## How It Works

1. **Profile** each weight matrix with diagonal Fisher information (sensitivity analysis)
2. **Allocate** per-matrix rank budgets via MCKP knapsack (spend bits where they matter)
3. **Compress sequentially** front-to-back with Fisher recomputed per layer (accounts for error propagation)
4. **Recover** via knowledge distillation on the model's original training data

The sequential compression is the key breakthrough: each layer is optimized for the *actual distorted activations* it will see at inference, not the pristine activations from the original model. This single change gives 67x better perplexity than naive SVD at the same compression ratio.

## Comparison: VAC vs Quantization

| Format | Size | Quality | Inference Speed |
|--------|------|---------|-----------------|
| Original (bf16) | 14.6 GB | Baseline | 1.0x |
| GPTQ 4-bit | 4.1 GB | Good | ~1.0x |
| **VAC 2x (bf16 factors)** | **~8 GB** | **Within 7 PPL** | **~2x faster** |

VAC is orthogonal to quantization: you can quantize the factorized matrices for additional compression. A VAC model with Q8 factors is *smaller than the original at Q4* while maintaining Q8 fidelity.

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

## Citation

If you use VAC in your research, please cite:

```bibtex
@software{vac2025,
  title={VAC: Variable Allocation Compression},
  author={Alex (asystemoffields)},
  year={2025},
  url={https://github.com/asystemoffields/v-a-c},
}
```

## Acknowledgments

- **Allen AI** for publishing OLMo with full training data, post-training datasets, and evaluation infrastructure. Their radical openness made this research possible.
- Built on [PyTorch](https://pytorch.org/) and [HuggingFace Transformers](https://huggingface.co/transformers/).

## License

Apache 2.0. See [LICENSE](LICENSE).
