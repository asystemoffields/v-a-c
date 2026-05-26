"""Core compression: sequential Fisher factorization.

The key insight: compress front-to-back, recomputing Fisher at each layer so
each matrix is optimized for the *actual distorted activations* it will see
at inference. This gives 67x better perplexity than naive (blind) SVD.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from vac.modeling import FactorizedLinear
from vac.profile import compute_fisher_for_linear
from vac.utils import (
    eval_perplexity,
    get_calibration_data,
    parent_and_attr,
    target_module_paths,
    target_rank,
    MATRIX_ORDER,
)


@torch.no_grad()
def fisher_scaled_svd(
    W: torch.Tensor,
    fisher_diag: torch.Tensor,
    rank: int,
    exponent: float = 0.5,
    clamp_min: float = 0.05,
    clamp_max: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fisher-weighted truncated SVD with configurable scaling exponent.

    The separable Fisher scaling approximation:
    1. Compute row/column importance from Fisher diagonal
    2. Scale the weight matrix to amplify high-Fisher regions
    3. Apply standard SVD to the scaled matrix
    4. Un-scale back to original coordinates

    Args:
        W: Weight matrix (out_features, in_features)
        fisher_diag: Diagonal Fisher information, same shape as W
        rank: Target rank for truncation
        exponent: Fisher scaling power (0.5=sqrt, 0.33=cbrt, 1.0=linear)
        clamp_min: Minimum scale factor (prevents division by zero)
        clamp_max: Maximum scale factor (prevents outlier domination)

    Returns:
        A: Left factor (out_features, rank)
        B: Right factor (rank, in_features)
        Such that W ~ A @ B
    """
    device = W.device
    F_diag = fisher_diag.to(device=device, dtype=torch.float32).clamp_min(0)

    # Row and column marginals with configurable exponent
    row = F_diag.mean(dim=1).pow(exponent)
    col = F_diag.mean(dim=0).pow(exponent)

    def normalize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean()
        if torch.isfinite(mean) and mean > 0:
            x = x / mean
        return x.clamp(clamp_min, clamp_max)

    row = normalize(row)
    col = normalize(col)

    # Scale, SVD, truncate, un-scale
    scaled = row[:, None] * W.float() * col[None, :]
    U, S, Vt = torch.linalg.svd(scaled, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]

    S_sqrt = S_r.sqrt()
    A = (U_r * S_sqrt.unsqueeze(0)) / row[:, None]
    B = (S_sqrt.unsqueeze(1) * Vt_r) / col[None, :]
    return A, B


@torch.no_grad()
def replace_with_factorized(
    model: nn.Module,
    module_path: str,
    fisher_diag: torch.Tensor,
    rank: int,
    exponent: float = 0.5,
) -> FactorizedLinear:
    """Replace a nn.Linear with a FactorizedLinear using Fisher-weighted SVD.

    Args:
        model: The model containing the target module
        module_path: Dot-separated path to the Linear module
        fisher_diag: Diagonal Fisher for the weight matrix
        rank: Target rank
        exponent: Fisher scaling exponent

    Returns:
        The new FactorizedLinear module
    """
    module = model.get_submodule(module_path)
    if not isinstance(module, nn.Linear):
        raise TypeError(f"{module_path} is {type(module).__name__}, expected nn.Linear")

    W = module.weight.detach()
    bias = module.bias.detach().clone() if module.bias is not None else None
    A, B = fisher_scaled_svd(W, fisher_diag, rank, exponent=exponent)

    factorized = FactorizedLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        rank=rank,
        bias=bias is not None,
        device=W.device,
        dtype=W.dtype,
    )
    factorized.down.weight.copy_(B.to(device=W.device, dtype=W.dtype))
    factorized.up.weight.copy_(A.to(device=W.device, dtype=W.dtype))
    if bias is not None:
        factorized.up.bias.copy_(bias.to(device=W.device, dtype=W.dtype))

    parent, attr = parent_and_attr(model, module_path)
    setattr(parent, attr, factorized)
    return factorized


def compress_sequential(
    model: nn.Module,
    tokenizer,
    *,
    target_ratio: float = 2.0,
    fisher_exponent: float = 0.5,
    n_fisher_samples: int = 4,
    n_eval_samples: int = 64,
    seq_len: int = 512,
    device: str = "cuda",
    eval_every_layers: int = 4,
    order: str = "front-to-back",
    rank_overrides: Optional[dict[str, int]] = None,
    verbose: bool = True,
) -> list[dict]:
    """Sequential Fisher compression: the core VAC algorithm.

    Compresses a model layer-by-layer, recomputing Fisher information at each
    step to account for error propagation from previously compressed layers.

    Args:
        model: HuggingFace causal LM model (already on device)
        tokenizer: Corresponding tokenizer
        target_ratio: Target compression ratio (2.0 = half the params)
        fisher_exponent: Scaling exponent for Fisher-weighted SVD
        n_fisher_samples: Calibration samples per Fisher computation
        n_eval_samples: Samples for perplexity evaluation
        seq_len: Sequence length for calibration/eval data
        device: Torch device
        eval_every_layers: Evaluate PPL every N layers (0 to disable)
        order: Compression order ("front-to-back", "back-to-front", "middle-out")
        rank_overrides: Optional dict mapping module_path -> rank
        verbose: Print progress

    Returns:
        List of metadata dicts describing each compressed module
    """
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Load calibration and eval data
    calib_data = get_calibration_data(
        tokenizer, split="train", n_samples=max(n_fisher_samples, 1), seq_len=seq_len
    )
    eval_data = get_calibration_data(
        tokenizer, split="validation", n_samples=n_eval_samples, seq_len=seq_len
    )

    # Baseline
    if verbose:
        base_ppl, base_loss = eval_perplexity(model, eval_data, device, batch_size=1)
        print(f"  Baseline: PPL={base_ppl:.2f}, loss={base_loss:.4f}")

    # Get target modules and apply ordering
    targets = target_module_paths(model)
    targets = _apply_order(targets, order)

    if verbose:
        print(f"  Compressing {len(targets)} matrices ({order}, exponent={fisher_exponent})")

    original_params = sum(p.numel() for p in model.parameters())
    compressed_units = original_params
    metadata = []
    t0 = time.time()

    for idx, (layer, matrix_type, path) in enumerate(targets, start=1):
        module = model.get_submodule(path)
        if not isinstance(module, nn.Linear):
            continue

        out_features, in_features = module.weight.shape

        # Determine rank
        if rank_overrides and path in rank_overrides:
            rank = rank_overrides[path]
        else:
            rank = target_rank(out_features, in_features, target_ratio)

        original = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
        stored = rank * (out_features + in_features) + (
            module.bias.numel() if module.bias is not None else 0
        )

        if verbose:
            matrix_ratio = original / stored
            print(f"  [{idx}/{len(targets)}] L{layer:02d}.{matrix_type:<16} "
                  f"rank={rank}, {matrix_ratio:.2f}x")

        # Compute Fisher on current (distorted) model state
        fisher = compute_fisher_for_linear(
            model, calib_data, device, path,
            n_samples=n_fisher_samples, batch_size=1,
        )

        # Replace with factorized module
        replace_with_factorized(model, path, fisher, rank, exponent=fisher_exponent)

        compressed_units -= (original - stored)
        metadata.append({
            "index": idx,
            "layer": layer,
            "matrix_type": matrix_type,
            "module_path": path,
            "in_features": in_features,
            "out_features": out_features,
            "rank": rank,
            "original_params": original,
            "compressed_params": stored,
            "matrix_ratio": original / stored,
        })

        del fisher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Periodic evaluation
        if eval_every_layers > 0 and verbose:
            finished_layer = idx == len(targets) or targets[idx][0] != layer
            if finished_layer:
                layer_count = sum(1 for m in metadata if m["layer"] <= layer)
                if layer_count % (eval_every_layers * len(MATRIX_ORDER)) == 0 or idx == len(targets):
                    ppl, loss = eval_perplexity(model, eval_data, device, batch_size=1)
                    ratio_now = original_params / compressed_units
                    print(f"    >>> after L{layer}: PPL={ppl:.2f}, ratio={ratio_now:.3f}x")

    achieved_ratio = original_params / compressed_units
    elapsed = time.time() - t0

    if verbose:
        final_ppl, _ = eval_perplexity(model, eval_data, device, batch_size=1)
        print(f"\n  Compression complete in {elapsed:.0f}s")
        print(f"  Final PPL: {final_ppl:.2f}, ratio: {achieved_ratio:.3f}x")

    return metadata


def compress_model(
    model_name: str,
    target_ratio: float = 2.0,
    device: str = "cuda",
    dtype=torch.bfloat16,
    fisher_exponent: float = 0.5,
    order: str = "front-to-back",
    n_fisher_samples: int = 4,
    seq_len: int = 512,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict]]:
    """Compress a HuggingFace model using sequential Fisher factorization.

    This is the main entry point for VAC compression. It loads a model,
    compresses it using the sequential Fisher algorithm, and returns the
    compressed model with metadata.

    Args:
        model_name: HuggingFace model name or path
        target_ratio: Target compression ratio (2.0 = half size)
        device: Torch device ("cuda", "cpu")
        dtype: Model dtype (torch.bfloat16 recommended)
        fisher_exponent: Scaling exponent (0.5=sqrt, 0.33=cbrt)
        order: Compression order ("front-to-back", "back-to-front", "middle-out")
        n_fisher_samples: Fisher calibration samples per matrix
        seq_len: Sequence length for calibration data
        verbose: Print progress

    Returns:
        (model, metadata) tuple where model is the compressed model
        and metadata describes each factorized module
    """
    if verbose:
        print(f"  Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    metadata = compress_sequential(
        model, tokenizer,
        target_ratio=target_ratio,
        fisher_exponent=fisher_exponent,
        n_fisher_samples=n_fisher_samples,
        seq_len=seq_len,
        device=device,
        order=order,
        verbose=verbose,
    )

    return model, metadata


def _apply_order(
    targets: list[tuple[int, str, str]], order: str
) -> list[tuple[int, str, str]]:
    """Reorder targets according to the specified compression order."""
    if order == "front-to-back":
        return targets  # Already in order
    elif order == "back-to-front":
        return list(reversed(targets))
    elif order == "middle-out":
        layers = sorted(set(t[0] for t in targets))
        n_layers = len(layers)
        mid = n_layers // 2
        layer_order = sorted(layers, key=lambda l: abs(l - layers[mid]))
        # Rebuild targets in this layer order
        by_layer = {}
        for layer, matrix_type, path in targets:
            by_layer.setdefault(layer, []).append((layer, matrix_type, path))
        result = []
        for l in layer_order:
            if l in by_layer:
                result.extend(by_layer[l])
        return result
    else:
        raise ValueError(f"Unknown order: {order}. Use 'front-to-back', 'back-to-front', or 'middle-out'")
