"""Fisher profiling: measure per-matrix sensitivity for rank allocation.

Computes diagonal Fisher information for each weight matrix, then evaluates
compression quality at multiple rank levels to build sensitivity curves.
These curves feed into the MCKP allocator.
"""

from __future__ import annotations

import gc
import math
import time
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from vac.utils import (
    eval_perplexity,
    get_calibration_data,
    parent_and_attr,
    target_module_paths,
    target_rank,
    MATRIX_ORDER,
)


def compute_fisher_for_linear(
    model: nn.Module,
    data: torch.Tensor,
    device: str,
    module_path: str,
    n_samples: int = 4,
    batch_size: int = 1,
) -> torch.Tensor:
    """Compute diagonal empirical Fisher for a Linear module's weight.

    The diagonal Fisher measures how sensitive the loss is to perturbations
    of each weight element: F_ij = E[(dL/dW_ij)^2].

    Only the target parameter has requires_grad=True during computation
    (all others frozen) for memory efficiency.

    Args:
        model: The model containing the target module
        data: Calibration data tensor (n_samples, seq_len)
        device: Torch device
        module_path: Dot-separated path to the Linear module
        n_samples: Number of calibration samples to average over
        batch_size: Batch size for Fisher computation

    Returns:
        Fisher diagonal tensor, same shape as the weight matrix
    """
    module = model.get_submodule(module_path)
    if not isinstance(module, nn.Linear):
        raise TypeError(f"{module_path} is {type(module).__name__}, expected nn.Linear")

    fisher = torch.zeros(module.weight.shape, dtype=torch.float32, device="cpu")
    model.train()

    # Freeze everything except the target
    for p in model.parameters():
        p.requires_grad_(False)
    module.weight.requires_grad_(True)

    n = 0
    for i in range(0, min(n_samples, len(data)), batch_size):
        batch = data[i:i + batch_size].to(device)
        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=batch, labels=batch)
        outputs.loss.backward()
        if module.weight.grad is not None:
            fisher += module.weight.grad.detach().float().cpu().pow(2)
        n += 1
        del batch, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.eval()
    module.weight.requires_grad_(False)
    return fisher / max(n, 1)


def profile_matrix(
    model: nn.Module,
    target_name: str,
    eval_data: torch.Tensor,
    fisher_diag: torch.Tensor,
    device: str,
    base_loss: float,
    ratios: Optional[list[float]] = None,
) -> list[dict]:
    """Profile a single matrix at multiple compression levels.

    For each compression ratio, applies truncated SVD and measures the
    resulting perplexity change (delta_loss).

    Args:
        model: The model
        target_name: Parameter name (e.g., "model.layers.0.self_attn.q_proj.weight")
        eval_data: Evaluation data tensor
        fisher_diag: Pre-computed Fisher diagonal
        device: Torch device
        base_loss: Baseline loss for delta computation
        ratios: Compression ratios to test (default: [2, 4, 8, 16, 32])

    Returns:
        List of result dicts with compression metrics
    """
    if ratios is None:
        ratios = [2.0, 4.0, 8.0, 16.0, 32.0]

    param = dict(model.named_parameters())[target_name]
    W = param.data.float()
    m, n = W.shape
    original_params = m * n

    results = []
    for ratio in ratios:
        rank = target_rank(m, n, ratio)
        stored = rank * (m + n)
        if stored >= original_params:
            continue

        # Compute Fisher-weighted SVD approximation
        from vac.compress import fisher_scaled_svd
        A, B = fisher_scaled_svd(W, fisher_diag, rank)
        approx = A @ B

        # Substitute and evaluate
        original_data = param.data.clone()
        param.data.copy_(approx.to(param.dtype).to(param.device))
        ppl, loss = eval_perplexity(model, eval_data, device)
        param.data.copy_(original_data)

        results.append({
            "matrix": target_name,
            "family": "fisher_svd",
            "original_params": original_params,
            "compressed_params": stored,
            "compression_ratio": original_params / stored,
            "delta_loss": loss - base_loss,
            "ppl": ppl,
            "rank": rank,
        })

        del approx, A, B, original_data
        gc.collect()

    return results


def profile_model(
    model_name: str,
    n_calib: int = 64,
    n_eval: int = 64,
    seq_len: int = 512,
    layers: Optional[list[int]] = None,
    device: str = "cuda",
    ratios: Optional[list[float]] = None,
    verbose: bool = True,
) -> list[dict]:
    """Profile all weight matrices in a model for rank allocation.

    Computes Fisher information and sensitivity curves for each compressible
    matrix. The output feeds directly into vac.allocate.solve_allocation().

    Args:
        model_name: HuggingFace model name or path
        n_calib: Number of calibration samples for Fisher
        n_eval: Number of evaluation samples for PPL
        seq_len: Sequence length
        layers: Which layers to profile (default: [0, mid, last])
        device: Torch device
        ratios: Compression ratios to evaluate per matrix
        verbose: Print progress

    Returns:
        List of profiling result dicts (one per matrix per ratio)
    """
    if verbose:
        print(f"  Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

    # Determine layers to profile
    all_targets = target_module_paths(model)
    all_layers = sorted(set(t[0] for t in all_targets))
    if layers is None:
        # Default: first, middle, last
        if len(all_layers) >= 3:
            layers = [all_layers[0], all_layers[len(all_layers) // 2], all_layers[-1]]
        else:
            layers = all_layers

    # Load data
    if verbose:
        print(f"  Loading calibration data (n={n_calib})...")
    calib_data = get_calibration_data(tokenizer, "train", n_samples=n_calib, seq_len=seq_len)
    eval_data = get_calibration_data(tokenizer, "validation", n_samples=n_eval, seq_len=seq_len)

    # Baseline
    if verbose:
        print(f"  Measuring baseline...")
    base_ppl, base_loss = eval_perplexity(model, eval_data, device)
    if verbose:
        print(f"  Baseline: PPL={base_ppl:.2f}, loss={base_loss:.4f}")

    # Profile target matrices
    targets = []
    for name, param in model.named_parameters():
        if param.ndim < 2 or param.numel() < 50000:
            continue
        for layer_idx in layers:
            if f"layers.{layer_idx}." in name and name.endswith(".weight"):
                targets.append(name)
                break

    if verbose:
        print(f"  Profiling {len(targets)} matrices across layers {layers}...")

    all_results = []
    for i, target_name in enumerate(targets):
        if verbose:
            print(f"  [{i+1}/{len(targets)}] {target_name}")

        fisher = compute_fisher_for_matrix(
            model, calib_data, device, target_name, n_samples=n_calib
        )
        results = profile_matrix(
            model, target_name, eval_data, fisher, device, base_loss, ratios=ratios
        )
        all_results.extend(results)

        del fisher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if verbose:
        print(f"  Profiling complete: {len(all_results)} measurements")

    return all_results


def compute_fisher_for_matrix(
    model: nn.Module,
    data: torch.Tensor,
    device: str,
    target_name: str,
    n_samples: int = 64,
    batch_size: int = 4,
) -> torch.Tensor:
    """Compute diagonal Fisher for a named parameter.

    Similar to compute_fisher_for_linear but takes a parameter name
    instead of a module path.
    """
    param = dict(model.named_parameters())[target_name]
    fisher = torch.zeros_like(param, device="cpu")
    model.train()

    for p in model.parameters():
        p.requires_grad_(False)
    param.requires_grad_(True)

    n = 0
    for i in range(0, min(n_samples, len(data)), batch_size):
        batch = data[i:i + batch_size].to(device)
        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=batch, labels=batch)
        outputs.loss.backward()
        if param.grad is not None:
            fisher += (param.grad.detach().cpu() ** 2)
        n += 1
        del batch, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.eval()
    param.requires_grad_(False)
    return fisher / max(n, 1)
