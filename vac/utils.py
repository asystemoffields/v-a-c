"""Shared utilities for VAC compression pipeline."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset


# Standard matrix compression order within each transformer layer
MATRIX_ORDER = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def target_rank(out_features: int, in_features: int, target_ratio: float) -> int:
    """Compute target rank for a given compression ratio.

    Given W (out_features x in_features) factorized as A @ B where
    A is (out_features x rank) and B is (rank x in_features):
        storage = rank * (out_features + in_features)
        ratio = (out_features * in_features) / storage

    Inverting: rank = (out * in) / (ratio * (out + in))

    Args:
        out_features: Output dimension of the weight matrix
        in_features: Input dimension of the weight matrix
        target_ratio: Desired compression ratio (2.0 = half the params)

    Returns:
        Target rank, clamped to feasible range [1, min(m,n)-1]
    """
    rank = int((out_features * in_features) / (target_ratio * (out_features + in_features)))
    return max(1, min(rank, min(out_features, in_features) - 1))


def parent_and_attr(model: nn.Module, module_path: str) -> tuple[nn.Module, str]:
    """Get the parent module and attribute name for a given path.

    Args:
        model: Root module
        module_path: Dot-separated path (e.g., "model.layers.0.self_attn.q_proj")

    Returns:
        (parent_module, attribute_name) tuple
    """
    parts = module_path.split(".")
    parent = model.get_submodule(".".join(parts[:-1]))
    return parent, parts[-1]


def target_module_paths(
    model: nn.Module,
    start_layer: int = 0,
    max_layers: Optional[int] = None,
) -> list[tuple[int, str, str]]:
    """Find all compressible Linear modules in a transformer model.

    Searches for standard transformer weight matrices (Q, K, V, O projections
    and MLP gate/up/down projections) across all layers.

    Args:
        model: HuggingFace model
        start_layer: First layer to include
        max_layers: Maximum number of layers (None = all)

    Returns:
        List of (layer_idx, matrix_type, full_path) tuples
    """
    paths = []
    end_layer = start_layer + max_layers if max_layers is not None else None

    for layer in range(start_layer, end_layer or 10_000):
        layer_paths = []
        for matrix_type in MATRIX_ORDER:
            path = f"model.layers.{layer}.{matrix_type}"
            try:
                module = model.get_submodule(path)
            except (AttributeError, ValueError):
                continue
            if isinstance(module, nn.Linear):
                layer_paths.append((layer, matrix_type, path))
        if not layer_paths:
            if layer == start_layer:
                continue
            break
        paths.extend(layer_paths)
    return paths


@torch.no_grad()
def eval_perplexity(
    model: nn.Module,
    data: torch.Tensor,
    device: str,
    batch_size: int = 8,
) -> tuple[float, float]:
    """Evaluate model perplexity on tokenized data.

    Args:
        model: Language model
        data: Tokenized data tensor (n_samples, seq_len)
        device: Torch device
        batch_size: Evaluation batch size

    Returns:
        (perplexity, average_loss) tuple
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].to(device)
        outputs = model(input_ids=batch, labels=batch)
        n_tokens = (batch.shape[1] - 1) * batch.shape[0]
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss


def get_calibration_data(
    tokenizer,
    split: str = "train",
    n_samples: int = 128,
    seq_len: int = 512,
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    skip: int = 0,
) -> torch.Tensor:
    """Load tokenized calibration/evaluation data.

    Streams from a HuggingFace dataset, taking full-length document-start
    chunks for proper context.

    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split ("train" or "validation")
        n_samples: Number of samples to collect
        seq_len: Sequence length per sample
        dataset_name: Dataset identifier
        dataset_config: Dataset configuration
        skip: Number of valid samples to skip (for separating calib from eval)

    Returns:
        Tensor of shape (n_samples, seq_len) with token IDs
    """
    try:
        dataset = load_dataset(dataset_name, dataset_config, split="validation", streaming=True)
    except Exception:
        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1",
            split="validation" if split == "validation" else "train",
        )
        dataset = iter([{"text": t} for t in dataset["text"] if len(t.strip()) > 200])

    samples = []
    skipped = 0
    for doc in dataset:
        if len(samples) >= n_samples:
            break
        text = doc.get("text", "").strip()
        if len(text) < 100:
            continue
        tokens = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=seq_len
        )["input_ids"][0]
        if len(tokens) >= seq_len:
            if skipped < skip:
                skipped += 1
                continue
            samples.append(tokens[:seq_len])

    if len(samples) < n_samples:
        import warnings
        warnings.warn(
            f"Only got {len(samples)}/{n_samples} full-length samples. "
            f"Results may be less reliable."
        )

    return torch.stack(samples[:n_samples]) if samples else torch.zeros(1, seq_len, dtype=torch.long)


def save_compressed_model(
    model: nn.Module,
    tokenizer,
    metadata: list[dict],
    output_path: str,
    extra_info: Optional[dict] = None,
):
    """Save a compressed model in HuggingFace-compatible format.

    Saves:
    - Model weights (safetensors)
    - Tokenizer
    - Config with VAC metadata
    - factorized_modules.json

    Args:
        model: Compressed model
        tokenizer: Tokenizer
        metadata: Factorization metadata (from compress_sequential)
        output_path: Output directory
        extra_info: Additional info to store in config
    """
    import json
    from pathlib import Path
    from safetensors.torch import save_file

    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)

    # Save config with VAC metadata
    config = model.config if hasattr(model, "config") else None
    if config is not None:
        config.vac_metadata = metadata
        if extra_info:
            config.vac_info = extra_info
        config.save_pretrained(path)

    # Save tokenizer
    tokenizer.save_pretrained(path)

    # Save weights as safetensors
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(path / "model.safetensors"))

    # Save metadata separately for easy access
    with open(path / "factorized_modules.json", "w") as f:
        json.dump(metadata, f, indent=2)
