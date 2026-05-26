"""HuggingFace-compatible model class for VAC compressed models.

Provides:
- FactorizedLinear: low-rank replacement for nn.Linear (x -> up(down(x)))
- VACModel: wrapper that loads compressed models from HuggingFace Hub

Usage:
    model = VACModel.from_pretrained(
        "asystemoffields/OLMo-3-3.5B-Think-VAC",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


class FactorizedLinear(nn.Module):
    """Low-rank replacement for nn.Linear: output = up(down(x)).

    A weight matrix W (out_features x in_features) is stored as two smaller
    matrices: down (rank x in_features) and up (out_features x rank).
    The forward pass computes x @ down.T @ up.T, which is equivalent to
    x @ W.T but with rank-constrained W.

    Inference speedup equals the compression ratio:
        speedup = (m * n) / (rank * (m + n))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

    def extra_repr(self) -> str:
        ratio = self.in_features * self.out_features / (self.rank * (self.in_features + self.out_features))
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, compression={ratio:.1f}x"
        )


def apply_factorized_modules(
    model: nn.Module,
    metadata: list[dict[str, Any]],
    device=None,
    dtype=None,
) -> nn.Module:
    """Replace Linear modules with FactorizedLinear according to metadata.

    Args:
        model: Base model with standard Linear layers
        metadata: List of dicts with keys: module_path, rank, in_features, out_features
        device: Target device
        dtype: Target dtype

    Returns:
        The modified model (in-place)
    """
    for entry in metadata:
        module_path = entry["module_path"]
        rank = int(entry["rank"])
        in_features = int(entry["in_features"])
        out_features = int(entry["out_features"])

        old = model.get_submodule(module_path)
        has_bias = isinstance(old, nn.Linear) and old.bias is not None

        replacement = FactorizedLinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            bias=has_bias,
            device=device,
            dtype=dtype,
        )

        parts = module_path.split(".")
        parent = model.get_submodule(".".join(parts[:-1]))
        setattr(parent, parts[-1], replacement)

    return model


class VACModel(nn.Module):
    """HuggingFace-compatible wrapper for VAC compressed models.

    Handles memory-efficient loading: the full dense model is never
    instantiated. Peak memory equals the compressed model size.

    Usage:
        model = VACModel.from_pretrained(
            "asystemoffields/OLMo-3-3.5B-Think-VAC",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        output = model.generate(input_ids, max_new_tokens=100)
    """

    def __init__(self):
        super().__init__()
        self._model = None
        self.config = None

    def forward(self, **kwargs):
        return self._model(**kwargs)

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self._model.get_input_embeddings()

    def get_output_embeddings(self):
        return self._model.get_output_embeddings()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self._model.prepare_inputs_for_generation(*args, **kwargs)

    @property
    def device(self):
        return next(self._model.parameters()).device

    def to(self, *args, **kwargs):
        self._model = self._model.to(*args, **kwargs)
        return self

    def parameters(self, recurse=True):
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self._model.named_parameters(prefix=prefix, recurse=recurse)

    def eval(self):
        self._model.eval()
        return self

    def train(self, mode=True):
        self._model.train(mode)
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load a VAC-compressed model.

        Supports both local directories and HuggingFace Hub model IDs.
        Peak memory = compressed model size (never instantiates full dense).

        Args:
            pretrained_model_name_or_path: Local path or HF model ID
            torch_dtype: Model dtype (default: bfloat16)
            device_map: Device placement ("auto", "cuda", "cpu")

        Returns:
            VACModel instance ready for inference
        """
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download, list_repo_files

        model_path = Path(pretrained_model_name_or_path)
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        device_map = kwargs.pop("device_map", None)

        is_local = model_path.is_dir()

        # Load config
        if is_local:
            config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        else:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True
            )

        # Load factorization metadata
        metadata = getattr(config, "vac_metadata", None)
        if metadata is None:
            metadata = getattr(config, "pmre_metadata", None)
        if metadata is None and is_local:
            meta_file = model_path / "factorized_modules.json"
            if meta_file.exists():
                with meta_file.open() as f:
                    metadata = json.load(f)

        if metadata is None:
            raise ValueError(
                "No VAC metadata found. Ensure the model has 'vac_metadata' in config "
                "or a 'factorized_modules.json' file."
            )

        # Build base model with factorized structure
        import copy
        base_config = copy.deepcopy(config)
        for field in ("vac_metadata", "pmre_metadata", "auto_map", "pmre_info"):
            if hasattr(base_config, field):
                delattr(base_config, field)

        base_model = AutoModelForCausalLM.from_config(
            base_config, trust_remote_code=True, torch_dtype=torch_dtype
        )
        apply_factorized_modules(base_model, metadata, device="cpu", dtype=torch_dtype)

        # Load weights
        if is_local:
            sf_files = sorted(model_path.glob("*.safetensors"))
        else:
            repo_files = list_repo_files(pretrained_model_name_or_path)
            sf_files = []
            for f in repo_files:
                if f.endswith(".safetensors"):
                    local = hf_hub_download(pretrained_model_name_or_path, f)
                    sf_files.append(Path(local))

        state_dict = {}
        for sf_path in sf_files:
            state_dict.update(load_file(str(sf_path), device="cpu"))

        base_model.load_state_dict(state_dict, strict=True)
        del state_dict

        # Move to device
        target_device = "cpu"
        if device_map == "auto" or device_map == "cuda":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(device_map, str):
            target_device = device_map

        base_model = base_model.to(target_device)

        # Wrap
        wrapper = cls()
        wrapper._model = base_model
        wrapper.config = config
        wrapper._model.eval()

        return wrapper
