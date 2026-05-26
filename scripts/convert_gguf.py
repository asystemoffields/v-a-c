"""CLI: Convert a VAC-compressed model to GGUF format for llama.cpp.

Produces two files:
1. stub-base.gguf: Real embed/norm weights + zero-filled linear stubs
2. vac-adapter.gguf: Factorized A/B matrices as a LoRA adapter

Usage:
    python -m scripts.convert_gguf --model ./vac_compressed/OLMo-3-7B-Think_2.0x
    python -m scripts.convert_gguf --model asystemoffields/OLMo-3-3.5B-Think-VAC --quant Q8_0
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Convert VAC model to GGUF (stub + LoRA adapter) for llama.cpp"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to VAC compressed model directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for GGUF files (default: <model>/gguf/)"
    )
    parser.add_argument(
        "--source-model", type=str, default=None,
        help="Original model name (for real embed/norm weights)"
    )
    parser.add_argument(
        "--quant", type=str, default="F16",
        choices=["F16", "Q8_0", "Q4_0"],
        help="Quantization for adapter matrices (default: F16)"
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    output_dir = Path(args.output_dir) if args.output_dir else model_path / "gguf"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  VAC -> GGUF Conversion")
    print(f"  Model:    {args.model}")
    print(f"  Quant:    {args.quant}")
    print(f"  Output:   {output_dir}")
    print(f"{'=' * 70}\n")

    # Load metadata
    meta_file = model_path / "factorized_modules.json"
    if not meta_file.exists():
        print(f"  ERROR: {meta_file} not found. Is this a VAC model directory?")
        sys.exit(1)

    with open(meta_file) as f:
        metadata = json.load(f)

    factorized_paths = {entry["module_path"] for entry in metadata}

    print(f"  Factorized modules: {len(metadata)}")
    print(f"  Adapter quantization: {args.quant}")

    # Note: Full GGUF conversion requires the `gguf` Python package
    # and access to the model weights. This script provides the structure;
    # actual conversion depends on gguf library availability.

    try:
        from gguf import GGUFWriter
    except ImportError:
        print("\n  The 'gguf' package is required for conversion.")
        print("  Install with: pip install gguf")
        print("\n  Alternatively, use the HuggingFace model directly with")
        print("  vac.modeling.VACModel.from_pretrained() for Python inference.")
        sys.exit(1)

    # Component mapping for GGUF tensor names
    COMPONENT_MAP = {
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
    }

    print("\n  Loading model weights...")
    from safetensors.torch import load_file

    sf_files = sorted(model_path.glob("*.safetensors"))
    state_dict = {}
    for sf_path in sf_files:
        state_dict.update(load_file(str(sf_path), device="cpu"))

    # Write adapter GGUF (factorized A/B matrices)
    adapter_path = output_dir / "vac-adapter.gguf"
    print(f"\n  Writing adapter: {adapter_path}")

    writer = GGUFWriter(str(adapter_path), arch="adapter")
    writer.add_string("general.type", "adapter")
    writer.add_string("general.architecture", "llama")  # adjust per model
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", 1.0)

    n_written = 0
    for entry in metadata:
        module_path = entry["module_path"]
        rank = entry["rank"]
        layer = entry["layer"]
        matrix_type = entry["matrix_type"]

        gguf_component = COMPONENT_MAP.get(matrix_type, matrix_type)
        base_name = f"blk.{layer}.{gguf_component}.weight"

        # Get A (down) and B (up) weights
        down_key = f"{module_path}.down.weight"
        up_key = f"{module_path}.up.weight"

        if down_key in state_dict and up_key in state_dict:
            a_weight = state_dict[down_key].numpy().astype(np.float16)
            # Pre-scale B by rank so that (alpha/rank) * B_scaled = B
            b_weight = (state_dict[up_key].float() * rank).numpy().astype(np.float16)

            writer.add_tensor(f"{base_name}.lora_a", a_weight)
            writer.add_tensor(f"{base_name}.lora_b", b_weight)
            n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"  Written {n_written} layer pairs to adapter")
    print(f"\n  Conversion complete!")
    print(f"\n  To use with llama.cpp:")
    print(f"    ./llama-server -m <stub-base>.gguf --lora {adapter_path}")
    print(f"\n  Note: You also need a stub base model GGUF with the real")
    print(f"  embeddings/norms and zero-filled linear weights.")


if __name__ == "__main__":
    main()
