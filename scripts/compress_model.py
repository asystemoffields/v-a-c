"""CLI: Compress a HuggingFace model using VAC.

Usage:
    python -m scripts.compress_model --model allenai/OLMo-3-7B-Think --ratio 2.0
    python -m scripts.compress_model --model meta-llama/Llama-2-7b-hf --ratio 2.0 --order middle-out
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Compress a transformer model using VAC (Variable Allocation Compression)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: ./vac_compressed/<model_name>)"
    )
    parser.add_argument(
        "--ratio", type=float, default=2.0,
        help="Target compression ratio (default: 2.0)"
    )
    parser.add_argument(
        "--order", type=str, default="front-to-back",
        choices=["front-to-back", "back-to-front", "middle-out"],
        help="Compression order (default: front-to-back)"
    )
    parser.add_argument(
        "--exponent", type=float, default=0.5,
        help="Fisher scaling exponent: 0.5=sqrt, 0.33=cbrt (default: 0.5)"
    )
    parser.add_argument(
        "--n-fisher", type=int, default=4,
        help="Fisher calibration samples per matrix (default: 4)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=512,
        help="Calibration sequence length (default: 512)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)"
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Determine output path
    if args.output is None:
        model_short = args.model.split("/")[-1]
        output_dir = f"./vac_compressed/{model_short}_{args.ratio}x"
    else:
        output_dir = args.output

    print(f"\n{'=' * 70}")
    print(f"  VAC: Variable Allocation Compression")
    print(f"  Model:     {args.model}")
    print(f"  Ratio:     {args.ratio}x")
    print(f"  Order:     {args.order}")
    print(f"  Exponent:  {args.exponent}")
    print(f"  Device:    {device}")
    print(f"  Output:    {output_dir}")
    print(f"{'=' * 70}\n")

    from vac import compress_model
    from vac.utils import save_compressed_model
    from transformers import AutoTokenizer

    model, metadata = compress_model(
        args.model,
        target_ratio=args.ratio,
        device=device,
        dtype=dtype,
        fisher_exponent=args.exponent,
        order=args.order,
        n_fisher_samples=args.n_fisher,
        seq_len=args.seq_len,
        verbose=True,
    )

    # Save
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    save_compressed_model(
        model, tokenizer, metadata, output_dir,
        extra_info={
            "source_model": args.model,
            "target_ratio": args.ratio,
            "order": args.order,
            "fisher_exponent": args.exponent,
        },
    )

    print(f"\n  Saved to: {output_dir}")
    print(f"  Modules compressed: {len(metadata)}")

    # Summary
    total_original = sum(m["original_params"] for m in metadata)
    total_compressed = sum(m["compressed_params"] for m in metadata)
    print(f"  Original params (compressed matrices): {total_original:,}")
    print(f"  Compressed params: {total_compressed:,}")
    print(f"  Effective ratio: {total_original / total_compressed:.3f}x")


if __name__ == "__main__":
    main()
