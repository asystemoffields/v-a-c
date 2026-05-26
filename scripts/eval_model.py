"""CLI: Evaluate a compressed model's perplexity.

Usage:
    python -m scripts.eval_model --model ./vac_compressed/OLMo-3-7B-Think_2.0x
    python -m scripts.eval_model --model asystemoffields/OLMo-3-3.5B-Think-VAC
"""

import argparse
import sys

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a VAC-compressed model's perplexity"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to compressed model directory or HuggingFace model ID"
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Original (uncompressed) model for comparison (optional)"
    )
    parser.add_argument(
        "--n-eval", type=int, default=64,
        help="Number of evaluation samples (default: 64)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=512,
        help="Evaluation sequence length (default: 512)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: cuda if available)"
    )
    parser.add_argument(
        "--dataset", type=str, default="allenai/c4",
        help="Evaluation dataset (default: allenai/c4)"
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 70}")
    print(f"  VAC Model Evaluation")
    print(f"  Model:    {args.model}")
    print(f"  Device:   {device}")
    print(f"  Samples:  {args.n_eval}")
    print(f"  Seq len:  {args.seq_len}")
    print(f"{'=' * 70}\n")

    from transformers import AutoTokenizer
    from vac.modeling import VACModel
    from vac.utils import eval_perplexity, get_calibration_data

    # Load compressed model
    print("  Loading compressed model...")
    model = VACModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Load tokenizer (from original model if specified, otherwise from compressed)
    tokenizer_source = args.baseline if args.baseline else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    # Load evaluation data
    print("  Loading evaluation data...")
    eval_data = get_calibration_data(
        tokenizer, split="validation", n_samples=args.n_eval,
        seq_len=args.seq_len, dataset_name=args.dataset,
    )

    # Evaluate compressed model
    print("  Evaluating compressed model...")
    comp_ppl, comp_loss = eval_perplexity(model._model, eval_data, device, batch_size=1)
    print(f"  Compressed model: PPL={comp_ppl:.2f}, loss={comp_loss:.4f}")

    # Optionally evaluate baseline
    if args.baseline:
        from transformers import AutoModelForCausalLM

        print(f"\n  Loading baseline: {args.baseline}")
        baseline = AutoModelForCausalLM.from_pretrained(
            args.baseline, torch_dtype=torch.bfloat16
        ).to(device)
        baseline.eval()

        base_ppl, base_loss = eval_perplexity(baseline, eval_data, device, batch_size=1)
        print(f"  Baseline model:   PPL={base_ppl:.2f}, loss={base_loss:.4f}")
        print(f"\n  Delta PPL:  {comp_ppl - base_ppl:+.2f}")
        print(f"  Delta loss: {comp_loss - base_loss:+.4f}")

        del baseline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Memory (bf16): {total_params * 2 / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
