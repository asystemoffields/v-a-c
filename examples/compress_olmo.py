"""Example: Compress OLMo-3-7B-Think using VAC.

This script demonstrates the full VAC compression pipeline:
1. Load a model
2. Compress with sequential Fisher factorization
3. (Optional) Run knowledge distillation for recovery
4. Save the compressed model

Requirements:
    pip install vac-compress[kd]
    # GPU with ~40 GB VRAM for compression, ~80 GB for KD (teacher + student)
"""

import torch
from transformers import AutoTokenizer

from vac import compress_model
from vac.kd import train_kd
from vac.utils import save_compressed_model, get_calibration_data, eval_perplexity


def main():
    # Configuration
    MODEL_NAME = "allenai/OLMo-3-7B-Think"
    TARGET_RATIO = 2.0
    FISHER_EXPONENT = 1.0 / 3.0  # cbrt -- 18% better than sqrt
    ORDER = "middle-out"          # 21% better than front-to-back
    DEVICE = "cuda"
    OUTPUT_DIR = "./vac_compressed/OLMo-3-3.5B-Think-VAC"

    print("=" * 70)
    print("  VAC Compression: OLMo-3-7B-Think -> ~3.5B effective params")
    print("=" * 70)

    # Step 1: Compress the model
    print("\n[Step 1] Sequential Fisher compression...")
    model, metadata = compress_model(
        MODEL_NAME,
        target_ratio=TARGET_RATIO,
        device=DEVICE,
        dtype=torch.bfloat16,
        fisher_exponent=FISHER_EXPONENT,
        order=ORDER,
        n_fisher_samples=4,
        seq_len=512,
        verbose=True,
    )

    # Step 2: Evaluate pre-KD quality
    print("\n[Step 2] Pre-KD evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    eval_data = get_calibration_data(tokenizer, "validation", n_samples=64, seq_len=512)
    pre_kd_ppl, pre_kd_loss = eval_perplexity(model, eval_data, DEVICE, batch_size=1)
    print(f"  Pre-KD PPL: {pre_kd_ppl:.2f}")

    # Step 3: Knowledge distillation (optional but recommended)
    # NOTE: This requires ~80 GB VRAM (teacher + student on same GPU)
    # If you don't have enough VRAM, skip this step and save the pre-KD model.
    RUN_KD = False  # Set to True if you have sufficient VRAM

    if RUN_KD:
        print("\n[Step 3] Knowledge distillation...")
        kd_result = train_kd(
            student=model,
            teacher_name=MODEL_NAME,
            tokenizer=tokenizer,
            device=DEVICE,
            # Use the model's actual training data for best results:
            # dataset_name="allenai/dolma",
            # For demonstration, use C4 (available without special access):
            dataset_name="allenai/c4",
            dataset_config="en",
            n_steps=5000,
            lr=3e-5,
            temperature=2.0,
            alpha=0.7,
            seq_len=512,
            eval_data=eval_data,
            base_loss=pre_kd_loss,
            output_dir=OUTPUT_DIR,
            verbose=True,
        )
        print(f"  Best KD PPL: {kd_result['best_ppl']:.2f}")
    else:
        print("\n[Step 3] Skipping KD (set RUN_KD=True to enable)")

    # Step 4: Save the compressed model
    print("\n[Step 4] Saving compressed model...")
    save_compressed_model(
        model, tokenizer, metadata, OUTPUT_DIR,
        extra_info={
            "source_model": MODEL_NAME,
            "target_ratio": TARGET_RATIO,
            "order": ORDER,
            "fisher_exponent": FISHER_EXPONENT,
            "pre_kd_ppl": pre_kd_ppl,
        },
    )

    # Summary
    total_original = sum(m["original_params"] for m in metadata)
    total_compressed = sum(m["compressed_params"] for m in metadata)
    print(f"\n{'=' * 70}")
    print(f"  Compression complete!")
    print(f"  Matrices compressed: {len(metadata)}")
    print(f"  Effective ratio: {total_original / total_compressed:.3f}x")
    print(f"  Pre-KD PPL: {pre_kd_ppl:.2f}")
    print(f"  Saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
