"""Knowledge distillation for compressed model recovery.

After compression, the model's perplexity is elevated (e.g., 90-144 vs baseline ~21).
Knowledge distillation trains the compressed student to match the original teacher's
output distribution, recovering most of the quality gap.

Key lessons applied:
- Use the model's ORIGINAL training data, not a convenience dataset like C4
- Interleave structured completions (e.g., <think> blocks) if the model uses them
- Use 8-bit Adam (not SGD) for 7B-scale models
- Train at the model's native context length
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from vac.utils import eval_perplexity, get_calibration_data


def stream_batches(
    tokenizer,
    batch_size: int,
    seq_len: int = 512,
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    split: str = "train",
):
    """Stream tokenized batches from a HuggingFace dataset.

    Concatenates documents and chunks into fixed-length sequences
    for efficient training.

    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Number of sequences per batch
        seq_len: Sequence length per sample
        dataset_name: Dataset identifier
        dataset_config: Dataset configuration
        split: Dataset split

    Yields:
        Batches of shape (batch_size, seq_len)
    """
    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    buffer = []
    buffer_tokens = torch.tensor([], dtype=torch.long)

    for doc in dataset:
        text = doc.get("text", "").strip()
        if len(text) < 100:
            continue
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        buffer_tokens = torch.cat([buffer_tokens, tokens])
        while len(buffer_tokens) >= seq_len:
            buffer.append(buffer_tokens[:seq_len])
            buffer_tokens = buffer_tokens[seq_len:]
            if len(buffer) >= batch_size:
                yield torch.stack(buffer)
                buffer = []


def train_kd(
    student: nn.Module,
    teacher_name: str,
    tokenizer,
    device: str,
    *,
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    n_steps: int = 5000,
    lr: float = 3e-5,
    temperature: float = 2.0,
    alpha: float = 0.7,
    batch_size: int = 1,
    grad_accum: int = 8,
    seq_len: int = 512,
    eval_every: int = 500,
    eval_data: Optional[torch.Tensor] = None,
    base_loss: float = 0.0,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Train a compressed student to match the teacher's output distribution.

    The loss combines KL divergence on softened logits with standard NLL:
        L = alpha * T^2 * KL(student_soft || teacher_soft) + (1-alpha) * NLL

    Args:
        student: Compressed model (already on device)
        teacher_name: HuggingFace name for the teacher model
        tokenizer: Shared tokenizer
        device: Torch device
        dataset_name: Training data source
        dataset_config: Dataset config
        n_steps: Number of training steps
        lr: Learning rate
        temperature: KD temperature (softens distributions)
        alpha: KL weight (1-alpha = NLL weight)
        batch_size: Micro-batch size
        grad_accum: Gradient accumulation steps
        seq_len: Training sequence length
        eval_every: Evaluate every N steps
        eval_data: Pre-loaded evaluation data (optional)
        base_loss: Baseline loss for delta computation
        output_dir: Save checkpoints here (optional)
        verbose: Print progress

    Returns:
        Dict with training history and best metrics
    """
    # Load teacher
    if verbose:
        print(f"  Loading teacher: {teacher_name}")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.bfloat16
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if hasattr(teacher.config, "use_cache"):
        teacher.config.use_cache = False

    # Set up student for training
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)

    # Optimizer: prefer 8-bit Adam for memory efficiency
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(student.parameters(), lr=lr, weight_decay=0.01)
        if verbose:
            print(f"  Using 8-bit Adam (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
        if verbose:
            print(f"  Using AdamW (bitsandbytes not available)")

    # Learning rate schedule: warmup + cosine decay
    warmup_steps = min(200, max(1, n_steps // 10))

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Data
    data_iter = stream_batches(tokenizer, batch_size, seq_len, dataset_name, dataset_config)

    # Training loop
    history = []
    best_ppl = float("inf")
    optimizer.zero_grad(set_to_none=True)

    if verbose:
        print(f"  KD: {n_steps} steps, batch={batch_size}x{grad_accum}, "
              f"T={temperature}, alpha={alpha}")

    for step in range(1, n_steps + 1):
        accum_loss = 0.0
        for _ in range(grad_accum):
            batch = next(data_iter).to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids=batch).logits

            student_out = student(input_ids=batch, labels=batch)
            student_logits = student_out.logits

            # KL divergence on softened distributions
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)

            # Combined loss
            nll_loss = student_out.loss
            loss = (alpha * kd_loss + (1.0 - alpha) * nll_loss) / grad_accum
            loss.backward()
            accum_loss += loss.item() * grad_accum

            del batch, teacher_logits, student_logits, student_out, kd_loss, nll_loss, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if verbose and step % 100 == 0:
            print(f"    step {step:>5d}/{n_steps}  loss={accum_loss/grad_accum:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        # Evaluation
        if eval_data is not None and (step % eval_every == 0 or step == n_steps):
            student.eval()
            ppl, loss_val = eval_perplexity(student, eval_data, device, batch_size=1)
            dloss = loss_val - base_loss
            history.append({"step": step, "ppl": ppl, "loss": loss_val, "dloss": dloss})

            if verbose:
                print(f"    >>> eval step {step}: PPL={ppl:.2f}, dloss={dloss:+.4f}")

            if ppl < best_ppl:
                best_ppl = ppl
                if output_dir:
                    _save_checkpoint(student, tokenizer, output_dir, step, ppl)

            student.train()

    # Cleanup
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "history": history,
        "best_ppl": best_ppl,
        "n_steps": n_steps,
    }


def _save_checkpoint(model, tokenizer, output_dir, step, ppl):
    """Save a training checkpoint."""
    path = Path(output_dir) / "best_checkpoint"
    path.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(model.state_dict(), path / "model.pt")
