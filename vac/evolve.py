"""Evolutionary search over compression strategies.

The insight: pre-KD perplexity is a fast, cheap evaluator of compression quality.
If we can find a compression strategy that starts at PPL 90 instead of 144,
everything downstream (KD, SFT, DPO) becomes easier.

The search space includes:
- Compression order (front-to-back, back-to-front, middle-out)
- Fisher scaling exponent (sqrt, cbrt, linear, per-matrix)
- Per-component ratios (attention vs MLP)
- Which matrices to keep at full rank

Each evaluation takes ~5-10 minutes on an H100 (compress + measure PPL).
"""

from __future__ import annotations

import copy
import gc
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from vac.compress import fisher_scaled_svd, replace_with_factorized
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


# =========================================================================
# Genome Definition
# =========================================================================

@dataclass
class MatrixGene:
    """Compression parameters for a single matrix."""
    ratio: float = 2.0
    fisher_exponent: float = 0.5
    fisher_samples: int = 4
    keep_full_rank: bool = False

    def clone(self):
        return MatrixGene(
            ratio=self.ratio,
            fisher_exponent=self.fisher_exponent,
            fisher_samples=self.fisher_samples,
            keep_full_rank=self.keep_full_rank,
        )


@dataclass
class CompressionGenome:
    """Full compression strategy for a transformer model."""
    matrix_genes: dict = field(default_factory=dict)
    compression_order: list = field(default_factory=list)
    global_fisher_exponent: float = 0.5
    global_fisher_samples: int = 4
    target_overall_ratio: float = 2.0
    genome_id: str = ""
    generation: int = 0
    parent_ids: list = field(default_factory=list)
    mutation_log: list = field(default_factory=list)

    def clone(self):
        return CompressionGenome(
            matrix_genes={k: v.clone() for k, v in self.matrix_genes.items()},
            compression_order=list(self.compression_order),
            global_fisher_exponent=self.global_fisher_exponent,
            global_fisher_samples=self.global_fisher_samples,
            target_overall_ratio=self.target_overall_ratio,
            genome_id="",
            generation=self.generation,
            parent_ids=list(self.parent_ids),
            mutation_log=list(self.mutation_log),
        )

    def to_dict(self) -> dict:
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutation_log": self.mutation_log,
            "global_fisher_exponent": self.global_fisher_exponent,
            "global_fisher_samples": self.global_fisher_samples,
            "target_overall_ratio": self.target_overall_ratio,
            "compression_order": [(l, m) for l, m in self.compression_order],
            "matrix_genes": {
                f"{l}_{m}": {
                    "layer": l, "matrix_type": m,
                    "ratio": gene.ratio, "fisher_exponent": gene.fisher_exponent,
                    "fisher_samples": gene.fisher_samples, "keep_full_rank": gene.keep_full_rank,
                }
                for (l, m), gene in self.matrix_genes.items()
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "CompressionGenome":
        g = CompressionGenome(
            genome_id=d["genome_id"],
            generation=d["generation"],
            parent_ids=d.get("parent_ids", []),
            mutation_log=d.get("mutation_log", []),
            global_fisher_exponent=d["global_fisher_exponent"],
            global_fisher_samples=d["global_fisher_samples"],
            target_overall_ratio=d["target_overall_ratio"],
            compression_order=[(l, m) for l, m in d["compression_order"]],
        )
        for key, gd in d["matrix_genes"].items():
            g.matrix_genes[(gd["layer"], gd["matrix_type"])] = MatrixGene(
                ratio=gd["ratio"], fisher_exponent=gd["fisher_exponent"],
                fisher_samples=gd["fisher_samples"], keep_full_rank=gd["keep_full_rank"],
            )
        return g


@dataclass
class FitnessResult:
    """Evaluation result for a genome."""
    genome_id: str
    pre_kd_ppl: float
    pre_kd_loss: float
    delta_loss: float
    achieved_ratio: float
    original_params: int
    compressed_params: int
    eval_time_s: float

    def dominates(self, other: "FitnessResult") -> bool:
        """Pareto dominance over (PPL, ratio)."""
        return (self.pre_kd_ppl <= other.pre_kd_ppl and
                self.achieved_ratio >= other.achieved_ratio and
                (self.pre_kd_ppl < other.pre_kd_ppl or
                 self.achieved_ratio > other.achieved_ratio))


# =========================================================================
# Seed Genomes
# =========================================================================

def create_seed_genomes(
    model: nn.Module,
    target_ratio: float = 2.0,
    n_layers: Optional[int] = None,
) -> list[CompressionGenome]:
    """Create diverse initial population of compression strategies.

    Includes:
    - Uniform front-to-back (sqrt) -- the v1 baseline
    - Uniform back-to-front (sqrt)
    - Middle-out order (sqrt)
    - Front-to-back with cbrt Fisher
    - Attention-heavy allocation
    - MLP-heavy allocation (control)
    """
    targets = target_module_paths(model, start_layer=0, max_layers=n_layers)
    keys = [(l, m) for l, m, _ in targets]
    n_layers_actual = max(l for l, _ in keys) + 1

    def make_base(name, exponent=0.5):
        g = CompressionGenome(
            global_fisher_exponent=exponent,
            global_fisher_samples=4,
            target_overall_ratio=target_ratio,
            genome_id=name,
        )
        for layer, matrix_type, _ in targets:
            g.matrix_genes[(layer, matrix_type)] = MatrixGene(
                ratio=target_ratio, fisher_exponent=exponent, fisher_samples=4
            )
            g.compression_order.append((layer, matrix_type))
        return g

    variants = []

    # 1. Front-to-back, sqrt
    variants.append(make_base("seed_frontback_sqrt"))

    # 2. Back-to-front, sqrt
    v = make_base("seed_backfront_sqrt")
    v.compression_order = list(reversed(v.compression_order))
    variants.append(v)

    # 3. Middle-out, sqrt
    v = make_base("seed_middleout_sqrt")
    mid = n_layers_actual // 2
    layer_order = sorted(range(n_layers_actual), key=lambda l: abs(l - mid))
    new_order = []
    for l in layer_order:
        for mt in MATRIX_ORDER:
            if (l, mt) in v.matrix_genes:
                new_order.append((l, mt))
    v.compression_order = new_order
    variants.append(v)

    # 4. Front-to-back, cbrt
    variants.append(make_base("seed_frontback_cbrt", exponent=1.0 / 3.0))

    # 5. Attention aggressive, MLP gentle
    v = make_base("seed_attn_heavy")
    for (l, mt), gene in v.matrix_genes.items():
        if "self_attn" in mt:
            gene.ratio = target_ratio * 1.5
        else:
            gene.ratio = target_ratio * 0.7
    variants.append(v)

    # 6. MLP aggressive, attention gentle (control)
    v = make_base("seed_mlp_heavy")
    for (l, mt), gene in v.matrix_genes.items():
        if "self_attn" in mt:
            gene.ratio = target_ratio * 0.7
        else:
            gene.ratio = target_ratio * 1.5
    variants.append(v)

    return variants


# =========================================================================
# Mutations
# =========================================================================

_COUNTER = 0


def _next_id(prefix: str = "gen") -> str:
    global _COUNTER
    _COUNTER += 1
    return f"{prefix}_{_COUNTER:05d}"


def mutate_ratio(genome: CompressionGenome) -> CompressionGenome:
    """Perturb compression ratios for a subset of matrices."""
    g = genome.clone()
    keys = [k for k, v in g.matrix_genes.items() if not v.keep_full_rank]
    n = random.randint(1, max(1, len(keys) // 7))
    for key in random.sample(keys, min(n, len(keys))):
        gene = g.matrix_genes[key]
        gene.ratio = max(1.1, min(gene.ratio * random.gauss(1.0, 0.3), 20.0))
    g.mutation_log.append(f"mutate_ratio(n={n})")
    return g


def mutate_exponent(genome: CompressionGenome) -> CompressionGenome:
    """Change Fisher scaling exponent."""
    g = genome.clone()
    options = [0.25, 1.0 / 3.0, 0.5, 0.67, 0.75, 1.0]
    if random.random() < 0.4:
        exp = random.choice(options)
        g.global_fisher_exponent = exp
        for gene in g.matrix_genes.values():
            gene.fisher_exponent = exp
        g.mutation_log.append(f"mutate_exp_global({exp:.3f})")
    else:
        keys = random.sample(list(g.matrix_genes.keys()),
                             random.randint(1, max(1, len(g.matrix_genes) // 4)))
        for key in keys:
            g.matrix_genes[key].fisher_exponent = random.choice(options)
        g.mutation_log.append(f"mutate_exp_local(n={len(keys)})")
    return g


def mutate_order(genome: CompressionGenome) -> CompressionGenome:
    """Shuffle compression order."""
    g = genome.clone()
    order = g.compression_order
    if len(order) < 2:
        return g
    # Swap two layers
    layers = sorted(set(l for l, _ in order))
    if len(layers) >= 2:
        l1, l2 = random.sample(layers, 2)
        pos1 = [i for i, (l, _) in enumerate(order) if l == l1]
        pos2 = [i for i, (l, _) in enumerate(order) if l == l2]
        if len(pos1) == len(pos2):
            for p1, p2 in zip(pos1, pos2):
                order[p1], order[p2] = order[p2], order[p1]
    g.compression_order = order
    g.mutation_log.append("mutate_order")
    return g


def mutate_component(genome: CompressionGenome) -> CompressionGenome:
    """Scale all ratios for one component type."""
    g = genome.clone()
    component = random.choice(["self_attn", "mlp"])
    factor = max(0.3, min(3.0, random.gauss(1.0, 0.3)))
    for (l, mt), gene in g.matrix_genes.items():
        if component in mt and not gene.keep_full_rank:
            gene.ratio = max(1.1, min(gene.ratio * factor, 20.0))
    g.mutation_log.append(f"mutate_component({component},{factor:.2f})")
    return g


_MUTATIONS = [
    (mutate_ratio, 0.35),
    (mutate_exponent, 0.20),
    (mutate_order, 0.20),
    (mutate_component, 0.25),
]


def mutate(genome: CompressionGenome) -> CompressionGenome:
    """Apply a random mutation."""
    mutations, weights = zip(*_MUTATIONS)
    fn = random.choices(mutations, weights=weights, k=1)[0]
    result = fn(genome)
    result.genome_id = _next_id(f"g{genome.generation + 1}")
    result.generation = genome.generation + 1
    result.parent_ids = [genome.genome_id]
    return result


def crossover(a: CompressionGenome, b: CompressionGenome) -> CompressionGenome:
    """Layer-level crossover between two parents."""
    child = a.clone()
    layers = sorted(set(l for l, _ in child.matrix_genes.keys()))
    cut = random.randint(1, len(layers) - 1)
    b_layers = set(layers[cut:])
    for (l, mt) in list(child.matrix_genes.keys()):
        if l in b_layers and (l, mt) in b.matrix_genes:
            child.matrix_genes[(l, mt)] = b.matrix_genes[(l, mt)].clone()
    child.genome_id = _next_id(f"g{max(a.generation, b.generation) + 1}")
    child.generation = max(a.generation, b.generation) + 1
    child.parent_ids = [a.genome_id, b.genome_id]
    child.mutation_log = [f"crossover(cut={cut})"]
    return child


# =========================================================================
# Evaluator
# =========================================================================

def evaluate_genome(
    genome: CompressionGenome,
    model_name: str,
    device: str,
    eval_data: torch.Tensor,
    calib_data: torch.Tensor,
    base_loss: float,
    n_layers: Optional[int] = None,
    verbose: bool = False,
) -> FitnessResult:
    """Compress a model with a genome and return pre-KD perplexity.

    Loads a fresh model, applies the genome's compression strategy,
    and measures the resulting perplexity. This is the fitness function.

    Args:
        genome: Compression strategy to evaluate
        model_name: HuggingFace model name
        device: Torch device
        eval_data: Pre-loaded evaluation data
        calib_data: Pre-loaded calibration data
        base_loss: Baseline loss for delta computation
        n_layers: Limit to first N layers (for testing)
        verbose: Print per-layer progress

    Returns:
        FitnessResult with pre-KD PPL and achieved ratio
    """
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    original_params = sum(p.numel() for p in model.parameters())
    compressed_units = original_params

    targets = target_module_paths(model, start_layer=0, max_layers=n_layers)
    path_lookup = {(l, m): p for l, m, p in targets}

    for layer, matrix_type in genome.compression_order:
        key = (layer, matrix_type)
        if key not in path_lookup:
            continue
        gene = genome.matrix_genes.get(key)
        if gene is None or gene.keep_full_rank:
            continue

        path = path_lookup[key]
        module = model.get_submodule(path)
        if not isinstance(module, nn.Linear):
            continue

        out_features, in_features = module.weight.shape
        rank = target_rank(out_features, in_features, gene.ratio)
        original = module.weight.numel()
        stored = rank * (out_features + in_features)

        fisher = compute_fisher_for_linear(
            model, calib_data, device, path,
            n_samples=gene.fisher_samples, batch_size=1,
        )

        W = module.weight.detach()
        A, B = fisher_scaled_svd(W, fisher, rank, exponent=gene.fisher_exponent)

        has_bias = module.bias is not None
        bias = module.bias.detach().clone() if has_bias else None
        factorized = FactorizedLinear(
            in_features=in_features, out_features=out_features, rank=rank,
            bias=has_bias, device=W.device, dtype=W.dtype,
        )
        with torch.no_grad():
            factorized.down.weight.copy_(B.to(W.device, W.dtype))
            factorized.up.weight.copy_(A.to(W.device, W.dtype))
            if bias is not None:
                factorized.up.bias.copy_(bias.to(W.device, W.dtype))

        parent, attr = parent_and_attr(model, path)
        setattr(parent, attr, factorized)
        compressed_units -= (original - stored)

        del fisher, W, A, B
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_ppl, final_loss = eval_perplexity(model, eval_data, device, batch_size=1)
    achieved_ratio = original_params / compressed_units

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return FitnessResult(
        genome_id=genome.genome_id,
        pre_kd_ppl=final_ppl,
        pre_kd_loss=final_loss,
        delta_loss=final_loss - base_loss,
        achieved_ratio=achieved_ratio,
        original_params=original_params,
        compressed_params=compressed_units,
        eval_time_s=time.time() - t0,
    )


# =========================================================================
# Main Loop
# =========================================================================

def run_evolution(
    model_name: str,
    target_ratio: float = 2.0,
    population_size: int = 30,
    n_generations: int = 50,
    children_per_gen: int = 10,
    mutation_rate: float = 0.7,
    tournament_size: int = 3,
    device: str = "cuda",
    seq_len: int = 512,
    n_eval: int = 64,
    n_calib: int = 16,
    n_layers: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[CompressionGenome, FitnessResult]:
    """Run evolutionary search over compression strategies.

    Args:
        model_name: HuggingFace model name
        target_ratio: Target compression ratio
        population_size: Number of genomes in the population
        n_generations: Number of generations to evolve
        children_per_gen: New candidates per generation
        mutation_rate: Probability of mutation vs crossover
        tournament_size: Tournament selection size
        device: Torch device
        seq_len: Sequence length for eval
        n_eval: Number of eval samples
        n_calib: Number of calibration samples
        n_layers: Limit layers (for testing)
        seed: Random seed
        verbose: Print progress

    Returns:
        (best_genome, best_fitness) tuple
    """
    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if verbose:
        print(f"  Loading eval/calibration data...")
    eval_data = get_calibration_data(tokenizer, "validation", n_samples=n_eval, seq_len=seq_len)
    calib_data = get_calibration_data(tokenizer, "train", n_samples=n_calib, seq_len=seq_len)

    # Baseline
    if verbose:
        print(f"  Measuring baseline...")
    tmp = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    tmp.eval()
    if hasattr(tmp.config, "use_cache"):
        tmp.config.use_cache = False
    base_ppl, base_loss = eval_perplexity(tmp, eval_data, device, batch_size=1)
    if verbose:
        print(f"  Baseline: PPL={base_ppl:.2f}")

    # Create initial population
    population = create_seed_genomes(tmp, target_ratio, n_layers)
    del tmp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pad population
    while len(population) < population_size:
        population.append(mutate(random.choice(population)))
    population = population[:population_size]

    # Evaluate initial population
    fitnesses = []
    best_fitness = None
    best_genome = None

    for i, genome in enumerate(population):
        if verbose:
            print(f"  Evaluating {genome.genome_id} ({i+1}/{len(population)})...")
        fitness = evaluate_genome(
            genome, model_name, device, eval_data, calib_data,
            base_loss, n_layers=n_layers, verbose=False,
        )
        fitnesses.append(fitness)
        if best_fitness is None or fitness.pre_kd_ppl < best_fitness.pre_kd_ppl:
            best_fitness = fitness
            best_genome = genome
            if verbose:
                print(f"    NEW BEST: PPL={fitness.pre_kd_ppl:.2f} ({genome.genome_id})")

    # Evolution loop
    for gen in range(1, n_generations + 1):
        if verbose:
            print(f"\n  Generation {gen}/{n_generations}, best={best_fitness.pre_kd_ppl:.2f}")

        for _ in range(children_per_gen):
            if random.random() < mutation_rate:
                parent = _tournament_select(population, fitnesses, tournament_size)
                child = mutate(parent)
            else:
                p1 = _tournament_select(population, fitnesses, tournament_size)
                p2 = _tournament_select(population, fitnesses, tournament_size)
                child = crossover(p1, p2)

            fitness = evaluate_genome(
                child, model_name, device, eval_data, calib_data,
                base_loss, n_layers=n_layers, verbose=False,
            )
            population.append(child)
            fitnesses.append(fitness)

            if fitness.pre_kd_ppl < best_fitness.pre_kd_ppl:
                best_fitness = fitness
                best_genome = child
                if verbose:
                    print(f"    NEW BEST: PPL={fitness.pre_kd_ppl:.2f} ({child.genome_id})")

        # Select survivors
        indexed = sorted(range(len(population)), key=lambda i: fitnesses[i].pre_kd_ppl)
        indexed = indexed[:population_size]
        population = [population[i] for i in indexed]
        fitnesses = [fitnesses[i] for i in indexed]

    if verbose:
        print(f"\n  Evolution complete. Best: PPL={best_fitness.pre_kd_ppl:.2f}, "
              f"ratio={best_fitness.achieved_ratio:.3f}x")

    return best_genome, best_fitness


def _tournament_select(population, fitnesses, k=3):
    candidates = random.sample(list(zip(population, fitnesses)), min(k, len(population)))
    return min(candidates, key=lambda x: x[1].pre_kd_ppl)[0]
