"""MCKP rank allocation: optimal per-matrix compression budgets.

Solves a Multiple-Choice Knapsack Problem (MCKP) to distribute a global
storage budget across all weight matrices, minimizing total functional loss.
Each matrix has multiple compression options (different ranks/families)
on its Pareto frontier, and the solver picks the combination that minimizes
total delta-loss under a global parameter budget.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# Quantization unit for DP (reduces DP table size)
DP_UNIT = 1024


# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class Option:
    """A single compression option for a weight matrix."""
    family: str
    compressed_params: int
    compression_ratio: float
    delta_loss: float
    weight_units: int = 0


@dataclass
class Group:
    """A weight matrix with its Pareto-optimal compression options."""
    group_id: str
    layer: int
    matrix_type: str
    original_params: int
    options: list[Option] = field(default_factory=list)


@dataclass
class Allocation:
    """The chosen compression option for a matrix."""
    group_id: str
    layer: int
    matrix_type: str
    original_params: int
    chosen: Option


# =========================================================================
# Architecture Detection
# =========================================================================

_LAYER_RE = re.compile(r"\.layers?\.(\d+)\.")


def parse_matrix_name(name: str) -> tuple[int, str]:
    """Extract (layer_index, matrix_type) from a HuggingFace parameter name."""
    m = _LAYER_RE.search(name)
    if not m:
        return -1, name
    layer_idx = int(m.group(1))
    suffix = name[m.end():]
    suffix = suffix.replace(".weight", "").replace(".bias", "")
    parts = suffix.split(".")
    if any(p.endswith("_proj") for p in parts):
        matrix_type = next(p for p in parts if p.endswith("_proj"))
    elif len(parts) >= 2:
        matrix_type = ".".join(parts[-2:])
    else:
        matrix_type = parts[-1]
    return layer_idx, matrix_type


def classify_component(matrix_type: str) -> str:
    """Classify a matrix type into a component category."""
    t = matrix_type.lower()
    if any(k in t for k in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "self_attn", "attention")):
        return "attn"
    if any(k in t for k in ("gate", "up_proj", "down_proj", "mlp", "fc1", "fc2")):
        return "mlp"
    if "expert" in t:
        return "expert"
    return "other"


# =========================================================================
# Group Construction
# =========================================================================

def pareto_frontier(options: list[Option]) -> list[Option]:
    """Extract Pareto-optimal options (fewer params AND lower loss)."""
    sorted_opts = sorted(options, key=lambda o: o.compressed_params)
    frontier = []
    best_dloss = float("inf")
    for opt in sorted_opts:
        if opt.delta_loss < best_dloss:
            frontier.append(opt)
            best_dloss = opt.delta_loss
    return frontier


def build_groups(
    profiling_results: list[dict],
    n_layers: int,
    embed_params: int = 0,
    compress_all: bool = False,
    preserve_layers: Optional[set[int]] = None,
) -> tuple[list[Group], dict]:
    """Build matrix groups from profiling results.

    Args:
        profiling_results: List of profiling result dicts with keys:
            matrix, original_params, compressed_params, compression_ratio, delta_loss, family
        n_layers: Total number of transformer layers
        embed_params: Non-layer parameters (treated as incompressible)
        compress_all: If True, remove "keep original" option (prevents
            routing/execution mismatch between compressed and intact components)
        preserve_layers: Layer indices exempt from compress_all

    Returns:
        (groups, config) tuple
    """
    if preserve_layers is None:
        preserve_layers = set()

    # Detect architecture from profiling
    by_source = {}
    matrix_types = {}
    profiled_layers = set()

    for config in profiling_results:
        layer_idx, mtype = parse_matrix_name(config["matrix"])
        if layer_idx < 0:
            continue
        profiled_layers.add(layer_idx)
        key = (layer_idx, mtype)
        by_source.setdefault(key, []).append(config)
        if mtype not in matrix_types:
            matrix_types[mtype] = config["original_params"]

    profiled_layers = sorted(profiled_layers)
    per_layer_params = sum(matrix_types.values())
    layer_params = per_layer_params * n_layers

    config = {
        "n_layers": n_layers,
        "embed_params": embed_params,
        "profiled_layers": profiled_layers,
        "matrix_types": sorted(matrix_types.keys()),
        "per_layer_params": per_layer_params,
        "layer_params": layer_params,
        "total_params": layer_params + embed_params,
    }

    # Build groups for all layers (extrapolate from profiled layers)
    groups = []
    for layer in range(n_layers):
        src = _nearest_profiled_layer(layer, profiled_layers, n_layers)
        for mtype in sorted(matrix_types.keys()):
            key = (src, mtype)
            orig = matrix_types[mtype]

            raw_options = []
            for entry in by_source.get(key, []):
                raw_options.append(Option(
                    family=entry.get("family", "low_rank"),
                    compressed_params=entry["compressed_params"],
                    compression_ratio=entry["compression_ratio"],
                    delta_loss=entry["delta_loss"],
                ))

            # Add "keep original" option unless compress_all
            allow_original = (not compress_all) or (layer in preserve_layers)
            if allow_original:
                raw_options.append(Option("original", orig, 1.0, 0.0))

            frontier = pareto_frontier(raw_options)
            for opt in frontier:
                opt.weight_units = opt.compressed_params // DP_UNIT

            groups.append(Group(
                group_id=f"L{layer:02d}.{mtype}",
                layer=layer,
                matrix_type=mtype,
                original_params=orig,
                options=frontier,
            ))

    return groups, config


def _nearest_profiled_layer(target: int, profiled: list[int], n_layers: int) -> int:
    """Map an unprofiled layer to the best profiled representative."""
    if target in profiled:
        return target
    if target == 0:
        return profiled[0]
    if target == n_layers - 1:
        return profiled[-1]
    middle = [p for p in profiled if p != 0 and p != n_layers - 1]
    if not middle:
        return min(profiled, key=lambda p: abs(target - p))
    return min(middle, key=lambda p: abs(target - p))


# =========================================================================
# MCKP Solver (numpy-vectorized dynamic programming)
# =========================================================================

def solve_knapsack(
    groups: list[Group],
    budget_params: int,
) -> tuple[Optional[list[Allocation]], float]:
    """Solve the Multiple-Choice Knapsack Problem via dynamic programming.

    Args:
        groups: List of matrix groups, each with Pareto-optimal options
        budget_params: Total parameter budget for all compressed matrices

    Returns:
        (allocations, total_dloss) or (None, inf) if infeasible
    """
    budget_units = budget_params // DP_UNIT
    INF = np.float64(1e30)

    dp = np.full(budget_units + 1, INF, dtype=np.float64)
    dp[0] = 0.0

    all_choices = []
    all_prev = []

    for group in groups:
        new_dp = np.full(budget_units + 1, INF, dtype=np.float64)
        choice_arr = np.full(budget_units + 1, -1, dtype=np.int32)
        prev_arr = np.full(budget_units + 1, -1, dtype=np.int32)

        for opt_idx, opt in enumerate(group.options):
            w = opt.weight_units
            if w > budget_units:
                continue
            c = opt.delta_loss
            end = budget_units - w + 1
            source = dp[:end]
            candidates = source + c
            target_slice = slice(w, w + end)
            improved = candidates < new_dp[target_slice]
            new_dp[target_slice] = np.where(improved, candidates, new_dp[target_slice])
            choice_arr[target_slice] = np.where(improved, opt_idx, choice_arr[target_slice])
            prev_arr[target_slice] = np.where(
                improved, np.arange(end, dtype=np.int32), prev_arr[target_slice]
            )

        dp = new_dp
        all_choices.append(choice_arr)
        all_prev.append(prev_arr)

    # Find best feasible solution
    feasible = dp.copy()
    feasible[feasible >= INF / 2] = INF
    if np.all(feasible >= INF / 2):
        return None, float("inf")

    best_u = int(np.argmin(feasible))
    total_dloss = float(dp[best_u])

    # Backtrack to find allocations
    allocations = []
    u = best_u
    for g in range(len(groups) - 1, -1, -1):
        opt_idx = int(all_choices[g][u])
        u = int(all_prev[g][u])
        allocations.append(Allocation(
            group_id=groups[g].group_id,
            layer=groups[g].layer,
            matrix_type=groups[g].matrix_type,
            original_params=groups[g].original_params,
            chosen=groups[g].options[opt_idx],
        ))
    allocations.reverse()
    return allocations, total_dloss


# =========================================================================
# High-Level API
# =========================================================================

def solve_allocation(
    profiling_results: list[dict],
    target_ratio: float,
    n_layers: int,
    embed_params: int = 0,
    compress_all: bool = False,
    preserve_layers: Optional[set[int]] = None,
) -> tuple[list[Allocation], dict]:
    """Solve optimal rank allocation from profiling results.

    This is the main entry point for the allocator. Given profiling data
    (per-matrix sensitivity measurements), it finds the globally optimal
    assignment of compression levels to minimize total functional loss
    under a storage budget.

    Args:
        profiling_results: Output from vac.profile.profile_model()
        target_ratio: Target overall compression ratio
        n_layers: Number of transformer layers
        embed_params: Incompressible parameters (embeddings, norms)
        compress_all: Force all matrices to be compressed
        preserve_layers: Layers exempt from compress_all

    Returns:
        (allocations, stats) tuple
    """
    groups, config = build_groups(
        profiling_results, n_layers, embed_params,
        compress_all=compress_all,
        preserve_layers=preserve_layers,
    )

    budget = int(config["layer_params"] / target_ratio)
    allocations, total_dloss = solve_knapsack(groups, budget)

    if allocations is None:
        raise ValueError(
            f"No feasible solution at {target_ratio}x. "
            f"Try a lower compression ratio."
        )

    # Compute stats
    total_compressed = sum(a.chosen.compressed_params for a in allocations)
    total_original = sum(a.original_params for a in allocations)
    stats = {
        "target_ratio": target_ratio,
        "achieved_layer_ratio": total_original / total_compressed if total_compressed > 0 else float("inf"),
        "achieved_overall_ratio": (total_original + embed_params) / (total_compressed + embed_params),
        "total_dloss": total_dloss,
        "n_compressed": sum(1 for a in allocations if a.chosen.family != "original"),
        "n_original": sum(1 for a in allocations if a.chosen.family == "original"),
    }

    return allocations, stats


def allocation_to_rank_overrides(allocations: list[Allocation], n_layers: int) -> dict[str, int]:
    """Convert allocations to a rank_overrides dict for compress_sequential().

    Args:
        allocations: Output from solve_allocation()
        n_layers: Number of transformer layers

    Returns:
        Dict mapping module_path -> target rank
    """
    overrides = {}
    for alloc in allocations:
        if alloc.chosen.family == "original":
            continue
        # Reconstruct module path
        # Infer the path pattern from matrix_type
        layer = alloc.layer
        mtype = alloc.matrix_type
        if "q_proj" in mtype or "k_proj" in mtype or "v_proj" in mtype or "o_proj" in mtype:
            path = f"model.layers.{layer}.self_attn.{mtype}"
        elif "gate" in mtype or "up_proj" in mtype or "down_proj" in mtype:
            path = f"model.layers.{layer}.mlp.{mtype}"
        else:
            path = f"model.layers.{layer}.{mtype}"

        # Compute rank from compressed_params and matrix dimensions
        # compressed_params = rank * (m + n), so rank = compressed_params / (m + n)
        # We don't have m, n here directly, so store compressed_params
        # and let the caller compute rank
        overrides[path] = alloc.chosen.compressed_params

    return overrides
