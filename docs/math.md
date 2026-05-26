# VAC: Mathematical Specification

## 1. Storage Formula (Low-Rank Factorization)

Given a weight matrix W in R^{m x n} (where m = out_features, n = in_features) compressed to rank r:

    W ~ A @ B    where A in R^{m x r}, B in R^{r x n}

**Storage in bf16-equivalent units (1 unit = 2 bytes):**

    storage_low_rank = r * (m + n)

The per-matrix compression ratio is:

    ratio = (m * n) / (r * (m + n))

## 2. Target Rank Formula

Given a desired compression ratio rho (e.g., 2.0 means 2x smaller):

    r = floor((m * n) / (rho * (m + n)))

Clamped to [1, min(m, n) - 1].

## 3. Fisher-Weighted SVD (Separable Approximation)

### The Objective

The true Fisher-weighted objective is:

    min_{A,B} sum_ij F_ij * (W_ij - (AB)_ij)^2

where F_ij = E[(dL/dW_ij)^2] is the diagonal empirical Fisher.

### Separable Scaling

We approximate via row and column importance scales:

    s_row_i = sqrt(mean_j(F_ij))
    s_col_j = sqrt(mean_i(F_ij))

Normalized to unit mean and clamped to [0.05, 20.0].

The Fisher-scaled weight is:

    W_tilde = diag(s_row) @ W @ diag(s_col)

Standard SVD on the scaled matrix:

    W_tilde = U S V^T

Truncate to rank r and un-scale back:

    A = (U_r * sqrt(S_r)) / s_row
    B = (sqrt(S_r) * V_r^T) / s_col

**Why this works:** Scaling amplifies high-Fisher regions before SVD, so truncation
preferentially discards functionally unimportant directions.

### Configurable Exponent

The exponent controls how aggressively Fisher information biases the SVD:

    s_row_i = mean_j(F_ij)^exponent

- exponent=0.5 (sqrt): moderate Fisher weighting (default)
- exponent=0.33 (cbrt): gentler weighting, 18% better than sqrt
- exponent=1.0 (linear): too aggressive, 38% worse than sqrt

The diagonal Fisher is already an approximation, so over-trusting it hurts.

## 4. Sequential Compression

### Algorithm

For each matrix in topological order:

1. Compute diagonal Fisher F^(t) for the current weight in the current model state
   (all previously compressed layers already substituted)
2. Apply Fisher-weighted SVD to obtain factors A^(t), B^(t)
3. Replace the weight with the factorized module
4. Proceed to the next matrix

### Why This Works

In a residual transformer:

    h_l = h_{l-1} + f_l(h_{l-1})

If layer l introduces error delta_l, all subsequent layers process corrupted input.
Blind SVD computes Fisher on the original model, but after compressing layer l,
the activations at layer l+1 have changed.

Sequential compression recomputes Fisher at each step, always measuring sensitivity
in the actual corrupted model. This gives 67x better perplexity (PPL 144 vs 9,739)
because it prevents catastrophic error amplification through the residual stream.

## 5. Compression Order

The order in which layers are compressed matters because it determines which layers
get accurate Fisher information:

- **Front-to-back:** Hard edge layers first (accurate Fisher on pristine model),
  easy middle layers last (stale Fisher, but doesn't matter)
- **Middle-out:** Easy middle layers first (almost zero error), then hard edge layers
  get Fisher computed on a barely-distorted model. **21% better than front-to-back.**

Middle-out wins because OLMo's middle layers are ~746x compressible (nearly lossless).
Compressing them first keeps the model almost pristine for the hard edge layer
compressions that follow.

## 6. Compound Error in Residual Networks

### First-Order Additive Model (Allocator Assumption)

    delta_L_total ~ sum_i delta_L_i

where delta_L_i is measured in isolation (all other matrices at original values).

### Why It Breaks

Compressed attention sends incorrect routing signals into MLP:
- Attention error epsilon with MLP gain G gives compound contribution ~ G * epsilon
- This is multiplicative, not additive
- Across L layers: true error >> sum of individual errors

Empirical compound factor: 2.5-3.2x (true loss / predicted additive loss).

**Mitigation:** The `compress_all` mode forces all matrices to be compressed uniformly,
preventing routing/execution mismatch between compressed and intact components.

## 7. MCKP Budget Allocation

### Problem

Given G groups (matrices), each with Pareto-optimal options O_g:

    min sum_g cost(o_g)
    s.t. sum_g weight(o_g) <= Budget
         o_g in O_g for all g

### Solution

Standard 1D dynamic programming over quantized weight units:

    dp[w] = min over all groups and options of dp_prev[w - w_o] + c_o

Time: O(G * |O_max| * Budget/DP_UNIT)

The DP_UNIT quantization (1024 params) makes this tractable for multi-billion
parameter models.

## 8. Knowledge Distillation Loss

    L_KD = alpha * L_KL + (1 - alpha) * L_NLL

where:

    L_KL = T^2 * KL(softmax(z_s/T) || softmax(z_t/T))

T=2.0 softens distributions so the student can learn from the teacher's
uncertainty, not just its top predictions.

## 9. Inference Speedup

For a FactorizedLinear with input x in R^{b x n}:

- Original: y = x @ W^T costs O(b * m * n) FLOPs
- Factorized: y = (x @ B^T) @ A^T costs O(b * r * (m + n)) FLOPs

Speedup = m*n / (r*(m+n)) = compression ratio.

At 2x compression, each factorized layer runs ~2x faster.
