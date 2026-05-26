# Key Lessons Learned

Hard-won insights from compressing OLMo-3-7B-Think!

## 1. Sequential Fisher Is the Breakthrough

Compressing front-to-back with Fisher recomputed on distorted activations gives
**67x better pre-KD perplexity** than blind SVD:

- Blind SVD 2x: PPL 9,739
- Sequential Fisher 2x: PPL 144

Each layer is optimized for the input it will *actually see* at inference.

## 2. Compression Order Matters More Than Expected

Simply changing the compression order -- no other changes -- gives 21% improvement:

- Front-to-back: PPL 147.91
- Back-to-front: PPL 122.48 (-17%)
- Middle-out: PPL 116.84 (-21%)

Middle-out wins because middle layers are ~746x compressible.
Compressing them first means the hard edge layers get Fisher computed on a
barely-distorted model.

## 3. The Fisher Exponent Is Tunable

- sqrt (0.5, default): PPL 147.91
- cbrt (0.33): PPL 121.84 (-18%)
- linear (1.0): PPL 204.07 (+38%, worse)

The diagonal Fisher is an approximation. Over-trusting it (linear) hurts more
than under-trusting it (cbrt).

## 4. KD Must Use the Model's Training Data

KD on C4 (generic web text) gave good C4 PPL but 0% on GSM8K/BBH/IFBench.
The fix: use the model's actual training data (DOLMA for OLMo).

**Rule:** Match the KD distribution to the training distribution.

## 5. PPL Recovery Does Not Equal Behavior Recovery

C4 PPL 27.86 (only 6 above teacher) but 0% on all benchmarks. The model could
not produce `<think>` tags correctly because they are multi-token BPE sequences
that were never exercised during C4 KD.

**Rule:** Always pair PPL with behavioral probes for the model's intended use case.

## 6. Compound Error Depends on Topology

Compressing ALL matrices uniformly gives lower compound error than selectively
compressing some harder. The gain/scale mismatch between compressed and intact
components amplifies errors.

## 7. 8-bit Adam, Not SGD, for Large Model KD

SGD plateaued at PPL 156. 8-bit Adam (bitsandbytes) reached PPL 27.86.
Memory: teacher 14GB + student 14GB + Adam8bit 14GB = fits one H100.

## 8. Replay Post-Training, Not Just Pretraining

For models with SFT/DPO/RLVR post-training, you must replay those stages
to recover the actual capabilities (instruction following, reasoning quality,
safety, format control).

## 9. Never Trust Profiling Predictions

Per-matrix profiling said "additive dloss = 1.6." Actual pre-KD PPL: 9,739.
Profiling measures each matrix in isolation; compound errors through 32 layers
of residual stream are multiplicative, not additive.

## The Full Pipeline (What Works)

1. Profile all matrices with Fisher information (on training data, not C4)
2. Allocate per-matrix rank via MCKP knapsack
3. Compress sequentially (evolved order, Fisher recomputed per layer)
4. KD on the original training data with structured completions
5. Post-training replay (SFT + DPO + RLVR)
6. Eval with proper stop tokens, behavioral probes, AND benchmarks
