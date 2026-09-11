# Shelved: Draft Adaptation Toolkit (2026-09-10)

This repository was first set up for adapting speculative-decoding draft models to quantized
llama.cpp targets. Full proposal: `~/ideas/proposta-draft-adaptation-toolkit.md`. The project
switched to grammar jump-forward decoding the same day; see `docs/decisions/0001-switch-to-grammar-jump-forward.md`.

## Why it was shelved

- **The core novel effect is probably small.** Drafts aligned to the Q4 target vs the high-precision
  target only get different training labels where the two targets disagree on the top-1 token.
  llama.cpp's Llama-3-8B KLD scoreboard shows roughly 92–98% top-1 agreement for 4-bit-and-up quants,
  so the expected C-over-B gain is a few percentage points at most. That estimate is unmeasured.
- **The pilot threshold measured the wrong thing.** The ≥10 pp threshold (adapted vs off-the-shelf
  draft) mostly measures domain and language fine-tuning, which is known to work.
- **The field is moving to heads attached to the target.** Standalone draft models have the lowest
  ceiling, and llama.cpp at `95ef7fc16` already supports `draft-eagle3`, `draft-mtp`, `draft-dflash`
  and `draft-dspark`.

## Facts verified while it was active (still true at `95ef7fc16`)

- Speculative flags were renamed: `-md/--spec-draft-model`, `--spec-type draft-simple` (default `none`),
  `--spec-draft-n-max` (default 3), `--spec-draft-n-min` (0), `--spec-draft-p-min` (0.00),
  `--spec-draft-p-split` (0.10), `-ngld`. `--draft-max` and `--draft-min` are removed.
- The server response `timings` object carries `draft_n` and `draft_n_accepted`.
- `--reasoning off` or `--reasoning-budget 0` disables Qwen3 thinking. Qwen3 thinks in English even
  for Portuguese prompts.
- `llama-perplexity --kl-divergence-base` / `--kl-divergence` reports "Same top p" (teacher-forced
  top-1 agreement), usable between any two models that share a vocabulary.
- `~/models/qwen3-{8b,4b}-q4_k_m` were built from AWQ-compatible checkpoints and are not plain quants of
  the official bf16 weights.
- PyTorch 2.14.0+cu130 runs on the 2080 Ti (`sm_75` present; fp16 matmul check passed).

## To resume

Step 0 still costs a day: measure Q4_K_M vs Q8_0 "Same top p" on pt-BR text before building anything.
It also connects to RQ4 of the current project: draft-model acceptance on grammar-forced positions.
