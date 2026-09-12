# Bounded native-forcing prototype protocol

Registered 2026-09-12 before engine changes or prototype measurements. Rafael authorized this
experiment after the existing-baseline report with "Certo, então faça". This is a prospective
experiment-only go, not a retroactive pass of the original Gate 0.

## Effort and implementation boundary

Spend at most two hours of active implementation/validation and one hour of isolated GPU
measurement. If a trustworthy instrument cannot be produced within that limit, stop and report
the limitation; do not expand the design. No new models, hardware, applications, public submissions
or full speculative-strategy tuning. No concurrent GPU jobs.

Add a default-off `--spec-type grammar` proposer using the current llguidance matcher's native
fast-forward tokens. Its state is already after the last sampled token; do not consume that token
again. Reuse the existing speculative target batching, real token-by-token verification, sampler
acceptance, stopping, output and KV handling. No synthetic acceptance, separate decode loop, new
tokenization policy, skipping target KV computation, or claim of numerical losslessness.
The initial cap is 32 forced draft tokens, configurable by the existing max-draft option. This cap
is fixed for evaluation. Unsupported/missing grammar and active reasoning-budget paths yield no
grammar draft; composition with other proposers is outside the measured experiment.

Use development cases from the previous 25-question cohort plus hand-written boundary checks.
Verify default-off behavior, ordinary native GBNF fallback, empty/free regions, forced spans,
Unicode, optional branches, token limits, stopping and repeated requests. A release CUDA build and
CPU debug checks must succeed before evaluation. Retain the unmodified runtime for comparison.

Pre-implementation safety finding: the pinned llguidance C FFI computes a bounded returned length
but copies the unbounded vector length in `llg_matcher_compute_ff_tokens`. Correct that copy length
locally in both build dependencies and retain a separate patch/hash and a bounded-output regression
check. No tokenizer or grammar semantics change is intended. Do not call the unsafe original API.

## Frozen general-scope evaluation

Select 200 previously ungenerated single-call questions from the already adapted full corpus:
40 each from simple, multiple, live_simple, live_multiple and Hermes. Exclude every question in the
earlier generated cohort. Within each subset, sort eligible questions by compact reference token
length (ID as tie-break), divide into four equal-count rank strata, and select ten per stratum by
SHA-256 of `20260912:prototype:<subset>:<id>`. Freeze all IDs before prototype generation. Never use
forced fraction or timing to select cases. Record adaptation eligibility and lengths; this is a
balanced benchmark mixture, not a deployment distribution. No oracle tool selection.

The previous generated cohort remains the development set. Do not change the patch or tune its cap
after held-out evaluation starts. If an instrument bug invalidates evaluation, preserve the failed
run and stop this decision round rather than reusing the same selection as a fresh holdout.

Primary comparison: identical compact G2/S1 grammar, target model, prompt, temperature zero,
thinking off, one slot, 512-token limit and cold cache. Compare patched-binary `none` against
`grammar`; the former is the strongest applicable configuration from the fixed baseline screen.
Use two repetitions in process-block order A0, B0, B1, A1, reversing question order in repetition
one. Warm each process outside timing. Verify `timings.cache_n == 0`. Record all raw responses,
tokens, end-to-end wall and server decode/prefill times, actual draft acceptance and observed VRAM.
This counterbalanced blocking reduces but does not eliminate time-block confounding.

If the primary cold gate passes, run the same grid with exact-prompt-repeat caching as a separate
secondary condition, retaining untimed priming requests. If it fails, warm evaluation may be
skipped to save time; cache cannot rescue the primary decision and no hit-rate mixture is assumed.

## Pre-registered go/stop rule

Performance requires median paired end-to-end speedup >= 1.4x, with a question-bootstrap 95%
interval lower endpoint above 1.0 (5000 resamples, seed 20260912), no subset median below 1.0, and
candidate pooled p95 request latency at most 1.10x baseline p95. Reduce repetitions within each
question by their median before computing paired ratios. Report decode separately; no ratio of
unpaired medians may substitute for the paired primary statistic.

Correctness requires no new invalid JSON/schema/UTF-8, truncation, request failures or token-byte
roundtrip failures. Report all failures, not just successful pairs. Parsed-output changes relative
to the identical-grammar baseline are conservatively treated as potential quality regressions,
even if a changed answer matches the chosen reference. A question counts once if any repetition
differs or is unstable. Require the one-sided exact binomial 95% upper bound for this rate to be
below 2%. With 200 questions and zero differences, that bound is below 2%; the sample cannot prove
zero risk or official task accuracy. Reference equality is descriptive only. Separately report
same bytes/different token IDs and changed bytes/same parsed JSON.

All performance and correctness criteria must pass to recommend further general-scope engineering
on this model/hardware. Missing measurements or sample inadequacy do not count as a pass. A win
only in a favorable subset, only decode, or only warm cache is not a pass. Failure means stop this
general-scope investment, not search for a niche or lower the threshold after seeing the result.
Even a pass would still require later multi-model, parallel-call, lazy-wrapper and concurrency
evaluation before making a broad production claim.

## Pre-evaluation input-length amendment

Before any held-out model generation, input inspection found one frozen reference longer than
the initial 512-token ceiling (914 compact tokens). Keep that question and all selected IDs, and
raise the common generation ceiling to 2048 tokens in both arms; do not change the batch cap,
performance target or quality rule. The original selection manifest retains the earlier protocol
hash. A separate evaluation seal records the updated protocol, runtime hashes and passed smoke
checks before the first held-out request. No further protocol or engine changes are permitted
after that seal. This avoids preordaining a truncation from a known input length.
