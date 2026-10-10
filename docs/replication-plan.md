# Divergence audit and second-model replication

Registered 2026-09-12 after the first experiment and before new model generations.
Rafael explicitly authorized this limited continuation. The first experiment remains STOP
under its original rule; this is a new investigation, not a retroactive threshold change.

## Scope and budget

Allow up to two hours of active implementation/analysis and one hour of isolated GPU work.
Reuse the sealed C++ runtime, including the llguidance FFI safety fix, with no engine edits,
performance tuning, new weights, paid infrastructure, niche selection, or upstream submission.
Preserve all old runs and default-off behavior. Use only owned servers; refuse a busy GPU.

The second target is the already pinned Qwen3-4B Q4_K_M in config/models.json. This tests another
model size in the same family/tokenizer, not cross-family or cross-hardware generalization.
Verify its GGUF hash and tokenizer/server parity before measuring. No second-target strategy
search is conducted: the denominator is the matched compact no-batching configuration, not a
claim to the strongest possible configuration for the 4B model or the market.

## First-model diagnosis

Audit all four parsed divergences and the one same-bytes/different-token case in the frozen 8B
results. Retain prompts, tools, chosen references and both outputs. Classify quality effects
against the supplied request/tool semantics, not exact reference equality alone. These are
post-hoc descriptive judgments, not official benchmark accuracy or a blinded evaluation.

Replay both token sequences through the offline matcher. Locate their first differing token,
check whether both alternatives are legal at the identical prefix, and whether that position
is forced. Verify native forced spans match sequential singleton masks. This can detect an
unsupported forcing interpretation; it cannot by itself prove the GPU state is correct.

Use five fixed diagnostic configurations, two repetitions of all five cases:
normal off; normal on (cap 32); grammar enabled with cap zero; microbatch-size-one off;
microbatch-size-one on (cap 32). Reverse cases on repetition one. Keep the original prompts,
grammar, seed, 2048-token ceiling, temperature zero and cold cache. The normal arms check
reproduction; cap zero checks the no-draft path; matched microbatch-one arms probe dependence
on physical batching. These slower controls are never candidates in the timing comparison.

Record real draft counters. If disagreements disappear with physical microbatch size one,
that supports execution-shape/numerical sensitivity but is not direct measurement of logits
or proof excluding every integration bug. The current speculative server path does not
populate per-token probability arrays; do not fabricate a logit-level explanation. Do not
change the engine to force a desired result. Diagnostic errors remain reportable limitations.

## Second-model replication

Use every one of the original 200 questions, including both previously truncated cases.
This is the same benchmark mixture, deliberately not a new unseen-question claim. No selection
based on observed gains or divergences. Freeze cases, protocol, runtime, model and driver hashes
before generation. Run two repetitions, process order off/on/on/off, reverse question order
on repetition one. Fixed compact G2/S1, cap 32, one slot, cold cache, 2048 output tokens,
temperature zero, thinking off, exact matched requests. No warm-cache branch in this stage.

Report paired median end-to-end and decode ratios with the existing question bootstrap (5000
resamples, seed 20260912), all five subsets, pooled p95 including timed failures, observed VRAM,
draft acceptance, invalid/truncated responses, parsed/byte/token divergence and stability.
Report the old 1.4x target descriptively; do not turn a smaller result into an old-gate pass.

Separate conclusions:

- Performance reproduction is supported if the paired median interval's lower endpoint is
  above 1, every subset median is at least 1, and candidate pooled p95 is at most 1.10x baseline.
  Require at least 190 complete pairs with at least 35 in each subset; otherwise inconclusive.
- Preserve the earlier conservative output-change screen: one-sided exact binomial 95% upper
  below 2%, with no candidate-only failures or new structural invalidity. Changed answers are
  potential regressions until assessed, not automatically errors or improvements.
- A speed result can be positive while preservation remains unresolved. Neither result proves
  production readiness, new algorithmic novelty, or superiority over all existing engines.

End this stage after the fixed diagnosis and replication. Report what is established and what
would require new authority; do not continue into optimization or a broader benchmark campaign.
