# 0001 — Switch from draft adaptation to grammar jump-forward

**Date:** 2026-09-10 · **Status:** accepted · **Decided by:** Rafael

## Context

Two candidate proposals, compared on complexity, relevance and risk:

- Draft Adaptation Toolkit: `~/ideas/proposta-draft-adaptation-toolkit.md`
- Grammar Jump-Forward Decoding: `~/ideas/grammar-jump-forward-llamacpp.md` (frozen copy: `docs/proposal.md`)

## Decision

Pursue grammar jump-forward. Front-load the cheap kill criteria (Gate 0, RQ1 emulation, pass-cost
estimate, early RQ4) in Python before writing any engine C++.

## Why

- **Relevance.** Local agents run tool calls under GBNF or llguidance. The gap is verified at
  `95ef7fc16`: no jump-forward path exists, and even the reasoning-budget feature forces tokens one
  forward pass at a time.
- **Risk can be tested early.** Forced-token fraction, pass cost and forced-tokenization quality
  can all be measured without touching the engine.
- **Draft adaptation's novel effect is likely small** (see `docs/archive/draft-adaptation.md`).
- **Fit.** Real C++ systems work in a major project, and continuity with the TCC through DSL
  determinism.

## Consequences and known threats (verified in source, 2026-09-10)

- llama.cpp's JSON-schema grammar allows optional whitespace between structural tokens
  (`common/json-schema-to-grammar.cpp:227`). Under the stock grammar, forced spans may be rare.
  Gate 0 must report both the stock grammar and a whitespace-canonical variant.
- Tool-call grammars are lazy and activate on a trigger (`grammar_lazy`, `grammar_triggers`).
- llguidance v1.0.1, which llama.cpp already builds, exposes `llg_matcher_compute_ff_tokens`,
  `llg_matcher_consume_tokens`, `llg_matcher_validate_tokens` and `llg_matcher_rollback`. llama.cpp's
  integration uses none of them. On the llguidance backend the engine work may reduce to wiring; the
  GBNF backend has no equivalent.
- The RQ1 novelty claim ("no published measurement") must be checked against DOMINO
  (Beurer-Kellner et al., ICML 2024) and guidance's token healing before it is written anywhere.
- The speculative framework churns quickly (types added, flags renamed). Keep the patch small and
  export it to `patches/` at every milestone.

## Alternatives not taken

- Keep draft adaptation. Rejected for the reasons above; archived with its verified facts.
- Do both at once. Rejected: RQ4 already connects them, and one well-gated project fits the time.
