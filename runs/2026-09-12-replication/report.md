# Divergence diagnosis and second-model replication

The original 8B experiment remains STOP under its registered criteria. This report
answers the separately authorized continuation, with the same unmodified engine patch.

## Qwen3-4B Q4_K_M replication

200 questions, 800 requests; 200 complete pairs.
Paired median end-to-end: **1.122x [1.101, 1.144]**. Decode: 1.243x [1.217, 1.269].
Baseline: identical compact llguidance, batching off, not a tuned market best.
The same 200 questions are reused across sizes, not a fresh question holdout.
Cold cache, temperature zero, one slot, same-family tokenizer and the same RTX 2080 Ti.

| Subset | Complete pairs | End-to-end ratio [95% bootstrap interval] |
|---|---:|---:|
| simple | 40 | 1.223x [1.187, 1.295] |
| multiple | 40 | 1.105x [1.079, 1.128] |
| live_simple | 40 | 1.083x [1.070, 1.152] |
| live_multiple | 40 | 1.034x [1.003, 1.066] |
| hermes | 40 | 1.153x [1.110, 1.188] |

Candidate/baseline pooled p95: 0.934; includes timed failures.
Failed requests: 0. Candidate-only failed questions: 0.
Parsed differences: 5; unstable questions: 0.
Potential-regression rate one-sided 95% upper bound: 5.18%.
Chosen-reference equality: off=123, on=123; not official accuracy.
Accepted/proposed grammar drafts: 5386/5386.
Observed VRAM MiB: off=3802, on=3802; snapshots, not a profiled peak.

| Separate replication criterion | Result |
|---|---|
| adequate_complete_pairs | True |
| performance_reproduced | True |
| preservation_screen_passed | False |
| old_1_4_target_met_descriptively | False |

## Original 8B differences: semantic review

Post-hoc assessments against supplied prompts/tools, not blinded judgments
or an official benchmark score. No dataset tool or external application was executed.

### multiple_175

both_inadequate: The user asks for women's tennis rankings. Both outputs select the musical instrument tool instead of the supplied sports ranking tool.
First differing token offset: 19; both alternatives grammar-legal: True; allowed tokens: 147039; forced tokens there: [].

### live_simple_237-125-0

likely_regression: Both identify the requested song. Only batching adds device_id=':null', an unsupported device identifier, distinct from the documented default 'null'. Explicit play=true alone is consistent with the documented default. No Spotify tool was executed.
First differing token offset: 22; both alternatives grammar-legal: True; allowed tokens: 147039; forced tokens there: [].

### live_multiple_221-95-0

both_inadequate: Both select segment but replace the explicit relative path './yoha.jpg' with unsupported absolute paths. The provided context establishes neither replacement.
First differing token offset: 11; both alternatives grammar-legal: True; allowed tokens: 147039; forced tokens there: [].

### live_multiple_749-169-4

same_bytes_tokenization_control: Parsed output and bytes are identical; token IDs differ.
First differing token offset: 3; both alternatives grammar-legal: True; allowed tokens: 3; forced tokens there: [].

### f595b8d5-531a-4699-85bc-02620d4e1de8

improvement: The user explicitly requests createTaxiBookingSkill with the supplied details. Batching follows that request; the baseline instead cancels an invented booking.
First differing token offset: 3; both alternatives grammar-legal: True; allowed tokens: 4; forced tokens there: [].

## Controlled execution checks

50 diagnostic requests; 0 failures.
Token equality requires both repetitions. Microbatch-one also changes prefill:
compare its off/on arms to each other, not to the normal baseline.

| Case | Off reproduced | On reproduced | Zero cap = off | Micro-one off = on |
|---|---|---|---|---|
| multiple_175 | True | True | True | True |
| live_simple_237-125-0 | True | True | True | True |
| live_multiple_221-95-0 | True | True | True | True |
| live_multiple_749-169-4 | True | True | True | True |
| f595b8d5-531a-4699-85bc-02620d4e1de8 | True | True | True | True |

Agreement under microbatch-one, with actual accepted drafts, supports dependence
on execution shape, not direct measurement of floating-point/logit differences
or proof the integration has no other bugs. Divergent free choices are not
incorrect forced tokens. The audit retains independent singleton-mask replays.

## Interpretation and limits

Speed reproduction and preservation are separate. A failed preservation screen
does not mean every changed answer is wrong, nor that possible harm can be ignored.
No new performance tuning, warm-cache rescue, niche selection, cross-family claim, or
state-of-the-art superiority claim follows. The 4B baseline was not tuned against
other speculative strategies. Wall time excludes model loading and tool execution.
The question bootstrap does not remove process-time confounding.

Protocol: docs/replication-plan.md. Driver: scripts/09_replication.py. All requests,
outputs, runtime/model/protocol hashes, controls and failures are preserved.
