# Gate 0 — preliminary feasibility report

Status: **incomplete**. Updated 2026-09-12T03:49:54Z.

The source audit and hardware microbenchmarks support investigating native llguidance batching.
They do not yet establish that realistic tool calls contain enough injectable spans, that batching
achieves the proposed end-to-end target, or that a whitespace change preserves task accuracy.
No engine implementation is authorized by a Gate 0 pass in this report.

## Completed evidence

- The pinned llguidance implementation already forces its native canonical tokens through
  singleton masks; see [policy audit](notes/llguidance-ff-tokens.md).
- The original broad novelty claim needs narrowing; see [prior-art audit](notes/prior-art.md).
- The stock CPU debug `test-grammar-llguidance` executable passed with the local target's GGUF
  vocabulary. This checks existing backend integration, not the proposed batching path.
- The batch-cost and prefill commands completed successfully. Timing tables below are derived
  programmatically from the raw JSONL outputs and [summary](../runs/2026-09-12-pass-cost/summary.json),
  not estimates of observed tool-call performance.

## Hardware cost

Configuration and exact commands: [manifest](../runs/2026-09-12-pass-cost/manifest.json).
Target: local Qwen3-8B Q4_K_M; RTX 2080 Ti; all layers on GPU; flash attention on.
The unmodified reference binary has llguidance disabled because this measurement isolates model
processing. The depth prefix is prepared outside the timed interval.

| Context depth | Batch tokens | Mean ms | Sample SD ms | Cost / one token |
|---:|---:|---:|---:|---:|
| 1024 | 1 | 11.580 | 1.473 | 1.00 |
| 1024 | 2 | 12.162 | 1.176 | 1.05 |
| 1024 | 4 | 16.364 | 1.833 | 1.41 |
| 1024 | 8 | 26.720 | 1.641 | 2.31 |
| 1024 | 16 | 19.826 | 1.561 | 1.71 |
| 1024 | 32 | 24.232 | 2.071 | 2.09 |
| 4096 | 1 | 12.589 | 1.252 | 1.00 |
| 4096 | 2 | 13.016 | 1.091 | 1.03 |
| 4096 | 4 | 17.168 | 1.754 | 1.36 |
| 4096 | 8 | 26.816 | 1.757 | 2.13 |
| 4096 | 16 | 20.532 | 1.687 | 1.63 |
| 4096 | 32 | 25.985 | 1.534 | 2.06 |

The cost per batch is non-monotonic at the measured sizes. A next-size interpolation is therefore
not automatically a conservative estimate. The measured curve must be retained; do not fit a
monotonic speedup and treat it as observed. Five within-process repetitions describe local
microbenchmark variation, not a confidence interval over tool-call workloads.

| Prompt tokens | Mean ms | Sample SD ms |
|---:|---:|---:|
| 1024 | 392.328 | 0.754 |
| 4096 | 1674.576 | 6.289 |

These prefill costs use synthetic prompts without prefix-cache hits. Real prompts and reused
prefixes need their own measurements. Neither prompt processing nor the remaining free tokens
can be assumed free in an end-to-end projection.

## Decision status

| Decision input | Status |
|---|---|
| Native PL algorithm and default-mask behavior | Source-audited |
| Existing llguidance integration with the GGUF vocabulary | Existing test executable passed |
| Cost of batching model tokens | Measured |
| Sequential mask agreement for complete PL spans and lazy triggers | Pending |
| Frozen pilot IDs and reference-adapter validation | Pending |
| BFCL/Hermes forced fractions, span lengths and coverage | Not measured |
| CPU extraction and grammar overhead | Not measured |
| Offline/native tokenizer parity | Not measured |
| Real model-generated output and task accuracy | Not measured |
| Draft latency and memory comparison | Not measured |
| Predicted or measured end-to-end tool-call speedup | Not available |

## Next dependency

The repository contains no measurement implementation yet. Its `CLAUDE.md` reserves code
authorship to Rafael. The user has been asked whether this investigation may introduce Phase 0
Python scripts; that choice is pending. Documentation and stock-binary checks were completed
without creating or modifying project code.

Once that choice is resolved, the next executable work is the registered pilot adapter and PL
span measurement. Use all candidate tools, preserve failures in the coverage denominator, and
check tokenizer/mask agreement before accepting the fractions. The pilot must not be labelled a
full-corpus Gate 0 result.

**Gate 0 verdict: incomplete.** The hardware result is encouraging, but a go/no-go decision still
depends on workload measurements.
