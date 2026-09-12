# Measured native-forcing prototype decision

Decision: **STOP** under the prospective protocol.

200 frozen questions; 800 requests; 198 complete pairs.
Cache condition: cold. Median paired end-to-end speedup: **1.358x [1.331, 1.396]**.
Decode-only speedup: 1.658x [1.602, 1.696].

The denominator is the same Qwen3-8B Q4_K_M, patched runtime with the flag off,
and the same compact llguidance G2/S1 grammar. The candidate enables native forced-token
batching through the real speculative verifier, with a 32-token cap and no draft model.
Both arms include target KV computation, grammar/sampling work and prompt processing.
Wall time is local inference, not model loading, tool execution or application latency.

## Actual timings and work

| Arm | Median wall ms | Median decode ms | Generated tokens* | Observed VRAM MiB | Draft accepted/proposed |
|---|---:|---:|---:|---:|---:|
| none | 717.8 | 446.7 | 29.5 | 5904 | 0/0 |
| grammar | 526.1 | 272.7 | 30.0 | 5904 | 5408/5408 |

*Token counts include EOS. VRAM is an observed snapshot maximum, not a peak.
Candidate/baseline p95 ratio: 0.856 (lower is better).
P95 includes every timed request, including failures. Paired medians require both
arms and repetitions to be valid. The completeness gate retains excluded failures.
Generation batch counts are derived as generated tokens - 1 - accepted drafts,
for this single-slot Transformer, not read from a GPU profiler. They exclude prefill
and rely on each verification block fitting in one decode microbatch.
Derived generation batches: off=23494, on=18216.

| Subset | Wall speedup [95% bootstrap interval] | Decode speedup [interval] |
|---|---:|---:|
| simple | 1.554x [1.510, 1.641] | 1.854x [1.796, 1.942] |
| multiple | 1.354x [1.329, 1.364] | 1.620x [1.537, 1.684] |
| live_simple | 1.484x [1.446, 1.560] | 1.774x [1.675, 1.944] |
| live_multiple | 1.287x [1.257, 1.313] | 1.619x [1.530, 1.673] |
| hermes | 1.232x [1.160, 1.299] | 1.432x [1.261, 1.469] |

## Correctness and pre-registered criteria

Failed requests: 8. Questions with changed parsed output: 4.
Questions unstable across repetitions: 0. Potential-regression rate one-sided 95% upper bound: 4.56%.
Same parsed JSON but changed bytes: 0; same bytes but changed token IDs: 1.
Every successful response was independently rechecked for schema validity, UTF-8
and token-byte roundtrip. Parsed differences count as potential regressions;
they are not assumed correct or incorrect from a single reference.

| Criterion | Pass |
|---|---|
| complete_no_failures | False |
| median_at_least_1_4 | False |
| ci_lower_above_1 | True |
| all_subsets_at_least_1 | True |
| p95_at_most_1_10 | True |
| quality_upper_below_2_percent | False |
| actual_grammar_drafts | True |

## Interpretation and limits

This is an actual batching measurement, not a cost projection. A STOP means this
bounded prototype did not justify the retained general-scope investment under the
registered criteria; it is not a theorem that all future implementations must fail.
Do not lower the target, substitute decode speedup, or seek a favorable niche
after these results. A failed cold gate does not require a warm-cache rescue run.

The balanced mixture spans five subsets and four reference-length rank strata,
but only single-call bodies on one model/hardware pair. This is not BFCL accuracy,
a deployment distribution, or proof across parallel calls/chat wrappers.
The binomial preservation bound assumes independent question outcomes; benchmark
selection and dependence limit its interpretation. It does not prove zero risk.
Process order was off/on/on/off with reversed questions on repetition one. Remaining
time-block confounding is not removed by the question-bootstrap intervals.

The llguidance 1.0.1 FFI required a local bounded-copy safety fix. Both measured
arms use that same fixed library; it does not change grammar or tokenization policy.
The shared 2048-token ceiling was sealed before generation because one reference
exceeded the initial ceiling. Selected IDs and the 1.4x target were unchanged.

## Reproduction

See docs/prototype-plan.md and scripts/08_prototype.py. The seal fixes the runtime,
dependency diff, protocol and input hashes. Raw outputs, commands, token IDs,
memory snapshots, pair results and hashes are retained beside this report.
