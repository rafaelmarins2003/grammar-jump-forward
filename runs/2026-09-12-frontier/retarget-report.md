# Retargeting follow-up — prompts, cache and output length

Recommendation: keep **retarget**. Cache creates a conditional opportunity, not an engine go.
The most important next input is a concrete application workload, not another broad implementation.

## What was actually measured

The recorded generated cohort supplied 25 questions. The controlled assay completed 225 one-token probes, with 0 validation failures and 25 complete questions. All candidate tool schemas and the existing G2 grammar were retained.
The engine remains unmodified. Each condition has three repetitions; reported latencies below
are medians across questions of within-question repetition medians. This is not live traffic,
a production cache-hit estimate, or a jump-forward latency benchmark.

Prompt tokens: median 359, p10–p90 207–936. Generated body tokens: median 38, p10–p90 23–76. The system/tool prefix occupied a median 88.8% of the prompt.

| Condition | Measured request-prefix wall ms | Prefill ms | Reused tokens | Newly processed tokens |
|---|---:|---:|---:|---:|
| Cold prompt | 167.79 | 159.44 | 0 | 359 |
| Warm system/tools, user suffix uncached | 33.41 | 30.55 | 295 | 37 |
| Exact prompt repeat (best-case control) | 16.42 | 13.59 | 358 | 1 |

Reused tokens come from `timings.cache_n`. At this pinned engine, top-level
`tokens_cached` is populated from the slot after generation, so it is not the number
reused by the request. The counters were checked against the exact native prompt token IDs.
All measured warm conditions were explicitly primed; those preparation requests and costs
are retained in `probes.jsonl`. They are not free. A real request stream must pay misses and
may lose prefixes to eviction, slot scheduling, tool-schema changes or changed earlier text.
Exact prompt repetition is a boundary condition, not an assumed normal application.

The collection was resumed 1 time(s) after an interruption, keeping all flushed trials. Warm trials recreate their own prefix state. Continuation manifests preserve the prior-prefix file hash.

## Projected batching after measured prefix cost

Both models use the conservative existing synthetic batch proxy at depth 1024. The model-only
case combines that body cost with measured request-prefix wall time. The residual-preserving
case additionally retains, in both arms, the difference between the earlier observed complete
decode time and synthetic sequential body cost. It does not pretend that all decode overhead
will be accelerated. The residual is a mixed timing discrepancy, not an attributed CPU cost.

23 of 25 prompt lengths were outside the measured 1024–4096 depth range (below the lower anchor). The model curve is therefore an approximation,
not a calibrated measurement at every actual depth. Thermal/run-to-run differences, grammar
and sampling costs, and altered numerical execution remain unisolated. Negative residuals
would invalidate that calibration, not be clamped away.

| Assumption | Cache condition | Median ratio | Questions reaching 1.5× | Ratio of total costs |
|---|---|---:|---:|---:|
| model_only | cold | 1.45× | 36.0% | 1.30× |
| model_only | tools | 1.60× | 64.0% | 1.39× |
| model_only | repeat | 1.68× | 68.0% | 1.43× |
| residual_preserving | cold | 1.34× | 16.0% | 1.23× |
| residual_preserving | tools | 1.39× | 48.0% | 1.28× |
| residual_preserving | repeat | 1.43× | 48.0% | 1.29× |

The denominator is the same-format unbatched G2/S2 baseline. These are hypothetical
body-cost replacements after an observed prefix, not measured patched/unpatched ratios.
A ratio of total costs weights long requests more heavily and must not replace the median
per-question target merely because it looks better.

## How much cache would be needed?

For baseline body cost B, batched cost J, target s and shared overhead O:

`(O+B)/(O+J) >= s` requires `O <= (B-s*J)/(s-1)`.

Negative overhead budgets mean batching cannot reach the target even with a free prefix.
For hit rate h, cold/hit prefix costs P/H, the mixture is `(1-h)*P+h*H`. Hit rate is the
fraction of requests with a warm prefix, not the fraction of tokens reused. The residual
is retained in both P and H where applicable. Scenarios below are hypothetical:

| Assumption | Warm condition | 0% hits | 50% hits | 90% hits | 100% hits | Possible questions / total |
|---|---|---:|---:|---:|---:|---:|
| model_only | tools | 1.45× | 1.53× | 1.58× | 1.60× | 16/25 |
| model_only | repeat | 1.45× | 1.54× | 1.65× | 1.68× | 17/25 |
| residual_preserving | tools | 1.34× | 1.36× | 1.38× | 1.39× | 12/25 |
| residual_preserving | repeat | 1.34× | 1.38× | 1.42× | 1.43× | 12/25 |

The possible subset is not a replacement sample. Per-question minimum rates and impossible
cases are retained in `questions.json`; no production hit rate has been inferred or fitted.

## Full reference corpus: common compact baseline

This reuses all complete reference questions, with the original coverage limitations.
Zero prefix overhead is an ideal limit, not a measured warm-request cost. The batching
column is conservative; each row uses the stated unbatched serialization as its reference.

| Unbatched → batched | Depth | Questions | Median, zero prefix cost | Median shared overhead budget for 1.5× (ms) |
|---|---:|---:|---:|---:|
| G1/S1 → G1/S1 | 1024 | 3095 | 1.35× | -88.57 |
| G1/S1 → G1/S1 | 4096 | 3095 | 1.36× | -84.97 |
| G1/S1 → G2/S1 | 1024 | 3095 | 1.49× | -8.15 |
| G1/S1 → G2/S1 | 4096 | 3095 | 1.51× | 3.47 |
| G1/S2 → G2/S2 | 1024 | 3095 | 1.69× | 110.49 |
| G1/S2 → G2/S2 | 4096 | 3095 | 1.72× | 136.58 |
| G1/S1 → G2/S2 | 1024 | 3095 | 1.36× | -92.02 |
| G1/S1 → G2/S2 | 4096 | 3095 | 1.38× | -82.22 |

Adding predictable spaces can improve the fraction-forced metric while losing to compact
JSON on absolute latency. Preserve a common compact baseline in the next experiment.

## Longer output is not automatically a better workload

Strata use compact body length for the same questions in every comparison. These are
descriptive post-hoc strata, not newly selected success cohorts. No string was artificially
repeated or padded to manufacture a long forced span.

| Comparison (depth 1024) | Compact body tokens | Questions | Median forced fraction | Median, zero prefix cost | Median, cold prefix |
|---|---|---:|---:|---:|---:|
| G1/S1 → G2/S1 | 1–32 | 1527 | 48.4% | 1.57× | 1.17× |
| G1/S1 → G2/S1 | 33–64 | 864 | 38.7% | 1.43× | 1.20× |
| G1/S1 → G2/S1 | 65–128 | 495 | 40.0% | 1.45× | 1.29× |
| G1/S1 → G2/S1 | 129–256 | 155 | 29.8% | 1.31× | 1.24× |
| G1/S1 → G2/S1 | >256 | 54 | 13.7% | 1.12× | 1.11× |
| G1/S2 → G2/S2 | 1–32 | 1527 | 56.7% | 1.83× | 1.25× |
| G1/S2 → G2/S2 | 33–64 | 864 | 47.4% | 1.60× | 1.30× |
| G1/S2 → G2/S2 | 65–128 | 495 | 49.6% | 1.64× | 1.43× |
| G1/S2 → G2/S2 | 129–256 | 155 | 41.0% | 1.47× | 1.38× |
| G1/S2 → G2/S2 | >256 | 54 | 18.4% | 1.17× | 1.16× |

## Decision and next bounded step

Do not start the engine patch or expand into draft composition on these estimates alone.
The narrower candidate is repeated structured work with stable system/tool prefixes and
enough genuinely deterministic structure, measured against compact JSON. Cache alone does
not remove non-accelerated decode cost, and long free-form values can dilute the opportunity.

Before changing the gate, collect representative traces from the intended application:

- prompt/body token counts and the actual system/tool prefix;
- observed `timings.cache_n`, prefill/decode/wall times, misses and slot/concurrency behavior;
- schema/grammar identity and raw output token IDs, with sensitive values kept local;
- semantic accuracy criteria and acceptable regression margin fixed before confirmation.

Use those traces to choose a small confirmatory workload and measure a matching-depth cost
curve. If it still cannot plausibly reach the target against compact JSON, stop or accept a
smaller learning/engineering objective explicitly. A positive result would justify a revised
go decision, not a retroactive claim that the original gate passed.

## Reproducibility

- [Cache measurements and summaries](../runs/2026-09-12-workload-cache/summary.json).
- [Collection manifest](../runs/2026-09-12-workload-cache/collect-manifest.json) and continuation subdirectories.
- [Full-cohort frontier and report hashes](../runs/2026-09-12-frontier/manifest.json).
- [Original Gate 0 report](gate0-report.md) remains an immutable-results reference.
- Scripts: `05_workload_cache.py` (collect/analyze), `06_retarget_frontier.py` (report).

The standalone report is generated from run artifacts; numbers are not transcribed manually.
No application telemetry was available. No tools described by the dataset were executed.

Decision: **retarget**, with no C++ go.
