# General-scope existing-baseline screen

**This is a measured comparison of existing configurations, not a jump-forward result.**
The earlier approximate 1.4x was a projection against no-draft llguidance with equal
serialization/cache assumptions. It was not measured against an unconstrained model or
the best existing alternative. The original Gate 0 remains unpassed.

Frozen cohort: 25 questions from five benchmark subsets;
350 requests, two repetitions per question/configuration.
Qwen3-8B Q4_K_M on the RTX 2080 Ti, temperature zero, thinking off, cold cache, one slot.
All candidate tools are included in the schema; no functions are executed.

## Measured configurations

Wall time is local HTTP request latency including prefill, not model loading or tool
execution. Decode is the server's predicted_ms. Each question is reduced to the median
of its two repetitions before aggregation. Token counts help expose length confounding.

| Configuration | Complete questions | Wall ms | Decode ms | Body tokens | VRAM MiB* |
|---|---:|---:|---:|---:|---:|
| none/stock | 25/25 | 732.5 | 512.1 | 39.0 | 5904 |
| none/S1 | 25/25 | 623.2 | 365.1 | 28.0 | 5904 |
| none/S2 | 25/25 | 692.0 | 507.5 | 39.0 | 5904 |
| draft-simple/S1 | 25/25 | 666.6 | 383.0 | 28.0 | 7452 |
| draft-simple/S2 | 25/25 | 695.6 | 450.3 | 36.0 | 7452 |
| ngram-simple/S1 | 25/25 | 670.3 | 399.3 | 28.0 | 5908 |
| ngram-simple/S2 | 25/25 | 772.2 | 543.4 | 39.0 | 5908 |

Stock sends json_schema to the native GBNF sampler, with its default whitespace.
S1 uses llguidance fixed-order compact JSON; S2 uses fixed-order spaced JSON.
draft-simple uses the pinned Qwen3-0.6B Q8_0, GPU offload, max draft 3, p-min 0.75.
ngram-simple uses max draft 8. These are untuned configurations, not a search over
every existing speculative method. *VRAM is the maximum observed process snapshot,
not a profiled peak; it includes the target and any draft model.

## Paired speedups

Each row is baseline time / candidate time, then the median across paired questions.
Above 1 means the candidate was faster. Brackets are question-bootstrap 95% intervals
(5000 resamples, seed 20260912). They describe this cohort, not general deployment.

| Baseline → candidate | Wall speedup [CI] | Decode speedup [CI] | Parsed differences |
|---|---:|---:|---:|
| none/S2 → none/S1 | 1.183x [1.103, 1.240] | 1.251x [1.190, 1.280] | 4/25 |
| none/stock → none/S1 | 1.170x [1.136, 1.206] | 1.240x [1.199, 1.295] | 4/25 |
| none/stock → none/S2 | 1.021x [0.995, 1.034] | 1.002x [0.994, 1.012] | 0/25 |
| none/S1 → draft-simple/S1 | 0.877x [0.846, 0.929] | 0.975x [0.934, 1.005] | 0/25 |
| none/S1 → ngram-simple/S1 | 0.920x [0.894, 0.938] | 0.932x [0.920, 0.939] | 0/25 |
| none/S2 → draft-simple/S2 | 0.942x [0.926, 1.018] | 1.061x [1.028, 1.086] | 1/25 |
| none/S2 → ngram-simple/S2 | 0.913x [0.898, 0.947] | 0.923x [0.905, 0.931] | 0/25 |

## Output and validity checks

Parsed differences are against the paired baseline, not an official quality score.
Chosen-reference equality is descriptive: datasets can allow other correct answers.
No lossless or quality non-inferiority claim follows from schema validity or this
small cohort. Equal-output-only sensitivity is post-hoc and can have selection bias.

| Configuration | Failed requests | Chosen reference, both repetitions | Draft accepted/proposed |
|---|---:|---:|---:|
| none/stock | 0 | 19/25 | n/a |
| none/S1 | 0 | 19/25 | n/a |
| none/S2 | 0 | 19/25 | n/a |
| draft-simple/S1 | 0 | 19/25 | 1752/1910 (91.7%) |
| draft-simple/S2 | 0 | 20/25 | 2104/2160 (97.4%) |
| ngram-simple/S1 | 0 | 19/25 | 142/1748 (8.1%) |
| ngram-simple/S2 | 0 | 19/25 | 154/2142 (7.2%) |

| Baseline → candidate | Equal-output pairs | Wall speedup [CI], equal outputs only | Unstable across repetitions |
|---|---:|---:|---:|
| none/S2 → none/S1 | 21 | 1.152x [1.103, 1.187] | 0 |
| none/stock → none/S1 | 21 | 1.167x [1.136, 1.194] | 0 |
| none/stock → none/S2 | 25 | 1.021x [0.995, 1.034] | 0 |
| none/S1 → draft-simple/S1 | 25 | 0.877x [0.846, 0.929] | 0 |
| none/S1 → ngram-simple/S1 | 25 | 0.920x [0.894, 0.938] | 0 |
| none/S2 → draft-simple/S2 | 24 | 0.940x [0.926, 0.998] | 0 |
| none/S2 → ngram-simple/S2 | 25 | 0.913x [0.898, 0.947] | 0 |

## Limits and decision boundary

The user retains general tool-calling scope. An application niche is not required.
This selection is an exploratory instrument screen of short single-call bodies, not
a representative test of all structured generation. Parallel calls, chat wrappers,
other models/hardware, concurrency and real cache hit rates remain untested.
Strategy processes ran in separate fixed-order time blocks; grammar order rotated and
the second repetition reversed question order. Bootstrap intervals do not remove this
time-block confounding. Prompt reuse was checked with timings.cache_n == 0.

These measurements identify baselines for an actual prototype; they do not measure
jump-forward, establish its incremental value, or authorize an engine go.
Do not multiply draft and hypothetical jump-forward speedups: these mechanisms
can accelerate overlapping tokens and their composition has not been implemented.
No more cache assumptions or favorable niches can replace that missing measurement.

## Artifacts and implementation audit

Sibling summary.json contains statistics; report-manifest.json hashes this report,
summary and raw outputs. inputs.json fixes prompts and schemas. Each strategy
directory retains exact server command/binary hash, requests/responses, token
IDs, warmup, counters, memory snapshots and logs. Completed runs are not overwritten.
The shared provenance helper's inherited offline scope label in older server metadata
does not describe this assay: exact requests and this report's explicit scope do.

At pinned engine 95ef7fc16054e63b427a3ef00188e055ef7586d8, common/sampling.cpp
routes the %llguidance prefix to Lark; other grammar strings use the native
llama_sampler_init_grammar implementation in src/llama-sampler.cpp. Thus stock is a
native GBNF baseline even though all arms share the llguidance-enabled binary.
