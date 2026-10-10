# General-scope decision and separately authorized continuation

## Completed continuation

The separately authorized [replication report](../runs/2026-09-12-replication/report.md)
finds a positive, smaller speed gain on the second local model. Its preservation screen still
fails despite valid completed outputs. The original divergences include an improvement,
a likely regression, and changes between answers that were already inadequate; they must not
be counted wholesale as demonstrated accuracy losses. This post-hoc review does not establish
quality non-inferiority.

Normal replays reproduce the original outputs. Disabling proposals reproduces the baseline;
matched physical-microbatch-one controls remove the token disagreements despite accepting
grammar drafts. Together with independent forcing-mask replay, this supports dependence on
execution shape, but not a proven logit-level cause or a guarantee of bug-free integration.

The mechanism is useful in both measured models, but its gain is not constant and its quality
impact remains the main open question. This supports targeted investigation of preservation
and task quality, not production integration or a broad fastest-method claim. This stage is
finished without engine tuning. Further work requires a new explicit decision and protocol.

## Subsequent user authorization

After the first report, Rafael judged the measured gain worth a limited follow-up and explicitly
authorized divergence diagnosis and reproduction on another model. The
[new protocol](replication-plan.md) governs that stage. The old STOP remains the outcome of its
registered experiment; it is not a universal impossibility claim or a prohibition on this new
user decision. No old metric or gate is changed. Performance and preservation are reported
separately, and superiority to every existing inference system has not been established.

The user explicitly retains general tool-calling scope. Selecting varied benchmark cases to test
the mechanism is not specializing the product. Requiring a favorable application niche would be
a change of scope and is no longer the recommended next step.

## First measured decision (2026-09-12)

Rafael's "Certo, então faça" explicitly authorized the private, minimal C++ prototype. The
[prospective protocol](prototype-plan.md) was registered before implementation and evaluation;
the [evaluation seal](../runs/2026-09-12-prototype/evaluation-seal.json) fixes the final inputs,
runtime, 32-token draft cap and common generation ceiling before held-out requests.

The [measured report](../runs/2026-09-12-prototype/cold-report.md) supports **STOP**, not further
general-scope engineering under these criteria. The experiment used 200 frozen questions across
five subsets, two repetitions per arm, 800 requests, and cold cache. Its denominator is the same
Qwen3-8B Q4_K_M and compact llguidance grammar with batching disabled, not "nothing" or a
whitespace-heavy baseline.

| Criterion | Observed result | Decision |
|---|---|---|
| Paired median end-to-end >= 1.4x | 1.358x; bootstrap 95% interval [1.331, 1.396] | Fail |
| Interval lower endpoint > 1.0 | 1.331 | Pass |
| Every subset median >= 1.0x | 1.232x to 1.554x | Pass |
| Candidate p95 <= 1.10x baseline | 0.856x; includes timed failures | Pass |
| Potential-regression rate upper bound < 2% | 4/198 changed; one-sided 95% upper 4.56% | Fail |

The gain is real: about 26.4% less paired request latency, and 1.658x decode-only speedup.
All 5,408 proposed forced tokens were accepted by the real target verifier. Both arms' observed
VRAM maximum was 5,904 MiB. Acceptance does not prove output preservation: batching can still
change subsequent unconstrained choices, and the four observed differences include arguments
and a function selection. They are potential regressions, not four proven task-accuracy errors.
Reference equality is 123/198 in both arms; this aggregate hides individual wins and losses and
does not replace the registered preservation check. No official BFCL accuracy is claimed.

Two questions hit the 2,048-token generation limit in every arm/repetition, producing eight
failed requests and 198 complete question pairs. No question failed only with batching. The
report's conservative `complete_no_failures` check also fails; this is stricter than treating
only *new* failures as regressions. The stop decision does not depend on that interpretation:
performance and output preservation independently miss their thresholds. Failures remain in
the raw data and pooled p95; their omission from complete-pair medians is explicit.

Do not round 1.358x into a 1.4x pass, substitute decode speedup, lower the quality standard, or
search for a favorable subset/cache condition after seeing the results. Warm-cache evaluation
was skipped as allowed by the protocol. This experiment is finished; retain the patch and
reproducible evidence as research artifacts, not production-ready integration. No commit or PR
was made. Reopening development would require an explicit new decision, not automatic tuning.

This is evidence about one bounded implementation on one model/GPU and single-call benchmark
bodies. It does not prove jump-forward is universally unviable. The speedup is meaningful but
insufficient to meet the agreed performance-and-preservation bar for continued investment.

## Earlier evidence and rationale (superseded next-step recommendation)

The approximate 1.4x in the cache follow-up is a **projection**, not a measured jump-forward gain.
Its denominator is the same target model with llguidance, no jump-forward and no draft, under
matched serialization and cache assumptions. It is not an unconstrained-generation denominator,
nor a demonstration of incremental value over the best existing optimization.

The [existing-baseline report](../runs/2026-09-12-general-baselines/baseline-report.md) now separates
default-schema llguidance, compact/spaced llguidance, small-model speculation and n-gram speculation.
The historical report's native-GBNF and n-gram-cap labels were wrong; read its
[metadata corrections](notes/baseline-audit-corrections.md). Raw measurements are unchanged.
Use its measured paired comparisons, output checks and limitations, not a multiplication of
unrelated projected speedups. This is an exploratory screen of the existing frozen cohort;
the fastest tested configuration is not necessarily the best possible configuration.

In this screen, compact llguidance without speculation is the most competitive tested latency
baseline. Neither fixed speculative configuration improves it. Compact and stock do not always
produce identical parsed answers, so the formatting comparison is not a lossless claim; consult
the report's equal-output sensitivity as well. None of these wins or regressions belongs to the
then-unimplemented jump-forward mechanism.

At that stage there was no implemented jump-forward arm. The original Gate 0 has not passed.
Historical reports remain intact; the separate prototype measurements above do not retroactively
turn their estimates into observations.

## Pre-experiment recommendation (now executed)

There is enough mechanism-level evidence to consider a small actual prototype, but not enough to
endorse the full engineering project. Further projection-only rounds will not answer the missing
question. The next investment, if explicitly authorized, should be a bounded experimental patch
with a go/stop decision, not production integration or a larger search for favorable workloads.

Before that patch or its confirmatory data, register:

- A fixed effort limit and an exact supported path: native forced-token batching, default off,
  without unrelated engine refactoring or a new tokenization policy.
- A **new, prospective** practical target. A measured 1.4x end-to-end median is a reasonable
  candidate given the user's stated preference; it is not an already-approved replacement for
  the original 1.5x gate. It must be against the strongest applicable tested baseline, including
  compact JSON, and with matched cache state. Also compare the identical grammar with batching off
  to isolate the mechanism. Do not credit whitespace removal to jump-forward.
- A new frozen evaluation selection spanning general tool-call structure and lengths, disjoint
  from prototype development cases. Retain all failures. Do not cherry-pick long forced spans.
- A quality margin and sample-size plan before seeing evaluation results. Compare parsed
  arguments, token/byte divergence and independent validity, not just formatting success.
- End-to-end and decode latency, uncertainty, per-subset results, tail latency and memory.
  Separate cold and warm-cache results without inventing a production cache-hit mixture.

Stop if the fixed budget cannot produce a valid instrument, if the measured gain fails the new
pre-registered target, or if quality/correctness does not meet its pre-registered criterion. A win
confined to a favorable subset does not establish the retained general-scope claim. Do not rescue a
failure by changing the denominator, lowering the target after results or requiring a niche.

The recommendation itself did not authorize C++. Rafael subsequently gave that explicit go;
the session exception is recorded in CLAUDE.md. It covered only this bounded experiment, not
production integration, unrestricted tuning, commits, or upstream submission.
