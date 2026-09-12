# Gate 0 — feasibility decision

Decision: **retarget**. The original end-to-end performance claim is not established.
No engine C++ implementation is green-lit by this report.

Continue only as a narrower engineering investigation of native llguidance batching,
with cached/short prompts or longer structured outputs. Do not proceed on the assumption
that ordinary tool calls will generally achieve the proposed measured-latency target.

## Evidence and coverage

The frozen pilot selected 200 questions. The expanded run enumerated 4,004 questions in the six selected subsets, with 3,095 complete questions per variant (77.3% coverage).
The adapter retained 905 failures; a further 4 questions failed replay.
No failures were replaced. These results describe the supported cohort, not every BFCL
category or all of Hermes. Adapter failures can reflect unsupported reference encodings,
not necessarily a bad original benchmark answer.

| Subset | Selected | Complete | G1/S2 median forced | G2/S2 median forced |
|---|---:|---:|---:|---:|
| simple | 400 | 395 | 28.6% | 60.0% |
| multiple | 200 | 198 | 25.7% | 55.3% |
| parallel | 200 | 199 | 28.2% | 58.1% |
| live_simple | 258 | 238 | 25.9% | 55.6% |
| live_multiple | 1,053 | 1,028 | 23.1% | 47.1% |
| hermes | 1,893 | 1,037 | 23.3% | 48.6% |

Every grammar includes all tools supplied with the question. Only `{name, arguments}`
bodies are counted: wrappers, triggers, choice of number of calls, transport and actual
prompt lengths are excluded. Parallel bodies are combined into one question before
computing distributions; prefill is charged once per question.

## Forced spans

G1 is llguidance JSON Schema with flexible whitespace; G2 fixes whitespace to match the
serialization. S1 is compact JSON; S2 inserts one space after commas and colons.

| Grammar / serialization | Forced p10 | Median | p90 | Median injected bytes | Pooled span median, tokens |
|---|---:|---:|---:|---:|---:|
| G1/S1 | 20.0% | 30.6% | 42.9% | 44.1% | 1 |
| G1/S2 | 16.3% | 24.4% | 33.7% | 41.4% | 1 |
| G2/S1 | 28.1% | 42.9% | 61.1% | 54.0% | 2 |
| G2/S2 | 34.8% | 51.4% | 69.2% | 59.3% | 4 |

Fractions are question-weighted; span lengths pool individual spans and therefore
are span-weighted. Spans never overlap, and every injected token was checked against a
separate sequential singleton mask. JSON/schema and UTF-8 byte roundtrip were checked
independently. Schema `format` is annotation-only in the initial jsonschema validation;
the llguidance replay can enforce more and reject such a reference.

The G2/S2 forced-fraction median exceeds the registered 25% floor. Even assigning every missing question a fraction of zero or one puts the selection-wide median between 46.2% and 56.5%. This sensitivity check is not an estimate for the missing data.
The fraction floor alone is not a performance gate.

## Cost projections, not observed server speedups

| Variant | Context | Decode, conservative–optimistic | Cold prompt, conservative–optimistic | Zero-forced-cost cold ceiling |
|---|---:|---:|---:|---:|
| G1/S1 | 1024 | 1.35×–1.40× | 1.14×–1.16× | 1.17× |
| G1/S1 | 4096 | 1.36×–1.41× | 1.05×–1.06× | 1.06× |
| G1/S2 | 1024 | 1.26×–1.30× | 1.13×–1.14× | 1.15× |
| G1/S2 | 4096 | 1.27×–1.30× | 1.05×–1.06× | 1.06× |
| G2/S1 | 1024 | 1.49×–1.56× | 1.19×–1.21× | 1.26× |
| G2/S1 | 4096 | 1.51×–1.57× | 1.07×–1.08× | 1.09× |
| G2/S2 | 1024 | 1.69×–1.82× | 1.28×–1.32× | 1.38× |
| G2/S2 | 4096 | 1.72×–1.83× | 1.11×–1.12× | 1.13× |

Each row compares batching with token-by-token processing of the **same grammar and
serialization**. The columns are medians of paired per-question ratios, not ratios of
median latencies or confidence intervals. Optimistic/conservative labels denote sensitivity
proxies, not guaranteed bounds. The zero-cost ceiling eliminates forced-token model work
only inside this fixed-prefill model; it is not a bound on an arbitrary deployed server.

The cost curve is an actual synthetic GPU measurement on the pinned Qwen3 GGUF and RTX
2080 Ti. Batches process every body token, including the final one for termination logits,
with a cap of 32 tokens. Initial forced prefixes are charged. The conservative proxy adds
measured FF extraction and bulk consumption, but neither arm models full server overhead,
ordinary mask cost, grammar compilation, changing context depth or cache-management cost.
Its monotone envelope handles the observed non-monotone batch-cost curve.

G2/S2 median FF extraction was 0.451 ms per question; median grammar compilation, excluded from both proxies, was 1.227 ms. Cold prefill means were 392.328 ms for 1024 tokens, 1674.576 ms for 4096 tokens.
Decode-only projections do not include cache lookup or other cached-request overhead.

### Serialization sanity check

S2 contains additional space tokens, so a larger forced fraction or within-S2 speedup
need not mean the fastest implementation. The following **post-hoc sensitivity** uses
compact G1/S1 as the common unbatched reference, with the conservative batch proxy:

| Candidate | Context | Decode vs compact baseline | Cold vs compact baseline |
|---|---:|---:|---:|
| G2/S1 | 1024 | 1.49× | 1.19× |
| G2/S1 | 4096 | 1.51× | 1.07× |
| G2/S2 | 1024 | 1.36× | 1.15× |
| G2/S2 | 4096 | 1.38× | 1.06× |

This is not a new success threshold. Optimize measured absolute latency, not forced
fraction by adding predictable whitespace. No tested cold-prompt median proxy reaches
the original measured-latency target.

## Native tokenizer and generated-output checks

Tokenizer parity: 200 call records, 400 serializations checked against the unmodified local llama-server `/tokenize`; 0 ID mismatches. This checks those body strings, not equality of all
GGUF/HF vocabulary entries, special tokens or arbitrary chat prefixes.

The preregistered exploratory generation check ran 25 single-call questions in both whitespace arms (50 outputs), temperature zero, thinking off, no draft and no prompt reuse. Neither arm had jump-forward implemented.
All supplied candidate tools remained available; no tool was actually executed.

| Arm | Valid JSON/schema | Raw-token replays | Median forced | Chosen-reference agreement | Noncanonical whole-text tokenizations |
|---|---:|---:|---:|---:|---:|
| G1 | 25/25 | 25/25 | 22.7% | 19/25 | 1 |
| G2 | 25/25 | 25/25 | 52.2% | 19/25 | 1 |

There were 0 parsed-output differences among 25 valid pairs. This small selected cohort cannot establish non-inferiority. Chosen-reference agreement is not official BFCL accuracy: alternative
valid argument realizations and task semantics require a separate evaluator. Raw generated
IDs, including noncanonical segmentations, were replayed unchanged; only a recorded
trailing EOS would be removed. Retokenizing the final text would measure a different trace.

## Replay exclusions and compatibility limits

| Question ID | Failed call bodies in G2/S2 | First diagnostic |
|---|---:|---|
| e932bfbe-a706-472b-8612-f36fdf5b2107 | 1 | Unknown format |
| f31ce443-73a1-470e-8318-a6d83f88ca3a | 1 | reference_token_not_allowed |
| cb24a2b8-69e0-4d6f-8afd-8fe69aaff4f0 | 4 | ff_reference_mismatch |
| ebeaf63c-750e-45bf-99bb-7c01d0b3ea96 | 3 | ff_reference_mismatch |

The `uri` format was not supported by this llguidance compiler. Two date references
contained year-month strings rather than complete dates; a date-time reference omitted
the timezone required by the grammar. These failed replay rather than being repaired.
An FF/reference mismatch is not automatically a tokenizer bug. Detailed IDs, token IDs,
grammar text and errors remain in the run artifacts. Missing/invalid Hermes JSON and
unsupported BFCL answer encodings account for much of the earlier adaptation loss.

G3 (native llama.cpp GBNF) was **not measured**: equivalence of an offline conversion to
the native runtime was not validated. G1 must not be presented as G3. The synthetic lazy
lexeme test passes, but actual chat-template trigger/cardinality integration remains open.
Other tokenizers, alternative forcing policies, draft latency/memory comparisons, CPU
inference and a confirmatory quality study remain deferred under the PL-first amendment.

## What to do next

1. Keep the project as an engineering/learning study; narrow its claim to batching tokens
   already forced by llguidance. Novel tokenization research is not established by this work.
2. Before engine implementation, measure real prompt lengths, prompt-cache hit rates and
   output lengths for the intended application. Recompute absolute latency with those
   distributions and compare compact serialization. Drop the original broad performance
   claim if the selected application also has no plausible path to its target.
3. If that retargeted workload passes, preregister a paired quality margin and sample size,
   compare G1/G2 under the same native token policy, and validate G3/real tool wrappers.
   A policy study must compare PL with `no_forcing` under the same grammar, not confuse
   existing singleton forcing with a new batching effect.
4. Only after an explicit revised go decision, implement a small default-off PL batching
   path and measure actual paired request latency, KV correctness and output divergence.
   Draft composition is a later comparison of latency and memory, not acceptance alone.

## Reproducibility and prior art

Generated by `scripts/04_gate0_report.py`; all reported measurements come from:

- [Full reference summary](../runs/2026-09-12-full/summary.json),
  [selection/provenance](../runs/2026-09-12-full/prepare-manifest.json) and
  [measurement manifest](../runs/2026-09-12-full/measure-manifest.json).
- [Pilot summary](../runs/2026-09-12-pilot/summary.json).
- [Synthetic GPU cost curve](../runs/2026-09-12-pass-cost/summary.json) and its
  [exact commands/build/model hashes](../runs/2026-09-12-pass-cost/manifest.json).
- [Tokenizer parity](../runs/2026-09-12-parity/summary.json) and
  [generated-output check](../runs/2026-09-12-generated/summary.json).
- [Report input hashes](../runs/2026-09-12-decision/manifest.json).

Raw bodies, tokens, spans and individual errors are retained locally and in lossless
gzip archives where large. Inspect the audit trail before generalizing these findings.
See the [native-policy audit](notes/llguidance-ff-tokens.md) and
[prior-art review](notes/prior-art.md). DOMINO, SGLang and XGrammar already study relevant
tokenization/jump-forward mechanisms. The potential contribution is their carefully
controlled evaluation and integration in this pinned consumer-hardware llama.cpp setting,
not the invention of jump-forward or proof of a novel quality effect.

Final decision: **retarget**, not an original-scope Gate 0 pass.
