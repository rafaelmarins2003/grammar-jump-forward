# Grammar Jump-Forward Decoding for llama.cpp

A prototype that batches grammar-forced tokens in llama.cpp.
The goal is to reduce tool-call latency without an auxiliary model or a loss in response quality.

This repository contains the Python experiments, results, and patches. The C++ implementation
lives in a separate worktree at `~/scripts/llama.cpp-jf`.
Published source: [llama.cpp-jf, branch grammar-jump-forward](https://github.com/rafaelmarins2003/llama.cpp-jf/tree/grammar-jump-forward).

## Implementation

llama.cpp already integrates [llguidance](https://github.com/ggml-org/llama.cpp/blob/master/docs/llguidance.md)
as an optional grammar backend, enabled with `LLAMA_LLGUIDANCE=ON`. During conventional generation,
the model computes logits and llguidance restricts the tokens allowed at the next position.
This format constraint is constrained decoding.

llguidance also identifies sequences of forced tokens. At the base commit used in these tests,
llama.cpp did not use that information to batch their generation. The patch adds a
`--spec-type grammar` proposer: it retrieves the sequence and passes it to the existing
speculative decoding framework for target-model verification and processing.

Drafting requires no auxiliary-model inference. The tokens still pass through the target model
and update the KV cache; the savings come from reducing sequential steps.
When there are no forced tokens, normal generation continues.

Jump-forward is an existing technique. This work integrates it into llama.cpp and evaluates
the result. The prototype uses llguidance's forced-token API without introducing a different
tokenization policy. It does not implement jump-forward for the native GBNF backend.
GBNF is the rule format used by llama.cpp's own grammar engine.

## Code

The repositories sit side by side on the development machine:

```text
~/scripts/
├── grammar-jump-forward/   # experiments, results, and patches
├── llama.cpp/             # reference repository, shared with other projects
└── llama.cpp-jf/           # worktree containing the C++ changes
```

The worktree is on the `grammar-jump-forward` branch, based on commit
`95ef7fc16054e63b427a3ef00188e055ef7586d8`. The scripts read its location from the
`worktree` field in [`config/llama_cpp.json`](config/llama_cpp.json).
The measured C++ changes are recorded in [commit b949b34f1](https://github.com/rafaelmarins2003/llama.cpp-jf/commit/b949b34f1).

Key files changed in the engine:

| File in `llama.cpp-jf` | Role |
|---|---|
| `common/llguidance.cpp` | Retrieves and validates forced tokens |
| `common/sampling.cpp` | Exposes the query through the active sampler |
| `common/speculative.cpp` | Implements the `grammar` proposer |
| `tools/server/server-context.cpp` | Connects the slot sampler to the proposer |
| `tests/test-grammar-llguidance.cpp` | Tests buffers, cloning, reset, and forced tokens |

Python runs the experiments; it does not replace the engine:

- [spans.py](src/jumpforward/spans.py): offline forced-span analysis with llguidance.
- [parity.py](src/jumpforward/parity.py): starts and controls the C++ `llama-server`.
- [prototype.py](src/jumpforward/prototype.py): primary evaluation, validation, and statistics.
- [replication.py](src/jumpforward/replication.py): replication and divergence diagnosis.

Entry points are in `scripts/`. Offline analyses produce estimates; the speedups below were
measured by running the C++ server. The [development guide](docs/development-guide.md) covers
the reading order and builds.

## Evaluation

The primary test uses the same model, runtime, and compact llguidance grammar in both arms:

- Baseline: `--spec-type none`.
- Prototype: `--spec-type grammar --spec-draft-n-max 32`.

Both arms include the llguidance safety fix. The comparison measures the proposer's effect,
not a change of backend.

Each evaluation used 200 questions and 800 requests on an RTX 2080 Ti, with cold cache,
temperature zero, one slot, and a single tool call per response.

| Q4_K_M model | End-to-end speedup [95% CI] | Decode speedup | Parsed-output changes |
|---|---|---|---|
| Qwen3-8B | 1.358x [1.331, 1.396] | 1.658x | 4/198 complete pairs |
| Qwen3-4B | 1.122x [1.101, 1.144] | 1.243x | 5/200 complete pairs |

Speedups are medians of per-question paired ratios. In the 8B test, the same two questions
hit the generation limit in both arms. All questions completed in the 4B test, which reused
the first evaluation's questions and used the same tokenizer family.

The speed gain was reproduced, but both models failed the output-preservation criterion.
A changed response is not necessarily an accuracy loss, and valid JSON does not guarantee
a correct answer. Diagnostics suggest sensitivity to batching execution shape without
establishing the exact numerical cause. The prototype remains experimental and disabled
by default.

The first experiment ended in STOP: it missed the registered 1.4x target and the preservation
criterion. Replication used the same patch without additional optimization.

Reports: [prototype](runs/2026-09-12-prototype/cold-report.md),
[replication](runs/2026-09-12-replication/report.md), and
[criteria and decision](docs/general-scope-decision.md).

### Comparison with alternatives

The [preliminary evaluation](runs/2026-09-12-general-baselines/baseline-report.md) tested a draft
model and n-gram speculation on a smaller selection. The tested configurations did not improve
on the compact baseline, but there was no comprehensive search for competitive configurations.

The test labeled `stock` used llguidance, not native GBNF. The n-gram cap label was also wrong;
the [corrections](docs/notes/baseline-audit-corrections.md) record what actually ran.
Native GBNF received functional tests, not a full performance comparison.

A comparison against native GBNF and other applicable llama.cpp strategies under equivalent
output requirements is still needed. Current results demonstrate a gain over the defined
baseline, not superiority over the best available method.

## Patches and distribution

The measured engine changes are preserved as a base commit and patches:

- [C++ prototype](patches/2026-09-12-grammar-prototype.patch).
- [Rust fix for llguidance's fast-forward buffer](patches/2026-09-12-llguidance-ffi-bound.patch).

To clone the published fork, run from this project's root.
The directory `../llama.cpp-jf` must not already exist; do not run this over the existing worktree.

```bash
git clone --branch grammar-jump-forward --single-branch https://github.com/rafaelmarins2003/llama.cpp-jf.git ../llama.cpp-jf
```

The engine changes are already committed in this branch; do not apply the C++ patch again.
Before running the proposer, apply the safety patch to each build's llguidance checkout and
rebuild the library. CMake does not apply this fix automatically. The fork includes the patch
and fresh-build commands in [README.jump-forward.md](https://github.com/rafaelmarins2003/llama.cpp-jf/blob/grammar-jump-forward/README.jump-forward.md).
Historical reproduction details are in [prototype reproduction](docs/prototype-reproduction.md).
Do not reapply the dependency patch to already corrected checkouts.

The patches capture the measured version; they do not automatically track later worktree
edits. The personal fork publishes the same C++ implementation, with separate documentation
and a copy of the required dependency patch. No upstream PR, issue, or message was submitted.
Builds, models, and binaries were not uploaded. Historical evaluation seals remain unchanged;
new commits or builds require a new run rather than resuming a sealed evaluation.

## Environment and documents

Set up the Python environment and tools:

```bash
mise install
uv sync
uv run prek install
```

Models, datasets, and binaries are not versioned here. Revisions and hashes are recorded
in manifests. Results are in `runs/`, configuration in `config/`, and Python tests in
`tests/`; C++ tests live in the engine worktree.

- [Environment and builds](docs/environment.md).
- [Data and tokenizers](data/raw/MANIFEST.json).
- [Initial-phase reproduction](docs/phase0-reproduction.md).
- [Replication protocol](docs/replication-plan.md) and [reproduction](docs/replication-reproduction.md).
- [Original proposal](docs/proposal.md) and [Gate 0 plan](docs/phase0-plan.md).
- [Gate 0 results](docs/gate0-report.md) and [retargeting](docs/retarget-report.md).

The proposal and protocols preserve the original hypotheses and criteria. Not everything
proposed was implemented. The original Gate 0 did not pass; subsequent development was
recorded separately. The scope remains general tool calling, without specialization
for a particular application.
