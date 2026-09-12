# Grammar Jump-Forward Decoding for llama.cpp

When llama.cpp generates a JSON tool call under a grammar, some output tokens are already fixed
by the grammar but still processed one at a time. This project measures how much latency could be
saved by processing native llguidance forced spans in batches. Injected tokens still require model
and KV-cache computation; batching and changing the tokenization policy are separate interventions.

Proposal: [`docs/proposal.md`](docs/proposal.md). Phase 0 plan and Gate 0 definition:
[`docs/phase0-plan.md`](docs/phase0-plan.md).

**Status:** Phase 0 investigation completed with a **retarget** recommendation, not an engine go.
The reference corpus, native tokenizer parity, GPU cost curve and a small generated-output check
support a narrower engineering study; they do not establish the original end-to-end speedup claim.
Results and next decision: [`docs/gate0-report.md`](docs/gate0-report.md).
Reproduction commands and limitations: [`docs/phase0-reproduction.md`](docs/phase0-reproduction.md).
Follow-up with measured cache-prefix costs and output-length sensitivity:
[`docs/retarget-report.md`](docs/retarget-report.md). Cache helps, but does not yet justify the
original performance target against compact JSON; an application workload is still needed.

## Layout

| Path | Role |
|---|---|
| `config/` | Frozen decisions: engine base commit and build flags, local models |
| `data/raw/` | Pinned datasets and tokenizers. Only `MANIFEST.json` is tracked |
| `docs/` | Proposal, Phase 0 plan, environment, decision records, archive |
| `src/jumpforward/` | Python package: analysis, llama-server driver, statistics |
| `scripts/` | Numbered pipeline steps |
| `tests/` | pytest; markers `gpu`, `server`, `engine` |
| `runs/` | Versioned run manifests and small outputs |
| `patches/` | `git format-patch` exports of the llama.cpp branch at milestones |
| `~/llama.cpp-jf` | llama.cpp worktree, branch `grammar-jump-forward`, where the C++ work happens |

## Setup

Full details and verification commands: [`docs/environment.md`](docs/environment.md).

```bash
mise install        # rust 1.98.1, ninja 1.13.2, ccache 4.14 (mise.toml)
uv sync             # Python 3.12 environment
uv run prek install # git hooks
```

The llama.cpp worktree, builds and data downloads are documented in `docs/environment.md`, pinned in
`config/llama_cpp.json` and `data/raw/MANIFEST.json`.
