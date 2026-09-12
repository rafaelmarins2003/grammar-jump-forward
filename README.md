# Grammar Jump-Forward Decoding for llama.cpp

When llama.cpp generates a JSON tool call under a grammar, most output tokens (braces, quotes,
keys, colons) are fixed by the grammar before the model runs, yet each one still costs a full
forward pass. This project measures how much of that work can be skipped by injecting grammar-forced
spans as one batch through llama.cpp's speculative decoding framework, and whether forcing a
tokenization degrades the values the model writes next.

Proposal: [`docs/proposal.md`](docs/proposal.md). Phase 0 plan and Gate 0 definition:
[`docs/phase0-plan.md`](docs/phase0-plan.md).

**Status:** Phase 0. Source/prior-art audit and synthetic GPU pass-cost curve completed;
forced-token fractions and tool-call evaluation remain pending. No project code yet.
Preliminary evidence and outstanding work: [`docs/gate0-report.md`](docs/gate0-report.md).

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
