# Grammar Jump-Forward Decoding for llama.cpp

When llama.cpp generates a JSON tool call under a grammar, some output tokens are already fixed
by the grammar but still processed one at a time. This project measures how much latency could be
saved by processing native llguidance forced spans in batches. Injected tokens still require model
and KV-cache computation; batching and changing the tokenization policy are separate interventions.

Proposal: [`docs/proposal.md`](docs/proposal.md). Phase 0 plan and Gate 0 definition:
[`docs/phase0-plan.md`](docs/phase0-plan.md).

**Status: STOP for further general-scope investment under the registered experiment.**
The authorized default-off C++ prototype was measured on 200 frozen questions / 800 requests.
Its paired end-to-end speedup is **1.358x [1.331, 1.396]**, against identical compact llguidance
without batching, below the prospective 1.4x target. Decode alone improves 1.658x. Four of 198
complete questions change parsed output; the preservation criterion also fails. The same two
questions hit the generation limit in both arms, not a new batching-only failure.
Measured [prototype report](runs/2026-09-12-prototype/cold-report.md),
[decision](docs/general-scope-decision.md) and [reproduction](docs/prototype-reproduction.md).
No warm-cache rescue or niche retargeting was run. This is a bounded result on one model/GPU,
not proof that every future implementation must fail. The original Gate 0 remains unpassed.

The earlier approximate 1.4x was only a projection. The prior
[existing-baseline screen](runs/2026-09-12-general-baselines/baseline-report.md) compares stock
JSON Schema, compact/spaced llguidance, a small draft model and n-gram speculation.
Read the [baseline metadata corrections](docs/notes/baseline-audit-corrections.md) with that report:
its stock request used default-schema llguidance, not native GBNF, and its n-gram cap label was wrong.
Reproduction commands and limitations: [`docs/phase0-reproduction.md`](docs/phase0-reproduction.md).

Historical evidence: [`docs/gate0-report.md`](docs/gate0-report.md) and
[`docs/retarget-report.md`](docs/retarget-report.md). Their results and original gate remain unchanged;
their recommendation to require an application niche is superseded by the user's general-scope
direction. The new measured decision closes this bounded general-scope feasibility round.

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
