# Grammar Jump-Forward — Claude Code Operating Manual

## Session exception — 2026-09-12

Rafael explicitly authorized the assistant to create the Phase 0 Python measurement scripts and
their validation tests in the project-review conversation. This exception covers the feasibility
investigation and its reports. It does not authorize engine C++ changes before Gate 0.

## Ground rule: Claude does not write code in this project

Rafael writes every line of code by hand, in this repo and in the llama.cpp worktree. The project is
a learning vehicle as much as a research artifact.

- **Never** create or modify code: `.py`, `.cpp`, `.h`, `CMakeLists.txt`, shell scripts, notebooks.
  This applies to Write/Edit/NotebookEdit **and** to Bash (no heredocs, redirection into source
  files, `sed -i`, `git apply`, or `git commit` of code in either repo).
- **Instead**, put proposed code in the chat reply: complete snippets, each headed with its target
  path, followed by what it does, why, and how to check it worked.
- Configuration, environment and docs (`pyproject.toml`, `mise.toml`, `config/*.json`,
  `data/raw/MANIFEST.json`, `.gitignore`, `docs/`, this file) may be changed **only when Rafael asks**.
- **Allowed without asking:** reading anything; read-only inspection (`ls`, `git log`, `--help`,
  `nvidia-smi`, grepping llama.cpp or llguidance source). **Ask before** any long GPU job, because it
  blocks the only GPU.
- When Rafael shares code, review it against "Methodology". Bugs that silently bias a measurement
  come first, then correctness at token boundaries, then style.

## The project

In grammar-constrained JSON tool calls, most tokens are forced by the grammar, yet llama.cpp spends a
full forward pass on each. The proposal is to detect forced spans, choose a tokenization, and inject
them as one batch through the speculative framework (`--spec-type grammar`, a zero-inference
proposer like `ngram-*`).

- Research questions: RQ1, does forced tokenization hurt the free values that follow? RQ2, how does
  that vary by policy and tokenizer? RQ3, what fraction of tool-call tokens are forced? RQ4, does
  jump-forward add to draft-model speculation or compete with it?
- Proposal, frozen: `docs/proposal.md`.
- Why this idea over draft adaptation: `docs/decisions/0001-switch-to-grammar-jump-forward.md`.

## Current phase: Phase 0 (no engine C++ yet)

Follow `docs/phase0-plan.md`, which is pre-registered; amend it only in its "Amendments" section:
- P0.1: llguidance ff-token semantics
- P0.2: prior-art check
- P0.3: Gate 0 forced fraction (≥ 25% median)
- P0.4: pass-cost curve
- P0.5: RQ1 emulation design
- P0.6: early RQ4 signal

Engine C++ starts only after the Gate 0 report says go.

## Verified engine facts at base commit `95ef7fc16` (2026-09-10)

- **No jump-forward exists anywhere.**
  - `common/llguidance.cpp` calls only `llg_matcher_compute_mask`, `get_mask`, `consume_token`,
    `reset` and `clone`.
  - The reasoning budget forces tokens by masking logits, one forward pass per token
    (`common/reasoning-budget.cpp:178`). That makes it a second consumer of a jump-forward path.
- **llguidance v1.0.1 C API** (`~/llama.cpp-jf/build/llguidance/source/parser/llguidance.h`):
  - `llg_matcher_compute_ff_tokens` (forced tokens)
  - `llg_matcher_consume_tokens` (multiple tokens)
  - `llg_matcher_validate_tokens` (how many of a draft are grammar-valid)
  - `llg_matcher_rollback` (undo rejected drafts)
  - Backtracking is always disabled in the matcher.
  - The tokenization policy behind ff tokens is **unverified** (P0.1).
- **llguidance 1.0.1 Python** (`LLMatcher`) exposes `compute_ff_tokens`, `compute_ff_bytes`,
  `validate_tokens`, `try_consume_tokens` and `rollback`. The module also has `gbnf_to_lark`, which
  loads llama.cpp GBNF grammars offline.
- **Stock JSON grammar has flexible whitespace:**
  `common/json-schema-to-grammar.cpp:227`
  `SPACE_RULE = | " " | "\n"{1,2} [ \t]{0,20}`.
  Required properties are emitted in a fixed order (`:752`); optional properties create branches.
- **Tool-call grammars are lazy** and start on a trigger: `grammar_lazy`, `grammar_triggers`
  (`common/chat.h:273`, `tools/server/server-common.cpp:1346`).
- **Speculative types:** `none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark,
  ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache`.
  - There is no grammar type.
  - When draft and draftless types are combined, **draftless takes precedence**
    (`docs/speculative.md`), so composing a grammar proposer with a draft model is a design problem
    (Issue #23184).
- **Speculative flags** were renamed: `-md`, `--spec-type`, `--spec-draft-n-max` (default 3),
  `--spec-draft-p-min`, `-ngld`. The response `timings` include `draft_n` and `draft_n_accepted`.
- **Qwen3 thinking:** `--reasoning off`. Verify outputs have no think block.

## Where things live

```
config/llama_cpp.json    base commit, worktree, llguidance version, exact build flags
config/models.json       local GGUFs (sha256), tokenizer dirs
data/raw/MANIFEST.json   pinned datasets (BFCL, Hermes FC) and tokenizers (Qwen3, Llama 3.1, Mistral v0.3, Gemma 3)
docs/proposal.md         frozen proposal
docs/phase0-plan.md      pre-registered Phase 0 and Gate 0
docs/environment.md      how to rebuild everything
docs/decisions/          decision records (ADR style, numbered)
docs/archive/            shelved draft-adaptation idea and its verified facts
src/jumpforward/         Python package          (Rafael writes)
scripts/                 numbered pipeline steps (Rafael writes)
tests/                   pytest; markers gpu, server, engine
runs/                    versioned run manifests
patches/<milestone>/     git format-patch exports of the llama.cpp branch
~/llama.cpp-jf           llama.cpp worktree, branch grammar-jump-forward; builds: build/ (Release, CUDA sm_75, llguidance), build-debug/ (Debug, CPU, llguidance)
~/llama.cpp              untouched reference at the same commit, shared with other projects; no-patch baseline
```

Toolchain: `mise.toml` pins rust 1.98.1, ninja 1.13.2, ccache 4.14. Run llama.cpp builds under
`mise exec`, because llguidance needs cargo. Python 3.12 via uv; `llguidance==1.0.1` matches the engine.
Hardware: RTX 2080 Ti 11 GB (`sm_75`), i9-9900K, 23 GB RAM.

## Python vs C++

- **Python:**
  - Gate 0 and RQ2/RQ3 analysis (llguidance Python + HF tokenizers)
  - RQ1 emulation through llama-server token-id prompts
  - benchmark orchestration, statistics, reports
- **C++:** the engine patch in `~/llama.cpp-jf`, only after Gate 0 passes. Keep it small, behind a
  flag, default off, and inside the speculative framework, with no parallel decode path.
- **Tokenizer parity:** HF tokenizers and llama.cpp's GGUF tokenizer can disagree. Check with
  llama-server `/tokenize` before trusting offline numbers for a tokenizer.

## Methodology

- **Temperature 0 everywhere.** Thinking off.
- **Every run writes a manifest:** base commit and branch HEAD (and `git diff` hash if dirty),
  grammar backend, grammar text hash, model sha256, tokenizer revision, full command line, seed.
  No number in a report is typed by hand.
- **Validity is checked independently.** Use JSON Schema validation plus byte-level UTF-8 checks.
  Never trust the engine's own grammar to certify itself; a known soundness hole exists for mixed-length
  UTF-8 character classes.
- **Jump-forward is not lossless the way draft speculation is.** A forced canonical tokenization
  can change the model's later output. Count divergences from the no-jump baseline and classify them
  as (a) the same bytes with different tokens, or (b) different bytes. Class (b) is RQ1.
- **Report decode-only and end-to-end times** (prefill included), plus forward passes per tool call.
  A CPU-only configuration is a secondary report.
- **Statistics:** pair by tool call. Bootstrap by call for timing deltas; McNemar for per-call
  argument exact match.
- **Every threshold** in `docs/phase0-plan.md` and the proposal's gates is fixed before its data exists.

## Voice

Direct and concrete. Name files, line numbers, flags and numbers. Say plainly when something is
unverified. Recommend instead of listing every option.
