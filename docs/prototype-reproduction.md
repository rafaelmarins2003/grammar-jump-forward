# Reproducing the bounded prototype

This is a private experimental patch, not a production integration or upstream submission.
The [prospective protocol](prototype-plan.md) fixes the decision criteria. The earlier baseline
report must be read with its [metadata corrections](notes/baseline-audit-corrections.md).

The completed [cold evaluation](../runs/2026-09-12-prototype/cold-report.md) returned STOP:
1.358x paired end-to-end speedup and four parsed-output changes among 198 complete questions.
The same two questions reached the generation ceiling in both arms. No warm run followed.
The [decision record](general-scope-decision.md) explains why this ends the bounded experiment.

## Runtime and safety prerequisite

The active original worktree is `~/scripts/llama.cpp-jf`; Python launchers resolve it from
`config/llama_cpp.json`. Its engine patch, fixed dependencies and original binaries are already
present. The patch-application steps below are for fresh checkouts, not this existing worktree.
Historical protocols and run manifests retain the paths used during their evaluations.

The engine diff is `patches/2026-09-12-grammar-prototype.patch`, against the pinned engine commit
in `config/llama_cpp.json`. It adds a default-off grammar proposer to the existing speculative
framework. It does not skip target verification, create another decode loop, or supply synthetic
acceptance. The main comparison is identical compact llguidance with `--spec-type none` versus
`--spec-type grammar --spec-draft-n-max 32`.

**Do not use this engine patch with the original llguidance 1.0.1 C fast-forward function.**
That function copied the full token vector even when the caller supplied a smaller buffer.
Apply `patches/2026-09-12-llguidance-ffi-bound.patch` to each configured build's llguidance source
before rebuilding its Rust library. The C++ regression test checks a canary beyond the declared
buffer capacity. A fresh dependency checkout requires reapplying this fix; the patch is not
automatically installed by CMake. Both measured arms use the same corrected library.

On a fresh private checkout at the pinned commits, use `git apply --check` before applying each
diff. Do not reapply patches to the already modified working copy. The unmodified runtime used
for the development off-switch check was preserved in `~/scripts/llama.cpp-jf/build-pre-grammar/bin`,
including its shared libraries, before any source change. The launcher explicitly selects those
libraries with `LD_LIBRARY_PATH`; copying the executable alone would not preserve the runtime.

From this project's root, after applying the diffs to the configured engine/dependency checkouts:

```bash
mise exec -- cargo build --release --package llguidance \
  --manifest-path /home/rafaelmarins2003/scripts/llama.cpp-jf/build/llguidance/source/Cargo.toml
mise exec -- cargo build --release --package llguidance \
  --manifest-path /home/rafaelmarins2003/scripts/llama.cpp-jf/build-debug/llguidance/source/Cargo.toml
mise exec -- cmake --build /home/rafaelmarins2003/scripts/llama.cpp-jf/build \
  --target llama-server test-grammar-llguidance test-sampling -j 4
mise exec -- cmake --build /home/rafaelmarins2003/scripts/llama.cpp-jf/build-debug \
  --target test-grammar-llguidance test-sampling -j 4
ctest --test-dir /home/rafaelmarins2003/scripts/llama.cpp-jf/build-debug \
  -R '^(test-grammar-llguidance|test-sampling)$' --output-on-failure
ctest --test-dir /home/rafaelmarins2003/scripts/llama.cpp-jf/build \
  -R '^(test-grammar-llguidance|test-sampling)$' --output-on-failure
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The published engine/dependency diffs are uncommitted patch exports. No commit, push or PR was
created. Existing user changes in the Python project and the separate reference engine were
preserved. The prototype remains off by default.

## Data, development checks and evaluation

The hf-cli skill was used only to verify the existing Qwen tokenizer at its pinned revision.
No new model was downloaded. All requests run locally and none of the described dataset tools
are executed. The user authorized unloading the separate 27B model from their model router;
the router itself was not stopped and no model files were removed.

The launcher refuses to start GPU work while another compute process is present. Run evaluation
alone; do not start another model while it is in progress. A parent-death signal terminates only
the owned child server if the controller is interrupted.

```bash
.venv/bin/python scripts/08_prototype.py prepare --run runs/my-prototype
.venv/bin/python scripts/08_prototype.py smoke --run runs/my-prototype-smoke
.venv/bin/python scripts/08_prototype.py seal --run runs/my-prototype \
  --smoke-run runs/my-prototype-smoke
.venv/bin/python scripts/08_prototype.py collect --block 0 --run runs/my-prototype
.venv/bin/python scripts/08_prototype.py collect --block 1 --run runs/my-prototype
.venv/bin/python scripts/08_prototype.py collect --block 2 --run runs/my-prototype
.venv/bin/python scripts/08_prototype.py collect --block 3 --run runs/my-prototype
.venv/bin/python scripts/08_prototype.py report --run runs/my-prototype
```

`prepare` reproduces the registered selection, not a fresh unseen confirmatory cohort. It retains
every selected ID, including the long reference that motivated the pre-evaluation ceiling amendment.
`seal` requires passed development checks and fixes the engine binary, shared libraries,
dependency diff, input and protocol hashes before held-out generation. `collect` refuses a changed
runtime or protocol. The output limit is shared; the draft cap and performance target are unchanged.

For an interrupted block, use the same command with `--resume`; completed flushed trials, including
errors, are not repeated. A continuation gets its own server manifest. Completed blocks and reports
are immutable. Do not alter or tune the engine after evaluation starts. The development smoke run
that exposed the misidentified native-backend assumption was retained separately and corrected
before sealing; no held-out outputs were used to repair it.

Only if the primary cold gate passes, repeat the four collection commands with `--condition repeat`
and render that condition separately. Priming responses are recorded. A cold failure is not rescued
by changing the cache mixture, output format, selected subset or target after observing results.

Generated reports contain per-question paired speedups, bootstrap intervals, p95, per-subset results,
validity checks, token/byte/parsed divergence, actual speculative counters and VRAM snapshots.
The binomial preservation bound is a limited benchmark screen, not proof of zero accuracy risk.
