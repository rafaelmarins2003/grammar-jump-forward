# Reproducing the Phase 0 investigation

The Python scripts were explicitly authorized for this investigation. No engine source was changed.
Use the pinned dependencies, input revisions and engine builds documented in `environment.md`.
Each command writes new artifacts exclusively and refuses to overwrite an existing run stage.
Choose a new run directory for a rerun; never delete a completed run to reuse its name.

## Reference measurements

From the repository root, with local inputs already installed:

```bash
.venv/bin/python scripts/01_phase0.py prepare --run runs/my-pilot
.venv/bin/python scripts/01_phase0.py measure --run runs/my-pilot
.venv/bin/python scripts/01_phase0.py report --run runs/my-pilot

.venv/bin/python scripts/01_phase0.py prepare --full --run runs/my-full
.venv/bin/python scripts/01_phase0.py measure --run runs/my-full
.venv/bin/python scripts/01_phase0.py report --run runs/my-full
```

`prepare` verifies required raw SHA-256 hashes, freezes IDs before adaptation, and retains failures.
`measure` records exact grammar text/hash, token IDs, non-overlapping spans, independent sequential
mask checks and cost proxies. `report` aggregates complete questions, not arbitrarily pooled calls.
`--curve` can select another recorded hardware curve; by default the original pinned curve is used.
The scripts never repair rejected references or select a correct tool before building the grammar.

## Native parity and exploratory generation

The parity script loads the GGUF on CPU without generation. The generated-output script occupies
the GPU and should run alone. Both create and stop only their own localhost server child process.
They verify the GGUF content hash and record the unmodified server binary/revision. Neither sends
data to a remote service or executes the functions described in dataset tool schemas.

```bash
.venv/bin/python scripts/02_tokenizer_parity.py \
  --cases runs/my-pilot/cases.jsonl --run runs/my-parity
.venv/bin/python scripts/03_generated_check.py \
  --cases runs/my-pilot/cases.jsonl --run runs/my-generated
```

The generated-output command implements the specific exploratory selection in the amendment;
it is not a general quality benchmark. Its selected reference agreement is not official BFCL
accuracy, and its raw server timings compare grammar arms without any jump-forward implementation.

## Report and validation

The final decision renderer is specific to the dated investigation: it reads the `2026-09-12-*`
run directories under `--runs` (default `runs/`), and emits Markdown plus an input-hash manifest into
a fresh `--out` directory. It does not overwrite the published report automatically.

```bash
.venv/bin/python scripts/04_gate0_report.py --out runs/my-report-copy
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src scripts tests
.venv/bin/python -m ruff format --check src scripts tests
```

Large `cases.jsonl`, `traces.jsonl` and `questions.json` files are ignored by git; their lossless
`.gz` archives are retained. On a fresh checkout, use `gzip -dk` on the specific archives needed
before rendering the report. Archives use `gzip -n` to omit filename/time metadata. Uncompressed
hashes are recorded in the run manifests; check them after extraction.

Timing extraction calls from Python is sensitive to machine load. The offline cost model is not
an actual server latency benchmark or confidence interval. Rerunning on another machine changes
CPU overhead; new model/GPU cost measurements require their own pinned manifest and isolated run.
Historical early summaries call the pooled-span distribution `span_length_call_weighted`; the
renderer correctly labels it span-weighted, and new summaries use `span_length_pooled`.

## Retargeting follow-up

The cache assay uses the existing generated cohort, not production traffic. It runs short one-token
requests on the GPU with cold, shared system/tool prefix, and exact-repeat cache states. Priming
requests are retained separately. Do not run it concurrently with another GPU measurement.

```bash
.venv/bin/python scripts/05_workload_cache.py collect --run runs/my-cache
.venv/bin/python scripts/05_workload_cache.py analyze --run runs/my-cache
.venv/bin/python scripts/06_retarget_frontier.py \
  --cache runs/my-cache --out runs/my-frontier
```

If collection was interrupted, rerun `collect --resume` with the same run and generated source.
This preserves existing trials and records continuation provenance; completed runs are immutable.
`analyze --resume` can finish missing analysis metadata only if existing derived artifacts exactly
match recomputation. It cannot revise a completed analysis. Linux launches use `setpriv --pdeathsig
TERM` when available, so controller termination also terminates the owned server.

The full-cohort `frontier.json` is also archived with `gzip -kn` and ignored in expanded form.
The published [retargeting report](retarget-report.md) distinguishes actual cache-prefix timings,
model-only estimates, residual-preserving sensitivity, ideal zero-prefix limits, and output-length
strata. The original Gate 0 report and its input hashes are preserved, not silently reinterpreted.
