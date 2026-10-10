# Reproducing the bounded continuation

The completed [report](../runs/2026-09-12-replication/report.md) and neighboring raw artifacts
record the positive speed reproduction and the unresolved preservation result separately.

See the [protocol](replication-plan.md). No additional engine patch is needed: use the exact
runtime from the [first prototype](prototype-reproduction.md), including its required llguidance
FFI bounded-copy fix. The second model is the local pinned Qwen3-4B Q4_K_M. No new weights,
paid GPU, external tool execution, or public submission is required.

From the project root, in a new run directory:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python scripts/09_replication.py prepare --run runs/my-replication
.venv/bin/python scripts/09_replication.py diagnose --diagnostic 0 --run runs/my-replication
.venv/bin/python scripts/09_replication.py diagnose --diagnostic 1 --run runs/my-replication
.venv/bin/python scripts/09_replication.py diagnose --diagnostic 2 --run runs/my-replication
.venv/bin/python scripts/09_replication.py diagnose --diagnostic 3 --run runs/my-replication
.venv/bin/python scripts/09_replication.py diagnose --diagnostic 4 --run runs/my-replication
.venv/bin/python scripts/09_replication.py parity --run runs/my-replication
.venv/bin/python scripts/09_replication.py collect --block 0 --run runs/my-replication
.venv/bin/python scripts/09_replication.py collect --block 1 --run runs/my-replication
.venv/bin/python scripts/09_replication.py collect --block 2 --run runs/my-replication
.venv/bin/python scripts/09_replication.py collect --block 3 --run runs/my-replication
.venv/bin/python scripts/09_replication.py report --run runs/my-replication
```

Each command runs one owned server and refuses a GPU already occupied by another compute process.
Run collection serially, without other GPU/benchmark work. Physical-microbatch-one diagnostics
are intentionally slow, including prompt processing. Their latencies are not performance claims.

The seal records all driver source hashes. Do not edit Python source, the C++ runtime, inputs or
protocol after preparation. `diagnose` and `collect` accept `--resume` for interrupted blocks:
flushed trials, including errors, are preserved and never repeated. Completed blocks/reports are
immutable. Reusing the run directory without `--resume` is rejected.

`divergence-audit.json` retains original prompts/tools, outputs, semantic assessments, first
token differences, legal alternatives and independent forced-span replay. The assessments are
post-hoc and are not official accuracy scores. `diagnostic-summary.json` compares exact token
sequences across controls. `replication-summary.json` separates speed reproduction from the
conservative preservation screen; the original gate is reported separately and not rewritten.

The raw data stay local. The hf-cli skill was used to verify existing pinned tokenizer files;
server tokenization parity additionally checks the selected reference strings on the second GGUF.
This is another model size using the same family/tokenizer and GPU, not broad portability proof.
