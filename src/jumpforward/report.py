"""Render the feasibility decision from immutable run artifacts, not typed-in results."""

# Report prose is kept as readable Markdown lines, including interpolated tables.
# ruff: noqa: E501

import argparse
import json
from pathlib import Path

from .corpus import digest, read_jsonl
from .cost import quantiles
from .phase0 import ROOT, provenance, write_json


def pct(value):
    return f"{100 * value:.1f}%"


def ratio(value):
    return f"{value:.2f}×"


def render(full, pilot, parity, generated, questions, cases, errors, curve):
    best = full["G2/S2"]["overall"]
    cold = best["cost"]["1024"]["cold_optimistic"]["median"]
    decision = "retarget" if cold < 1.5 else "incomplete"
    lines = [
        "# Gate 0 — feasibility decision",
        "",
        f"Decision: **{decision}**. The original end-to-end performance claim is not established.",
        "No engine C++ implementation is green-lit by this report.",
        "",
        "Continue only as a narrower engineering investigation of native llguidance batching,",
        "with cached/short prompts or longer structured outputs. Do not proceed on the assumption",
        "that ordinary tool calls will generally achieve the proposed measured-latency target.",
        "",
        "## Evidence and coverage",
        "",
        f"The frozen pilot selected {pilot['G2/S2']['overall']['selected']:,} questions. "
        f"The expanded run enumerated {best['selected']:,} questions in the six selected subsets, "
        f"with {best['complete']:,} complete questions per variant ({pct(best['coverage'])} coverage).",
        f"The adapter retained {sum(c['status'] != 'ok' for c in cases):,} failures; a further "
        f"{sum(c['status'] == 'ok' for c in cases) - best['complete']:,} questions failed replay.",
        "No failures were replaced. These results describe the supported cohort, not every BFCL",
        "category or all of Hermes. Adapter failures can reflect unsupported reference encodings,",
        "not necessarily a bad original benchmark answer.",
        "",
        "| Subset | Selected | Complete | G1/S2 median forced | G2/S2 median forced |",
        "|---|---:|---:|---:|---:|",
    ]
    for subset, row in full["G2/S2"]["subsets"].items():
        baseline = full["G1/S2"]["subsets"][subset]
        lines.append(
            f"| {subset} | {row['selected']:,} | {row['complete']:,} | "
            f"{pct(baseline['forced_fraction']['median'])} | "
            f"{pct(row['forced_fraction']['median'])} |"
        )
    lines.extend(
        [
            "",
            "Every grammar includes all tools supplied with the question. Only `{name, arguments}`",
            "bodies are counted: wrappers, triggers, choice of number of calls, transport and actual",
            "prompt lengths are excluded. Parallel bodies are combined into one question before",
            "computing distributions; prefill is charged once per question.",
            "",
            "## Forced spans",
            "",
            "G1 is llguidance JSON Schema with flexible whitespace; G2 fixes whitespace to match the",
            "serialization. S1 is compact JSON; S2 inserts one space after commas and colons.",
            "",
            "| Grammar / serialization | Forced p10 | Median | p90 | Median injected bytes | "
            "Pooled span median, tokens |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, value in full.items():
        r = value["overall"]
        f = r["forced_fraction"]
        # Early manifests used this legacy key for the same pooled-span distribution.
        spans = r.get("span_length_pooled", r.get("span_length_call_weighted"))
        lines.append(
            f"| {key} | {pct(f['p10'])} | {pct(f['median'])} | {pct(f['p90'])} | "
            f"{pct(r['byte_fraction']['median'])} | {spans['median']:.0f} |"
        )
    best_questions = [q for q in questions if q["variant"] == "G2" and q["serialization"] == "S2"]
    fractions = [q["forced_fraction"] for q in best_questions if q["status"] == "ok"]
    missing = best["selected"] - len(fractions)
    low = quantiles(fractions + [0.0] * missing)["median"]
    high = quantiles(fractions + [1.0] * missing)["median"]
    lines.extend(
        [
            "",
            "Fractions are question-weighted; span lengths pool individual spans and therefore",
            "are span-weighted. Spans never overlap, and every injected token was checked against a",
            "separate sequential singleton mask. JSON/schema and UTF-8 byte roundtrip were checked",
            "independently. Schema `format` is annotation-only in the initial jsonschema validation;",
            "the llguidance replay can enforce more and reject such a reference.",
            "",
            f"The G2/S2 forced-fraction median exceeds the registered 25% floor. Even assigning every "
            f"missing question a fraction of zero or one puts the selection-wide median between "
            f"{pct(low)} and {pct(high)}. This sensitivity check is not an estimate for the missing data.",
            "The fraction floor alone is not a performance gate.",
            "",
            "## Cost projections, not observed server speedups",
            "",
            "| Variant | Context | Decode, conservative–optimistic | Cold prompt, "
            "conservative–optimistic | Zero-forced-cost cold ceiling |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, value in full.items():
        for depth, c in value["overall"]["cost"].items():
            lines.append(
                f"| {key} | {depth} | {ratio(c['decode_conservative']['median'])}–"
                f"{ratio(c['decode_optimistic']['median'])} | "
                f"{ratio(c['cold_conservative']['median'])}–"
                f"{ratio(c['cold_optimistic']['median'])} | "
                f"{ratio(c['zero_cost_cold_ceiling']['median'])} |"
            )
    lines.extend(
        [
            "",
            "Each row compares batching with token-by-token processing of the **same grammar and",
            "serialization**. The columns are medians of paired per-question ratios, not ratios of",
            "median latencies or confidence intervals. Optimistic/conservative labels denote sensitivity",
            "proxies, not guaranteed bounds. The zero-cost ceiling eliminates forced-token model work",
            "only inside this fixed-prefill model; it is not a bound on an arbitrary deployed server.",
            "",
            "The cost curve is an actual synthetic GPU measurement on the pinned Qwen3 GGUF and RTX",
            "2080 Ti. Batches process every body token, including the final one for termination logits,",
            "with a cap of 32 tokens. Initial forced prefixes are charged. The conservative proxy adds",
            "measured FF extraction and bulk consumption, but neither arm models full server overhead,",
            "ordinary mask cost, grammar compilation, changing context depth or cache-management cost.",
            "Its monotone envelope handles the observed non-monotone batch-cost curve.",
            "",
            f"G2/S2 median FF extraction was {best['ff_ms']['median']:.3f} ms per question; "
            f"median grammar compilation, excluded from both proxies, was "
            f"{best['compile_ms']['median']:.3f} ms. Cold prefill means were "
            + ", ".join(f"{r['mean_ms']:.3f} ms for {r['tokens']} tokens" for r in curve["prefill"])
            + ".",
            "Decode-only projections do not include cache lookup or other cached-request overhead.",
            "",
            "### Serialization sanity check",
            "",
            "S2 contains additional space tokens, so a larger forced fraction or within-S2 speedup",
            "need not mean the fastest implementation. The following **post-hoc sensitivity** uses",
            "compact G1/S1 as the common unbatched reference, with the conservative batch proxy:",
            "",
            "| Candidate | Context | Decode vs compact baseline | Cold vs compact baseline |",
            "|---|---:|---:|---:|",
        ]
    )
    base = {
        (q["subset"], q["id"]): q
        for q in questions
        if q["status"] == "ok" and q["variant"] == "G1" and q["serialization"] == "S1"
    }
    for serial in ("S1", "S2"):
        rows = [
            q
            for q in questions
            if q["status"] == "ok" and q["variant"] == "G2" and q["serialization"] == serial
        ]
        for i, depth in enumerate((1024, 4096)):
            decode, cold_values = [], []
            for q in rows:
                c = q["cost"][i]
                b = base[(q["subset"], q["id"])]["cost"][i]["baseline_ms"]
                decode.append(b / c["conservative_ms"])
                cold_values.append((b + c["prefill_ms"]) / (c["conservative_ms"] + c["prefill_ms"]))
            lines.append(
                f"| G2/{serial} | {depth} | {ratio(quantiles(decode)['median'])} | "
                f"{ratio(quantiles(cold_values)['median'])} |"
            )
    lines.extend(
        [
            "",
            "This is not a new success threshold. Optimize measured absolute latency, not forced",
            "fraction by adding predictable whitespace. No tested cold-prompt median proxy reaches",
            "the original measured-latency target.",
            "",
            "## Native tokenizer and generated-output checks",
            "",
            f"Tokenizer parity: {parity['n_calls']} call records, {parity['n_serializations']} "
            f"serializations checked against the unmodified local llama-server `/tokenize`; "
            f"{parity['mismatches']} ID mismatches. This checks those body strings, not equality of all",
            "GGUF/HF vocabulary entries, special tokens or arbitrary chat prefixes.",
            "",
            f"The preregistered exploratory generation check ran {generated['n_questions']} single-call "
            f"questions in both whitespace arms ({generated['n_outputs']} outputs), temperature zero, "
            "thinking off, no draft and no prompt reuse. Neither arm had jump-forward implemented.",
            "All supplied candidate tools remained available; no tool was actually executed.",
            "",
            "| Arm | Valid JSON/schema | Raw-token replays | Median forced | "
            "Chosen-reference agreement | Noncanonical whole-text tokenizations |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, r in generated["arms"].items():
        lines.append(
            f"| {arm} | {r['json_schema_valid']}/{r['selected']} | "
            f"{r['replay_ok']}/{r['selected']} | {pct(r['forced_fraction']['median'])} | "
            f"{r['chosen_reference_equal']}/{r['selected']} | "
            f"{r['noncanonical_tokenizations']} |"
        )
    lines.extend(
        [
            "",
            f"There were {generated['parsed_output_differences']} parsed-output differences among "
            f"{generated['valid_pairs']} valid pairs. This small selected cohort cannot establish "
            "non-inferiority. Chosen-reference agreement is not official BFCL accuracy: alternative",
            "valid argument realizations and task semantics require a separate evaluator. Raw generated",
            "IDs, including noncanonical segmentations, were replayed unchanged; only a recorded",
            "trailing EOS would be removed. Retokenizing the final text would measure a different trace.",
            "",
            "## Replay exclusions and compatibility limits",
            "",
            "| Question ID | Failed call bodies in G2/S2 | First diagnostic |",
            "|---|---:|---|",
        ]
    )
    grouped = {}
    for row in errors:
        grouped.setdefault(row["id"], []).append(row)
    for ident, rows in grouped.items():
        reason = rows[0]["error"].split(":")[0].split(" at ")[0]
        lines.append(f"| {ident} | {len(rows)} | {reason} |")
    lines.extend(
        [
            "",
            "The `uri` format was not supported by this llguidance compiler. Two date references",
            "contained year-month strings rather than complete dates; a date-time reference omitted",
            "the timezone required by the grammar. These failed replay rather than being repaired.",
            "An FF/reference mismatch is not automatically a tokenizer bug. Detailed IDs, token IDs,",
            "grammar text and errors remain in the run artifacts. Missing/invalid Hermes JSON and",
            "unsupported BFCL answer encodings account for much of the earlier adaptation loss.",
            "",
            "G3 (native llama.cpp GBNF) was **not measured**: equivalence of an offline conversion to",
            "the native runtime was not validated. G1 must not be presented as G3. The synthetic lazy",
            "lexeme test passes, but actual chat-template trigger/cardinality integration remains open.",
            "Other tokenizers, alternative forcing policies, draft latency/memory comparisons, CPU",
            "inference and a confirmatory quality study remain deferred under the PL-first amendment.",
            "",
            "## What to do next",
            "",
            "1. Keep the project as an engineering/learning study; narrow its claim to batching tokens",
            "   already forced by llguidance. Novel tokenization research is not established by this work.",
            "2. Before engine implementation, measure real prompt lengths, prompt-cache hit rates and",
            "   output lengths for the intended application. Recompute absolute latency with those",
            "   distributions and compare compact serialization. Drop the original broad performance",
            "   claim if the selected application also has no plausible path to its target.",
            "3. If that retargeted workload passes, preregister a paired quality margin and sample size,",
            "   compare G1/G2 under the same native token policy, and validate G3/real tool wrappers.",
            "   A policy study must compare PL with `no_forcing` under the same grammar, not confuse",
            "   existing singleton forcing with a new batching effect.",
            "4. Only after an explicit revised go decision, implement a small default-off PL batching",
            "   path and measure actual paired request latency, KV correctness and output divergence.",
            "   Draft composition is a later comparison of latency and memory, not acceptance alone.",
            "",
            "## Reproducibility and prior art",
            "",
            "Generated by `scripts/04_gate0_report.py`; all reported measurements come from:",
            "",
            "- [Full reference summary](../runs/2026-09-12-full/summary.json),",
            "  [selection/provenance](../runs/2026-09-12-full/prepare-manifest.json) and",
            "  [measurement manifest](../runs/2026-09-12-full/measure-manifest.json).",
            "- [Pilot summary](../runs/2026-09-12-pilot/summary.json).",
            "- [Synthetic GPU cost curve](../runs/2026-09-12-pass-cost/summary.json) and its",
            "  [exact commands/build/model hashes](../runs/2026-09-12-pass-cost/manifest.json).",
            "- [Tokenizer parity](../runs/2026-09-12-parity/summary.json) and",
            "  [generated-output check](../runs/2026-09-12-generated/summary.json).",
            "- [Report input hashes](../runs/2026-09-12-decision/manifest.json).",
            "",
            "Raw bodies, tokens, spans and individual errors are retained locally and in lossless",
            "gzip archives where large. Inspect the audit trail before generalizing these findings.",
            "See the [native-policy audit](notes/llguidance-ff-tokens.md) and",
            "[prior-art review](notes/prior-art.md). DOMINO, SGLang and XGrammar already study relevant",
            "tokenization/jump-forward mechanisms. The potential contribution is their carefully",
            "controlled evaluation and integration in this pinned consumer-hardware llama.cpp setting,",
            "not the invention of jump-forward or proof of a novel quality effect.",
            "",
            f"Final decision: **{decision}**, not an original-scope Gate 0 pass.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: args.runs / f"2026-09-12-{name}"
        for name in ("full", "pilot", "parity", "generated", "pass-cost")
    }
    inputs = {name: json.loads((p / "summary.json").read_text()) for name, p in paths.items()}
    questions = json.loads((paths["full"] / "questions.json").read_text())
    cases = read_jsonl(paths["full"] / "cases.jsonl")
    errors = [
        r
        for r in read_jsonl(paths["full"] / "traces.jsonl")
        if r["status"] == "error" and r["variant"] == "G2" and r["serialization"] == "S2"
    ]
    report = render(
        inputs["full"],
        inputs["pilot"],
        inputs["parity"],
        inputs["generated"],
        questions,
        cases,
        errors,
        inputs["pass-cost"],
    )
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "gate0-report.md").open("x") as stream:
        stream.write(report)
    artifacts = [p / "summary.json" for p in paths.values()]
    artifacts += [
        paths["full"] / name for name in ("questions.json", "cases.jsonl", "traces.jsonl")
    ]
    write_json(
        args.out / "manifest.json",
        {
            **provenance(),
            "inputs_sha256": {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in artifacts},
            "report_sha256": digest(report.encode()),
            "report_interpretation": "retarget original scope; no engine authorization",
        },
    )
    print(args.out / "gate0-report.md")
