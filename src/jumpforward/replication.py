"""Post-hoc divergence diagnosis and unchanged-engine, second-model replication."""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from llguidance import LLMatcher

from .corpus import digest, read_jsonl
from .parity import file_hash, owned_server
from .phase0 import ROOT, provenance, write_json
from .prototype import BLOCKS, analyze_rows, collect, runtime_identity, tokenizer, validate_response
from .spans import allows, make_grammar, trace_text

ORIGINAL = ROOT / "runs/2026-09-12-prototype"
PROTOCOL = ROOT / "docs/replication-plan.md"
MODEL = "qwen3-4b-q4_k_m"
DIAGNOSTICS = (
    ("normal-off", "none", 32, None),
    ("normal-on", "grammar", 32, None),
    ("zero-cap", "grammar", 0, None),
    ("micro-one-off", "none", 32, 1),
    ("micro-one-on", "grammar", 32, 1),
)
JUDGMENTS = {
    "multiple_175": {
        "classification": "both_inadequate",
        "reason": "The user asks for women's tennis rankings. Both outputs select the musical "
        "instrument tool instead of the supplied sports ranking tool.",
    },
    "live_simple_237-125-0": {
        "classification": "likely_regression",
        "reason": "Both identify the requested song. Only batching adds device_id=':null', an "
        "unsupported device identifier, distinct from the documented default 'null'. Explicit "
        "play=true alone is consistent with the documented default. No Spotify tool was executed.",
    },
    "live_multiple_221-95-0": {
        "classification": "both_inadequate",
        "reason": "Both select segment but replace the explicit relative path './yoha.jpg' with "
        "unsupported absolute paths. The provided context establishes neither replacement.",
    },
    "f595b8d5-531a-4699-85bc-02620d4e1de8": {
        "classification": "improvement",
        "reason": "The user explicitly requests createTaxiBookingSkill with the supplied details. "
        "Batching follows that request; the baseline instead cancels an invented booking.",
    },
}


def first_difference(left, right):
    return next(
        (i for i, (a, b) in enumerate(zip(left, right, strict=False)) if a != b),
        min(len(left), len(right)),
    )


def audit_pair(case, pair, tok):
    arms = pair["arms"]
    tokens = {s: list(arms[s]["tokens"][0]) for s in ("none", "grammar")}
    offset = first_difference(tokens["none"], tokens["grammar"])
    grammar = make_grammar(case["schema"], "G2", "S1")
    matcher = LLMatcher(tok, grammar)
    for token in tokens["none"][:offset]:
        if not matcher.consume_token(token):
            raise ValueError("invalid shared-prefix token")
    mask = matcher.compute_bitmask()
    alternatives = {s: t[offset] if offset < len(t) else None for s, t in tokens.items()}
    traces = {}
    for strategy, ids in tokens.items():
        body = ids[:-1] if ids and ids[-1] == tok.eos_token else ids
        try:
            trace = trace_text(tok, grammar, arms[strategy]["bytes"][0], body)
            traces[strategy] = {
                k: trace[k] for k in ("singleton_agreement", "n_tokens", "n_forced", "spans")
            }
        except ValueError as exc:
            traces[strategy] = {"error": str(exc)}
    return {
        "subset": case["subset"],
        "id": case["id"],
        "parsed_changed": pair["parsed_changed"],
        "messages": case["messages"],
        "tools": case["tools"],
        "reference": case["calls"][0],
        "off": arms["none"]["parsed"][0],
        "on": arms["grammar"]["parsed"][0],
        "assessment": JUDGMENTS.get(
            case["id"],
            {
                "classification": "same_bytes_tokenization_control",
                "reason": "Parsed output and bytes are identical; token IDs differ.",
            },
        ),
        "first_divergence": {
            "token_offset": offset,
            "shared_prefix": tok.decode_bytes(tokens["none"][:offset]).decode("utf-8", "replace"),
            "tokens": alternatives,
            "pieces": {
                s: tok.decode_bytes([t]).decode("utf-8", "replace") if t is not None else None
                for s, t in alternatives.items()
            },
            "both_legal": all(t is not None and allows(mask, t) for t in alternatives.values()),
            "allowed_token_count": int.from_bytes(mask, "little").bit_count(),
            "ff_tokens": matcher.compute_ff_tokens(),
        },
        "native_forcing_replay": traces,
    }


def prepare(args):
    old_seal = json.loads((ORIGINAL / "evaluation-seal.json").read_text())
    identity = runtime_identity()
    if identity != old_seal["runtime"]:
        raise ValueError("the original engine changed; do not call this unchanged replication")
    old_report = json.loads((ORIGINAL / "cold-report-manifest.json").read_text())
    for path, expected in old_report["raw_outputs_sha256"].items():
        if digest((ORIGINAL / path).read_bytes()) != expected:
            raise ValueError("original raw results changed")
    if digest((ORIGINAL / "cases.json").read_bytes()) != old_seal["cases_sha256"]:
        raise ValueError("original questions changed")
    model = json.loads((ROOT / "config/models.json").read_text())["targets"][MODEL]
    if file_hash(Path(model["path"]).expanduser()) != model["sha256"]:
        raise ValueError("second-model hash mismatch")
    cases = json.loads((ORIGINAL / "cases.json").read_text())
    by_id = {(c["subset"], c["id"]): c for c in cases}
    pairs = json.loads((ORIGINAL / "cold-pairs.json").read_text())
    selected = [p for p in pairs if p["tokens_changed"]]
    if len(selected) != 5 or sum(p["parsed_changed"] for p in selected) != 4:
        raise ValueError("unexpected original divergence cohort")
    tok = tokenizer()
    audit = [audit_pair(by_id[(p["subset"], p["id"])], p, tok) for p in selected]
    original_rows = read_jsonl(ORIGINAL / "cold-00-none/outputs.jsonl")
    requests = {(r["subset"], r["id"]): r["request"] for r in original_rows}
    args.run.mkdir(exist_ok=False)
    write_json(args.run / "cases.json", cases)
    write_json(args.run / "divergence-audit.json", audit)
    write_json(
        args.run / "diagnostic-inputs.json",
        [
            {
                "case": by_id[(p["subset"], p["id"])],
                "original": p,
                "request": requests[(p["subset"], p["id"])],
            }
            for p in selected
        ],
    )
    with (args.run / "registered-protocol.md").open("x") as stream:
        stream.write(PROTOCOL.read_text())
    write_json(
        args.run / "evaluation-seal.json",
        {
            **provenance(),
            "model_key": MODEL,
            "model_config": model,
            "scope": "same-family second-model replication, not a fresh question holdout",
            "runtime": identity,
            "cases_sha256": digest((args.run / "cases.json").read_bytes()),
            "protocol_sha256": digest(PROTOCOL.read_bytes()),
            "n_predict": 2048,
            "draft_cap": 32,
            "diagnostic_inputs_sha256": digest((args.run / "diagnostic-inputs.json").read_bytes()),
            "original_report_sha256": old_report["report_sha256"],
        },
    )
    print(
        json.dumps(
            {
                "selected": len(cases),
                "diagnostic_cases": len(audit),
                "assessments": Counter(a["assessment"]["classification"] for a in audit),
                "first_divergences": [a["first_divergence"] for a in audit],
            }
        ),
        flush=True,
    )


def verify_seal(run):
    seal = json.loads((run / "evaluation-seal.json").read_text())
    if (
        runtime_identity() != seal["runtime"]
        or digest(PROTOCOL.read_bytes()) != seal["protocol_sha256"]
    ):
        raise ValueError("sealed engine or protocol changed")
    if provenance()["source_sha256"] != seal["source_sha256"]:
        raise ValueError("sealed driver source changed")
    if digest((run / "cases.json").read_bytes()) != seal["cases_sha256"]:
        raise ValueError("sealed cases changed")
    if digest((run / "diagnostic-inputs.json").read_bytes()) != seal["diagnostic_inputs_sha256"]:
        raise ValueError("diagnostic inputs changed")
    return seal


def parity(args):
    verify_seal(args.run)
    directory = args.run / "second-model-parity"
    directory.mkdir(exist_ok=False)
    cases = json.loads((args.run / "cases.json").read_text())
    tok, rows = tokenizer(), []
    with owned_server(directory, gpu=True, model_key=MODEL) as client:
        for case in cases:
            text = json.dumps(case["calls"][0], ensure_ascii=False, separators=(",", ":"))
            response = client.post(
                "/tokenize",
                json={
                    "content": text,
                    "add_special": False,
                    "parse_special": False,
                },
            )
            response.raise_for_status()
            native = response.json()["tokens"]
            offline = tok.tokenize_str(text)
            rows.append(
                {
                    "id": case["id"],
                    "subset": case["subset"],
                    "native": native,
                    "offline": offline,
                    "equal": native == offline,
                }
            )
    write_json(directory / "comparisons.json", rows)
    summary = {
        "checked": len(rows),
        "mismatches": sum(not r["equal"] for r in rows),
        "scope": "reference token IDs; not an exhaustive vocabulary equivalence proof",
    }
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def diagnose(args):
    verify_seal(args.run)
    inputs = json.loads((args.run / "diagnostic-inputs.json").read_text())
    name, strategy, cap, ubatch = DIAGNOSTICS[args.diagnostic]
    directory = args.run / f"diagnostic-{args.diagnostic:02d}-{name}"
    runtime, prior = directory, []
    if args.resume:
        if (directory / "complete.json").exists():
            raise ValueError("completed diagnostic is immutable")
        prior = read_jsonl(directory / "outputs.jsonl")
        runtime = directory / f"continuation-{len(list(directory.glob('continuation-*'))) + 1:02d}"
    runtime.mkdir(exist_ok=False)
    done = {(r["subset"], r["id"], r["repetition"]) for r in prior}
    if len(done) != len(prior):
        raise ValueError("duplicate diagnostic trials")
    tok, count = tokenizer(), len(prior)
    with (
        owned_server(
            runtime, gpu=True, speculation=strategy, grammar_cap=cap, ubatch_size=ubatch
        ) as client,
        (directory / "outputs.jsonl").open("a" if args.resume else "x") as stream,
    ):
        warm = client.post(
            "/completion",
            json={
                "prompt": "Say hello.",
                "n_predict": 16,
                "temperature": 0,
                "cache_prompt": False,
            },
        )
        warm.raise_for_status()
        write_json(runtime / "warmup.json", warm.json())
        for repetition in range(2):
            for item in inputs if repetition == 0 else inputs[::-1]:
                case = item["case"]
                if (case["subset"], case["id"], repetition) in done:
                    continue
                row = {
                    "id": case["id"],
                    "subset": case["subset"],
                    "configuration": name,
                    "repetition": repetition,
                    "request": item["request"],
                }
                try:
                    start = time.perf_counter()
                    response = client.post("/completion", json=item["request"])
                    row["wall_ms"] = (time.perf_counter() - start) * 1000
                    response.raise_for_status()
                    row["response"] = response.json()
                    parsed = validate_response(case, row["response"], tok)
                    row.update(status="ok", parsed=parsed)
                except Exception as exc:
                    row.update(status="error", error=str(exc))
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                count += 1
                print(f"{name}: {count}/{len(inputs) * 2} {row['status']}", flush=True)
    write_json(
        directory / "complete.json",
        {
            "trials": count,
            "outputs_sha256": digest((directory / "outputs.jsonl").read_bytes()),
        },
    )


def diagnosis_summary(run):
    inputs = json.loads((run / "diagnostic-inputs.json").read_text())
    table, failures = {}, []
    for number, (name, _, _, _) in enumerate(DIAGNOSTICS):
        directory = run / f"diagnostic-{number:02d}-{name}"
        complete = json.loads((directory / "complete.json").read_text())
        path = directory / "outputs.jsonl"
        if digest(path.read_bytes()) != complete["outputs_sha256"]:
            raise ValueError("completed diagnostic changed")
        rows = read_jsonl(path)
        expected = {(i["case"]["subset"], i["case"]["id"], r) for i in inputs for r in range(2)}
        if (
            len(rows) != len(expected)
            or {(r["subset"], r["id"], r["repetition"]) for r in rows} != expected
        ):
            raise ValueError("diagnostic trial grid mismatch")
        table[name] = {(r["subset"], r["id"], r["repetition"]): r for r in rows}
        failures.extend(r for r in rows if r["status"] != "ok")
    per_case = []
    for item in inputs:
        case = item["case"]

        def response(name, rep, case=case):
            return table[name][(case["subset"], case["id"], rep)].get("response", {})

        def same(a, b):
            return all(
                response(a, rep).get("tokens") is not None
                and response(a, rep).get("tokens") == response(b, rep).get("tokens")
                for rep in range(2)
            )

        per_case.append(
            {
                "id": case["id"],
                "subset": case["subset"],
                "normal_off_matches_original": all(
                    response("normal-off", r).get("tokens")
                    == item["original"]["arms"]["none"]["tokens"][r]
                    for r in range(2)
                ),
                "normal_on_matches_original": all(
                    response("normal-on", r).get("tokens")
                    == item["original"]["arms"]["grammar"]["tokens"][r]
                    for r in range(2)
                ),
                "normal_arms_same_tokens": same("normal-off", "normal-on"),
                "zero_cap_matches_off": same("zero-cap", "normal-off"),
                "micro_one_arms_same_tokens": same("micro-one-off", "micro-one-on"),
                "configurations": {
                    name: {
                        "parsed": [
                            table[name][(case["subset"], case["id"], r)].get("parsed")
                            for r in range(2)
                        ],
                        "drafted": sum(
                            response(name, r).get("timings", {}).get("draft_n", 0) for r in range(2)
                        ),
                        "accepted": sum(
                            response(name, r).get("timings", {}).get("draft_n_accepted", 0)
                            for r in range(2)
                        ),
                    }
                    for name in table
                },
            }
        )
    return {
        "requests": sum(len(t) for t in table.values()),
        "failed_requests": len(failures),
        "failures": failures,
        "cases": per_case,
    }


def replication_criteria(summary):
    complete = summary["complete_pairs"] >= 190 and all(
        s["wall_speedup"]["n"] >= 35 for s in summary["subsets"].values()
    )
    performance = complete and all(
        summary["criteria"][k]
        for k in (
            "ci_lower_above_1",
            "all_subsets_at_least_1",
            "p95_at_most_1_10",
            "actual_grammar_drafts",
        )
    )
    preserved = (
        complete
        and summary["criteria"]["quality_upper_below_2_percent"]
        and not (summary["failure_questions"]["candidate_only"])
        and all(f["error"] == "generation limit or truncation" for f in summary["failures"])
    )
    return {
        "adequate_complete_pairs": complete,
        "performance_reproduced": performance,
        "preservation_screen_passed": preserved,
        "old_1_4_target_met_descriptively": summary["criteria"]["median_at_least_1_4"],
    }


def report(args):
    seal = verify_seal(args.run)
    cases = json.loads((args.run / "cases.json").read_text())
    by_id = {(c["subset"], c["id"]): c for c in cases}
    rows, hashes = [], {}
    tok = tokenizer()
    for number, (strategy, _) in enumerate(BLOCKS):
        directory = args.run / f"cold-{number:02d}-{strategy}"
        path = directory / "outputs.jsonl"
        hashes[str(path.relative_to(args.run))] = digest(path.read_bytes())
        complete = json.loads((directory / "complete.json").read_text())
        if digest(path.read_bytes()) != complete["outputs_sha256"]:
            raise ValueError("completed replication block changed")
        for m in directory.rglob("server-manifest.json"):
            metadata = json.loads(m.read_text())
            if (
                metadata["model_key"] != MODEL
                or metadata["verified_model_sha256"] != seal["model_config"]["sha256"]
            ):
                raise ValueError("wrong measured model")
            if metadata["engine_diff_sha256"] != seal["runtime"]["engine_diff_sha256"]:
                raise ValueError("wrong measured engine")
            if (
                metadata["binary_sha256"]
                != seal["runtime"]["files_sha256"]["build/bin/llama-server"]
            ):
                raise ValueError("wrong measured binary")
            if any(
                seal["runtime"]["files_sha256"].get(f"build/bin/{p}") != h
                for p, h in metadata["runtime_libraries_sha256"].items()
            ):
                raise ValueError("wrong measured libraries")
        block_rows = read_jsonl(path)
        if len(block_rows) != complete["trials"]:
            raise ValueError("incomplete replication block")
        rows.extend(block_rows)
    requests = defaultdict(list)
    for row in rows:
        requests[(row["subset"], row["id"])].append(row["request"])
        if row["status"] == "ok":
            parsed = validate_response(by_id[(row["subset"], row["id"])], row["response"], tok)
            if parsed != row["parsed"]:
                raise ValueError("revalidated JSON differs")
    if any(any(r != group[0] for r in group) for group in requests.values()):
        raise ValueError("requests differ, including failed trials")
    summary, pairs = analyze_rows(cases, rows)
    summary["original_experiment_rule_result"] = summary.pop("decision")
    summary["replication_criteria"] = replication_criteria(summary)
    summary["model_key"] = MODEL
    for strategy in ("none", "grammar"):
        snapshots = [
            s
            for p in args.run.glob(f"cold-*-{strategy}/**/memory.json")
            for s in json.loads(p.read_text())
        ]
        summary["arms"][strategy]["observed_vram_mib"] = max(
            (s["vram_mib"] for s in snapshots if "vram_mib" in s), default=None
        )
    diagnostics = diagnosis_summary(args.run)
    write_json(args.run / "diagnostic-summary.json", diagnostics)
    write_json(args.run / "replication-summary.json", summary)
    write_json(args.run / "replication-pairs.json", pairs)
    markdown = render_report(
        summary, diagnostics, json.loads((args.run / "divergence-audit.json").read_text())
    )
    with (args.run / "report.md").open("x") as stream:
        stream.write(markdown)
    write_json(
        args.run / "report-manifest.json",
        {
            **provenance(),
            "model_key": MODEL,
            "model_config": seal["model_config"],
            "scope": "post-hoc 8B diagnosis and frozen 4B replication; no engine tuning",
            "raw_outputs_sha256": hashes,
            "files_sha256": {
                p: digest((args.run / p).read_bytes())
                for p in (
                    "report.md",
                    "replication-summary.json",
                    "replication-pairs.json",
                    "diagnostic-summary.json",
                    "divergence-audit.json",
                    "evaluation-seal.json",
                )
            },
        },
    )
    print(
        json.dumps(
            {
                "replication": summary,
                "diagnostic_requests": diagnostics["requests"],
                "diagnostic_failures": diagnostics["failed_requests"],
            }
        ),
        flush=True,
    )


def render_report(s, diagnostic, audit):
    def ratio(result):
        if result["median"] is None:
            return "n/a"
        return f"{result['median']:.3f}x [{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}]"

    lines = [
        "# Divergence diagnosis and second-model replication",
        "",
        "The original 8B experiment remains STOP under its registered criteria. This report",
        "answers the separately authorized continuation, with the same unmodified engine patch.",
        "",
        "## Qwen3-4B Q4_K_M replication",
        "",
        f"{s['selected']} questions, {s['requests']} requests; "
        f"{s['complete_pairs']} complete pairs.",
        f"Paired median end-to-end: **{ratio(s['wall_speedup'])}**. "
        f"Decode: {ratio(s['decode_speedup'])}.",
        "Baseline: identical compact llguidance, batching off, not a tuned market best.",
        "The same 200 questions are reused across sizes, not a fresh question holdout.",
        "Cold cache, temperature zero, one slot, same-family tokenizer and the same RTX 2080 Ti.",
        "",
        "| Subset | Complete pairs | End-to-end ratio [95% bootstrap interval] |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {k} | {v['wall_speedup']['n']} | {ratio(v['wall_speedup'])} |"
        for k, v in s["subsets"].items()
    )
    lines.extend(
        [
            "",
            f"Candidate/baseline pooled p95: {s['p95_latency_ratio']:.3f}; "
            "includes timed failures.",
            f"Failed requests: {len(s['failures'])}. Candidate-only failed questions: "
            f"{len(s['failure_questions']['candidate_only'])}.",
            f"Parsed differences: {s['quality']['parsed_changed']}; "
            f"unstable questions: {s['quality']['unstable']}.",
            "Potential-regression rate one-sided 95% upper bound: "
            f"{s['quality']['one_sided_95_upper']:.2%}.",
            f"Chosen-reference equality: off={s['arms']['none']['chosen_reference_equal']}, "
            f"on={s['arms']['grammar']['chosen_reference_equal']}; not official accuracy.",
            f"Accepted/proposed grammar drafts: {s['arms']['grammar']['accepted']}/"
            f"{s['arms']['grammar']['drafted']}.",
            f"Observed VRAM MiB: off={s['arms']['none']['observed_vram_mib']}, "
            f"on={s['arms']['grammar']['observed_vram_mib']}; snapshots, not a profiled peak.",
            "",
            "| Separate replication criterion | Result |",
            "|---|---|",
        ]
    )
    lines.extend(f"| {k} | {v} |" for k, v in s["replication_criteria"].items())
    lines.extend(
        [
            "",
            "## Original 8B differences: semantic review",
            "",
            "Post-hoc assessments against supplied prompts/tools, not blinded judgments",
            "or an official benchmark score. No dataset tool or external application was executed.",
            "",
        ]
    )
    for a in audit:
        lines.extend(
            [
                f"### {a['id']}",
                "",
                f"{a['assessment']['classification']}: {a['assessment']['reason']}",
                f"First differing token offset: {a['first_divergence']['token_offset']}; "
                f"both alternatives grammar-legal: {a['first_divergence']['both_legal']}; "
                f"allowed tokens: {a['first_divergence']['allowed_token_count']}; "
                f"forced tokens there: {a['first_divergence']['ff_tokens']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Controlled execution checks",
            "",
            f"{diagnostic['requests']} diagnostic requests; "
            f"{diagnostic['failed_requests']} failures.",
            "Token equality requires both repetitions. Microbatch-one also changes prefill:",
            "compare its off/on arms to each other, not to the normal baseline.",
            "",
            "| Case | Off reproduced | On reproduced | Zero cap = off | Micro-one off = on |",
            "|---|---|---|---|---|",
        ]
    )
    lines.extend(
        f"| {c['id']} | {c['normal_off_matches_original']} | {c['normal_on_matches_original']} | "
        f"{c['zero_cap_matches_off']} | {c['micro_one_arms_same_tokens']} |"
        for c in diagnostic["cases"]
    )
    lines.extend(
        [
            "",
            "Agreement under microbatch-one, with actual accepted drafts, supports dependence",
            "on execution shape, not direct measurement of floating-point/logit differences",
            "or proof the integration has no other bugs. Divergent free choices are not",
            "incorrect forced tokens. The audit retains independent singleton-mask replays.",
            "",
            "## Interpretation and limits",
            "",
            "Speed reproduction and preservation are separate. A failed preservation screen",
            "does not mean every changed answer is wrong, nor that possible harm can be ignored.",
            "No new performance tuning, warm-cache rescue, niche selection, cross-family claim, or",
            "state-of-the-art superiority claim follows. The 4B baseline was not tuned against",
            "other speculative strategies. Wall time excludes model loading and tool execution.",
            "The question bootstrap does not remove process-time confounding.",
            "",
            "Protocol: docs/replication-plan.md. Driver: scripts/09_replication.py. All requests,",
            "outputs, runtime/model/protocol hashes, controls and failures are preserved.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "parity", "diagnose", "collect", "report"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--diagnostic", type=int, choices=range(len(DIAGNOSTICS)), default=0)
    parser.add_argument("--block", type=int, choices=range(4), default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "collect":
        verify_seal(args.run)
        checked = json.loads((args.run / "second-model-parity/summary.json").read_text())
        if checked["mismatches"] or checked["checked"] != 200:
            raise ValueError("second-model tokenizer parity failed")
        args.condition = "cold"
        collect(args, model_key=MODEL, protocol_path=PROTOCOL)
    else:
        {"prepare": prepare, "parity": parity, "diagnose": diagnose, "report": report}[args.action](
            args
        )
