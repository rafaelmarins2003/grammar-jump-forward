"""Small measured baseline screen, deliberately separate from unimplemented jump-forward."""

import argparse
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator
from llguidance import LLTokenizer

from .corpus import SEED, digest, read_jsonl
from .cost import quantiles
from .parity import owned_server
from .phase0 import ROOT, provenance, write_json
from .spans import make_grammar

STRATEGIES = ("none", "draft-simple", "ngram-simple")


def grammar_request(schema, form):
    if form == "stock":
        return {"json_schema": schema}
    if form not in ("S1", "S2"):
        raise ValueError("invalid form")
    compiled = json.loads(make_grammar(schema, "G2", form))["grammars"][0]["json_schema"]
    return {"grammar": "%llguidance {}\nstart: %json " + json.dumps(compiled, ensure_ascii=False)}


def forms_for(strategy):
    return ("stock", "S1", "S2") if strategy == "none" else ("S1", "S2")


def trial_key(row):
    return (row["subset"], row["id"], row["repetition"], row["form"])


def validate_trial_grid(rows, inputs, strategy):
    expected = {
        (case["subset"], case["id"], repetition, form)
        for case in inputs
        for repetition in range(2)
        for form in forms_for(strategy)
    }
    if (
        len(rows) != len(expected)
        or {trial_key(r) for r in rows} != expected
        or any(r["strategy"] != strategy for r in rows)
    ):
        raise ValueError("incomplete or mismatched trial grid")


def memory_snapshot(pid):
    result = {"pid": pid}
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
        )
        for line in output.splitlines():
            parts = line.split(",")
            if int(parts[0].strip()) == pid:
                result["vram_mib"] = int(parts[1].strip())
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                result["rss_kib"] = int(line.split()[1])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    return result


def prepare(args):
    raw = args.generated / "outputs.jsonl"
    original = json.loads((args.generated / "summary.json").read_text())
    if digest(raw.read_bytes()) != original["outputs_sha256"]:
        raise ValueError("recorded generated outputs changed")
    rows = [r for r in read_jsonl(raw) if r["variant"] == "G2"]
    selection = json.loads((args.generated / "selection.json").read_text())
    if [{"subset": r["subset"], "id": r["id"]} for r in rows] != selection:
        raise ValueError("frozen generated selection mismatch")
    cases = {(r["subset"], r["id"]): r for r in read_jsonl(args.cases)}
    inputs = []
    for row in rows:
        case = cases[(row["subset"], row["id"])]
        inputs.append(
            {
                "subset": row["subset"],
                "id": row["id"],
                "prompt": row["request"]["prompt"],
                "schema": case["schema"],
                "reference": case["calls"][0],
            }
        )
    args.run.mkdir(parents=True, exist_ok=False)
    write_json(args.run / "inputs.json", inputs)
    write_json(
        args.run / "prepare-manifest.json",
        {
            **provenance(),
            "scope": "general-scope exploratory baseline screen; no jump-forward arm",
            "inputs_sha256": digest((args.run / "inputs.json").read_bytes()),
            "source_sha256": {str(p.resolve()): digest(p.read_bytes()) for p in (raw, args.cases)},
            "strategies": STRATEGIES,
            "repetitions": 2,
            "bootstrap_resamples": 5000,
            "cache_prompt": False,
            "strategy_order": STRATEGIES,
        },
    )
    print(f"Frozen {len(inputs)} questions", flush=True)


def collect(args):
    manifest = json.loads((args.run / "prepare-manifest.json").read_text())
    if digest((args.run / "inputs.json").read_bytes()) != manifest["inputs_sha256"]:
        raise ValueError("frozen inputs changed")
    inputs = json.loads((args.run / "inputs.json").read_text())
    arm = args.run / args.strategy
    existing = []
    runtime = arm
    if args.resume:
        if (arm / "complete.json").exists():
            raise ValueError("completed arms are immutable")
        existing = read_jsonl(arm / "outputs.jsonl")
        if len({trial_key(r) for r in existing}) != len(existing):
            raise ValueError("duplicate trial keys")
        runtime = arm / f"continuation-{len(list(arm.glob('continuation-*'))) + 1:02d}"
        runtime.mkdir(exist_ok=False)
        write_json(
            runtime / "resume-manifest.json",
            {
                **provenance(),
                "preserved_trials": len(existing),
                "prior_outputs_sha256": digest((arm / "outputs.jsonl").read_bytes()),
            },
        )
    else:
        arm.mkdir(exist_ok=False)
    done = {trial_key(r) for r in existing}
    forms = forms_for(args.strategy)
    total = len(inputs) * 2 * len(forms)
    mode = "a" if args.resume else "x"
    with (
        owned_server(runtime, gpu=True, speculation=args.strategy) as client,
        (arm / "outputs.jsonl").open(mode) as stream,
    ):
        pid = json.loads((runtime / "server-process.json").read_text())["pid"]
        warmup = client.post(
            "/completion",
            json={
                "prompt": "Write a short greeting.",
                "n_predict": 16,
                "cache_prompt": False,
                "temperature": 0,
                "seed": int(SEED),
            },
        )
        warmup.raise_for_status()
        write_json(runtime / "warmup.json", warmup.json())
        snapshots = [memory_snapshot(pid)]
        for repetition in range(2):
            order = inputs if repetition == 0 else inputs[::-1]
            for i, case in enumerate(order):
                offset = (i + repetition) % len(forms)
                for form in forms[offset:] + forms[:offset]:
                    key = {
                        "subset": case["subset"],
                        "id": case["id"],
                        "repetition": repetition,
                        "form": form,
                    }
                    if trial_key(key) in done:
                        continue
                    request = {
                        "prompt": case["prompt"],
                        **grammar_request(case["schema"], form),
                        "temperature": 0,
                        "seed": int(SEED),
                        "n_predict": 512,
                        "cache_prompt": False,
                        "return_tokens": True,
                        "stream": False,
                    }
                    row = {**key, "strategy": args.strategy, "request": request}
                    started = time.perf_counter()
                    try:
                        response = client.post("/completion", json=request)
                        row["wall_ms"] = (time.perf_counter() - started) * 1000
                        response.raise_for_status()
                        output = response.json()
                        row["response"] = output
                        if output["timings"]["cache_n"] != 0:
                            raise ValueError("cold request reused cache")
                        if output.get("truncated") or output["stop_type"] == "limit":
                            raise ValueError("generation limit or truncation")
                        parsed = json.loads(output["content"])
                        Draft202012Validator(case["schema"]).validate(parsed)
                        row.update(
                            status="ok", parsed=parsed, reference_equal=parsed == case["reference"]
                        )
                    except Exception as exc:
                        row.update(status="error", error_type=type(exc).__name__, error=str(exc))
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    done.add(trial_key(row))
                snapshots.append(memory_snapshot(pid))
                if len(done) % 10 == 0 or len(done) == total:
                    print(f"{args.strategy}: {len(done)}/{total} requests", flush=True)
        write_json(runtime / "memory.json", snapshots)
    write_json(
        arm / "complete.json",
        {"outputs_sha256": digest((arm / "outputs.jsonl").read_bytes()), "trials": len(done)},
    )


def bootstrap_median(values, samples=5000):
    if not values:
        return {"n": 0, "median": None, "ci95": [None, None]}
    rng = random.Random(int(SEED))
    medians = sorted(
        quantiles(rng.choices(values, k=len(values)))["median"] for _ in range(samples)
    )
    return {
        "n": len(values),
        "median": quantiles(values)["median"],
        "ci95": [medians[int(0.025 * samples)], medians[int(0.975 * samples)]],
    }


def paired_comparison(base, candidate):
    shared = sorted(set(base) & set(candidate))
    result = {"pairs": len(shared), "output_differences": 0, "repetition_unstable_pairs": 0}
    equal_pairs = []
    for key in shared:
        a, b = base[key], candidate[key]
        if any(x != a["parsed"][0] for x in a["parsed"]) or any(
            x != b["parsed"][0] for x in b["parsed"]
        ):
            result["repetition_unstable_pairs"] += 1
        if a["parsed"] != b["parsed"]:
            result["output_differences"] += 1
        else:
            equal_pairs.append(key)
    for metric in ("wall_ms", "decode_ms"):
        result[metric] = bootstrap_median([base[k][metric] / candidate[k][metric] for k in shared])
        result["equal_output_" + metric] = bootstrap_median(
            [base[k][metric] / candidate[k][metric] for k in equal_pairs]
        )
    return result


def check_output_bytes(tokenizer, output):
    tokens = list(output["tokens"])
    trailing_eos = bool(tokens and tokens[-1] == tokenizer.eos_token)
    if trailing_eos:
        tokens.pop()
    decoded = tokenizer.decode_bytes(tokens)
    decoded.decode("utf-8", errors="strict")
    if decoded != output["content"].encode("utf-8", errors="strict"):
        raise ValueError("output token byte roundtrip mismatch")
    return trailing_eos, len(tokens)


def audit(args):
    """Post-collection byte/schema verification, without modifying completed results."""
    tokenizer = LLTokenizer(
        str(ROOT / "data/raw/tokenizers/qwen3/tokenizer.json"), eos_token=151645
    )
    inputs = json.loads((args.run / "inputs.json").read_text())
    cases = {(c["subset"], c["id"]): c for c in inputs}
    prepare_manifest = json.loads((args.run / "prepare-manifest.json").read_text())
    if digest((args.run / "inputs.json").read_bytes()) != prepare_manifest["inputs_sha256"]:
        raise ValueError("frozen inputs changed")
    result = {
        **provenance(),
        "scope": "post-collection UTF-8, token-byte roundtrip and independent schema audit",
        "token_count_clarification": (
            "The baseline report column 'Body tokens' uses server tokens_predicted, which includes "
            "the terminal EOS. It is a generated-token count, not EOS-stripped body length. "
            "This label clarification does not change any latency or speedup statistic."
        ),
        "arms": {},
    }
    for strategy in STRATEGIES:
        path = args.run / strategy / "outputs.jsonl"
        completion = json.loads((path.parent / "complete.json").read_text())
        if digest(path.read_bytes()) != completion["outputs_sha256"]:
            raise ValueError("completed arm hash mismatch")
        rows = read_jsonl(path)
        validate_trial_grid(rows, inputs, strategy)
        arm = {"checked": len(rows), "terminal_eos_removed": 0, "failures": []}
        for row in rows:
            try:
                output = row["response"]
                trailing_eos, _ = check_output_bytes(tokenizer, output)
                arm["terminal_eos_removed"] += int(trailing_eos)
                parsed = json.loads(output["content"])
                Draft202012Validator(cases[(row["subset"], row["id"])]["schema"]).validate(parsed)
                if parsed != row["parsed"]:
                    raise ValueError("recorded parsed output mismatch")
            except Exception as exc:
                arm["failures"].append({"key": trial_key(row), "error": str(exc)})
        result["arms"][strategy] = {**arm, "outputs_sha256": digest(path.read_bytes())}
    write_json(args.run / "byte-audit.json", result)
    print(json.dumps(result["arms"]), flush=True)


def report(args):
    manifest = json.loads((args.run / "prepare-manifest.json").read_text())
    if digest((args.run / "inputs.json").read_bytes()) != manifest["inputs_sha256"]:
        raise ValueError("frozen inputs changed")
    inputs = json.loads((args.run / "inputs.json").read_text())
    n = len(inputs)
    groups = defaultdict(list)
    all_rows = []
    for strategy in STRATEGIES:
        arm = args.run / strategy
        complete = json.loads((arm / "complete.json").read_text())
        if digest((arm / "outputs.jsonl").read_bytes()) != complete["outputs_sha256"]:
            raise ValueError("completed arm hash mismatch")
        rows = read_jsonl(arm / "outputs.jsonl")
        validate_trial_grid(rows, inputs, strategy)
        if complete["trials"] != len(rows):
            raise ValueError("completion count mismatch")
        all_rows.extend(rows)
        for row in rows:
            groups[(strategy, row["form"], row["subset"], row["id"])].append(row)
    aggregate = defaultdict(dict)
    summary = {
        "selected_questions": n,
        "measured_requests": len(all_rows),
        "arms": {},
        "comparisons": {},
    }
    for (strategy, form, subset, ident), rows in groups.items():
        rows.sort(key=lambda r: r["repetition"])
        if any(r["status"] != "ok" for r in rows):
            continue
        aggregate[f"{strategy}/{form}"][(subset, ident)] = {
            "wall_ms": quantiles([r["wall_ms"] for r in rows])["median"],
            "decode_ms": quantiles([r["response"]["timings"]["predicted_ms"] for r in rows])[
                "median"
            ],
            "prefill_ms": quantiles([r["response"]["timings"]["prompt_ms"] for r in rows])[
                "median"
            ],
            "tokens": quantiles([r["response"]["tokens_predicted"] for r in rows])["median"],
            "parsed": [r["parsed"] for r in rows],
            "reference_equal": all(r["reference_equal"] for r in rows),
        }
    for key in (f"{s}/{f}" for s in STRATEGIES for f in forms_for(s)):
        values = aggregate[key]
        strategy, form = key.split("/")
        arm_rows = [r for r in all_rows if r["strategy"] == strategy and r["form"] == form]
        memory_files = list((args.run / strategy).rglob("memory.json"))
        memory = [s for p in memory_files for s in json.loads(p.read_text())]
        draft = sum(r.get("response", {}).get("timings", {}).get("draft_n", 0) for r in arm_rows)
        accepted = sum(
            r.get("response", {}).get("timings", {}).get("draft_n_accepted", 0) for r in arm_rows
        )
        summary["arms"][key] = {
            "complete_questions": len(values),
            "selected_questions": n,
            **{
                metric: quantiles([v[metric] for v in values.values()])
                for metric in ("wall_ms", "decode_ms", "prefill_ms", "tokens")
            },
            "chosen_reference_equal": sum(v["reference_equal"] for v in values.values()),
            "draft_tokens": draft,
            "accepted_draft_tokens": accepted,
            "acceptance": accepted / draft if draft else None,
            "max_observed_vram_mib": max(
                (s["vram_mib"] for s in memory if "vram_mib" in s), default=None
            ),
            "max_observed_rss_kib": max(
                (s["rss_kib"] for s in memory if "rss_kib" in s), default=None
            ),
            "failures": [
                {k: r.get(k) for k in ("subset", "id", "repetition", "error")}
                for r in arm_rows
                if r["status"] != "ok"
            ],
        }
    comparisons = [("none/S2", "none/S1"), ("none/stock", "none/S1"), ("none/stock", "none/S2")]
    comparisons.extend(
        (f"none/{form}", f"{strategy}/{form}")
        for form in ("S1", "S2")
        for strategy in ("draft-simple", "ngram-simple")
    )
    for base, candidate in comparisons:
        summary["comparisons"][f"{base} → {candidate}"] = paired_comparison(
            aggregate[base], aggregate[candidate]
        )
    write_json(args.run / "summary.json", summary)
    with (args.run / "baseline-report.md").open("x") as stream:
        stream.write(render_report(summary))
    write_json(
        args.run / "report-manifest.json",
        {
            **provenance(),
            "frozen_inputs_sha256": manifest["inputs_sha256"],
            "summary_sha256": digest((args.run / "summary.json").read_bytes()),
            "report_sha256": digest((args.run / "baseline-report.md").read_bytes()),
            "raw_outputs_sha256": {
                strategy: digest((args.run / strategy / "outputs.jsonl").read_bytes())
                for strategy in STRATEGIES
            },
            "grammar_backend": "default-schema llguidance stock; llguidance G2/S1 and G2/S2",
            "scope": "observed existing baselines only; no implemented jump-forward speedup",
        },
    )
    print(json.dumps(summary), flush=True)


def render_report(summary):
    def number(value, places=1):
        return "n/a" if value is None else f"{value:.{places}f}"

    def ratio(value):
        if value["median"] is None:
            return "n/a"
        low, high = value["ci95"]
        return f"{value['median']:.3f}x [{low:.3f}, {high:.3f}]"

    lines = [
        "# General-scope existing-baseline screen",
        "",
        "**This is a measured comparison of existing configurations, not a jump-forward result.**",
        "The earlier approximate 1.4x was a projection against no-draft llguidance with equal",
        "serialization/cache assumptions. It was not measured against an unconstrained model or",
        "the best existing alternative. The original Gate 0 remains unpassed.",
        "",
        f"Frozen cohort: {summary['selected_questions']} questions from five benchmark subsets;",
        f"{summary['measured_requests']} requests, two repetitions per question/configuration.",
        "Qwen3-8B Q4_K_M on the RTX 2080 Ti, temperature zero, thinking off, cold cache, one slot.",
        "All candidate tools are included in the schema; no functions are executed.",
        "",
        "## Measured configurations",
        "",
        "Wall time is local HTTP request latency including prefill, not model loading or tool",
        "execution. Decode is the server's predicted_ms. Each question is reduced to the median",
        "of its two repetitions before aggregation. Token counts help expose length confounding.",
        "",
        "| Configuration | Complete questions | Wall ms | Decode ms | Body tokens | VRAM MiB* |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, arm in summary["arms"].items():
        lines.append(
            f"| {key} | {arm['complete_questions']}/{arm['selected_questions']} | "
            f"{number(arm['wall_ms']['median'])} | {number(arm['decode_ms']['median'])} | "
            f"{number(arm['tokens']['median'])} | {number(arm['max_observed_vram_mib'], 0)} |"
        )
    lines.extend(
        [
            "",
            "Stock sends json_schema, converted to default-schema llguidance in this build.",
            "S1 uses llguidance fixed-order compact JSON; S2 uses fixed-order spaced JSON.",
            "draft-simple uses the pinned Qwen3-0.6B Q8_0, GPU offload, max draft 3, p-min 0.75.",
            "ngram-simple retains its size_m cap of 48; --spec-draft-n-max 8 does not set it.",
            "These are untuned configurations, not a search over",
            "every existing speculative method. *VRAM is the maximum observed process snapshot,",
            "not a profiled peak; it includes the target and any draft model.",
            "",
            "## Paired speedups",
            "",
            "Each row is baseline time / candidate time, then the median across paired questions.",
            "Above 1 means the candidate was faster. Brackets are question-bootstrap 95% intervals",
            "(5000 resamples, seed 20260912). They describe this cohort, not general deployment.",
            "",
            "| Baseline → candidate | Wall speedup [CI] | Decode speedup [CI] |"
            " Parsed differences |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, result in summary["comparisons"].items():
        lines.append(
            f"| {key} | {ratio(result['wall_ms'])} | {ratio(result['decode_ms'])} | "
            f"{result['output_differences']}/{result['pairs']} |"
        )
    lines.extend(
        [
            "",
            "## Output and validity checks",
            "",
            "Parsed differences are against the paired baseline, not an official quality score.",
            "Chosen-reference equality is descriptive: datasets can allow other correct answers.",
            "No lossless or quality non-inferiority claim follows from schema validity or this",
            "small cohort. Equal-output-only sensitivity is post-hoc and can have selection bias.",
            "",
            "| Configuration | Failed requests | Chosen reference, both repetitions |"
            " Draft accepted/proposed |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, arm in summary["arms"].items():
        acceptance = (
            f"{arm['accepted_draft_tokens']}/{arm['draft_tokens']} ({arm['acceptance']:.1%})"
            if arm["draft_tokens"]
            else "n/a"
        )
        lines.append(
            f"| {key} | {len(arm['failures'])} | "
            f"{arm['chosen_reference_equal']}/{arm['selected_questions']} | {acceptance} |"
        )
    lines.extend(
        [
            "",
            "| Baseline → candidate | Equal-output pairs | Wall speedup [CI], equal outputs only |"
            " Unstable across repetitions |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, result in summary["comparisons"].items():
        lines.append(
            f"| {key} | {result['equal_output_wall_ms']['n']} | "
            f"{ratio(result['equal_output_wall_ms'])} | {result['repetition_unstable_pairs']} |"
        )
    lines.extend(
        [
            "",
            "## Limits and decision boundary",
            "",
            "The user retains general tool-calling scope. An application niche is not required.",
            "This selection is an exploratory instrument screen of short single-call bodies, not",
            "a representative test of all structured generation. Parallel calls, chat wrappers,",
            "other models/hardware, concurrency and real cache hit rates remain untested.",
            "Strategy processes ran in separate fixed-order time blocks; grammar order rotated and",
            "the second repetition reversed question order. Bootstrap intervals do not remove this",
            "time-block confounding. Prompt reuse was checked with timings.cache_n == 0.",
            "",
            "These measurements identify baselines for an actual prototype; they do not measure",
            "jump-forward, establish its incremental value, or authorize an engine go.",
            "Do not multiply draft and hypothetical jump-forward speedups: these mechanisms",
            "can accelerate overlapping tokens and their composition has not been implemented.",
            "No more cache assumptions or favorable niches can replace that missing measurement.",
            "",
            "## Artifacts and implementation audit",
            "",
            "Sibling summary.json contains statistics; report-manifest.json hashes this report,",
            "summary and raw outputs. inputs.json fixes prompts and schemas. Each strategy",
            "directory retains exact server command/binary hash, requests/responses, token",
            "IDs, warmup, counters, memory snapshots and logs. Completed runs are not overwritten.",
            "The shared provenance helper's inherited offline scope label in older server metadata",
            "does not describe this assay: exact requests and this report's explicit scope do.",
            "",
            "At pinned engine 95ef7fc16054e63b427a3ef00188e055ef7586d8,",
            "common/json-schema-to-grammar.cpp converts json_schema to %llguidance unless",
            "force_gbnf is true. Responses confirm this prefix. Stock is therefore not native",
            "GBNF. See docs/notes/baseline-audit-corrections.md for the historical report errata.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect", "report", "audit"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, default="none")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generated", type=Path, default=ROOT / "runs/2026-09-12-generated")
    parser.add_argument("--cases", type=Path, default=ROOT / "runs/2026-09-12-pilot/cases.jsonl")
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "report": report, "audit": audit}[args.action](args)
