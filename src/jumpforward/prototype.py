"""Frozen held-out evaluation of the minimal native-forcing engine prototype."""

import argparse
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator
from llguidance import LLTokenizer

from .baselines import bootstrap_median, check_output_bytes, grammar_request, memory_snapshot
from .corpus import SEED, digest, read_jsonl
from .cost import quantiles
from .parity import assert_gpu_idle, file_hash, owned_server
from .phase0 import ROOT, provenance, write_json

SUBSETS = ("simple", "multiple", "live_simple", "live_multiple", "hermes")
BLOCKS = (("none", 0), ("grammar", 0), ("grammar", 1), ("none", 1))


def tokenizer():
    return LLTokenizer(str(ROOT / "data/raw/tokenizers/qwen3/tokenizer.json"), eos_token=151645)


def select_holdout(cases, excluded, tok, per_stratum=10):
    selected, eligible_counts = [], {}
    for subset in SUBSETS:
        eligible = [
            c
            for c in cases
            if c["subset"] == subset
            and c["status"] == "ok"
            and len(c["calls"]) == 1
            and (subset, c["id"]) not in excluded
        ]
        ranked = []
        for case in eligible:
            text = json.dumps(case["calls"][0], ensure_ascii=False, separators=(",", ":"))
            ranked.append({**case, "reference_tokens": len(tok.tokenize_str(text))})
        eligible = ranked
        eligible.sort(key=lambda c: (c["reference_tokens"], c["id"]))
        eligible_counts[subset] = len(eligible)
        for stratum in range(4):
            pool = eligible[len(eligible) * stratum // 4 : len(eligible) * (stratum + 1) // 4]
            if len(pool) < per_stratum:
                raise ValueError(f"insufficient frozen stratum {subset}/{stratum}")
            pool.sort(key=lambda c: digest(f"{SEED}:prototype:{subset}:{c['id']}".encode()))
            selected.extend({**c, "length_stratum": stratum} for c in pool[:per_stratum])
    return selected, eligible_counts


def prepare(args):
    source = ROOT / "runs/2026-09-12-full/cases.jsonl"
    excluded_path = ROOT / "runs/2026-09-12-generated/selection.json"
    excluded = {(c["subset"], c["id"]) for c in json.loads(excluded_path.read_text())}
    selected, counts = select_holdout(read_jsonl(source), excluded, tokenizer())
    args.run.mkdir(parents=True, exist_ok=False)
    write_json(args.run / "cases.json", selected)
    write_json(
        args.run / "prepare-manifest.json",
        {
            **provenance(),
            "scope": "held-out generated prototype evaluation; not unseen reference data",
            "cases_sha256": digest((args.run / "cases.json").read_bytes()),
            "source_sha256": digest(source.read_bytes()),
            "excluded_sha256": digest(excluded_path.read_bytes()),
            "protocol_sha256": digest((ROOT / "docs/prototype-plan.md").read_bytes()),
            "eligible_counts": counts,
            "selected": len(selected),
            "blocks": BLOCKS,
        },
    )
    print(f"Frozen {len(selected)} held-out questions: {counts}", flush=True)


def messages_for(case):
    system = (
        "Select the appropriate function and arguments to answer the user's request. "
        'Return exactly one JSON object with keys "name" and "arguments". '
        "Do not include commentary or XML tags. Available functions:\n"
        + json.dumps(case["tools"], ensure_ascii=False)
    )
    return [{"role": "system", "content": system}, *case["messages"]]


def runtime_identity():
    engine = Path.home() / "llama.cpp-jf"
    paths = [engine / "build/bin/llama-server"]
    paths.extend(p for p in (engine / "build/bin").glob("*.so*") if not p.is_symlink())
    return {
        "files_sha256": {str(p.relative_to(engine)): file_hash(p) for p in sorted(paths)},
        "engine_diff_sha256": digest(
            subprocess.check_output(["git", "-C", str(engine), "diff", "HEAD"])
        ),
        "dependency_diff_sha256": digest(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(engine / "build/llguidance/source"),
                    "diff",
                    "HEAD",
                ]
            )
        ),
    }


def seal(args):
    if list(args.run.glob("cold-*")) or list(args.run.glob("repeat-*")):
        raise ValueError("evaluation already started")
    smoke_path = args.smoke_run / "summary.json"
    smoke_summary = json.loads(smoke_path.read_text())
    if not smoke_summary["instrument_pass"]:
        raise ValueError("development smoke checks have not passed")
    prepare_manifest = json.loads((args.run / "prepare-manifest.json").read_text())
    if digest((args.run / "cases.json").read_bytes()) != prepare_manifest["cases_sha256"]:
        raise ValueError("selection changed before sealing")
    write_json(
        args.run / "evaluation-seal.json",
        {
            **provenance(),
            "scope": "prospective immutable engine and evaluation seal",
            "runtime": runtime_identity(),
            "cases_sha256": prepare_manifest["cases_sha256"],
            "initial_protocol_sha256": prepare_manifest["protocol_sha256"],
            "protocol_sha256": digest((ROOT / "docs/prototype-plan.md").read_bytes()),
            "smoke_summary_sha256": digest(smoke_path.read_bytes()),
            "n_predict": 2048,
            "draft_cap": 32,
            "amendment": "raise shared generation ceiling before evaluation; keep frozen IDs",
        },
    )
    print("Sealed runtime and protocol before held-out generation", flush=True)


def validate_response(case, output, tok, cold=True):
    if cold and output["timings"]["cache_n"] != 0:
        raise ValueError("cold request reused a prompt")
    if output.get("truncated") or output.get("stop_type") == "limit":
        raise ValueError("generation limit or truncation")
    check_output_bytes(tok, output)
    parsed = json.loads(output["content"])
    Draft202012Validator(case["schema"]).validate(parsed)
    return parsed


def collect(args):
    assert_gpu_idle()
    manifest = json.loads((args.run / "evaluation-seal.json").read_text())
    if digest((args.run / "cases.json").read_bytes()) != manifest["cases_sha256"]:
        raise ValueError("frozen selection changed")
    if digest((ROOT / "docs/prototype-plan.md").read_bytes()) != manifest["protocol_sha256"]:
        raise ValueError("registered protocol changed")
    if runtime_identity() != manifest["runtime"]:
        raise ValueError("engine changed after the evaluation seal")
    cases = json.loads((args.run / "cases.json").read_text())
    strategy, repetition = BLOCKS[args.block]
    block = args.run / f"{args.condition}-{args.block:02d}-{strategy}"
    runtime = block
    prior = []
    if args.resume:
        if (block / "complete.json").exists():
            raise ValueError("completed block is immutable")
        prior = read_jsonl(block / "outputs.jsonl")
        runtime = block / f"continuation-{len(list(block.glob('continuation-*'))) + 1:02d}"
        runtime.mkdir(exist_ok=False)
        write_json(
            runtime / "resume.json",
            {
                "preserved": len(prior),
                "prior_sha256": digest((block / "outputs.jsonl").read_bytes()),
            },
        )
    else:
        block.mkdir(exist_ok=False)
    done = {(r["subset"], r["id"]) for r in prior}
    if len(done) != len(prior):
        raise ValueError("duplicate prior trials")
    tok = tokenizer()
    with (
        owned_server(runtime, gpu=True, speculation=strategy) as client,
        (block / "outputs.jsonl").open("a" if args.resume else "x") as stream,
    ):
        pid = json.loads((runtime / "server-process.json").read_text())["pid"]
        warm = client.post(
            "/completion",
            json={"prompt": "Say hello.", "n_predict": 16, "cache_prompt": False, "temperature": 0},
        )
        warm.raise_for_status()
        write_json(runtime / "warmup.json", warm.json())
        snapshots = [memory_snapshot(pid)]
        for case in cases if repetition == 0 else cases[::-1]:
            if (case["subset"], case["id"]) in done:
                continue
            row = {
                "subset": case["subset"],
                "id": case["id"],
                "strategy": strategy,
                "repetition": repetition,
                "condition": args.condition,
            }
            try:
                template = client.post("/apply-template", json={"messages": messages_for(case)})
                template.raise_for_status()
                request = {
                    "prompt": template.json()["prompt"],
                    **grammar_request(case["schema"], "S1"),
                    "temperature": 0,
                    "seed": int(SEED),
                    "n_predict": manifest["n_predict"],
                    "cache_prompt": args.condition == "repeat",
                    "return_tokens": True,
                    "stream": False,
                }
                row["request"] = request
                if args.condition == "repeat":
                    prime = client.post("/completion", json={**request, "n_predict": 1})
                    prime.raise_for_status()
                    row["priming"] = prime.json()
                started = time.perf_counter()
                response = client.post("/completion", json=request)
                row["wall_ms"] = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                output = response.json()
                row["response"] = output
                parsed = validate_response(case, output, tok, args.condition == "cold")
                row.update(status="ok", parsed=parsed, reference_equal=parsed == case["calls"][0])
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__, error=str(exc))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            done.add((case["subset"], case["id"]))
            if len(done) % 10 == 0:
                snapshots.append(memory_snapshot(pid))
                print(f"{block.name}: {len(done)}/{len(cases)}", flush=True)
        write_json(runtime / "memory.json", snapshots)
    write_json(
        block / "complete.json",
        {"trials": len(done), "outputs_sha256": digest((block / "outputs.jsonl").read_bytes())},
    )


def upper_binomial(changed, n, alpha=0.05):
    if not n or changed == n:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(70):
        p = (low + high) / 2
        cdf = sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(changed + 1))
        if cdf > alpha:
            low = p
        else:
            high = p
    return (low + high) / 2


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def smoke(args):
    assert_gpu_idle()
    args.run.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((ROOT / "runs/2026-09-12-general-baselines/inputs.json").read_text())
    tests = [
        {
            "id": f"{c['subset']}:{c['id']}",
            "kind": "schema",
            "schema": c["schema"],
            "request": {
                "prompt": c["prompt"],
                **grammar_request(c["schema"], "S1"),
                "n_predict": 512,
            },
        }
        for c in inputs
    ]
    constant = {"const": {"name": "echo", "arguments": {"text": "olá 🙂 " * 40}}}
    optional = {
        "type": "object",
        "properties": {"x": {"enum": ["left", "right"]}, "y": {"type": "string"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    for name, schema in (
        ("unicode_long", constant),
        ("optional", optional),
        ("free", {"type": "string"}),
    ):
        tests.append(
            {
                "id": name,
                "kind": "schema",
                "schema": schema,
                "request": {
                    "prompt": "Return valid JSON.\n",
                    **grammar_request(schema, "S1"),
                    "n_predict": 512,
                },
            }
        )
    tests.extend(
        [
            {
                "id": "limit",
                "kind": "limit",
                "request": {
                    "prompt": "Return JSON.",
                    **grammar_request(constant, "S1"),
                    "n_predict": 1,
                },
            },
            {
                "id": "json_schema_default",
                "kind": "schema",
                "schema": optional,
                "request": {
                    "prompt": "Return valid JSON.",
                    "json_schema": optional,
                    "n_predict": 64,
                },
            },
            {
                "id": "no_grammar",
                "kind": "unconstrained",
                "request": {"prompt": "Say hello.", "n_predict": 8},
            },
            {
                "id": "stop",
                "kind": "stop",
                "request": {
                    "prompt": "Return the text.",
                    "grammar": '%llguidance {}\nstart: "BEGIN END tail"',
                    "stop": ["END"],
                    "n_predict": 64,
                },
            },
        ]
    )
    tests.append(
        {
            "id": "native",
            "kind": "schema",
            "schema": optional,
            "request": {
                "prompt": "Return JSON.",
                "grammar": "root ::= " + json.dumps('{"x":"left"}'),
                "n_predict": 64,
            },
        }
    )
    write_json(args.run / "tests.json", tests)
    results = defaultdict(dict)
    errors = []
    tok = tokenizer()
    engine = Path.home() / "llama.cpp-jf"
    for name, strategy, build in (
        ("original", "none", engine / "build-pre-grammar"),
        ("off", "none", engine / "build"),
        ("on", "grammar", engine / "build"),
    ):
        runtime = args.run / name
        runtime.mkdir()
        with (
            owned_server(runtime, gpu=True, speculation=strategy, build_dir=build) as client,
            (runtime / "outputs.jsonl").open("x") as stream,
        ):
            for test in tests:
                request = {
                    **test["request"],
                    "temperature": 0,
                    "seed": int(SEED),
                    "cache_prompt": False,
                    "return_tokens": True,
                    "stream": False,
                }
                row = {"id": test["id"], "request": request}
                try:
                    response = client.post("/completion", json=request)
                    response.raise_for_status()
                    output = response.json()
                    row["response"] = output
                    if test["kind"] == "schema":
                        row["parsed"] = validate_response(test, output, tok)
                    elif test["kind"] == "limit":
                        if output["tokens_predicted"] != 1 or output["stop_type"] != "limit":
                            raise ValueError("one-token limit violated")
                    elif test["kind"] == "stop":
                        if output["content"] != "BEGIN " or output["stop_type"] != "word":
                            raise ValueError("stop-string boundary violated")
                    else:
                        check_output_bytes(tok, output)
                    if test["id"] in ("native", "no_grammar") and output["timings"].get(
                        "draft_n", 0
                    ):
                        raise ValueError("unsupported grammar unexpectedly drafted")
                    row["status"] = "ok"
                except Exception as exc:
                    row.update(status="error", error=str(exc))
                    errors.append({"arm": name, "id": test["id"], "error": str(exc)})
                results[name][test["id"]] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            print(f"smoke {name}: {len(tests)} requests", flush=True)
    off_differences, on_differences = [], []
    for test in tests:
        original, off, on = [results[a][test["id"]] for a in ("original", "off", "on")]
        if all(r["status"] == "ok" for r in (original, off, on)):
            fields = ("content", "tokens", "stop_type")
            if any(original["response"][f] != off["response"][f] for f in fields):
                off_differences.append(test["id"])
            if off.get("parsed") != on.get("parsed"):
                on_differences.append(test["id"])
    drafted = sum(
        r.get("response", {}).get("timings", {}).get("draft_n", 0) for r in results["on"].values()
    )
    accepted = sum(
        r.get("response", {}).get("timings", {}).get("draft_n_accepted", 0)
        for r in results["on"].values()
    )
    summary = {
        "errors": errors,
        "off_differences": off_differences,
        "on_parsed_differences": on_differences,
        "drafted": drafted,
        "accepted": accepted,
        "instrument_pass": not errors and not off_differences and drafted > 0,
    }
    write_json(args.run / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def analyze_rows(cases, rows):
    expected = {
        (c["subset"], c["id"], strategy, repetition)
        for c in cases
        for strategy, repetition in BLOCKS
    }
    actual = {(r["subset"], r["id"], r["strategy"], r["repetition"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("incomplete, duplicate or replaced evaluation trials")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["subset"], row["id"])].append(row)
    pairs, failures = [], []
    for key, group in groups.items():
        if any(r["status"] != "ok" for r in group):
            failures.extend(
                {
                    "key": key,
                    "strategy": r["strategy"],
                    "repetition": r["repetition"],
                    "error": r.get("error"),
                }
                for r in group
                if r["status"] != "ok"
            )
            continue
        requests = [
            {
                k: r["request"][k]
                for k in ("prompt", "grammar", "n_predict", "cache_prompt", "temperature", "seed")
            }
            for r in group
        ]
        if any(request != requests[0] for request in requests):
            raise ValueError("paired requests differ")
        arms = {}
        for strategy in ("none", "grammar"):
            arm = sorted(
                (r for r in group if r["strategy"] == strategy), key=lambda r: r["repetition"]
            )
            arms[strategy] = {
                "wall_ms": quantiles([r["wall_ms"] for r in arm])["median"],
                "decode_ms": quantiles([r["response"]["timings"]["predicted_ms"] for r in arm])[
                    "median"
                ],
                "prefill_ms": quantiles([r["response"]["timings"]["prompt_ms"] for r in arm])[
                    "median"
                ],
                "generated_tokens": quantiles([r["response"]["tokens_predicted"] for r in arm])[
                    "median"
                ],
                "parsed": [r["parsed"] for r in arm],
                "bytes": [r["response"]["content"] for r in arm],
                "tokens": [r["response"]["tokens"] for r in arm],
                "reference_equal": all(r["reference_equal"] for r in arm),
            }
        base, candidate = arms["none"], arms["grammar"]
        unstable = any(a["parsed"][0] != a["parsed"][1] for a in arms.values())
        pairs.append(
            {
                "subset": key[0],
                "id": key[1],
                "arms": arms,
                "parsed_changed": base["parsed"] != candidate["parsed"],
                "unstable": unstable,
                "bytes_changed": base["bytes"] != candidate["bytes"],
                "tokens_changed": base["tokens"] != candidate["tokens"],
                "wall_speedup": base["wall_ms"] / candidate["wall_ms"],
                "decode_speedup": base["decode_ms"] / candidate["decode_ms"],
            }
        )
    summary = {
        "selected": len(cases),
        "requests": len(rows),
        "complete_pairs": len(pairs),
        "failures": failures,
        "arms": {},
        "wall_speedup": bootstrap_median([p["wall_speedup"] for p in pairs]),
        "decode_speedup": bootstrap_median([p["decode_speedup"] for p in pairs]),
        "subsets": {},
    }
    for strategy in ("none", "grammar"):
        arm_rows = [r for r in rows if r["strategy"] == strategy]
        response_rows = [r for r in arm_rows if "response" in r]
        timed_rows = [r for r in arm_rows if "wall_ms" in r]
        drafts = sum(r["response"]["timings"].get("draft_n", 0) for r in response_rows)
        accepted = sum(r["response"]["timings"].get("draft_n_accepted", 0) for r in response_rows)
        summary["arms"][strategy] = {
            **{
                metric: quantiles([p["arms"][strategy][metric] for p in pairs])
                for metric in ("wall_ms", "decode_ms", "prefill_ms", "generated_tokens")
            },
            "timed_requests_including_failures": len(timed_rows),
            "pooled_p95_wall_ms": percentile([r["wall_ms"] for r in timed_rows], 0.95)
            if timed_rows
            else None,
            "drafted": drafts,
            "accepted": accepted,
            "acceptance": accepted / drafts if drafts else None,
            "verification_steps": sum(
                r["response"]["timings"].get("draft_verification_steps", 0) for r in response_rows
            ),
            "generation_batches_counter_derived": sum(
                r["response"]["tokens_predicted"]
                - 1
                - r["response"]["timings"].get("draft_n_accepted", 0)
                for r in response_rows
            ),
            "chosen_reference_equal": sum(p["arms"][strategy]["reference_equal"] for p in pairs),
        }
    for subset in SUBSETS:
        subset_pairs = [p for p in pairs if p["subset"] == subset]
        summary["subsets"][subset] = {
            "wall_speedup": bootstrap_median([p["wall_speedup"] for p in subset_pairs]),
            "decode_speedup": bootstrap_median([p["decode_speedup"] for p in subset_pairs]),
        }
    changed = sum(p["parsed_changed"] or p["unstable"] for p in pairs)
    summary["quality"] = {
        "potential_regression_questions": changed,
        "one_sided_95_upper": upper_binomial(changed, len(pairs)),
        "parsed_changed": sum(p["parsed_changed"] for p in pairs),
        "unstable": sum(p["unstable"] for p in pairs),
        "same_parsed_changed_bytes": sum(
            not p["parsed_changed"] and p["bytes_changed"] for p in pairs
        ),
        "same_bytes_changed_tokens": sum(
            not p["bytes_changed"] and p["tokens_changed"] for p in pairs
        ),
        "divergent_ids": [
            {
                k: p[k]
                for k in (
                    "subset",
                    "id",
                    "parsed_changed",
                    "unstable",
                    "bytes_changed",
                    "tokens_changed",
                )
            }
            for p in pairs
            if p["parsed_changed"] or p["unstable"] or p["tokens_changed"]
        ],
    }
    failed_questions = {
        strategy: {
            (r["subset"], r["id"])
            for r in rows
            if r["strategy"] == strategy and r["status"] != "ok"
        }
        for strategy in ("none", "grammar")
    }
    summary["failure_questions"] = {
        "both_arms": sorted(failed_questions["none"] & failed_questions["grammar"]),
        "baseline_only": sorted(failed_questions["none"] - failed_questions["grammar"]),
        "candidate_only": sorted(failed_questions["grammar"] - failed_questions["none"]),
    }
    base_p95 = summary["arms"]["none"]["pooled_p95_wall_ms"]
    candidate_p95 = summary["arms"]["grammar"]["pooled_p95_wall_ms"]
    summary["p95_latency_ratio"] = candidate_p95 / base_p95 if base_p95 and candidate_p95 else None
    timing = summary["wall_speedup"]
    criteria = {
        "complete_no_failures": len(pairs) == len(cases) and not failures,
        "median_at_least_1_4": timing["median"] is not None and timing["median"] >= 1.4,
        "ci_lower_above_1": timing["ci95"][0] is not None and timing["ci95"][0] > 1.0,
        "all_subsets_at_least_1": all(
            (s["wall_speedup"]["median"] or 0) >= 1.0 for s in summary["subsets"].values()
        ),
        "p95_at_most_1_10": summary["p95_latency_ratio"] is not None
        and summary["p95_latency_ratio"] <= 1.10,
        "quality_upper_below_2_percent": summary["quality"]["one_sided_95_upper"] < 0.02,
        "actual_grammar_drafts": summary["arms"]["grammar"]["drafted"] > 0,
    }
    summary["criteria"] = criteria
    summary["decision"] = "go" if all(criteria.values()) else "stop"
    return summary, pairs


def report(args):
    seal_manifest = json.loads((args.run / "evaluation-seal.json").read_text())
    cases_path = args.run / "cases.json"
    if digest(cases_path.read_bytes()) != seal_manifest["cases_sha256"]:
        raise ValueError("frozen cases changed")
    if digest((ROOT / "docs/prototype-plan.md").read_bytes()) != seal_manifest["protocol_sha256"]:
        raise ValueError("protocol changed after seal")
    rows, hashes, runtime_hashes = [], {}, set()
    for block, (strategy, _) in enumerate(BLOCKS):
        directory = args.run / f"{args.condition}-{block:02d}-{strategy}"
        path = directory / "outputs.jsonl"
        complete = json.loads((directory / "complete.json").read_text())
        hashes[str(path.relative_to(args.run))] = digest(path.read_bytes())
        if hashes[str(path.relative_to(args.run))] != complete["outputs_sha256"]:
            raise ValueError("completed block hash mismatch")
        block_rows = read_jsonl(path)
        if len(block_rows) != complete["trials"] or any(
            r["condition"] != args.condition for r in block_rows
        ):
            raise ValueError("block completion mismatch")
        rows.extend(block_rows)
        for p in directory.rglob("server-manifest.json"):
            m = json.loads(p.read_text())
            runtime_hashes.add(
                json.dumps(
                    {
                        k: m[k]
                        for k in ("binary_sha256", "runtime_libraries_sha256", "engine_diff_sha256")
                    },
                    sort_keys=True,
                )
            )
    if len(runtime_hashes) != 1:
        raise ValueError("runtime differs across blocks")
    cases = json.loads(cases_path.read_text())
    tok = tokenizer()
    by_id = {(c["subset"], c["id"]): c for c in cases}
    for row in rows:
        if row["status"] == "ok":
            parsed = validate_response(
                by_id[(row["subset"], row["id"])], row["response"], tok, args.condition == "cold"
            )
            if parsed != row["parsed"]:
                raise ValueError("stored parsed result differs")
    summary, pairs = analyze_rows(cases, rows)
    summary["condition"] = args.condition
    summary["selection_by_subset"] = dict(Counter(c["subset"] for c in cases))
    for strategy in ("none", "grammar"):
        snapshots = [
            s
            for p in args.run.glob(f"{args.condition}-*-{strategy}/**/memory.json")
            for s in json.loads(p.read_text())
        ]
        summary["arms"][strategy]["observed_vram_mib"] = max(
            (s["vram_mib"] for s in snapshots if "vram_mib" in s), default=None
        )
    write_json(args.run / f"{args.condition}-summary.json", summary)
    write_json(args.run / f"{args.condition}-pairs.json", pairs)
    markdown = render_report(summary)
    with (args.run / f"{args.condition}-report.md").open("x") as stream:
        stream.write(markdown)
    write_json(
        args.run / f"{args.condition}-report-manifest.json",
        {
            **provenance(),
            "scope": "actual patched engine paired latency and preservation screen",
            "raw_outputs_sha256": hashes,
            "evaluation_seal_sha256": digest((args.run / "evaluation-seal.json").read_bytes()),
            "summary_sha256": digest((args.run / f"{args.condition}-summary.json").read_bytes()),
            "report_sha256": digest(markdown.encode()),
        },
    )
    print(json.dumps(summary), flush=True)


def render_report(s):
    def fmt(value, places=3):
        return "n/a" if value is None else f"{value:.{places}f}"

    def ratio(value):
        return f"{fmt(value['median'])}x [{fmt(value['ci95'][0])}, {fmt(value['ci95'][1])}]"

    lines = [
        "# Measured native-forcing prototype decision",
        "",
        f"Decision: **{s['decision'].upper()}** under the prospective protocol.",
        "",
        f"{s['selected']} frozen questions; {s['requests']} requests; "
        f"{s['complete_pairs']} complete pairs.",
        f"Cache condition: {s['condition']}. Median paired end-to-end speedup: "
        f"**{ratio(s['wall_speedup'])}**.",
        f"Decode-only speedup: {ratio(s['decode_speedup'])}.",
        "",
        "The denominator is the same Qwen3-8B Q4_K_M, patched runtime with the flag off,",
        "and the same compact llguidance G2/S1 grammar. The candidate enables native forced-token",
        "batching through the real speculative verifier, with a 32-token cap and no draft model.",
        "Both arms include target KV computation, grammar/sampling work and prompt processing.",
        "Wall time is local inference, not model loading, tool execution or application latency.",
        "",
        "## Actual timings and work",
        "",
        "| Arm | Median wall ms | Median decode ms | Generated tokens* |"
        " Observed VRAM MiB | Draft accepted/proposed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, a in s["arms"].items():
        lines.append(
            f"| {strategy} | {fmt(a['wall_ms']['median'], 1)} | "
            f"{fmt(a['decode_ms']['median'], 1)} | {fmt(a['generated_tokens']['median'], 1)} | "
            f"{fmt(a['observed_vram_mib'], 0)} | {a['accepted']}/{a['drafted']} |"
        )
    lines.extend(
        [
            "",
            "*Token counts include EOS. VRAM is an observed snapshot maximum, not a peak.",
            f"Candidate/baseline p95 ratio: {fmt(s['p95_latency_ratio'])} (lower is better).",
            "P95 includes every timed request, including failures. Paired medians require both",
            "arms and repetitions to be valid. The completeness gate retains excluded failures.",
            "Generation batch counts are derived as generated tokens - 1 - accepted drafts,",
            "for this single-slot Transformer, not read from a GPU profiler. They exclude prefill",
            "and rely on each verification block fitting in one decode microbatch.",
            "Derived generation batches: "
            f"off={s['arms']['none']['generation_batches_counter_derived']}, "
            f"on={s['arms']['grammar']['generation_batches_counter_derived']}.",
            "",
            "| Subset | Wall speedup [95% bootstrap interval] | Decode speedup [interval] |",
            "|---|---:|---:|",
        ]
    )
    for subset, result in s["subsets"].items():
        lines.append(
            f"| {subset} | {ratio(result['wall_speedup'])} | {ratio(result['decode_speedup'])} |"
        )
    q = s["quality"]
    lines.extend(
        [
            "",
            "## Correctness and pre-registered criteria",
            "",
            f"Failed requests: {len(s['failures'])}. "
            f"Questions with changed parsed output: {q['parsed_changed']}.",
            f"Questions unstable across repetitions: {q['unstable']}. "
            f"Potential-regression rate one-sided 95% upper bound: {q['one_sided_95_upper']:.2%}.",
            f"Same parsed JSON but changed bytes: {q['same_parsed_changed_bytes']}; "
            f"same bytes but changed token IDs: {q['same_bytes_changed_tokens']}.",
            "Every successful response was independently rechecked for schema validity, UTF-8",
            "and token-byte roundtrip. Parsed differences count as potential regressions;",
            "they are not assumed correct or incorrect from a single reference.",
            "",
            "| Criterion | Pass |",
            "|---|---|",
        ]
    )
    lines.extend(f"| {criterion} | {passed} |" for criterion, passed in s["criteria"].items())
    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            "This is an actual batching measurement, not a cost projection. A STOP means this",
            "bounded prototype did not justify the retained general-scope investment under the",
            "registered criteria; it is not a theorem that all future implementations must fail.",
            "Do not lower the target, substitute decode speedup, or seek a favorable niche",
            "after these results. A failed cold gate does not require a warm-cache rescue run.",
            "",
            "The balanced mixture spans five subsets and four reference-length rank strata,",
            "but only single-call bodies on one model/hardware pair. This is not BFCL accuracy,",
            "a deployment distribution, or proof across parallel calls/chat wrappers.",
            "The binomial preservation bound assumes independent question outcomes; benchmark",
            "selection and dependence limit its interpretation. It does not prove zero risk.",
            "Process order was off/on/on/off with reversed questions on repetition one. Remaining",
            "time-block confounding is not removed by the question-bootstrap intervals.",
            "",
            "The llguidance 1.0.1 FFI required a local bounded-copy safety fix. Both measured",
            "arms use that same fixed library; it does not change grammar or tokenization policy.",
            "The shared 2048-token ceiling was sealed before generation because one reference",
            "exceeded the initial ceiling. Selected IDs and the 1.4x target were unchanged.",
            "",
            "## Reproduction",
            "",
            "See docs/prototype-plan.md and scripts/08_prototype.py. The seal fixes the runtime,",
            "dependency diff, protocol and input hashes. Raw outputs, commands, token IDs,",
            "memory snapshots, pair results and hashes are retained beside this report.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect", "smoke", "seal", "report"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--block", type=int, choices=range(4), default=0)
    parser.add_argument("--condition", choices=("cold", "repeat"), default="cold")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke-run", type=Path, default=ROOT / "runs/2026-09-12-prototype-smoke-03"
    )
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "smoke": smoke, "seal": seal, "report": report}[
        args.action
    ](args)
