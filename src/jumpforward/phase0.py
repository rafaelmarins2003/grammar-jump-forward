"""Reproducible Phase 0 pilot: frozen selection, strict replay, explicit coverage."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from llguidance import LLTokenizer

from .corpus import BFCL_SUBSETS, SEED, adapt_record, digest, read_jsonl, select_records
from .cost import project_cost, quantiles
from .spans import SERIALIZATIONS, characterization, trace_call

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, data) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def provenance() -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("llguidance", "jsonschema", "transformers", "huggingface-hub")
        },
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "git_diff_sha256": digest(git("diff", "HEAD").encode()),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p.read_bytes())
            for p in sorted(
                list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
            )
        },
        "plan_sha256": digest((ROOT / "docs/phase0-plan.md").read_bytes()),
        "engine_config": json.loads((ROOT / "config/llama_cpp.json").read_text()),
        "model_config": json.loads((ROOT / "config/models.json").read_text())["targets"][
            "qwen3-8b-q4_k_m"
        ],
        "seed": SEED,
        "grammar_backend": "llguidance 1.0.1, default native forcing (PL)",
        "scope": "offline reference call bodies, not model-generated server latency",
    }


def verify_inputs(root: Path) -> dict:
    manifest = json.loads((root / "MANIFEST.json").read_text())
    required = {
        f"datasets/bfcl/{prefix}BFCL_v3_{subset}.json"
        for subset in BFCL_SUBSETS
        for prefix in ("", "possible_answer/")
    }
    required.add("datasets/hermes-fc-v1/func-calling-singleturn.json")
    required.add("tokenizers/qwen3/tokenizer.json")
    checked = {}
    for category in ("datasets", "tokenizers"):
        for entry in manifest[category].values():
            for item in entry["files"]:
                if item["path"] not in required:
                    continue
                actual = digest((root / item["path"]).read_bytes())
                if actual != item["sha256"]:
                    raise ValueError(f"input hash mismatch: {item['path']}")
                checked[item["path"]] = actual
    if required != set(checked):
        raise ValueError(f"unmanifested required inputs: {required - set(checked)}")
    return {
        "checked_sha256": checked,
        "raw_manifest_sha256": digest((root / "MANIFEST.json").read_bytes()),
        "tokenizer": manifest["tokenizers"]["qwen3"],
    }


def prepare(args) -> None:
    inputs = verify_inputs(args.raw)
    selected = select_records(args.raw, args.full)
    args.run.mkdir(parents=True, exist_ok=False)
    selection = [{"subset": subset, "id": row["id"]} for subset, row, _ in selected]
    write_json(args.run / "selection.json", selection)
    write_json(
        args.run / "prepare-manifest.json",
        {
            **provenance(),
            **inputs,
            "full_corpus": args.full,
            "n_selected": len(selection),
            "selection_sha256": digest((args.run / "selection.json").read_bytes()),
        },
    )
    counts = Counter()
    with (args.run / "cases.jsonl").open("x") as stream:
        for subset, row, answer in selected:
            try:
                case = adapt_record(subset, row, answer)
            except Exception as exc:  # Preserve every adapter/schema failure, never replace a row.
                case = {
                    "subset": subset,
                    "id": row["id"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            counts[case["status"]] += 1
            stream.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(args.run / "adaptation-summary.json", dict(counts))
    print(json.dumps({"selected": len(selection), **counts}), flush=True)


def measure(args) -> None:
    inputs = verify_inputs(args.raw)
    prepared = json.loads((args.run / "prepare-manifest.json").read_text())
    if inputs["checked_sha256"] != prepared["checked_sha256"]:
        raise ValueError("prepared and current raw inputs differ")
    if digest((args.run / "selection.json").read_bytes()) != prepared["selection_sha256"]:
        raise ValueError("frozen selection changed")
    cases = read_jsonl(args.run / "cases.jsonl")
    selected = json.loads((args.run / "selection.json").read_text())
    if [{"subset": c["subset"], "id": c["id"]} for c in cases] != selected:
        raise ValueError("adapted cases do not match frozen selection")
    curve = json.loads(args.curve.read_text())
    tokenizer = LLTokenizer(str(args.raw / "tokenizers/qwen3/tokenizer.json"), eos_token=151645)
    write_json(
        args.run / "measure-manifest.json",
        {
            **provenance(),
            **inputs,
            "cases_sha256": digest((args.run / "cases.jsonl").read_bytes()),
            "curve_path": str(args.curve.resolve()),
            "curve_sha256": digest(args.curve.read_bytes()),
            "eos_token": tokenizer.eos_token,
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_is_canonical": tokenizer.is_canonical,
            "variants": ["G1", "G2"],
            "serializations": list(SERIALIZATIONS),
            "endpoint": "process every body token for termination logits; exclude EOS ID",
        },
    )
    checks = characterization(tokenizer)
    write_json(args.run / "characterization.json", checks)
    if any(c["status"] != "ok" for c in checks):
        raise RuntimeError("matcher characterization failed; inspect characterization.json")
    with (args.run / "traces.jsonl").open("x") as stream:
        for i, case in enumerate(cases):
            for variant in ("G1", "G2"):
                for serialization in SERIALIZATIONS:
                    key = {
                        "subset": case["subset"],
                        "id": case["id"],
                        "variant": variant,
                        "serialization": serialization,
                    }
                    if case["status"] != "ok":
                        stream.write(
                            json.dumps(
                                {**key, "status": "adaptation_error", "error": case["error"]}
                            )
                            + "\n"
                        )
                        continue
                    for j, call in enumerate(case["calls"]):
                        try:
                            result = trace_call(
                                tokenizer, case["schema"], call, variant, serialization
                            )
                            if result["status"] == "ok":
                                result["cost"] = [
                                    project_cost(result, curve, d) for d in (1024, 4096)
                                ]
                        except Exception as exc:
                            result = {
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        stream.write(
                            json.dumps(
                                {**key, "call_index": j, "n_calls": len(case["calls"]), **result},
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    stream.flush()
            if (i + 1) % 20 == 0 or i + 1 == len(cases):
                print(f"Measured {i + 1}/{len(cases)} questions", flush=True)
    write_json(
        args.run / "measurement-complete.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "n_questions": len(cases),
            "traces_sha256": digest((args.run / "traces.jsonl").read_bytes()),
        },
    )


def question_summary(traces: list[dict]) -> dict:
    first = traces[0]
    base = {k: first[k] for k in ("subset", "id", "variant", "serialization")}
    if any(t["status"] != "ok" for t in traces):
        return {
            **base,
            "status": "error",
            "errors": [t.get("error") for t in traces if t["status"] != "ok"],
        }
    if len(traces) != first["n_calls"]:
        raise ValueError("incomplete trace file: missing call bodies")
    nt = sum(t["n_tokens"] for t in traces)
    nf = sum(t["n_forced"] for t in traces)
    costs = []
    for i, depth in enumerate((1024, 4096)):
        summed = {
            key: sum(t["cost"][i][key] for t in traces)
            for key in ("baseline_ms", "optimistic_ms", "conservative_ms")
        }
        prefill = first["cost"][i]["prefill_ms"]  # Once per question, not once per parallel call.
        zero = summed["baseline_ms"] * (nt - nf) / nt
        costs.append(
            {
                "depth": depth,
                **summed,
                "prefill_ms": prefill,
                "decode_optimistic": summed["baseline_ms"] / summed["optimistic_ms"],
                "decode_conservative": summed["baseline_ms"] / summed["conservative_ms"],
                "cold_optimistic": (prefill + summed["baseline_ms"])
                / (prefill + summed["optimistic_ms"]),
                "cold_conservative": (prefill + summed["baseline_ms"])
                / (prefill + summed["conservative_ms"]),
                "zero_cost_cold_ceiling": (prefill + summed["baseline_ms"]) / (prefill + zero),
            }
        )
    return {
        **base,
        "status": "ok",
        "n_calls": len(traces),
        "n_tokens": nt,
        "n_forced": nf,
        "forced_fraction": nf / nt,
        "injected_byte_fraction": sum(t["n_injected_bytes"] for t in traces)
        / sum(t["n_bytes"] for t in traces),
        "span_lengths": [s["length"] for t in traces for s in t["spans"]],
        "ff_ms": sum(t["ff_ns"] for t in traces) / 1e6,
        "compile_ms": sum(t["compile_ns"] for t in traces) / 1e6,
        "cost": costs,
    }


def aggregate(questions: list[dict]) -> dict:
    ok = [q for q in questions if q["status"] == "ok"]
    costs = {}
    for i, depth in enumerate((1024, 4096)):
        costs[str(depth)] = {
            key: quantiles([q["cost"][i][key] for q in ok])
            for key in (
                "decode_optimistic",
                "decode_conservative",
                "cold_optimistic",
                "cold_conservative",
                "zero_cost_cold_ceiling",
            )
        }
    return {
        "selected": len(questions),
        "complete": len(ok),
        "coverage": len(ok) / len(questions),
        "forced_fraction": quantiles([q["forced_fraction"] for q in ok]),
        "byte_fraction": quantiles([q["injected_byte_fraction"] for q in ok]),
        "n_tokens": quantiles([q["n_tokens"] for q in ok]),
        "span_length_pooled": quantiles([s for q in ok for s in q["span_lengths"]]),
        "ff_ms": quantiles([q["ff_ms"] for q in ok]),
        "compile_ms": quantiles([q["compile_ms"] for q in ok]),
        "cost": costs,
        "errors": Counter(
            e.split(":")[0].split(" at ")[0]
            for q in questions
            if q["status"] != "ok"
            for e in q["errors"]
        ),
    }


def report(args) -> None:
    complete = json.loads((args.run / "measurement-complete.json").read_text())
    if digest((args.run / "traces.jsonl").read_bytes()) != complete["traces_sha256"]:
        raise ValueError("completed trace hash mismatch")
    groups = defaultdict(list)
    for trace in read_jsonl(args.run / "traces.jsonl"):
        groups[(trace["subset"], trace["id"], trace["variant"], trace["serialization"])].append(
            trace
        )
    questions = [question_summary(traces) for traces in groups.values()]
    if len(questions) != complete["n_questions"] * 4:
        raise ValueError("incomplete question/variant grid")
    summary = {}
    for variant in ("G1", "G2"):
        for serialization in SERIALIZATIONS:
            key = f"{variant}/{serialization}"
            matching = [
                q
                for q in questions
                if q["variant"] == variant and q["serialization"] == serialization
            ]
            summary[key] = {
                "overall": aggregate(matching),
                "subsets": {
                    subset: aggregate([q for q in matching if q["subset"] == subset])
                    for subset in (*BFCL_SUBSETS, "hermes")
                },
            }
    write_json(args.run / "questions.json", questions)
    write_json(args.run / "summary.json", summary)
    write_json(
        args.run / "report-manifest.json",
        {
            **provenance(),
            "traces_sha256": complete["traces_sha256"],
            "summary_sha256": digest((args.run / "summary.json").read_bytes()),
        },
    )
    for key, value in summary.items():
        row = value["overall"]
        print(
            key,
            json.dumps({k: row[k] for k in ("selected", "complete", "forced_fraction")}),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "measure", "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument(
        "--curve", type=Path, default=ROOT / "runs/2026-09-12-pass-cost/summary.json"
    )
    parser.add_argument("--full", action="store_true", help="all records instead of frozen pilot")
    args = parser.parse_args()
    {"prepare": prepare, "measure": measure, "report": report}[args.action](args)
