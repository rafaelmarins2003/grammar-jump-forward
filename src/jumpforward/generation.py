"""Bounded S3 instrument check; not a quality non-inferiority or batching experiment."""

import argparse
import json
import time
from pathlib import Path

from jsonschema import Draft202012Validator
from llguidance import LLTokenizer

from .corpus import SEED, digest, read_jsonl
from .cost import project_cost, quantiles
from .parity import owned_server
from .phase0 import ROOT, provenance, write_json
from .spans import make_grammar, trace_text


def server_grammar(schema, variant):
    spec = json.loads(make_grammar(schema, variant, "S2"))
    schema_with_options = spec["grammars"][0]["json_schema"]
    return "%llguidance {}\nstart: %json " + json.dumps(schema_with_options, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    cases = read_jsonl(args.cases)
    selected = []
    for subset in ("simple", "multiple", "live_simple", "live_multiple", "hermes"):
        eligible = [
            c
            for c in cases
            if c["subset"] == subset and c["status"] == "ok" and len(c["calls"]) == 1
        ]
        if len(eligible) < 5:
            raise ValueError(f"insufficient preregistered eligible questions: {subset}")
        selected.extend(eligible[:5])
    args.run.mkdir(parents=True, exist_ok=False)
    write_json(
        args.run / "selection.json", [{"subset": c["subset"], "id": c["id"]} for c in selected]
    )
    write_json(
        args.run / "manifest.json",
        {
            **provenance(),
            "cases_sha256": digest(args.cases.read_bytes()),
            "selection_sha256": digest((args.run / "selection.json").read_bytes()),
            "scope": "S3 raw-token replay and descriptive whitespace comparison; no batching",
            "temperature": 0,
            "seed": int(SEED),
            "max_generated_tokens": 512,
            "cache_prompt": False,
            "order": "G1 first on even question indices, G2 first on odd",
        },
    )
    tokenizer = LLTokenizer(
        str(ROOT / "data/raw/tokenizers/qwen3/tokenizer.json"), eos_token=151645
    )
    curve = json.loads((ROOT / "runs/2026-09-12-pass-cost/summary.json").read_text())
    results = []
    with owned_server(args.run, gpu=True) as client, (args.run / "outputs.jsonl").open("x") as out:
        for i, case in enumerate(selected):
            system = (
                "Select the appropriate function and arguments to answer the user's request. "
                'Return exactly one JSON object with keys "name" and "arguments". '
                "Do not include commentary or XML tags. Available functions:\n"
                + json.dumps(case["tools"], ensure_ascii=False)
            )
            messages = [{"role": "system", "content": system}, *case["messages"]]
            response = client.post("/apply-template", json={"messages": messages})
            response.raise_for_status()
            prompt = response.json()["prompt"]
            for variant in ("G1", "G2") if i % 2 == 0 else ("G2", "G1"):
                grammar = server_grammar(case["schema"], variant)
                request = {
                    "prompt": prompt,
                    "grammar": grammar,
                    "temperature": 0,
                    "seed": int(SEED),
                    "n_predict": 512,
                    "cache_prompt": False,
                    "return_tokens": True,
                    "stream": False,
                }
                row = {
                    "subset": case["subset"],
                    "id": case["id"],
                    "variant": variant,
                    "request": request,
                    "grammar_sha256": digest(grammar.encode()),
                }
                started = time.perf_counter()
                try:
                    response = client.post("/completion", json=request)
                    row["wall_ms"] = (time.perf_counter() - started) * 1000
                    response.raise_for_status()
                    result = response.json()
                    row["response"] = result
                    if result.get("truncated") or result.get("stop_type") == "limit":
                        raise ValueError("generation hit a limit")
                    text = result["content"]
                    parsed = json.loads(text)
                    Draft202012Validator(case["schema"]).validate(parsed)
                    row["json_schema_valid"] = True
                    row["parsed"] = parsed
                    row["chosen_reference_equal"] = parsed == case["calls"][0]
                    tokens = list(result["tokens"])
                    row["trailing_eos_removed"] = bool(tokens and tokens[-1] == tokenizer.eos_token)
                    if row["trailing_eos_removed"]:
                        tokens.pop()
                    row["whole_text_canonical_ids_equal"] = tokens == tokenizer.tokenize_str(text)
                    trace = trace_text(tokenizer, grammar, text, token_ids=tokens)
                    trace["cost"] = [project_cost(trace, curve, depth) for depth in (1024, 4096)]
                    row["trace"] = trace
                    row["status"] = "ok"
                except Exception as exc:
                    row["status"] = "error"
                    row["error_type"] = type(exc).__name__
                    row["error"] = str(exc)
                results.append(row)
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                out.flush()
            print(f"Generated paired question {i + 1}/{len(selected)}", flush=True)
    summary = {"n_questions": len(selected), "n_outputs": len(results), "arms": {}}
    for variant in ("G1", "G2"):
        rows = [r for r in results if r["variant"] == variant]
        ok = [r for r in rows if r["status"] == "ok"]
        summary["arms"][variant] = {
            "selected": len(rows),
            "replay_ok": len(ok),
            "json_schema_valid": sum(r.get("json_schema_valid", False) for r in rows),
            "chosen_reference_equal": sum(r.get("chosen_reference_equal", False) for r in rows),
            "noncanonical_tokenizations": sum(
                not r.get("whole_text_canonical_ids_equal", True) for r in rows
            ),
            "forced_fraction": quantiles([r["trace"]["forced_fraction"] for r in ok]),
            "errors": [{"id": r["id"], "error": r["error"]} for r in rows if r["status"] != "ok"],
        }
    paired = []
    for case in selected:
        pair = [r for r in results if r["subset"] == case["subset"] and r["id"] == case["id"]]
        if all("parsed" in r for r in pair):
            paired.append(
                {
                    "subset": case["subset"],
                    "id": case["id"],
                    "parsed_equal": pair[0]["parsed"] == pair[1]["parsed"],
                }
            )
    summary["valid_pairs"] = len(paired)
    summary["parsed_output_differences"] = sum(not p["parsed_equal"] for p in paired)
    summary["pairs"] = paired
    summary["outputs_sha256"] = digest((args.run / "outputs.jsonl").read_bytes())
    write_json(args.run / "summary.json", summary)
    print(json.dumps(summary), flush=True)
