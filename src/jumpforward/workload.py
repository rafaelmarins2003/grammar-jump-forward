"""Observed-prompt cache assay and transparent retargeting algebra; no batching patch."""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from .corpus import SEED, digest, read_jsonl
from .cost import quantiles
from .parity import owned_server
from .phase0 import ROOT, provenance, write_json

TARGET = 1.5
CONDITIONS = ("cold", "tools", "repeat")
HIT_RATES = (0.0, 0.5, 0.9, 1.0)


def overhead_budget(baseline: float, batched: float, target: float = TARGET) -> float:
    if not all(math.isfinite(v) for v in (baseline, batched, target)):
        raise ValueError("cost inputs must be finite")
    if baseline <= 0 or batched <= 0 or target <= 1:
        raise ValueError("positive costs and target greater than one required")
    return (baseline - target * batched) / (target - 1)


def required_hit_rate(budget: float, cold: float, hit: float) -> float | None:
    """None means impossible even with all requests warm; zero means cold already suffices."""
    if not all(math.isfinite(v) for v in (budget, cold, hit)) or min(cold, hit) < 0:
        raise ValueError("invalid overheads")
    if cold <= budget:
        return 0.0
    if hit > budget or hit >= cold:
        return None
    return (cold - budget) / (cold - hit)


def speedup(baseline: float, batched: float, overhead: float) -> float:
    if baseline <= 0 or batched <= 0 or overhead < 0:
        raise ValueError("invalid costs")
    return (overhead + baseline) / (overhead + batched)


def common_prefix(left: list[int], right: list[int]) -> list[int]:
    n = 0
    while n < min(len(left), len(right)) and left[n] == right[n]:
        n += 1
    return left[:n]


def cache_counts(response: dict) -> tuple[int, int]:
    """Top-level tokens_cached is slot occupancy, not reuse at this pinned engine revision."""
    timing = response["timings"]
    reused, processed = timing["cache_n"], timing["prompt_n"]
    if reused < 0 or processed < 0:
        raise ValueError("negative cache counters")
    return reused, processed


def tokenize(client, text):
    response = client.post(
        "/tokenize", json={"content": text, "add_special": False, "parse_special": True}
    )
    response.raise_for_status()
    return response.json()["tokens"]


def completed_trials(rows):
    keys = [(r["subset"], r["id"], r["repetition"], r["condition"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate cache trials")
    if any(k[2] not in range(3) or k[3] not in CONDITIONS for k in keys):
        raise ValueError("invalid cache trial key")
    return set(keys)


def collect(args):
    manifest = json.loads((args.generated / "summary.json").read_text())
    raw = args.generated / "outputs.jsonl"
    if digest(raw.read_bytes()) != manifest["outputs_sha256"]:
        raise ValueError("generated-output artifact changed")
    rows = [r for r in read_jsonl(raw) if r["variant"] == "G2"]
    selection = json.loads((args.generated / "selection.json").read_text())
    if [{"subset": r["subset"], "id": r["id"]} for r in rows] != selection:
        raise ValueError("selected generated questions missing or reordered")
    done = set()
    runtime_run = args.run
    if args.resume:
        if (args.run / "collection-complete.json").exists():
            raise ValueError("completed collections are immutable")
        previous = json.loads((args.run / "collect-manifest.json").read_text())
        if previous["generated_outputs_sha256"] != digest(raw.read_bytes()):
            raise ValueError("cannot resume with changed input")
        if json.loads((args.run / "selection.json").read_text()) != selection:
            raise ValueError("cannot resume with changed selection")
        done = completed_trials(read_jsonl(args.run / "probes.jsonl"))
        index = len(list(args.run.glob("continuation-*"))) + 1
        runtime_run = args.run / f"continuation-{index:02d}"
        runtime_run.mkdir(exist_ok=False)
        write_json(
            runtime_run / "resume-manifest.json",
            {
                **provenance(),
                "preserved_trials": len(done),
                "preserved_probes_sha256": digest((args.run / "probes.jsonl").read_bytes()),
                "reason": "resume interrupted collection; do not replace any flushed measurement",
            },
        )
    else:
        args.run.mkdir(parents=True, exist_ok=False)
        write_json(args.run / "selection.json", selection)
        write_json(
            args.run / "collect-manifest.json",
            {
                **provenance(),
                "generated_outputs_sha256": digest(raw.read_bytes()),
                "conditions": CONDITIONS,
                "repetitions": 3,
                "predicted_tokens": 1,
                "scope": "prefix-only cache assay, not deployment hit rate or jump-forward",
            },
        )
    mode = "a" if args.resume else "x"
    with (
        owned_server(runtime_run, gpu=True) as client,
        (args.run / "probes.jsonl").open(mode) as out,
    ):
        # Discard only this fixed instrumentation warmup; it is outside the frozen cohort.
        warmup = client.post(
            "/completion",
            json={"prompt": "Warmup", "n_predict": 1, "temperature": 0, "cache_prompt": False},
        )
        warmup.raise_for_status()
        write_json(runtime_run / "warmup.json", warmup.json())
        for i, source in enumerate(rows):
            key = {"subset": source["subset"], "id": source["id"]}
            if source["status"] != "ok":
                out.write(json.dumps({**key, "status": "source_error"}) + "\n")
                continue
            prompt = source["request"]["prompt"]
            marker = "<|im_start|>user\n"
            if marker not in prompt:
                raise ValueError("Qwen user boundary absent; no inferred prefix allowed")
            full = tokenize(client, prompt)
            prefix = common_prefix(full, tokenize(client, prompt.split(marker, 1)[0]))
            if not prefix or len(prefix) >= len(full):
                raise ValueError("invalid shared system/tool prefix")
            if len(full) != source["response"]["timings"]["prompt_n"]:
                raise ValueError("native token count differs from recorded cold prompt")
            for repetition in range(3):
                offset = (i + repetition) % len(CONDITIONS)
                order = CONDITIONS[offset:] + CONDITIONS[:offset]
                for condition in order:
                    if (source["subset"], source["id"], repetition, condition) in done:
                        continue
                    row = {
                        **key,
                        "repetition": repetition,
                        "condition": condition,
                        "prompt_tokens": full,
                        "system_prefix_tokens": prefix,
                    }
                    if condition != "cold":
                        seed = prefix if condition == "tools" else full
                        prime_request = {
                            "prompt": seed,
                            "n_predict": 1,
                            "temperature": 0,
                            "seed": int(SEED),
                            "cache_prompt": False,
                        }
                        started = time.perf_counter()
                        response = client.post("/completion", json=prime_request)
                        response.raise_for_status()
                        row["prime"] = {
                            "request": prime_request,
                            "response": response.json(),
                            "wall_ms": (time.perf_counter() - started) * 1000,
                        }
                    request = {
                        **source["request"],
                        "prompt": full,
                        "n_predict": 1,
                        "cache_prompt": condition != "cold",
                    }
                    started = time.perf_counter()
                    response = client.post("/completion", json=request)
                    wall = (time.perf_counter() - started) * 1000
                    response.raise_for_status()
                    result = response.json()
                    cached, processed = cache_counts(result)
                    row.update(
                        {
                            "request": request,
                            "response": result,
                            "wall_ms": wall,
                            "reused_tokens": cached,
                            "processed_tokens": processed,
                            "status": "ok",
                        }
                    )
                    if cached + processed != len(full):
                        row.update(status="error", error="cache counts do not partition prompt")
                    elif condition == "cold" and cached != 0:
                        row.update(status="error", error="cold request unexpectedly reused cache")
                    elif condition == "tools" and not len(prefix) - 1 <= cached <= len(prefix) + 1:
                        row.update(status="error", error="shared-prefix reuse did not match setup")
                    elif condition == "repeat" and cached < len(full) - 1:
                        row.update(status="error", error="full-repeat cache was not reused")
                    elif result.get("truncated") or result["tokens_predicted"] != 1:
                        row.update(status="error", error="not a one-token, untruncated probe")
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
            print(f"Cache conditions measured for {i + 1}/{len(rows)} questions", flush=True)
    write_json(
        args.run / "collection-complete.json",
        {
            "probes_sha256": digest((args.run / "probes.jsonl").read_bytes()),
            "n_selected": len(rows),
        },
    )


def analyze(args):
    if (args.run / "analysis-manifest.json").exists():
        raise ValueError("completed analyses are immutable")
    completion = json.loads((args.run / "collection-complete.json").read_text())
    probes_path = args.run / "probes.jsonl"
    if digest(probes_path.read_bytes()) != completion["probes_sha256"]:
        raise ValueError("completed probe hash mismatch")
    probes = read_jsonl(probes_path)
    completed_trials(probes)
    collection = json.loads((args.run / "collect-manifest.json").read_text())
    if (
        digest((args.generated / "outputs.jsonl").read_bytes())
        != collection["generated_outputs_sha256"]
    ):
        raise ValueError("generated source changed after collection")
    generated = [r for r in read_jsonl(args.generated / "outputs.jsonl") if r["variant"] == "G2"]
    selected_keys = {(r["subset"], r["id"]) for r in generated}
    if any((r["subset"], r["id"]) not in selected_keys for r in probes):
        raise ValueError("probe question is outside frozen selection")
    grouped = defaultdict(list)
    for row in probes:
        grouped[(row["subset"], row["id"])].append(row)
    questions = []
    for source in generated:
        rows = grouped[(source["subset"], source["id"])]
        q = {"subset": source["subset"], "id": source["id"]}
        if len(rows) != 9 or any(r["status"] != "ok" for r in rows):
            questions.append({**q, "status": "error", "reason": "incomplete valid probe grid"})
            continue
        timing = source["response"]["timings"]
        cost = source["trace"]["cost"][0]
        assert cost["depth"] == 1024
        residual = timing["predicted_ms"] - cost["baseline_ms"]
        q.update(
            {
                "status": "ok",
                "prompt_tokens": len(rows[0]["prompt_tokens"]),
                "body_tokens": source["trace"]["n_tokens"],
                "system_prefix_tokens": len(rows[0]["system_prefix_tokens"]),
                "recorded_cold_wall_ms": source["wall_ms"],
                "recorded_prefill_ms": timing["prompt_ms"],
                "recorded_decode_ms": timing["predicted_ms"],
                "decode_residual_ms": residual,
                "calibration_valid": residual >= 0,
                "prompt_outside_measured_depth_range": not 1024
                <= len(rows[0]["prompt_tokens"])
                <= 4096,
                "probe": {},
                "models": {},
            }
        )
        for condition in CONDITIONS:
            rs = [r for r in rows if r["condition"] == condition]
            q["probe"][condition] = {
                "wall_ms": quantiles([r["wall_ms"] for r in rs]),
                "prefill_ms": quantiles([r["response"]["timings"]["prompt_ms"] for r in rs]),
                "cache_tokens": quantiles([r["reused_tokens"] for r in rs]),
                "uncached_tokens": quantiles([r["processed_tokens"] for r in rs]),
                "prime_wall_ms": quantiles([r["prime"]["wall_ms"] for r in rs if "prime" in r]),
            }
        for model, shared_residual in (("model_only", 0.0), ("residual_preserving", residual)):
            if shared_residual < 0:
                q["models"][model] = {"status": "invalid_negative_residual"}
                continue
            r = {
                "status": "ok",
                "baseline_body_ms": cost["baseline_ms"],
                "batched_body_ms": cost["conservative_ms"],
                "shared_residual_ms": shared_residual,
                "overhead_budget_ms": overhead_budget(cost["baseline_ms"], cost["conservative_ms"]),
                "conditions": {},
                "hit_rates": {},
            }
            b, j = cost["baseline_ms"], cost["conservative_ms"]
            for condition in CONDITIONS:
                overhead = q["probe"][condition]["wall_ms"]["median"] + shared_residual
                r["conditions"][condition] = {
                    "overhead_ms": overhead,
                    "baseline_ms": overhead + b,
                    "batched_ms": overhead + j,
                    "speedup": speedup(b, j, overhead),
                    "hits_target": speedup(b, j, overhead) >= TARGET,
                }
            p = r["conditions"]["cold"]["overhead_ms"]
            for condition in ("tools", "repeat"):
                h = r["conditions"][condition]["overhead_ms"]
                r["hit_rates"][condition] = {
                    "minimum_for_target": required_hit_rate(r["overhead_budget_ms"], p, h),
                    "scenarios": [
                        {
                            "hit_rate": rate,
                            "baseline_ms": (1 - rate) * p + rate * h + b,
                            "batched_ms": (1 - rate) * p + rate * h + j,
                            "speedup": speedup(b, j, (1 - rate) * p + rate * h),
                        }
                        for rate in HIT_RATES
                    ],
                }
            q["models"][model] = r
        questions.append(q)
    ok = [q for q in questions if q["status"] == "ok"]
    if not ok:
        raise ValueError("no complete cache questions; inspect probes")
    summary = {
        "selected": len(questions),
        "complete": len(ok),
        "target": TARGET,
        "n_probes": len(probes),
        "invalid_probes": sum(r["status"] != "ok" for r in probes),
        "scope": "controlled benchmark prompt cache; deployment hit rate unknown",
        "prompt_tokens": quantiles([q["prompt_tokens"] for q in ok]),
        "body_tokens": quantiles([q["body_tokens"] for q in ok]),
        "prefix_fraction": quantiles([q["system_prefix_tokens"] / q["prompt_tokens"] for q in ok]),
        "out_of_depth_range": sum(q["prompt_outside_measured_depth_range"] for q in ok),
        "recorded_cold_wall_ms": quantiles([q["recorded_cold_wall_ms"] for q in ok]),
        "models": {},
        "probe": {},
    }
    for condition in CONDITIONS:
        summary["probe"][condition] = {
            key: quantiles([q["probe"][condition][key]["median"] for q in ok])
            for key in ("wall_ms", "prefill_ms", "cache_tokens", "uncached_tokens")
        }
    for model in ("model_only", "residual_preserving"):
        rows = [q["models"][model] for q in ok if q["models"][model]["status"] == "ok"]
        m = {"complete": len(rows), "conditions": {}, "hit_rates": {}}
        if not rows:
            m["status"] = "no_valid_calibrations"
            summary["models"][model] = m
            continue
        for condition in CONDITIONS:
            rs = [r["conditions"][condition] for r in rows]
            m["conditions"][condition] = {
                "speedup": quantiles([r["speedup"] for r in rs]),
                "fraction_hitting_target": sum(r["hits_target"] for r in rs) / len(rs),
                "ratio_of_total_costs": sum(r["baseline_ms"] for r in rs)
                / sum(r["batched_ms"] for r in rs),
            }
        for condition in ("tools", "repeat"):
            rates = [r["hit_rates"][condition]["minimum_for_target"] for r in rows]
            possible = [r for r in rates if r is not None]
            m["hit_rates"][condition] = {
                "possible": len(possible),
                "impossible": len(rates) - len(possible),
                "minimum_rate_among_possible": quantiles(possible),
                "scenarios": [],
            }
            for i, rate in enumerate(HIT_RATES):
                rs = [r["hit_rates"][condition]["scenarios"][i] for r in rows]
                m["hit_rates"][condition]["scenarios"].append(
                    {
                        "hit_rate": rate,
                        "speedup": quantiles([r["speedup"] for r in rs]),
                        "fraction_hitting_target": sum(r["speedup"] >= TARGET for r in rs)
                        / len(rs),
                        "ratio_of_total_costs": sum(r["baseline_ms"] for r in rs)
                        / sum(r["batched_ms"] for r in rs),
                    }
                )
        summary["models"][model] = m
    for name, data in (("questions.json", questions), ("summary.json", summary)):
        path = args.run / name
        if args.resume and path.exists():
            if json.loads(path.read_text()) != data:
                raise ValueError("partial analysis does not match recomputed artifacts")
        else:
            write_json(path, data)
    write_json(
        args.run / "analysis-manifest.json",
        {
            **provenance(),
            "inputs_sha256": {
                str(p.resolve()): digest(p.read_bytes())
                for p in (
                    probes_path,
                    args.generated / "outputs.jsonl",
                    ROOT / "runs/2026-09-12-pass-cost/summary.json",
                )
            },
            "summary_sha256": digest((args.run / "summary.json").read_bytes()),
        },
    )
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "analyze"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="preserve already flushed trials")
    parser.add_argument("--generated", type=Path, default=ROOT / "runs/2026-09-12-generated")
    args = parser.parse_args()
    {"collect": collect, "analyze": analyze}[args.action](args)
