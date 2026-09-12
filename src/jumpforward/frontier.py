"""Retargeting report: measured cache overhead plus full-cohort sensitivity, not a new gate."""

# Markdown prose and generated table rows are kept readable.
# ruff: noqa: E501

import argparse
import json
import os
from pathlib import Path

from .corpus import digest
from .cost import quantiles
from .phase0 import ROOT, provenance, write_json
from .workload import HIT_RATES, overhead_budget, required_hit_rate, speedup


def length_bin(tokens):
    for limit, label in ((32, "1–32"), (64, "33–64"), (128, "65–128"), (256, "129–256")):
        if tokens <= limit:
            return label
    return ">256"


def reference_frontier(questions):
    ok = [q for q in questions if q["status"] == "ok"]
    keyed = {(q["subset"], q["id"], q["variant"], q["serialization"]): q for q in ok}
    comparisons = (
        ("G1/S1 → G1/S1", "G1", "S1", "S1"),
        ("G1/S1 → G2/S1", "G2", "S1", "S1"),
        ("G1/S2 → G2/S2", "G2", "S2", "S2"),
        ("G1/S1 → G2/S2", "G2", "S2", "S1"),
    )
    frontier = []
    for name, variant, serial, baseline_serial in comparisons:
        matching = [q for q in ok if q["variant"] == variant and q["serialization"] == serial]
        for index, depth in enumerate((1024, 4096)):
            rows = []
            for q in matching:
                base = keyed.get((q["subset"], q["id"], "G1", baseline_serial))
                compact = keyed.get((q["subset"], q["id"], "G1", "S1"))
                if base is None or compact is None:
                    continue
                b = base["cost"][index]["baseline_ms"]
                j = q["cost"][index]["conservative_ms"]
                p = q["cost"][index]["prefill_ms"]
                budget = overhead_budget(b, j)
                rows.append(
                    {
                        "subset": q["subset"],
                        "id": q["id"],
                        "compact_body_tokens": compact["n_tokens"],
                        "length_bin": length_bin(compact["n_tokens"]),
                        "forced_fraction": q["forced_fraction"],
                        "overhead_budget_ms": budget,
                        "ideal_cached_speedup": b / j,
                        "cold_speedup": speedup(b, j, p),
                        "min_hit_rate_if_hit_cost_zero": required_hit_rate(budget, p, 0),
                        "scenarios": [speedup(b, j, (1 - h) * p) for h in HIT_RATES],
                    }
                )
            fractions = [
                r["min_hit_rate_if_hit_cost_zero"]
                for r in rows
                if r["min_hit_rate_if_hit_cost_zero"] is not None
            ]
            summary = {
                "comparison": name,
                "depth": depth,
                "complete": len(rows),
                "ideal_cached_speedup": quantiles([r["ideal_cached_speedup"] for r in rows]),
                "cold_speedup": quantiles([r["cold_speedup"] for r in rows]),
                "shared_overhead_budget_ms": quantiles([r["overhead_budget_ms"] for r in rows]),
                "possible_even_with_zero_hit_cost": len(fractions),
                "min_hit_rate_among_possible": quantiles(fractions),
                "length_bins": {},
                "scenarios": [
                    {"hit_rate": h, "speedup": quantiles([r["scenarios"][i] for r in rows])}
                    for i, h in enumerate(HIT_RATES)
                ],
                "rows": rows,
            }
            for label in ("1–32", "33–64", "65–128", "129–256", ">256"):
                rs = [r for r in rows if r["length_bin"] == label]
                summary["length_bins"][label] = {
                    "questions": len(rs),
                    "forced_fraction": quantiles([r["forced_fraction"] for r in rs]),
                    "ideal_cached_speedup": quantiles([r["ideal_cached_speedup"] for r in rs]),
                    "cold_speedup": quantiles([r["cold_speedup"] for r in rs]),
                    "shared_overhead_budget_ms": quantiles([r["overhead_budget_ms"] for r in rs]),
                }
            frontier.append(summary)
    return frontier


def render(cache, frontier, resumed, manifest_link):
    def pct(value):
        return f"{100 * value:.1f}%"

    def ratio(value):
        return f"{value:.2f}×"

    lines = [
        "# Retargeting follow-up — prompts, cache and output length",
        "",
        "Recommendation: keep **retarget**. Cache creates a conditional opportunity, not an engine go.",
        "The most important next input is a concrete application workload, not another broad implementation.",
        "",
        "## What was actually measured",
        "",
        f"The recorded generated cohort supplied {cache['selected']} questions. The controlled assay completed "
        f"{cache['n_probes']} one-token probes, with {cache['invalid_probes']} validation failures and "
        f"{cache['complete']} complete questions. All candidate tool schemas and the existing G2 grammar were retained.",
        "The engine remains unmodified. Each condition has three repetitions; reported latencies below",
        "are medians across questions of within-question repetition medians. This is not live traffic,",
        "a production cache-hit estimate, or a jump-forward latency benchmark.",
        "",
        f"Prompt tokens: median {cache['prompt_tokens']['median']:.0f}, p10–p90 "
        f"{cache['prompt_tokens']['p10']:.0f}–{cache['prompt_tokens']['p90']:.0f}. "
        f"Generated body tokens: median {cache['body_tokens']['median']:.0f}, p10–p90 "
        f"{cache['body_tokens']['p10']:.0f}–{cache['body_tokens']['p90']:.0f}. "
        f"The system/tool prefix occupied a median {pct(cache['prefix_fraction']['median'])} of the prompt.",
        "",
        "| Condition | Measured request-prefix wall ms | Prefill ms | Reused tokens | Newly processed tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    names = {
        "cold": "Cold prompt",
        "tools": "Warm system/tools, user suffix uncached",
        "repeat": "Exact prompt repeat (best-case control)",
    }
    for condition, row in cache["probe"].items():
        lines.append(
            f"| {names[condition]} | {row['wall_ms']['median']:.2f} | "
            f"{row['prefill_ms']['median']:.2f} | {row['cache_tokens']['median']:.0f} | "
            f"{row['uncached_tokens']['median']:.0f} |"
        )
    lines.extend(
        [
            "",
            "Reused tokens come from `timings.cache_n`. At this pinned engine, top-level",
            "`tokens_cached` is populated from the slot after generation, so it is not the number",
            "reused by the request. The counters were checked against the exact native prompt token IDs.",
            "All measured warm conditions were explicitly primed; those preparation requests and costs",
            "are retained in `probes.jsonl`. They are not free. A real request stream must pay misses and",
            "may lose prefixes to eviction, slot scheduling, tool-schema changes or changed earlier text.",
            "Exact prompt repetition is a boundary condition, not an assumed normal application.",
            "",
            f"The collection was resumed {resumed} time(s) after an interruption, keeping all flushed trials. "
            "Warm trials recreate their own prefix state. Continuation manifests preserve the prior-prefix file hash.",
            "",
            "## Projected batching after measured prefix cost",
            "",
            "Both models use the conservative existing synthetic batch proxy at depth 1024. The model-only",
            "case combines that body cost with measured request-prefix wall time. The residual-preserving",
            "case additionally retains, in both arms, the difference between the earlier observed complete",
            "decode time and synthetic sequential body cost. It does not pretend that all decode overhead",
            "will be accelerated. The residual is a mixed timing discrepancy, not an attributed CPU cost.",
            "",
            f"{cache['out_of_depth_range']} of {cache['complete']} prompt lengths were outside the measured "
            "1024–4096 depth range (below the lower anchor). The model curve is therefore an approximation,",
            "not a calibrated measurement at every actual depth. Thermal/run-to-run differences, grammar",
            "and sampling costs, and altered numerical execution remain unisolated. Negative residuals",
            "would invalidate that calibration, not be clamped away.",
            "",
            "| Assumption | Cache condition | Median ratio | Questions reaching 1.5× | Ratio of total costs |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model, m in cache["models"].items():
        for condition, r in m["conditions"].items():
            lines.append(
                f"| {model} | {condition} | {ratio(r['speedup']['median'])} | "
                f"{pct(r['fraction_hitting_target'])} | {ratio(r['ratio_of_total_costs'])} |"
            )
    lines.extend(
        [
            "",
            "The reference is the same-format unbatched G2/S2 baseline. These are hypothetical",
            "body-cost replacements after an observed prefix, not measured patched/unpatched ratios.",
            "A ratio of total costs weights long requests more heavily and must not replace the median",
            "per-question target merely because it looks better.",
            "",
            "## How much cache would be needed?",
            "",
            "For baseline body cost B, batched cost J, target s and shared overhead O:",
            "",
            "`(O+B)/(O+J) >= s` requires `O <= (B-s*J)/(s-1)`.",
            "",
            "Negative overhead budgets mean batching cannot reach the target even with a free prefix.",
            "For hit rate h, cold/hit prefix costs P/H, the mixture is `(1-h)*P+h*H`. Hit rate is the",
            "fraction of requests with a warm prefix, not the fraction of tokens reused. The residual",
            "is retained in both P and H where applicable. Scenarios below are hypothetical:",
            "",
            "| Assumption | Warm condition | 0% hits | 50% hits | 90% hits | 100% hits | Possible questions / total |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, m in cache["models"].items():
        for condition, r in m["hit_rates"].items():
            values = " | ".join(ratio(x["speedup"]["median"]) for x in r["scenarios"])
            lines.append(f"| {model} | {condition} | {values} | {r['possible']}/{m['complete']} |")
    lines.extend(
        [
            "",
            "The possible subset is not a replacement sample. Per-question minimum rates and impossible",
            "cases are retained in `questions.json`; no production hit rate has been inferred or fitted.",
            "",
            "## Full reference corpus: common compact baseline",
            "",
            "This reuses all complete reference questions, with the original coverage limitations.",
            "Zero prefix overhead is an ideal limit, not a measured warm-request cost. The batching",
            "column is conservative; each row uses the stated unbatched serialization as its reference.",
            "",
            "| Unbatched → batched | Depth | Questions | Median, zero prefix cost | Median shared overhead budget for 1.5× (ms) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for f in frontier:
        lines.append(
            f"| {f['comparison']} | {f['depth']} | {f['complete']} | "
            f"{ratio(f['ideal_cached_speedup']['median'])} | "
            f"{f['shared_overhead_budget_ms']['median']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Adding predictable spaces can improve the fraction-forced metric while losing to compact",
            "JSON on absolute latency. Preserve a common compact baseline in the next experiment.",
            "",
            "## Longer output is not automatically a better workload",
            "",
            "Strata use compact body length for the same questions in every comparison. These are",
            "descriptive post-hoc strata, not newly selected success cohorts. No string was artificially",
            "repeated or padded to manufacture a long forced span.",
            "",
            "| Comparison (depth 1024) | Compact body tokens | Questions | Median forced fraction | Median, zero prefix cost | Median, cold prefix |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for f in frontier:
        if f["depth"] != 1024 or f["comparison"] not in ("G1/S1 → G2/S1", "G1/S2 → G2/S2"):
            continue
        for label, r in f["length_bins"].items():
            if r["questions"]:
                lines.append(
                    f"| {f['comparison']} | {label} | {r['questions']} | "
                    f"{pct(r['forced_fraction']['median'])} | "
                    f"{ratio(r['ideal_cached_speedup']['median'])} | {ratio(r['cold_speedup']['median'])} |"
                )
    lines.extend(
        [
            "",
            "## Decision and next bounded step",
            "",
            "Do not start the engine patch or expand into draft composition on these estimates alone.",
            "The narrower candidate is repeated structured work with stable system/tool prefixes and",
            "enough genuinely deterministic structure, measured against compact JSON. Cache alone does",
            "not remove non-accelerated decode cost, and long free-form values can dilute the opportunity.",
            "",
            "Before changing the gate, collect representative traces from the intended application:",
            "",
            "- prompt/body token counts and the actual system/tool prefix;",
            "- observed `timings.cache_n`, prefill/decode/wall times, misses and slot/concurrency behavior;",
            "- schema/grammar identity and raw output token IDs, with sensitive values kept local;",
            "- semantic accuracy criteria and acceptable regression margin fixed before confirmation.",
            "",
            "Use those traces to choose a small confirmatory workload and measure a matching-depth cost",
            "curve. If it still cannot plausibly reach the target against compact JSON, stop or accept a",
            "smaller learning/engineering objective explicitly. A positive result would justify a revised",
            "go decision, not a retroactive claim that the original gate passed.",
            "",
            "## Reproducibility",
            "",
            "- [Cache measurements and summaries](../runs/2026-09-12-workload-cache/summary.json).",
            "- [Collection manifest](../runs/2026-09-12-workload-cache/collect-manifest.json) and continuation subdirectories.",
            f"- [Full-cohort frontier and report hashes]({manifest_link}).",
            "- [Original Gate 0 report](gate0-report.md) remains an immutable-results reference.",
            "- Scripts: `05_workload_cache.py` (collect/analyze), `06_retarget_frontier.py` (report).",
            "",
            "The standalone report is generated from run artifacts; numbers are not transcribed manually.",
            "No application telemetry was available. No tools described by the dataset were executed.",
            "",
            "Decision: **retarget**, with no C++ go.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/2026-09-12-workload-cache")
    parser.add_argument("--full", type=Path, default=ROOT / "runs/2026-09-12-full")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cache = json.loads((args.cache / "summary.json").read_text())
    full = json.loads((args.full / "questions.json").read_text())
    frontier = reference_frontier(full)
    resumed = len(list(args.cache.glob("continuation-*")))
    manifest_link = os.path.relpath(args.out.resolve() / "manifest.json", ROOT / "docs")
    report = render(cache, frontier, resumed, manifest_link)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "frontier.json", frontier)
    with (args.out / "retarget-report.md").open("x") as stream:
        stream.write(report)
    write_json(
        args.out / "manifest.json",
        {
            **provenance(),
            "inputs_sha256": {
                str(p.resolve()): digest(p.read_bytes())
                for p in (args.cache / "summary.json", args.full / "questions.json")
            },
            "frontier_sha256": digest((args.out / "frontier.json").read_bytes()),
            "report_sha256": digest(report.encode()),
            "decision": "retarget",
        },
    )
    print(args.out / "retarget-report.md")
