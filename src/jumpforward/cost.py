"""Explicit body-processing proxies; these are not server benchmarks."""

import math


def batch_schedule(forced: list[bool], cap: int = 32) -> list[int]:
    """Process a pending token together with following grammar-forced tokens."""
    if cap < 1:
        raise ValueError("batch cap must be positive")
    batches = []
    offset = 0
    while offset < len(forced):
        end = offset + 1
        while end < len(forced) and forced[end] and end - offset < cap:
            end += 1
        batches.append(end - offset)
        offset = end
    return batches


def project_cost(trace: dict, curve: dict, depth: int) -> dict:
    rows = {row["batch"]: row["mean_ms"] for row in curve["batch_cost"] if row["depth"] == depth}
    prefill = next(row["mean_ms"] for row in curve["prefill"] if row["tokens"] == depth)
    sizes = sorted(rows)
    batches = batch_schedule(trace["forced"], sizes[-1])
    optimistic = conservative = 0.0
    for n in batches:
        upper = next(size for size in sizes if size >= n)
        lower = max(size for size in sizes if size <= n)
        optimistic += rows[n] if n in rows else min(rows[lower], rows[upper])
        conservative += max(rows[size] for size in sizes if size <= upper)
    conservative += (trace["ff_ns"] + trace["bulk_consume_ns"]) / 1e6
    baseline = len(trace["forced"]) * rows[1]
    free_cost = (len(trace["forced"]) - sum(trace["forced"])) * rows[1]
    return {
        "depth": depth,
        "batches": batches,
        "baseline_ms": baseline,
        "optimistic_ms": optimistic,
        "conservative_ms": conservative,
        "prefill_ms": prefill,
        "decode_speedup_optimistic": baseline / optimistic,
        "decode_speedup_conservative": baseline / conservative,
        "e2e_speedup_optimistic": (prefill + baseline) / (prefill + optimistic),
        "e2e_speedup_conservative": (prefill + baseline) / (prefill + conservative),
        "zero_forced_cost_ceiling_decode": baseline / free_cost if free_cost else None,
        "zero_forced_cost_ceiling_e2e": (prefill + baseline) / (prefill + free_cost),
    }


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p10": None, "median": None, "p90": None, "mean": None}
    values = sorted(values)

    def q(p):
        i = (len(values) - 1) * p
        low = math.floor(i)
        high = math.ceil(i)
        return values[low] + (values[high] - values[low]) * (i - low)

    return {
        "n": len(values),
        "p10": q(0.1),
        "median": q(0.5),
        "p90": q(0.9),
        "mean": sum(values) / len(values),
    }
