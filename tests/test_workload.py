import math

import pytest

from jumpforward.frontier import length_bin, reference_frontier
from jumpforward.workload import (
    cache_counts,
    common_prefix,
    completed_trials,
    overhead_budget,
    required_hit_rate,
    speedup,
)


def test_overhead_budget_exactly_reaches_target():
    budget = overhead_budget(100, 50)
    assert budget == 50
    assert speedup(100, 50, budget) == 1.5
    assert speedup(100, 50, budget + 1) < 1.5


def test_cache_cannot_rescue_insufficient_decode_gain():
    budget = overhead_budget(100, 80)
    assert budget < 0
    assert required_hit_rate(budget, 50, 0) is None


def test_request_hit_rate_is_not_fraction_of_tokens_reused():
    rate = required_hit_rate(50, 100, 20)
    assert rate == 0.625
    assert (1 - rate) * 100 + rate * 20 == 50
    assert required_hit_rate(50, 40, 20) == 0
    assert required_hit_rate(50, 100, 60) is None
    assert required_hit_rate(50, 100, 100) is None
    assert required_hit_rate(50, 100, 50) == 1


def test_shared_residual_can_eliminate_a_model_only_opportunity():
    assert speedup(100, 50, 20) > 1.5
    assert speedup(100, 50, 20 + 40) < 1.5
    assert overhead_budget(100, 50) - 40 == 10


def test_cache_counters_use_reuse_not_post_generation_occupancy():
    assert cache_counts({"tokens_cached": 250, "timings": {"cache_n": 0, "prompt_n": 217}}) == (
        0,
        217,
    )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([], [1], []),
        ([1, 2], [1, 2, 3], [1, 2]),
        ([1, 2], [1, 3], [1]),
        ([1], [2], []),
        ([1, 2, 3], [1], [1]),
    ],
)
def test_prefix_stops_at_first_token_difference(left, right, expected):
    assert common_prefix(left, right) == expected


@pytest.mark.parametrize("target", [1.0, 0.0, -1.0, math.nan, math.inf])
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError):
        overhead_budget(100, 50, target)


def test_resume_preserves_failed_trials_and_rejects_duplicates():
    trial = {"subset": "simple", "id": "x", "repetition": 0, "condition": "cold", "status": "error"}
    assert completed_trials([trial]) == {("simple", "x", 0, "cold")}
    with pytest.raises(ValueError, match="duplicate"):
        completed_trials([trial, trial])


@pytest.mark.parametrize(
    ("tokens", "label"),
    [(32, "1–32"), (33, "33–64"), (128, "65–128"), (129, "129–256"), (257, ">256")],
)
def test_length_strata_are_fixed(tokens, label):
    assert length_bin(tokens) == label


def test_frontier_keeps_compact_baseline_and_paired_length_strata():
    rows = []
    for variant, serial, n, baseline, batched in (
        ("G1", "S1", 32, 100, 80),
        ("G1", "S2", 45, 150, 120),
        ("G2", "S1", 32, 100, 60),
        ("G2", "S2", 45, 150, 70),
    ):
        rows.append(
            {
                "subset": "simple",
                "id": "x",
                "status": "ok",
                "variant": variant,
                "serialization": serial,
                "n_tokens": n,
                "forced_fraction": 0.5,
                "cost": [
                    {"baseline_ms": baseline, "conservative_ms": batched, "prefill_ms": 200}
                    for _ in (1024, 4096)
                ],
            }
        )
    result = {r["comparison"]: r for r in reference_frontier(rows) if r["depth"] == 1024}
    assert result["G1/S2 → G2/S2"]["ideal_cached_speedup"]["median"] == 150 / 70
    assert result["G1/S1 → G2/S2"]["ideal_cached_speedup"]["median"] == 100 / 70
    assert result["G1/S1 → G2/S2"]["length_bins"]["1–32"]["questions"] == 1
    assert result["G1/S1 → G2/S2"]["length_bins"]["33–64"]["questions"] == 0
