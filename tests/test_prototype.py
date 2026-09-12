import copy
from collections import Counter

import pytest

from jumpforward.parity import assert_gpu_idle
from jumpforward.prototype import BLOCKS, SUBSETS, analyze_rows, select_holdout, upper_binomial


def test_exact_binomial_bound_requires_enough_questions():
    assert upper_binomial(0, 200) == pytest.approx(1 - 0.05 ** (1 / 200))
    assert upper_binomial(0, 200) < 0.02
    assert upper_binomial(1, 200) > 0.02
    assert upper_binomial(0, 25) > 0.02
    assert upper_binomial(0, 0) == 1.0


def test_selection_is_disjoint_balanced_and_does_not_mutate_cases():
    class Tokenizer:
        def tokenize_str(self, text):
            return list(text)

    cases = [
        {"subset": s, "id": str(i), "status": "ok", "calls": [{"text": "a" * i}]}
        for s in SUBSETS
        for i in range(21)
    ]
    original = copy.deepcopy(cases)
    excluded = {(s, "0") for s in SUBSETS}
    selected, counts = select_holdout(cases, excluded, Tokenizer(), per_stratum=2)
    assert cases == original
    assert all((c["subset"], c["id"]) not in excluded for c in selected)
    assert set(Counter((c["subset"], c["length_stratum"]) for c in selected).values()) == {2}
    assert counts == dict.fromkeys(SUBSETS, 20)
    assert select_holdout(cases[::-1], excluded, Tokenizer(), per_stratum=2)[0] == selected


def test_gpu_guard_refuses_existing_process_without_touching_it(monkeypatch):
    monkeypatch.setattr(
        "jumpforward.parity.subprocess.check_output", lambda *a, **k: "123, user, 5"
    )
    with pytest.raises(RuntimeError, match="refusing to start"):
        assert_gpu_idle()
    monkeypatch.setattr("jumpforward.parity.subprocess.check_output", lambda *a, **k: "\n")
    assert_gpu_idle()


def fixture_rows():
    cases = [{"subset": s, "id": str(i)} for s in SUBSETS for i in range(40)]
    rows = []
    request = {
        "prompt": "p",
        "grammar": "g",
        "n_predict": 2048,
        "cache_prompt": False,
        "temperature": 0,
        "seed": 20260912,
    }
    for c in cases:
        for strategy, repetition in BLOCKS:
            rows.append(
                {
                    **c,
                    "strategy": strategy,
                    "repetition": repetition,
                    "status": "ok",
                    "request": request,
                    "wall_ms": 100 if strategy == "none" else 50,
                    "parsed": {},
                    "reference_equal": True,
                    "response": {
                        "content": "{}",
                        "tokens": [1, 2, 3],
                        "tokens_predicted": 3,
                        "timings": {
                            "predicted_ms": 80,
                            "prompt_ms": 20,
                            "draft_n": 1 if strategy == "grammar" else 0,
                            "draft_n_accepted": 1 if strategy == "grammar" else 0,
                        },
                    },
                }
            )
    return cases, rows


def test_decision_does_not_pass_on_speed_if_quality_or_coverage_fails(monkeypatch):
    # Keep this decision test independent of the separately tested bootstrap implementation.
    monkeypatch.setattr(
        "jumpforward.prototype.bootstrap_median",
        lambda v: {
            "n": len(v),
            "median": min(v) if v else None,
            "ci95": [min(v), max(v)] if v else [None, None],
        },
    )
    cases, rows = fixture_rows()
    summary, _ = analyze_rows(cases, rows)
    assert summary["decision"] == "go"
    assert summary["arms"]["none"]["generation_batches_counter_derived"] == 800
    assert summary["arms"]["grammar"]["generation_batches_counter_derived"] == 400
    rows[1]["parsed"] = {"changed": True}
    summary, _ = analyze_rows(cases, rows)
    assert summary["decision"] == "stop"
    assert not summary["criteria"]["quality_upper_below_2_percent"]
    rows[1]["status"] = "error"
    summary, _ = analyze_rows(cases, rows)
    assert summary["decision"] == "stop"
    assert not summary["criteria"]["complete_no_failures"]
    assert summary["arms"]["grammar"]["timed_requests_including_failures"] == 400
    assert summary["failure_questions"]["candidate_only"] == [(cases[0]["subset"], "0")]
    with pytest.raises(ValueError, match="evaluation trials"):
        analyze_rows(cases, rows[:-1])


def test_mismatched_prompts_and_duplicate_trials_are_errors():
    cases, rows = fixture_rows()
    rows[1]["request"] = {**rows[1]["request"], "prompt": "different"}
    with pytest.raises(ValueError, match="requests differ"):
        analyze_rows(cases, rows)
    with pytest.raises(ValueError, match="evaluation trials"):
        analyze_rows(cases, rows + [rows[0]])
