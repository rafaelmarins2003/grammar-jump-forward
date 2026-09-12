import pytest

from jumpforward.baselines import (
    bootstrap_median,
    check_output_bytes,
    forms_for,
    grammar_request,
    paired_comparison,
    trial_key,
    validate_trial_grid,
)


def test_stock_and_llguidance_are_distinct_requests():
    schema = {"type": "object"}
    assert grammar_request(schema, "stock") == {"json_schema": schema}
    assert grammar_request(schema, "S1")["grammar"].startswith("%llguidance")
    assert grammar_request(schema, "S1") != grammar_request(schema, "S2")
    with pytest.raises(ValueError):
        grammar_request(schema, "unknown")


def test_prespecified_request_grid():
    assert sum(25 * 2 * len(forms_for(s)) for s in ("none", "draft-simple", "ngram-simple")) == 350
    assert trial_key({"subset": "simple", "id": "x", "form": "S1", "repetition": 0}) == (
        "simple",
        "x",
        0,
        "S1",
    )


def test_bootstrap_is_deterministic_and_not_ratio_of_unpaired_medians():
    assert bootstrap_median([2.0, 2.0, 2.0]) == {"n": 3, "median": 2.0, "ci95": [2.0, 2.0]}
    assert bootstrap_median([1.0, 2.0, 3.0]) == bootstrap_median([1.0, 2.0, 3.0])


def test_different_outputs_are_flagged_not_counted_as_lossless_speedup():
    base = {
        ("s", "1"): {"wall_ms": 100, "decode_ms": 80, "parsed": [{"v": 1}, {"v": 1}]},
        ("s", "2"): {"wall_ms": 200, "decode_ms": 100, "parsed": [{"v": 2}, {"v": 2}]},
    }
    candidate = {
        ("s", "1"): {"wall_ms": 50, "decode_ms": 40, "parsed": [{"v": 1}, {"v": 1}]},
        ("s", "2"): {"wall_ms": 50, "decode_ms": 25, "parsed": [{"v": 3}, {"v": 3}]},
    }
    result = paired_comparison(base, candidate)
    assert result["pairs"] == 2
    assert result["output_differences"] == 1
    assert result["wall_ms"]["median"] == 3.0
    assert result["equal_output_wall_ms"]["median"] == 2.0
    assert result["equal_output_wall_ms"]["n"] == 1


def test_grid_rejects_replacements_even_when_counts_match():
    inputs = [{"subset": "s", "id": "one"}]
    rows = [
        {"subset": "s", "id": "one", "repetition": rep, "form": form, "strategy": "none"}
        for rep in range(2)
        for form in forms_for("none")
    ]
    validate_trial_grid(rows, inputs, "none")
    rows[0]["id"] = "replacement"
    with pytest.raises(ValueError, match="trial grid"):
        validate_trial_grid(rows, inputs, "none")


def test_empty_comparison_does_not_hide_missing_pairs():
    result = paired_comparison({}, {})
    assert result["pairs"] == 0
    assert result["wall_ms"]["median"] is None


def test_byte_audit_removes_only_trailing_eos_and_rejects_mismatch():
    class Tokenizer:
        eos_token = 9

        def decode_bytes(self, tokens):
            return b"".join({1: b"{", 2: b"}", 9: b"<eos>"}[t] for t in tokens)

    tokenizer = Tokenizer()
    assert check_output_bytes(tokenizer, {"tokens": [1, 2, 9], "content": "{}"}) == (True, 2)
    assert check_output_bytes(tokenizer, {"tokens": [1, 2], "content": "{}"}) == (False, 2)
    with pytest.raises(ValueError, match="roundtrip"):
        check_output_bytes(tokenizer, {"tokens": [1, 9, 2], "content": "{}"})
