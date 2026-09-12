import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from llguidance import LLMatcher, LLTokenizer

from jumpforward.corpus import adapt_bfcl_arguments, normalize_schema, schema_order, tool_union
from jumpforward.cost import batch_schedule, project_cost, quantiles
from jumpforward.generation import server_grammar
from jumpforward.phase0 import aggregate, question_summary
from jumpforward.spans import TraceError, characterization, trace_call, trace_text


@pytest.fixture(scope="module")
def tokenizer():
    path = Path(__file__).resolve().parents[1] / "data/raw/tokenizers/qwen3/tokenizer.json"
    if not path.exists():
        pytest.skip("pinned offline tokenizer not installed")
    return LLTokenizer(str(path), eos_token=151645)


def test_normalization_only_touches_schema_nodes():
    schema = {
        "type": "dict",
        "properties": {"type": {"type": "string", "enum": ["dict"]}, "v": {"type": "float"}},
        "const": {"type": "dict"},
    }
    changes = []
    output = normalize_schema(schema, changes)
    assert output["type"] == "object"
    assert output["properties"]["type"]["enum"] == ["dict"]
    assert output["const"] == {"type": "dict"}
    assert schema["type"] == "dict"
    assert len(changes) == 2


def test_alternatives_are_not_argument_values_and_omission_is_recorded():
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "array", "items": {"type": "integer"}},
            "unit": {"type": "string"},
        },
        "required": ["x"],
    }
    choices = []
    result = adapt_bfcl_arguments({"x": [[1, 2], [3]], "unit": ["", "m"]}, schema, choices)
    assert result == {"x": [1, 2]}
    assert choices[1]["omitted"]


def test_schema_order_preserves_values_and_all_tools_are_allowed():
    params = {"type": "object", "properties": {"b": {}, "a": {}}}
    value = {"a": {"type": "dict"}, "b": 3}
    assert list(schema_order(value, params)) == ["b", "a"]
    assert schema_order(value, params) == value
    schema = tool_union([{"name": name, "parameters": params} for name in ("f", "g")])
    validator = Draft202012Validator(schema)
    assert validator.is_valid({"name": "g", "arguments": value})
    assert not validator.is_valid({"name": "h", "arguments": value})


@pytest.mark.parametrize(
    ("flags", "cap", "expected"),
    [
        ([], 32, []),
        ([False] * 3, 32, [1, 1, 1]),
        ([False, True, True, False, True], 32, [3, 2]),
        ([True] * 65, 32, [32, 32, 1]),
        ([True, True, False, True], 2, [2, 2]),
    ],
)
def test_schedule_counts_every_token_once(flags, cap, expected):
    assert batch_schedule(flags, cap) == expected
    assert sum(expected) == len(flags)


def test_nonmonotonic_cost_proxy_is_not_an_interpolated_claim():
    curve = {
        "batch_cost": [
            {"depth": 10, "batch": n, "mean_ms": cost}
            for n, cost in ((1, 10), (2, 12), (4, 20), (8, 15))
        ],
        "prefill": [{"tokens": 10, "mean_ms": 100}],
    }
    trace = {"forced": [False] + [True] * 6, "ff_ns": 1_000_000, "bulk_consume_ns": 0}
    result = project_cost(trace, curve, 10)
    assert result["batches"] == [7]
    assert result["optimistic_ms"] == 15
    assert result["conservative_ms"] == 21
    assert result["baseline_ms"] == 70


def test_distribution_weights_questions_not_parallel_calls():
    assert quantiles([0, 1, 2, 3])["median"] == 1.5
    curve = {
        "batch_cost": [
            {"depth": d, "batch": n, "mean_ms": 10 + n}
            for d in (1024, 4096)
            for n in (1, 2, 4, 8, 16, 32)
        ],
        "prefill": [{"tokens": d, "mean_ms": 100} for d in (1024, 4096)],
    }
    trace = {
        "subset": "parallel",
        "id": "x",
        "variant": "G2",
        "serialization": "S2",
        "status": "ok",
        "n_calls": 2,
        "n_tokens": 3,
        "n_forced": 2,
        "n_injected_bytes": 2,
        "n_bytes": 3,
        "spans": [{"length": 2}],
        "forced": [False, True, True],
        "ff_ns": 0,
        "bulk_consume_ns": 0,
        "compile_ns": 0,
    }
    trace["cost"] = [project_cost(trace, curve, d) for d in (1024, 4096)]
    q = question_summary([trace, trace])
    assert q["n_tokens"] == 6
    assert q["cost"][0]["prefill_ms"] == 100
    assert aggregate([q])["forced_fraction"]["n"] == 1
    failed = {**trace, "status": "error", "error": "bad token"}
    assert question_summary([trace, failed])["status"] == "error"


def test_native_ff_agrees_with_sequential_masks_and_lazy_activation(tokenizer):
    checks = characterization(tokenizer)
    assert all(c["status"] == "ok" for c in checks)
    assert checks[0]["n_forced"] == checks[0]["n_tokens"]
    assert checks[1]["n_forced"] == checks[1]["n_tokens"]
    assert not any(checks[2]["forced"][:5])
    assert all(checks[2]["forced"][5:])


def test_forbidden_reference_is_never_consumed(tokenizer):
    with pytest.raises(TraceError, match="ff_reference_mismatch"):
        trace_text(tokenizer, 'start: "abc"', "def")


def test_whitespace_variants_validate_without_oracle_values(tokenizer):
    params = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    schema = tool_union(
        [{"name": name, "parameters": params} for name in ("weather_current", "weather_forecast")]
    )
    call = {"name": "weather_forecast", "arguments": {"value": "ação 🦙"}}
    for variant in ("G1", "G2"):
        for serial in ("S1", "S2"):
            result = trace_call(tokenizer, schema, call, variant, serial)
            assert result["status"] == "ok", result.get("error")
            assert json.loads(result["text"]) == call
            assert sum(result["forced"]) == sum(s["length"] for s in result["spans"])
            assert (
                result["n_forced"] < result["n_tokens"]
            )  # The free value wasn't const-constrained.


def test_no_forcing_is_a_different_mask_policy(tokenizer):
    normal = LLMatcher(tokenizer, 'start: "abcdefghijk"')
    unforced = LLMatcher(tokenizer, '%llguidance {"no_forcing":true}\nstart: "abcdefghijk"')
    assert int.from_bytes(normal.compute_bitmask(), "little").bit_count() == 1
    assert int.from_bytes(unforced.compute_bitmask(), "little").bit_count() > 1


def test_recorded_token_ids_are_not_silently_retokenized(tokenizer):
    ids = [tokenizer.tokenize_str(letter)[0] for letter in "abc"]
    trace = trace_text(tokenizer, "start: /[a-z]+/", "abc", token_ids=ids)
    assert trace["tokens"] == ids
    assert len(trace["tokens"]) == 3
    assert len(tokenizer.tokenize_str("abc")) == 1


def test_server_lark_envelope_has_same_schema_options(tokenizer):
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    for variant in ("G1", "G2"):
        expected = trace_call(tokenizer, schema, {"x": 7}, variant, "S2")
        actual = trace_text(tokenizer, server_grammar(schema, variant), expected["text"])
        assert actual["tokens"] == expected["tokens"]
        assert actual["forced"] == expected["forced"]


def test_datetime_format_is_stricter_than_annotation_only_validation(tokenizer):
    schema = {"type": "string", "format": "date-time"}
    missing_zone = "2023-06-01T08:00:00"
    assert Draft202012Validator(schema).is_valid(missing_zone)
    rejected = trace_call(tokenizer, schema, missing_zone, "G2", "S2")
    accepted = trace_call(tokenizer, schema, missing_zone + "Z", "G2", "S2")
    assert rejected["status"] == "error"
    assert accepted["status"] == "ok"
