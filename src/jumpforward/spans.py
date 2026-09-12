"""Replay canonical reference tokens through PL and an independent sequential matcher."""

import json
from time import perf_counter_ns

from jsonschema import Draft202012Validator
from llguidance import LLMatcher, LLTokenizer

from .corpus import digest

SERIALIZATIONS = {"S1": (",", ":"), "S2": (", ", ": ")}


class TraceError(ValueError):
    pass


def check_matcher(matcher: LLMatcher) -> None:
    if matcher.is_error():
        raise TraceError(matcher.get_error())


def allows(mask: bytes, token: int) -> bool:
    return 0 <= token < len(mask) * 8 and bool(mask[token // 8] & (1 << (token % 8)))


def make_grammar(schema: dict, variant: str, serialization: str) -> str:
    overrides = {"whitespace_flexible": True}
    if variant == "G2":
        item, key = SERIALIZATIONS[serialization]
        overrides = {"whitespace_flexible": False, "item_separator": item, "key_separator": key}
    elif variant != "G1":
        raise ValueError(f"unsupported grammar variant: {variant}")
    return LLMatcher.grammar_from_json_schema(schema, overrides=overrides)


def trace_text(
    tokenizer: LLTokenizer, grammar: str, text: str, token_ids: list[int] | None = None
) -> dict:
    tokens = tokenizer.tokenize_str(text) if token_ids is None else list(token_ids)
    source = text.encode("utf-8", errors="strict")
    if tokenizer.decode_bytes(tokens) != source:
        raise TraceError("reference tokenizer byte roundtrip mismatch")
    t0 = perf_counter_ns()
    matcher = LLMatcher(tokenizer, grammar, log_level=0)
    compile_ns = perf_counter_ns() - t0
    shadow = LLMatcher(tokenizer, grammar, log_level=0)
    check_matcher(matcher)
    check_matcher(shadow)
    spans, flags = [], [False] * len(tokens)
    ff_ns = consume_ns = 0
    forced_bytes = 0
    offset = 0
    while offset < len(tokens):
        t0 = perf_counter_ns()
        proposed = matcher.compute_ff_tokens()
        ff_ns += perf_counter_ns() - t0
        check_matcher(matcher)
        if proposed:
            actual = tokens[offset : offset + len(proposed)]
            if actual != proposed:
                raise TraceError(
                    f"ff_reference_mismatch at token {offset}: reference={actual}, PL={proposed}"
                )
            byte_span = matcher.compute_ff_bytes()
            check_matcher(matcher)
            injected = tokenizer.decode_bytes(proposed)
            if not byte_span.startswith(injected):
                raise TraceError("FF token bytes are not a prefix of forced bytes")
            for token in proposed:
                mask = shadow.compute_bitmask()
                check_matcher(shadow)
                if not allows(mask, token) or int.from_bytes(mask, "little").bit_count() != 1:
                    raise TraceError(f"PL differs from sequential singleton mask at {offset}")
                if not shadow.consume_token(token):
                    check_matcher(shadow)
                    raise TraceError("sequential consumption failed")
            t0 = perf_counter_ns()
            accepted = matcher.consume_tokens(proposed)
            consume_ns += perf_counter_ns() - t0
            check_matcher(matcher)
            if not accepted:
                raise TraceError("bulk consumption failed")
            spans.append(
                {
                    "start": offset,
                    "length": len(proposed),
                    "bytes": len(injected),
                    "available_forced_bytes": len(byte_span),
                }
            )
            flags[offset : offset + len(proposed)] = [True] * len(proposed)
            forced_bytes += len(injected)
            offset += len(proposed)
        else:
            mask = matcher.compute_bitmask()
            reference_mask = shadow.compute_bitmask()
            check_matcher(matcher)
            check_matcher(shadow)
            if mask != reference_mask:
                raise TraceError(f"bulk/sequential state divergence at {offset}")
            token = tokens[offset]
            if not allows(mask, token):
                raise TraceError(f"reference_token_not_allowed at {offset}: {token}")
            if not matcher.consume_token(token) or not shadow.consume_token(token):
                raise TraceError(f"free-token consumption failed at {offset}")
            offset += 1
    if not matcher.is_accepting() or not shadow.is_accepting():
        raise TraceError("reference does not reach an accepting grammar state")
    check_matcher(matcher)
    return {
        "status": "ok",
        "text": text,
        "tokens": tokens,
        "forced": flags,
        "spans": spans,
        "n_tokens": len(tokens),
        "n_forced": sum(flags),
        "n_bytes": len(source),
        "n_injected_bytes": forced_bytes,
        "forced_fraction": sum(flags) / len(tokens),
        "injected_byte_fraction": forced_bytes / len(source),
        "compile_ns": compile_ns,
        "ff_ns": ff_ns,
        "bulk_consume_ns": consume_ns,
        "singleton_agreement": True,
        "warnings": matcher.get_grammar_warnings(),
    }


def trace_call(tokenizer: LLTokenizer, schema: dict, call: dict, variant: str, serialization: str):
    text = json.dumps(
        call, ensure_ascii=False, allow_nan=False, separators=SERIALIZATIONS[serialization]
    )
    Draft202012Validator(schema).validate(json.loads(text))
    grammar = make_grammar(schema, variant, serialization)
    try:
        result = trace_text(tokenizer, grammar, text)
    except (TraceError, ValueError) as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "text": text,
            "tokens": tokenizer.tokenize_str(text),
        }
    return {**result, "grammar": grammar, "grammar_sha256": digest(grammar.encode())}


def characterization(tokenizer: LLTokenizer) -> list[dict]:
    cases = [
        ("literal", 'start: "abcdefghijk"', "abcdefghijk"),
        ("unicode", 'start: "ação🦙"', "ação🦙"),
        (
            "lazy_trigger",
            'start: before body\nbefore[lazy]: /(.|\\n)*<tool>/\nbody: "abcdefghijk"',
            "explanation<tool>abcdefghijk",
        ),
    ]
    results = []
    for name, grammar, text in cases:
        try:
            result = trace_text(tokenizer, grammar, text)
        except (TraceError, ValueError) as exc:
            result = {"status": "error", "error": str(exc)}
        results.append({"case": name, **result})
    return results
