"""Pinned corpus adapters. Failed records are retained; tools are never oracle-selected."""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator

BFCL_SUBSETS = ("simple", "multiple", "parallel", "live_simple", "live_multiple")
SEED = "20260912"


class CorpusError(ValueError):
    """A reference cannot be used without inventing or repairing information."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_schema(schema, changes: list, path: str = "parameters"):
    if isinstance(schema, bool):
        return schema
    if not isinstance(schema, dict):
        raise CorpusError(f"invalid schema at {path}")
    out = deepcopy(schema)
    aliases = {"dict": "object", "float": "number", "tuple": "array"}
    kind = out.get("type")
    if isinstance(kind, str) and kind in aliases:
        out["type"] = aliases[kind]
        changes.append({"path": path, "from": kind, "to": aliases[kind]})
    elif kind == "any":
        del out["type"]
        changes.append({"path": path, "from": "any", "to": "unconstrained"})
    for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
        if key in out:
            out[key] = {
                name: normalize_schema(value, changes, f"{path}/{key}/{name}")
                for name, value in out[key].items()
            }
    for key in ("items", "additionalProperties", "contains", "not", "if", "then", "else"):
        if key in out:
            out[key] = normalize_schema(out[key], changes, f"{path}/{key}")
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if key in out:
            out[key] = [
                normalize_schema(value, changes, f"{path}/{key}/{i}")
                for i, value in enumerate(out[key])
            ]
    Draft202012Validator.check_schema(out)
    return out


def schema_order(value, schema):
    """JSON object order is a serialization choice, not an argument-value change."""
    if not isinstance(schema, dict):
        return value
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        keys = [key for key in properties if key in value]
        keys.extend(key for key in value if key not in properties)
        return {
            key: schema_order(value[key], properties.get(key, schema.get("additionalProperties", {})))
            for key in keys
        }
    if isinstance(value, list):
        prefix = schema.get("prefixItems", [])
        return [
            schema_order(item, prefix[i] if i < len(prefix) else schema.get("items", {}))
            for i, item in enumerate(value)
        ]
    return value


def tool_union(tools: list[dict]) -> dict:
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {"name": {"const": tool["name"]}, "arguments": tool["parameters"]},
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
            for tool in tools
        ]
    }


def adapt_bfcl_arguments(alternatives: dict, schema: dict, choices: list) -> dict:
    args = {}
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for name, values in alternatives.items():
        if not isinstance(values, list) or not values:
            raise CorpusError(f"answer alternatives for {name!r} are not a nonempty list")
        validator = Draft202012Validator(properties.get(name, schema.get("additionalProperties", {})))
        for i, value in enumerate(values):
            if value == "" and name not in required:
                choices.append({"argument": name, "alternative_index": i, "omitted": True})
                break
            if validator.is_valid(value):
                args[name] = value
                choices.append({"argument": name, "alternative_index": i, "omitted": False})
                break
        else:
            raise CorpusError(f"no schema-valid alternative for {name!r}")
    if not Draft202012Validator(schema).is_valid(args):
        errors = [error.message for error in Draft202012Validator(schema).iter_errors(args)]
        raise CorpusError(f"reference arguments invalid: {errors}")
    return schema_order(args, schema)


def adapt_record(subset: str, record: dict, answer: dict | None) -> dict:
    changes, choices = [], []
    raw_tools = record["function"] if subset in BFCL_SUBSETS else json.loads(record["tools"])
    tools = []
    for raw in raw_tools:
        raw = raw.get("function", raw)
        tools.append({
            "name": raw["name"],
            "description": raw.get("description", ""),
            "parameters": normalize_schema(raw["parameters"], changes, raw["name"]),
        })
    by_name = {tool["name"]: tool for tool in tools}
    if not tools or len(by_name) != len(tools):
        raise CorpusError("missing tools or ambiguous duplicate tool names")
    calls = []
    if subset in BFCL_SUBSETS:
        if answer is None or not answer.get("ground_truth"):
            raise CorpusError("missing ground truth")
        for ground in answer["ground_truth"]:
            if len(ground) != 1:
                raise CorpusError("ambiguous ground-truth function choice")
            name, alternatives = next(iter(ground.items()))
            if name not in by_name:
                raise CorpusError(f"reference tool {name!r} not in supplied tools")
            call_choices = []
            args = adapt_bfcl_arguments(alternatives, by_name[name]["parameters"], call_choices)
            calls.append({"name": name, "arguments": args})
            choices.append({"call_index": len(calls) - 1, "choices": call_choices})
        messages = [message for turn in record["question"] for message in turn]
    else:
        for turn in record["conversations"]:
            if turn["from"] != "gpt":
                continue
            bodies = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", turn["value"], re.S)
            for body in bodies:
                call = json.loads(body)
                if set(call) != {"name", "arguments"} or call["name"] not in by_name:
                    raise CorpusError("invalid Hermes tool-call envelope")
                schema = by_name[call["name"]]["parameters"]
                Draft202012Validator(schema).validate(call["arguments"])
                calls.append({"name": call["name"], "arguments": schema_order(call["arguments"], schema)})
        if not calls:
            raise CorpusError("no JSON tool-call reference")
        messages = [
            {"role": "user", "content": turn["value"]}
            for turn in record["conversations"] if turn["from"] == "human"
        ]
    union = tool_union(tools)
    for call in calls:
        Draft202012Validator(union).validate(call)
    return {
        "subset": subset, "id": record["id"], "status": "ok", "tools": tools,
        "schema": union, "calls": calls, "messages": messages,
        "normalizations": changes, "reference_choices": choices,
    }


def select_records(root: Path, full: bool = False) -> list[tuple[str, dict, dict | None]]:
    selected = []
    for subset in (*BFCL_SUBSETS, "hermes"):
        if subset == "hermes":
            records = json.loads((root / "datasets/hermes-fc-v1/func-calling-singleturn.json").read_text())
            answers = {}
            size = 50
        else:
            records = read_jsonl(root / f"datasets/bfcl/BFCL_v3_{subset}.json")
            answer_rows = read_jsonl(root / f"datasets/bfcl/possible_answer/BFCL_v3_{subset}.json")
            answers = {row["id"]: row for row in answer_rows}
            if len(answers) != len(answer_rows):
                raise CorpusError(f"duplicate answer IDs: {subset}")
            size = 30
        ids = [record["id"] for record in records]
        if len(set(ids)) != len(ids):
            raise CorpusError(f"duplicate question IDs: {subset}")
        records.sort(key=lambda row: digest(f"{SEED}:{subset}:{row['id']}".encode()))
        for row in records if full else records[:size]:
            selected.append((subset, row, answers.get(row["id"])))
    return selected
