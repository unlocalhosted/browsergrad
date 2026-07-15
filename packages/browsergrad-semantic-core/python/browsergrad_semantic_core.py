#!/usr/bin/env python3
"""Dependency-free Python parity oracle for BrowserGrad layout wire v1."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1
SAFE_INTEGER_MAX = (1 << 53) - 1
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_DEPTH = 64
MAX_NODES = 100_000
MAX_STRING_BYTES = 2 * 1024 * 1024
MAX_ARRAY_LENGTH = 100_000
MAX_OBJECT_PROPERTIES = 10_000
MAX_INTEGER_BITS = 256
MAX_OPERATIONS = 200_000
I64_PATTERN = re.compile(r"^(?:0|-[1-9][0-9]*|[1-9][0-9]*)$", re.ASCII)
LOCAL_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$", re.ASCII)
SYMBOL_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$", re.ASCII)
DTYPE_BYTES = {
    "bool": 1,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "i32": 4,
    "u32": 4,
    "i64": 8,
    "u64": 8,
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "f64": 8,
}


class WireError(ValueError):
    pass


class DuplicateAwareObject(dict[str, Any]):
    pass


def _object_pairs(pairs: list[tuple[str, Any]]) -> DuplicateAwareObject:
    result: DuplicateAwareObject = DuplicateAwareObject()
    for key, value in pairs:
        if key in result:
            raise WireError(f"duplicate key: {key}")
        result[key] = value
    return result


def _parse_safe_integer(lexeme: str) -> int:
    if lexeme == "-0" or not I64_PATTERN.fullmatch(lexeme):
        raise WireError(f"noncanonical JSON integer: {lexeme}")
    value = int(lexeme, 10)
    if abs(value) > SAFE_INTEGER_MAX:
        raise WireError(f"unsafe JSON integer: {lexeme}")
    return value


def parse_wire_json_bytes(data: bytes) -> Any:
    if len(data) > MAX_DOCUMENT_BYTES:
        raise WireError("document byte budget exceeded")
    try:
        source = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise WireError("invalid UTF-8") from error
    try:
        value = json.loads(
            source,
            object_pairs_hook=_object_pairs,
            parse_int=_parse_safe_integer,
            parse_float=lambda lexeme: (_ for _ in ()).throw(WireError(f"floats forbidden: {lexeme}")),
            parse_constant=lambda lexeme: (_ for _ in ()).throw(WireError(f"constant forbidden: {lexeme}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise WireError("invalid JSON") from error
    validate_json_tree(value)
    return value


def validate_json_tree(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    seen: set[int] = set()
    nodes = 0
    string_bytes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            raise WireError("JSON resource budget exceeded")
        if current is None or isinstance(current, bool):
            continue
        if isinstance(current, str):
            _validate_unicode(current)
            string_bytes += len(current.encode("utf-8"))
            if string_bytes > MAX_STRING_BYTES:
                raise WireError("JSON string byte budget exceeded")
            continue
        if isinstance(current, int) and not isinstance(current, bool):
            if abs(current) > SAFE_INTEGER_MAX:
                raise WireError("unsafe JSON integer")
            continue
        if not isinstance(current, (list, dict)):
            raise WireError(f"non-JSON value: {type(current).__name__}")
        identity = id(current)
        if identity in seen:
            raise WireError("shared/cyclic JSON tree")
        seen.add(identity)
        if isinstance(current, list):
            if len(current) > MAX_ARRAY_LENGTH:
                raise WireError("JSON array length budget exceeded")
            stack.extend((entry, depth + 1) for entry in reversed(current))
        else:
            if len(current) > MAX_OBJECT_PROPERTIES:
                raise WireError("JSON object property budget exceeded")
            for key, entry in reversed(list(current.items())):
                if not isinstance(key, str):
                    raise WireError("non-string object key")
                _validate_unicode(key)
                string_bytes += len(key.encode("utf-8"))
                if string_bytes > MAX_STRING_BYTES:
                    raise WireError("JSON string byte budget exceeded")
                stack.append((entry, depth + 1))


def _validate_unicode(value: str) -> None:
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise WireError("lone Unicode surrogate")


def canonicalize(value: Any) -> str:
    validate_json_tree(value)
    result = _canonical(value)
    if len(result.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise WireError("canonical document byte budget exceeded")
    return result


def _canonical(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(_canonical(entry) for entry in value) + "]"
    keys = sorted(value, key=lambda key: key.encode("utf-16-be"))
    return "{" + ",".join(
        json.dumps(key, ensure_ascii=False, separators=(",", ":")) + ":" + _canonical(value[key])
        for key in keys
    ) + "}"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonicalize(value).encode("utf-8")).hexdigest()


def semantic_hash(envelope: dict[str, Any]) -> str:
    return canonical_hash({
        "domain": "browsergrad.semantic-artifact.v1",
        "schema": envelope["schema"],
        "version": envelope["version"],
        "requiredExtensions": sorted(envelope["requiredExtensions"]),
        "payload": envelope["payload"],
    })


def _exact(record: Any, fields: set[str], name: str, optional: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise WireError(f"{name} must be an object")
    unknown = set(record) - fields
    missing = fields - set(record) - (optional or set())
    if unknown or missing:
        raise WireError(f"{name} fields differ: unknown={sorted(unknown)} missing={sorted(missing)}")
    return record


def _local_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not LOCAL_ID_PATTERN.fullmatch(value):
        raise WireError(f"invalid {name}")
    return value


def _symbol_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SYMBOL_ID_PATTERN.fullmatch(value):
        raise WireError(f"invalid {name}")
    return value


def _wire_i64(value: Any, name: str) -> str:
    if not isinstance(value, str) or not I64_PATTERN.fullmatch(value):
        raise WireError(f"invalid {name}")
    parsed = int(value, 10)
    if parsed < I64_MIN or parsed > I64_MAX:
        raise WireError(f"{name} outside i64")
    return value


def _alignment(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > (1 << 31) or value & (value - 1):
        raise WireError(f"invalid {name}")
    return value


def _validate_dim(expression: Any, depth: int = 1) -> dict[str, Any]:
    if depth > MAX_DEPTH:
        raise WireError("dimension depth exceeded")
    record = _exact(expression, set(expression) if isinstance(expression, dict) else set(), "dimension expression")
    kind = record.get("kind")
    if kind == "const":
        _exact(record, {"kind", "value"}, "dimension constant")
        _wire_i64(record["value"], "dimension constant")
    elif kind == "symbol":
        _exact(record, {"kind", "id"}, "dimension symbol")
        _symbol_id(record["id"], "dimension symbol")
    elif kind == "add":
        _exact(record, {"kind", "terms"}, "dimension add")
        if not isinstance(record["terms"], list) or not record["terms"]:
            raise WireError("dimension add requires terms")
        for term in record["terms"]:
            _validate_dim(term, depth + 1)
    elif kind == "mul":
        _exact(record, {"kind", "lhs", "rhs"}, "dimension mul")
        _validate_dim(record["lhs"], depth + 1)
        _validate_dim(record["rhs"], depth + 1)
    elif kind in {"floorDiv", "ceilDiv", "mod"}:
        _exact(record, {"kind", "value", "divisor"}, "dimension division")
        _validate_dim(record["value"], depth + 1)
        _validate_dim(record["divisor"], depth + 1)
    elif kind in {"min", "max"}:
        _exact(record, {"kind", "values"}, "dimension nary")
        if not isinstance(record["values"], list) or not record["values"]:
            raise WireError("dimension nary requires values")
        for entry in record["values"]:
            _validate_dim(entry, depth + 1)
    else:
        raise WireError(f"unknown dimension kind: {kind}")
    return record


def _validate_index(expression: Any, rank: int, depth: int = 1) -> dict[str, Any]:
    if depth > MAX_DEPTH:
        raise WireError("index depth exceeded")
    if not isinstance(expression, dict):
        raise WireError("index expression must be object")
    kind = expression.get("kind")
    if kind == "const":
        _exact(expression, {"kind", "value"}, "index constant")
        _wire_i64(expression["value"], "index constant")
    elif kind == "coordinate":
        _exact(expression, {"kind", "axis"}, "coordinate")
        axis = expression["axis"]
        if not isinstance(axis, int) or isinstance(axis, bool) or axis < 0 or axis >= rank:
            raise WireError("coordinate axis outside rank")
    elif kind == "dimension":
        _exact(expression, {"kind", "symbolId"}, "index dimension")
        _symbol_id(expression["symbolId"], "index dimension")
    elif kind == "add":
        _exact(expression, {"kind", "terms"}, "index add")
        if not isinstance(expression["terms"], list) or not expression["terms"]:
            raise WireError("index add requires terms")
        for term in expression["terms"]:
            _validate_index(term, rank, depth + 1)
    elif kind == "mul":
        _exact(expression, {"kind", "lhs", "rhs"}, "index mul")
        _validate_index(expression["lhs"], rank, depth + 1)
        _validate_index(expression["rhs"], rank, depth + 1)
    elif kind in {"floorDiv", "ceilDiv", "mod"}:
        _exact(expression, {"kind", "value", "divisor"}, "index division")
        _validate_index(expression["value"], rank, depth + 1)
        _validate_index(expression["divisor"], rank, depth + 1)
    elif kind in {"min", "max"}:
        _exact(expression, {"kind", "values"}, "index nary")
        if not isinstance(expression["values"], list) or not expression["values"]:
            raise WireError("index nary requires values")
        for entry in expression["values"]:
            _validate_index(entry, rank, depth + 1)
    else:
        raise WireError(f"unknown index kind: {kind}")
    return expression


def _validate_predicate(expression: Any, rank: int, depth: int = 1) -> dict[str, Any]:
    if depth > MAX_DEPTH or not isinstance(expression, dict):
        raise WireError("invalid predicate")
    kind = expression.get("kind")
    if kind == "bool":
        _exact(expression, {"kind", "value"}, "boolean predicate")
        if not isinstance(expression["value"], bool):
            raise WireError("predicate value must be boolean")
    elif kind in {"equal", "lessEqual"}:
        _exact(expression, {"kind", "lhs", "rhs"}, "comparison predicate")
        _validate_index(expression["lhs"], rank, depth + 1)
        _validate_index(expression["rhs"], rank, depth + 1)
    elif kind in {"and", "or"}:
        _exact(expression, {"kind", "values"}, "boolean nary")
        if not isinstance(expression["values"], list) or not expression["values"]:
            raise WireError("predicate nary requires values")
        for entry in expression["values"]:
            _validate_predicate(entry, rank, depth + 1)
    elif kind == "not":
        _exact(expression, {"kind", "value"}, "predicate not")
        _validate_predicate(expression["value"], rank, depth + 1)
    else:
        raise WireError(f"unknown predicate kind: {kind}")
    return expression


def _validate_constraint(expression: Any, depth: int = 1) -> dict[str, Any]:
    if depth > MAX_DEPTH or not isinstance(expression, dict):
        raise WireError("invalid constraint")
    kind = expression.get("kind")
    if kind in {"equal", "lessEqual"}:
        _exact(expression, {"kind", "lhs", "rhs"}, "comparison constraint")
        _validate_dim(expression["lhs"], depth + 1)
        _validate_dim(expression["rhs"], depth + 1)
    elif kind in {"nonNegative", "positive"}:
        _exact(expression, {"kind", "value"}, "unary constraint")
        _validate_dim(expression["value"], depth + 1)
    elif kind == "divisible":
        _exact(expression, {"kind", "value", "divisor"}, "divisibility constraint")
        _validate_dim(expression["value"], depth + 1)
        _validate_dim(expression["divisor"], depth + 1)
    else:
        raise WireError(f"unknown constraint kind: {kind}")
    return expression


def _validate_memory_space(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WireError("memory space must be object")
    kind = value.get("kind")
    if kind in {"host", "global", "constant"}:
        _exact(value, {"kind"}, "memory space")
    elif kind == "shared":
        _exact(value, {"kind", "scope"}, "shared memory space")
        if value["scope"] not in {"subgroup", "workgroup", "cluster"}:
            raise WireError("invalid shared-memory scope")
    elif kind == "local":
        _exact(value, {"kind", "scope"}, "local memory space")
        if value["scope"] != "invocation":
            raise WireError("invalid local-memory scope")
    elif kind == "target":
        _exact(value, {"kind", "targetId", "spaceId"}, "target memory space")
        _local_id(value["targetId"], "target ID")
        _local_id(value["spaceId"], "target space ID")
    else:
        raise WireError(f"unknown memory-space kind: {kind}")
    return value


def validate_envelope(value: Any) -> dict[str, Any]:
    envelope = _exact(
        value,
        {"schema", "version", "producer", "artifactId", "payload", "requiredExtensions", "optionalMetadata"},
        "envelope",
        {"optionalMetadata"},
    )
    if envelope["schema"] != "browsergrad.layout":
        raise WireError("wrong schema")
    version = _exact(envelope["version"], {"major", "minor"}, "version")
    if (
        not isinstance(version["major"], int)
        or isinstance(version["major"], bool)
        or version["major"] != 1
        or not isinstance(version["minor"], int)
        or isinstance(version["minor"], bool)
        or version["minor"] < 0
        or version["minor"] > SAFE_INTEGER_MAX
    ):
        raise WireError("unsupported version")
    producer = _exact(envelope["producer"], {"id", "version"}, "producer")
    if not all(isinstance(producer[key], str) and producer[key] for key in ("id", "version")):
        raise WireError("invalid producer")
    if not isinstance(envelope["artifactId"], str) or not envelope["artifactId"]:
        raise WireError("invalid artifact ID")
    if envelope["requiredExtensions"] != []:
        raise WireError("layout v1 parity oracle has no required extensions")
    if "optionalMetadata" in envelope and not isinstance(envelope["optionalMetadata"], dict):
        raise WireError("metadata must be object")
    return envelope


def validate_payload(value: Any) -> dict[str, Any]:
    payload = _exact(value, {"symbols", "constraints", "allocations", "indexMaps", "views"}, "layout payload")
    if not all(isinstance(payload[field], list) for field in payload):
        raise WireError("payload declarations must be arrays")
    symbol_ids: set[str] = set()
    for symbol in payload["symbols"]:
        _exact(symbol, {"id", "domain"}, "symbol")
        symbol_id = _symbol_id(symbol["id"], "symbol ID")
        if symbol_id.startswith("__bg_") or symbol_id in symbol_ids:
            raise WireError("duplicate/reserved symbol ID")
        symbol_ids.add(symbol_id)
        domain = _exact(symbol["domain"], {"min", "max"}, "symbol domain", {"max"})
        minimum = int(_wire_i64(domain["min"], "symbol minimum"), 10)
        if "max" in domain and int(_wire_i64(domain["max"], "symbol maximum"), 10) < minimum:
            raise WireError("invalid symbol domain")
    for constraint in payload["constraints"]:
        _validate_constraint(constraint)

    allocation_ids: set[str] = set()
    for allocation in payload["allocations"]:
        _exact(allocation, {"allocationId", "byteLength", "memorySpace", "alignmentBytes", "aliasSetId"}, "allocation")
        allocation_id = _local_id(allocation["allocationId"], "allocation ID")
        if allocation_id in allocation_ids:
            raise WireError("duplicate allocation ID")
        allocation_ids.add(allocation_id)
        _validate_dim(allocation["byteLength"])
        _validate_memory_space(allocation["memorySpace"])
        _alignment(allocation["alignmentBytes"], "allocation alignment")
        _local_id(allocation["aliasSetId"], "alias-set ID")

    map_ids: set[str] = set()
    for index_map in payload["indexMaps"]:
        _exact(index_map, {"indexMapId", "coordinateRank", "locationUnit", "location", "inBounds"}, "index map")
        map_id = _local_id(index_map["indexMapId"], "index-map ID")
        if map_id in map_ids:
            raise WireError("duplicate index-map ID")
        map_ids.add(map_id)
        rank = index_map["coordinateRank"]
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 or rank > 64:
            raise WireError("invalid rank")
        if index_map["locationUnit"] not in {"element", "byte"}:
            raise WireError("invalid location unit")
        _validate_index(index_map["location"], rank)
        _validate_predicate(index_map["inBounds"], rank)

    view_ids: set[str] = set()
    maps_by_id = {entry["indexMapId"]: entry for entry in payload["indexMaps"]}
    allocations_by_id = {entry["allocationId"]: entry for entry in payload["allocations"]}
    for view in payload["views"]:
        _exact(view, {"viewId", "allocationId", "dtype", "byteOffset", "shape", "indexMapId", "requiredAlignmentBytes"}, "view")
        view_id = _local_id(view["viewId"], "view ID")
        if view_id in view_ids:
            raise WireError("duplicate view ID")
        view_ids.add(view_id)
        if view["allocationId"] not in allocations_by_id or view["indexMapId"] not in maps_by_id:
            raise WireError("dangling view reference")
        if view["dtype"] not in DTYPE_BYTES:
            raise WireError("unknown dtype")
        _validate_dim(view["byteOffset"])
        if not isinstance(view["shape"], list):
            raise WireError("view shape must be array")
        for dimension in view["shape"]:
            _validate_dim(dimension)
        if len(view["shape"]) != maps_by_id[view["indexMapId"]]["coordinateRank"]:
            raise WireError("view/map rank mismatch")
        required_alignment = _alignment(view["requiredAlignmentBytes"], "view alignment")
        dtype_alignment = DTYPE_BYTES[view["dtype"]]
        if required_alignment < dtype_alignment or required_alignment % dtype_alignment:
            raise WireError("dtype alignment not satisfied")
        allocation_alignment = allocations_by_id[view["allocationId"]]["alignmentBytes"]
        if allocation_alignment < required_alignment or allocation_alignment % required_alignment:
            raise WireError("allocation alignment not satisfied")
    return payload


def _dim_symbols(expression: dict[str, Any]) -> set[str]:
    kind = expression["kind"]
    if kind == "symbol":
        return {expression["id"]}
    if kind == "add":
        return set().union(*(_dim_symbols(entry) for entry in expression["terms"]))
    if kind == "mul":
        return _dim_symbols(expression["lhs"]) | _dim_symbols(expression["rhs"])
    if kind in {"floorDiv", "ceilDiv", "mod"}:
        return _dim_symbols(expression["value"]) | _dim_symbols(expression["divisor"])
    if kind in {"min", "max"}:
        return set().union(*(_dim_symbols(entry) for entry in expression["values"]))
    return set()


def _index_symbols(expression: dict[str, Any]) -> set[str]:
    kind = expression["kind"]
    if kind == "dimension":
        return {expression["symbolId"]}
    if kind == "add":
        return set().union(*(_index_symbols(entry) for entry in expression["terms"]))
    if kind == "mul":
        return _index_symbols(expression["lhs"]) | _index_symbols(expression["rhs"])
    if kind in {"floorDiv", "ceilDiv", "mod"}:
        return _index_symbols(expression["value"]) | _index_symbols(expression["divisor"])
    if kind in {"min", "max"}:
        return set().union(*(_index_symbols(entry) for entry in expression["values"]))
    return set()


def _predicate_symbols(expression: dict[str, Any]) -> set[str]:
    kind = expression["kind"]
    if kind in {"equal", "lessEqual"}:
        return _index_symbols(expression["lhs"]) | _index_symbols(expression["rhs"])
    if kind in {"and", "or"}:
        return set().union(*(_predicate_symbols(entry) for entry in expression["values"]))
    if kind == "not":
        return _predicate_symbols(expression["value"])
    return set()


def _constraint_symbols(constraint: dict[str, Any]) -> set[str]:
    kind = constraint["kind"]
    if kind in {"equal", "lessEqual"}:
        return _dim_symbols(constraint["lhs"]) | _dim_symbols(constraint["rhs"])
    if kind in {"nonNegative", "positive"}:
        return _dim_symbols(constraint["value"])
    return _dim_symbols(constraint["value"]) | _dim_symbols(constraint["divisor"])


def _consume_lower_bound_operation(budget: list[int]) -> bool:
    budget[0] += 1
    return budget[0] <= MAX_OPERATIONS


def _bounded_sum(values: list[int], budget: list[int]) -> int | None:
    result = values[0]
    for value in values[1:]:
        if not _consume_lower_bound_operation(budget):
            return None
        result += value
        if max(1, abs(result).bit_length()) > MAX_INTEGER_BITS:
            return None
    return result


def _bounded_product(lhs: int, rhs: int, budget: list[int]) -> int | None:
    if not _consume_lower_bound_operation(budget):
        return None
    if lhs and rhs and max(1, abs(lhs).bit_length()) + max(1, abs(rhs).bit_length()) - 1 > MAX_INTEGER_BITS:
        return None
    result = lhs * rhs
    return result if max(1, abs(result).bit_length()) <= MAX_INTEGER_BITS else None


def _bounded_extreme(values: list[int], kind: str, budget: list[int]) -> int | None:
    result = values[0]
    for value in values[1:]:
        if not _consume_lower_bound_operation(budget):
            return None
        result = min(result, value) if kind == "min" else max(result, value)
    return result


def _dim_lower_bound(
    expression: dict[str, Any],
    minima: dict[str, int],
    budget: list[int] | None = None,
) -> int | None:
    if budget is None:
        budget = [0]
    kind = expression["kind"]
    if kind == "const":
        return int(expression["value"], 10)
    if kind == "symbol":
        return minima.get(expression["id"])
    if kind == "add":
        values = [_dim_lower_bound(entry, minima, budget) for entry in expression["terms"]]
        return None if any(value is None for value in values) else _bounded_sum([value for value in values if value is not None], budget)
    if kind == "mul":
        lhs = _dim_lower_bound(expression["lhs"], minima, budget)
        rhs = _dim_lower_bound(expression["rhs"], minima, budget)
        return _bounded_product(lhs, rhs, budget) if lhs is not None and rhs is not None and lhs >= 0 and rhs >= 0 else None
    if kind in {"min", "max"}:
        values = [_dim_lower_bound(entry, minima, budget) for entry in expression["values"]]
        if any(value is None for value in values):
            return None
        resolved = [value for value in values if value is not None]
        return _bounded_extreme(resolved, kind, budget)
    if kind == "mod":
        return 0 if _dim_lower_bound(expression["divisor"], minima, budget) is not None else None
    return None


def _index_lower_bound(
    expression: dict[str, Any],
    minima: dict[str, int],
    budget: list[int] | None = None,
) -> int | None:
    if budget is None:
        budget = [0]
    kind = expression["kind"]
    if kind == "const":
        return int(expression["value"], 10)
    if kind == "dimension":
        return minima.get(expression["symbolId"])
    if kind == "coordinate":
        return None
    if kind == "add":
        values = [_index_lower_bound(entry, minima, budget) for entry in expression["terms"]]
        return None if any(value is None for value in values) else _bounded_sum([value for value in values if value is not None], budget)
    if kind == "mul":
        lhs = _index_lower_bound(expression["lhs"], minima, budget)
        rhs = _index_lower_bound(expression["rhs"], minima, budget)
        return _bounded_product(lhs, rhs, budget) if lhs is not None and rhs is not None and lhs >= 0 and rhs >= 0 else None
    if kind in {"min", "max"}:
        values = [_index_lower_bound(entry, minima, budget) for entry in expression["values"]]
        if any(value is None for value in values):
            return None
        resolved = [value for value in values if value is not None]
        return _bounded_extreme(resolved, kind, budget)
    if kind == "mod":
        return 0 if _index_lower_bound(expression["divisor"], minima, budget) is not None else None
    return None


def _validate_dim_divisors(expression: dict[str, Any], minima: dict[str, int]) -> None:
    kind = expression["kind"]
    if kind in {"floorDiv", "ceilDiv", "mod"}:
        lower = _dim_lower_bound(expression["divisor"], minima)
        if lower is None or lower <= 0:
            raise WireError("dimension divisor positivity is not statically proved")
        _validate_dim_divisors(expression["value"], minima)
        _validate_dim_divisors(expression["divisor"], minima)
    elif kind == "add":
        for entry in expression["terms"]:
            _validate_dim_divisors(entry, minima)
    elif kind == "mul":
        _validate_dim_divisors(expression["lhs"], minima)
        _validate_dim_divisors(expression["rhs"], minima)
    elif kind in {"min", "max"}:
        for entry in expression["values"]:
            _validate_dim_divisors(entry, minima)


def _validate_index_divisors(expression: dict[str, Any], minima: dict[str, int]) -> None:
    kind = expression["kind"]
    if kind in {"floorDiv", "ceilDiv", "mod"}:
        lower = _index_lower_bound(expression["divisor"], minima)
        if lower is None or lower <= 0:
            raise WireError("index divisor positivity is not statically proved")
        _validate_index_divisors(expression["value"], minima)
        _validate_index_divisors(expression["divisor"], minima)
    elif kind == "add":
        for entry in expression["terms"]:
            _validate_index_divisors(entry, minima)
    elif kind == "mul":
        _validate_index_divisors(expression["lhs"], minima)
        _validate_index_divisors(expression["rhs"], minima)
    elif kind in {"min", "max"}:
        for entry in expression["values"]:
            _validate_index_divisors(entry, minima)


def _validate_predicate_divisors(expression: dict[str, Any], minima: dict[str, int]) -> None:
    kind = expression["kind"]
    if kind in {"equal", "lessEqual"}:
        _validate_index_divisors(expression["lhs"], minima)
        _validate_index_divisors(expression["rhs"], minima)
    elif kind in {"and", "or"}:
        for entry in expression["values"]:
            _validate_predicate_divisors(entry, minima)
    elif kind == "not":
        _validate_predicate_divisors(expression["value"], minima)


def _validate_constraint_divisors(constraint: dict[str, Any], minima: dict[str, int]) -> None:
    kind = constraint["kind"]
    if kind in {"equal", "lessEqual"}:
        _validate_dim_divisors(constraint["lhs"], minima)
        _validate_dim_divisors(constraint["rhs"], minima)
    elif kind in {"nonNegative", "positive"}:
        _validate_dim_divisors(constraint["value"], minima)
    else:
        lower = _dim_lower_bound(constraint["divisor"], minima)
        if lower is None or lower <= 0:
            raise WireError("constraint divisor positivity is not statically proved")
        _validate_dim_divisors(constraint["value"], minima)
        _validate_dim_divisors(constraint["divisor"], minima)


def _eval_dim_partial(
    expression: dict[str, Any],
    bindings: dict[str, int],
    budget: "EvaluationBudget",
) -> int | None:
    kind = expression["kind"]
    if kind == "const":
        return int(expression["value"], 10)
    if kind == "symbol":
        return bindings.get(expression["id"])
    if kind == "add":
        values = [_eval_dim_partial(entry, bindings, budget) for entry in expression["terms"]]
        if any(value is None for value in values):
            return None
        result = values[0]
        if result is None:
            return None
        for value in values[1:]:
            if value is None:
                return None
            result = budget.operation(result + value)
        return result
    if kind == "mul":
        lhs = _eval_dim_partial(expression["lhs"], bindings, budget)
        rhs = _eval_dim_partial(expression["rhs"], bindings, budget)
        return None if lhs is None or rhs is None else budget.operation(lhs * rhs)
    if kind in {"floorDiv", "ceilDiv", "mod"}:
        value = _eval_dim_partial(expression["value"], bindings, budget)
        divisor = _eval_dim_partial(expression["divisor"], bindings, budget)
        if divisor is not None and divisor <= 0:
            raise WireError("nonpositive divisor")
        if value is None or divisor is None:
            return None
        if kind == "floorDiv":
            return budget.operation(value // divisor)
        if kind == "ceilDiv":
            return budget.operation(-((-value) // divisor))
        return budget.operation(value % divisor)
    values = [_eval_dim_partial(entry, bindings, budget) for entry in expression["values"]]
    if any(value is None for value in values):
        return None
    resolved = [value for value in values if value is not None]
    result = resolved[0]
    for value in resolved[1:]:
        budget.operation(value)
        result = min(result, value) if kind == "min" else max(result, value)
    return result


def _eval_index_partial(
    expression: dict[str, Any],
    coordinates: list[int],
    bindings: dict[str, int],
    budget: "EvaluationBudget",
) -> int | None:
    kind = expression["kind"]
    if kind == "const":
        return int(expression["value"], 10)
    if kind == "coordinate":
        return coordinates[expression["axis"]]
    if kind == "dimension":
        return bindings.get(expression["symbolId"])
    if kind == "add":
        values = [_eval_index_partial(entry, coordinates, bindings, budget) for entry in expression["terms"]]
        if any(value is None for value in values):
            return None
        result = values[0]
        if result is None:
            return None
        for value in values[1:]:
            if value is None:
                return None
            result = budget.operation(result + value)
        return result
    if kind == "mul":
        lhs = _eval_index_partial(expression["lhs"], coordinates, bindings, budget)
        rhs = _eval_index_partial(expression["rhs"], coordinates, bindings, budget)
        return None if lhs is None or rhs is None else budget.operation(lhs * rhs)
    if kind in {"floorDiv", "ceilDiv", "mod"}:
        value = _eval_index_partial(expression["value"], coordinates, bindings, budget)
        divisor = _eval_index_partial(expression["divisor"], coordinates, bindings, budget)
        if divisor is not None and divisor <= 0:
            raise WireError("nonpositive divisor")
        if value is None or divisor is None:
            return None
        if kind == "floorDiv":
            return budget.operation(value // divisor)
        if kind == "ceilDiv":
            return budget.operation(-((-value) // divisor))
        return budget.operation(value % divisor)
    values = [_eval_index_partial(entry, coordinates, bindings, budget) for entry in expression["values"]]
    if any(value is None for value in values):
        return None
    resolved = [value for value in values if value is not None]
    result = resolved[0]
    for value in resolved[1:]:
        budget.operation(value)
        result = min(result, value) if kind == "min" else max(result, value)
    return result


def _eval_predicate_partial(
    expression: dict[str, Any],
    coordinates: list[int],
    bindings: dict[str, int],
    budget: "EvaluationBudget",
) -> bool | None:
    kind = expression["kind"]
    if kind == "bool":
        return expression["value"]
    if kind in {"equal", "lessEqual"}:
        lhs = _eval_index_partial(expression["lhs"], coordinates, bindings, budget)
        rhs = _eval_index_partial(expression["rhs"], coordinates, bindings, budget)
        if lhs is None or rhs is None:
            return None
        return lhs == rhs if kind == "equal" else lhs <= rhs
    if kind in {"and", "or"}:
        values = [_eval_predicate_partial(entry, coordinates, bindings, budget) for entry in expression["values"]]
        resolved = [value for value in values if value is not None]
        if kind == "and" and any(value is False for value in resolved):
            return False
        if kind == "or" and any(value is True for value in resolved):
            return True
        if any(value is None for value in values):
            return None
        return all(resolved) if kind == "and" else any(resolved)
    value = _eval_predicate_partial(expression["value"], coordinates, bindings, budget)
    return None if value is None else not value


def _eval_constraint_partial(
    constraint: dict[str, Any],
    bindings: dict[str, int],
    budget: "EvaluationBudget",
) -> bool | None:
    kind = constraint["kind"]
    if kind in {"equal", "lessEqual"}:
        lhs = _eval_dim_partial(constraint["lhs"], bindings, budget)
        rhs = _eval_dim_partial(constraint["rhs"], bindings, budget)
        if lhs is None or rhs is None:
            return None
        return lhs == rhs if kind == "equal" else lhs <= rhs
    if kind in {"nonNegative", "positive"}:
        value = _eval_dim_partial(constraint["value"], bindings, budget)
        if value is None:
            return None
        return value >= 0 if kind == "nonNegative" else value > 0
    value = _eval_dim_partial(constraint["value"], bindings, budget)
    divisor = _eval_dim_partial(constraint["divisor"], bindings, budget)
    if divisor is not None and divisor <= 0:
        raise WireError("nonpositive constraint divisor")
    if value is None or divisor is None:
        return None
    return value % divisor == 0


def verify_semantics(payload: dict[str, Any]) -> None:
    domains = {
        symbol["id"]: (
            int(symbol["domain"]["min"], 10),
            int(symbol["domain"]["max"], 10) if "max" in symbol["domain"] else None,
        )
        for symbol in payload["symbols"]
    }
    declared = set(domains)
    minima = {symbol_id: domain[0] for symbol_id, domain in domains.items()}

    expressions: list[set[str]] = []
    expressions.extend(_constraint_symbols(entry) for entry in payload["constraints"])
    expressions.extend(_dim_symbols(entry["byteLength"]) for entry in payload["allocations"])
    for index_map in payload["indexMaps"]:
        expressions.append(_index_symbols(index_map["location"]))
        expressions.append(_predicate_symbols(index_map["inBounds"]))
    for view in payload["views"]:
        expressions.append(_dim_symbols(view["byteOffset"]))
        expressions.extend(_dim_symbols(entry) for entry in view["shape"])
    referenced = set().union(*expressions) if expressions else set()
    undeclared = referenced - declared
    if undeclared:
        raise WireError(f"undeclared symbols: {sorted(undeclared)}")

    constraint_budget = EvaluationBudget()
    for constraint in payload["constraints"]:
        _validate_constraint_divisors(constraint, minima)
        result = _eval_constraint_partial(constraint, {}, constraint_budget)
        if result is False:
            raise WireError("statically violated constraint")

    for allocation in payload["allocations"]:
        _validate_dim_divisors(allocation["byteLength"], minima)
        value = _eval_dim_partial(allocation["byteLength"], {}, EvaluationBudget())
        if value is not None and (value < 0 or value > U64_MAX):
            raise WireError("allocation byte length outside u64")

    for index_map in payload["indexMaps"]:
        _validate_index_divisors(index_map["location"], minima)
        _validate_predicate_divisors(index_map["inBounds"], minima)
        coordinates = [0] * index_map["coordinateRank"]
        _eval_index_partial(index_map["location"], coordinates, {}, EvaluationBudget())
        _eval_predicate_partial(index_map["inBounds"], coordinates, {}, EvaluationBudget())

    allocations = {entry["allocationId"]: entry for entry in payload["allocations"]}
    for view in payload["views"]:
        allocation = allocations[view["allocationId"]]
        _validate_dim_divisors(view["byteOffset"], minima)
        offset = _eval_dim_partial(view["byteOffset"], {}, EvaluationBudget())
        allocation_length = _eval_dim_partial(allocation["byteLength"], {}, EvaluationBudget())
        if offset is not None and (offset < 0 or offset > U64_MAX):
            raise WireError("view byte offset outside u64")
        if offset is not None and offset % view["requiredAlignmentBytes"]:
            raise WireError("view byte offset violates alignment")
        if offset is not None and allocation_length is not None and offset > allocation_length:
            raise WireError("view byte offset exceeds allocation")
        for dimension in view["shape"]:
            _validate_dim_divisors(dimension, minima)
            value = _eval_dim_partial(dimension, {}, EvaluationBudget())
            if value is not None and (value < 0 or value > U64_MAX):
                raise WireError("view extent outside u64")


def remap_payload(raw: dict[str, Any], scope: str) -> dict[str, Any]:
    result = copy.deepcopy(raw)

    def entity_id(kind: str, ordinal: int) -> str:
        if scope == "provisional":
            return f"@{kind}/{ordinal}"
        return f"bg.entity.{kind}.scope-sha256.{scope}.ordinal.{ordinal}"

    allocations = {
        entry["allocationId"]: entity_id("allocation", index)
        for index, entry in enumerate(raw["allocations"])
    }
    index_maps = {
        entry["indexMapId"]: entity_id("index-map", index)
        for index, entry in enumerate(raw["indexMaps"])
    }
    aliases: dict[str, str] = {}
    for entry in raw["allocations"]:
        aliases.setdefault(entry["aliasSetId"], entity_id("alias-set", len(aliases)))
    for index, entry in enumerate(result["allocations"]):
        raw_entry = raw["allocations"][index]
        entry["allocationId"] = allocations[raw_entry["allocationId"]]
        entry["aliasSetId"] = aliases[raw_entry["aliasSetId"]]
    for index, entry in enumerate(result["indexMaps"]):
        entry["indexMapId"] = index_maps[raw["indexMaps"][index]["indexMapId"]]
    for index, entry in enumerate(result["views"]):
        raw_entry = raw["views"][index]
        entry["viewId"] = entity_id("view", index)
        entry["allocationId"] = allocations[raw_entry["allocationId"]]
        entry["indexMapId"] = index_maps[raw_entry["indexMapId"]]
    return result


def verify_layout_artifact(value: Any) -> dict[str, Any]:
    envelope = validate_envelope(value)
    raw = validate_payload(envelope["payload"])
    provisional = remap_payload(raw, "provisional")
    scope = canonical_hash({"domain": "browsergrad.layout-id-scope.v1", "payload": provisional})
    normalized = remap_payload(raw, scope)
    verify_semantics(normalized)
    normalized_envelope = dict(envelope)
    normalized_envelope["requiredExtensions"] = sorted(envelope["requiredExtensions"])
    normalized_envelope["payload"] = normalized
    canonicalize(normalized_envelope)
    return normalized_envelope


class EvaluationBudget:
    def __init__(self) -> None:
        self.operations = 0

    def operation(self, value: int) -> int:
        self.operations += 1
        if self.operations > MAX_OPERATIONS or max(1, abs(value).bit_length()) > MAX_INTEGER_BITS:
            raise WireError("arithmetic budget exceeded")
        return value


def eval_dim(expression: dict[str, Any], bindings: dict[str, int], budget: EvaluationBudget) -> int:
    kind = expression["kind"]
    if kind == "const":
        return int(expression["value"], 10)
    if kind == "symbol":
        if expression["id"] not in bindings:
            raise WireError(f"missing binding: {expression['id']}")
        return bindings[expression["id"]]
    if kind == "add":
        values = [eval_dim(entry, bindings, budget) for entry in expression["terms"]]
        result = values[0]
        for value in values[1:]:
            result = budget.operation(result + value)
        return result
    if kind == "mul":
        return budget.operation(eval_dim(expression["lhs"], bindings, budget) * eval_dim(expression["rhs"], bindings, budget))
    if kind in {"floorDiv", "ceilDiv", "mod"}:
        value = eval_dim(expression["value"], bindings, budget)
        divisor = eval_dim(expression["divisor"], bindings, budget)
        if divisor <= 0:
            raise WireError("nonpositive divisor")
        if kind == "floorDiv":
            return budget.operation(value // divisor)
        if kind == "ceilDiv":
            return budget.operation(-((-value) // divisor))
        return budget.operation(value % divisor)
    values = [eval_dim(entry, bindings, budget) for entry in expression["values"]]
    result = values[0]
    for value in values[1:]:
        budget.operation(value)
        result = min(result, value) if kind == "min" else max(result, value)
    return result


def eval_index(expression: dict[str, Any], coordinates: list[int], bindings: dict[str, int], budget: EvaluationBudget) -> int:
    kind = expression["kind"]
    if kind == "const":
        return int(expression["value"], 10)
    if kind == "coordinate":
        return coordinates[expression["axis"]]
    if kind == "dimension":
        if expression["symbolId"] not in bindings:
            raise WireError(f"missing binding: {expression['symbolId']}")
        return bindings[expression["symbolId"]]
    if kind == "add":
        values = [eval_index(entry, coordinates, bindings, budget) for entry in expression["terms"]]
        result = values[0]
        for value in values[1:]:
            result = budget.operation(result + value)
        return result
    if kind == "mul":
        return budget.operation(
            eval_index(expression["lhs"], coordinates, bindings, budget)
            * eval_index(expression["rhs"], coordinates, bindings, budget)
        )
    if kind in {"floorDiv", "ceilDiv", "mod"}:
        value = eval_index(expression["value"], coordinates, bindings, budget)
        divisor = eval_index(expression["divisor"], coordinates, bindings, budget)
        if divisor <= 0:
            raise WireError("nonpositive divisor")
        if kind == "floorDiv":
            return budget.operation(value // divisor)
        if kind == "ceilDiv":
            return budget.operation(-((-value) // divisor))
        return budget.operation(value % divisor)
    values = [eval_index(entry, coordinates, bindings, budget) for entry in expression["values"]]
    result = values[0]
    for value in values[1:]:
        budget.operation(value)
        result = min(result, value) if kind == "min" else max(result, value)
    return result


def eval_predicate(expression: dict[str, Any], coordinates: list[int], bindings: dict[str, int], budget: EvaluationBudget) -> bool:
    kind = expression["kind"]
    if kind == "bool":
        return expression["value"]
    if kind in {"equal", "lessEqual"}:
        lhs = eval_index(expression["lhs"], coordinates, bindings, budget)
        rhs = eval_index(expression["rhs"], coordinates, bindings, budget)
        return lhs == rhs if kind == "equal" else lhs <= rhs
    if kind == "and":
        values = [eval_predicate(entry, coordinates, bindings, budget) for entry in expression["values"]]
        return all(values)
    if kind == "or":
        values = [eval_predicate(entry, coordinates, bindings, budget) for entry in expression["values"]]
        return any(values)
    return not eval_predicate(expression["value"], coordinates, bindings, budget)


def trace_coordinate(payload: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    _exact(case, {"viewOrdinal", "coordinates", "bindings"}, "trace case", {"bindings"})
    ordinal = case["viewOrdinal"]
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0 or ordinal >= len(payload["views"]):
        raise WireError("invalid view ordinal")
    if not isinstance(case["coordinates"], list):
        raise WireError("trace coordinates must be an array")
    view = payload["views"][ordinal]
    allocation = next(entry for entry in payload["allocations"] if entry["allocationId"] == view["allocationId"])
    index_map = next(entry for entry in payload["indexMaps"] if entry["indexMapId"] == view["indexMapId"])
    coordinates = [int(_wire_i64(value, "coordinate"), 10) for value in case["coordinates"]]
    if len(coordinates) != index_map["coordinateRank"]:
        raise WireError("coordinate rank mismatch")
    bindings: dict[str, int] = {}
    domains = {entry["id"]: entry["domain"] for entry in payload["symbols"]}
    raw_bindings = case.get("bindings", {})
    if not isinstance(raw_bindings, dict):
        raise WireError("bindings must be an object")
    for key, value in raw_bindings.items():
        if key not in domains:
            raise WireError("binding for undeclared symbol")
        parsed = int(_wire_i64(value, "binding"), 10)
        domain = domains[key]
        if parsed < int(domain["min"], 10) or ("max" in domain and parsed > int(domain["max"], 10)):
            raise WireError("binding outside domain")
        bindings[key] = parsed
    constraint_budget = EvaluationBudget()
    for constraint in payload["constraints"]:
        result = _eval_constraint_partial(constraint, bindings, constraint_budget)
        if result is None:
            raise WireError("missing binding required by constraint")
        if not result:
            raise WireError("runtime binding violates constraint")
    shape = [eval_dim(entry, bindings, EvaluationBudget()) for entry in view["shape"]]
    byte_offset = eval_dim(view["byteOffset"], bindings, EvaluationBudget())
    allocation_bytes = eval_dim(allocation["byteLength"], bindings, EvaluationBudget())
    location = eval_index(index_map["location"], coordinates, bindings, EvaluationBudget())
    predicate = _eval_predicate_partial(index_map["inBounds"], coordinates, bindings, EvaluationBudget())
    if predicate is None:
        raise WireError("missing binding required by index predicate")
    dtype_bytes = DTYPE_BYTES[view["dtype"]]
    delta = location * dtype_bytes if index_map["locationUnit"] == "element" else location
    root_start = byte_offset + delta
    root_end = root_start + dtype_bytes
    if any(value < 0 or value > U64_MAX for value in shape):
        raise WireError("resolved view extent outside u64")
    if byte_offset < 0 or byte_offset > U64_MAX:
        raise WireError("resolved view byte offset outside u64")
    if allocation_bytes < 0 or allocation_bytes > U64_MAX:
        raise WireError("resolved allocation length outside u64")
    if byte_offset % view["requiredAlignmentBytes"]:
        raise WireError("resolved view byte offset violates alignment")
    if location < I64_MIN or location > U64_MAX:
        raise WireError("resolved map location outside address range")
    if root_start < I64_MIN or root_start > U64_MAX or root_end < I64_MIN or root_end > U64_MAX:
        raise WireError("resolved byte address outside address range")
    logical = all(coordinate >= 0 and coordinate < shape[axis] for axis, coordinate in enumerate(coordinates))
    allocation_in_bounds = root_start >= 0 and root_end <= allocation_bytes
    if logical and predicate and allocation_in_bounds and root_start % dtype_bytes:
        raise WireError("resolved access violates dtype alignment")
    return {
        "viewId": view["viewId"],
        "allocationId": allocation["allocationId"],
        "aliasSetId": allocation["aliasSetId"],
        "logicalCoordinates": [str(value) for value in coordinates],
        "logicalShape": [str(value) for value in shape],
        "indexMapId": index_map["indexMapId"],
        "mapLocation": {"unit": index_map["locationUnit"], "value": str(location)},
        "viewByteOffset": str(byte_offset),
        "rootByteStart": str(root_start),
        "rootByteEndExclusive": str(root_end),
        "allocationByteLength": str(allocation_bytes),
        "logicalInBounds": logical,
        "predicateInBounds": predicate,
        "allocationInBounds": allocation_in_bounds,
        "accessInBounds": logical and predicate and allocation_in_bounds,
    }


def parity_result(input_path: Path, cases_path: Path) -> dict[str, Any]:
    envelope = verify_layout_artifact(parse_wire_json_bytes(input_path.read_bytes()))
    cases = parse_wire_json_bytes(cases_path.read_bytes())
    if not isinstance(cases, list):
        raise WireError("trace cases must be an array")
    traces = [trace_coordinate(envelope["payload"], case) for case in cases]
    return {
        "normalizedArtifact": envelope,
        "normalizedPayload": envelope["payload"],
        "canonicalArtifact": canonicalize(envelope),
        "canonicalPayload": canonicalize(envelope["payload"]),
        "canonicalOrderingProbe": canonicalize({"\ue000": 1, "\U0001f600": 2, "a": [2, 1]}),
        "semanticHash": semantic_hash(envelope),
        "traces": traces,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: browsergrad_semantic_core.py INPUT_JSON CASES_JSON", file=sys.stderr)
        return 2
    result = parity_result(Path(argv[1]), Path(argv[2]))
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
