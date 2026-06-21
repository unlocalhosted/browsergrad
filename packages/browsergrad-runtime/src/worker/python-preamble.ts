/**
 * Python source the worker runs once after Pyodide boots, registering the
 * `browsergrad` module that user code imports.
 *
 * Lives in its own file so integration tests can pin the same string into a
 * Pyodide-in-node session and verify the protocol's Python side end-to-end.
 * Internal — not exported from index.ts.
 *
 * Implementation note: helpers capture every name they reference via
 * keyword-only default arguments. Without that, deleting `_bg_post_artifact`
 * etc. at the end of the preamble would break the helpers, because Python
 * resolves free names in function bodies at *call* time against module
 * globals — the helpers must own their references at definition time.
 */
export const PY_PREAMBLE = `
import json as _bg_json
import importlib as _bg_importlib
import os as _bg_os
import sys as _bg_sys
import types as _bg_types
import traceback as _bg_traceback
import _bg_native as _bg_native_

_bg_mod = _bg_types.ModuleType("browsergrad")
_bg_mod.__doc__ = "Structured assertion + artifact emission for browsergrad runtime"

def _bg_post_assertion(payload, *, _native=_bg_native_, _json=_bg_json):
    _native.postAssertion(_json.dumps(payload))

def _bg_post_artifact(payload, *, _native=_bg_native_, _json=_bg_json):
    _native.postArtifact(_json.dumps(payload, default=str))

def _bg_assert_pass(name, duration_ms=None, *, _post=_bg_post_assertion):
    _post({"kind": "pass", "name": name, "durationMs": duration_ms})

def _bg_assert_fail(name, message, expected=None, actual=None, duration_ms=None,
                   *, _post=_bg_post_assertion):
    _post({
        "kind": "fail",
        "name": name,
        "message": message,
        "expectedRepr": None if expected is None else repr(expected),
        "actualRepr": None if actual is None else repr(actual),
        "durationMs": duration_ms,
    })

def _bg_assert_error(name, message, exc=None, duration_ms=None,
                    *, _post=_bg_post_assertion, _tb=_bg_traceback):
    tb = None
    if exc is not None:
        tb = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
    _post({
        "kind": "error",
        "name": name,
        "message": message,
        "traceback": tb,
        "durationMs": duration_ms,
    })

def _bg_log(name, data, level="info", *, _post=_bg_post_artifact):
    _post({"kind": "log", "name": name, "level": level, "data": str(data)})

def _bg_emit_json(name, data, *, _post=_bg_post_artifact):
    _post({"kind": "json", "name": name, "data": data})

def _bg_emit_image(name, mime, data_base64, *, _post=_bg_post_artifact):
    _post({
        "kind": "image",
        "name": name,
        "mime": mime,
        "dataBase64": data_base64,
    })

def _bg_oracle(name, *, _import_module=_bg_importlib.import_module):
    if not isinstance(name, str) or not name:
        raise ValueError("BrowserGrad oracle module name must be a non-empty string")
    try:
        return _import_module(name)
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != name:
            raise
        raise ImportError(
            f"BrowserGrad oracle module is not registered: {name}"
        ) from exc

def _bg_assignment_context(*, _os=_bg_os, _json=_bg_json):
    def _load_json_env(name, default):
        raw = _os.environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            return _json.loads(raw)
        except Exception as exc:
            raise ValueError(f"{name} must contain valid JSON") from exc

    return {
        "id": _os.environ.get("BROWSERGRAD_ASSIGNMENT_ID"),
        "root": _os.environ.get("BROWSERGRAD_ASSIGNMENT_ROOT"),
        "fixtures_path": _os.environ.get("BROWSERGRAD_FIXTURES_PATH"),
        "allowed_tests": _load_json_env("BROWSERGRAD_ALLOWED_TESTS_JSON", []),
        "behavioral_gates": _load_json_env("BROWSERGRAD_BEHAVIORAL_GATES_JSON", []),
    }

class _bg_StreamingGateViolation(AssertionError):
    pass

class _bg_StreamingInput:
    def __init__(self, gate, iterable):
        self._gate = gate
        self._iterator = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._iterator)
        self._gate._record_input_read()
        return item

class _bg_StreamingOutput:
    def __init__(self, gate, iterable):
        self._gate = gate
        self._iterator = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._iterator)
        self._gate.mark_output_yielded()
        return item

class _bg_StreamingGate:
    def __init__(
        self,
        name,
        iterable,
        max_chunks_before_first_yield,
        *,
        _input_cls=_bg_StreamingInput,
        _output_cls=_bg_StreamingOutput,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("BrowserGrad streaming gate name must be a non-empty string")
        if (
            not isinstance(max_chunks_before_first_yield, int)
            or max_chunks_before_first_yield < 0
        ):
            raise ValueError("max_chunks_before_first_yield must be a non-negative integer")
        self.name = name
        self.max_chunks_before_first_yield = max_chunks_before_first_yield
        self.chunks_consumed = 0
        self.first_output_yielded = False
        self.input = _input_cls(self, iterable)
        self._output_cls = _output_cls

    def _record_input_read(self, *, _violation_cls=_bg_StreamingGateViolation):
        self.chunks_consumed += 1
        if (
            not self.first_output_yielded
            and self.chunks_consumed > self.max_chunks_before_first_yield
        ):
            raise _violation_cls(
                f"{self.name} consumed input eagerly: read "
                f"{self.chunks_consumed} chunks before first output"
            )

    def mark_output_yielded(self):
        self.first_output_yielded = True

    def wrap_output(self, iterable):
        return self._output_cls(self, iterable)

def _bg_streaming_gate(
    name,
    iterable,
    max_chunks_before_first_yield=None,
    *,
    _gate_cls=_bg_StreamingGate,
    _context=_bg_assignment_context,
):
    if max_chunks_before_first_yield is None:
        for gate in _context().get("behavioral_gates", []):
            if gate.get("name") == name and gate.get("kind") == "streaming":
                max_chunks_before_first_yield = gate.get("options", {}).get(
                    "max_chunks_before_first_yield"
                )
                break
    if max_chunks_before_first_yield is None:
        raise ValueError(
            f"BrowserGrad streaming gate is not configured: {name}"
        )
    return _gate_cls(name, iterable, max_chunks_before_first_yield)

class _bg_ForbiddenReadViolation(AssertionError):
    pass

class _bg_ForbiddenReadGate:
    def __init__(self, name, text, methods):
        if not isinstance(name, str) or not name:
            raise ValueError("BrowserGrad forbidden-read gate name must be a non-empty string")
        if not isinstance(text, str):
            raise ValueError("forbidden_read_gate content must be text")
        if not isinstance(methods, (list, tuple, set)) or not all(
            isinstance(method, str) for method in methods
        ):
            raise ValueError("forbidden_read_gate methods must be strings")
        self.name = name
        self._text = text
        self._pos = 0
        self._methods = set(methods)

    def _forbid(self, method, *, _violation_cls=_bg_ForbiddenReadViolation):
        raise _violation_cls(f"{self.name} forbids eager {method}()")

    def read(self, size=-1):
        if "read" in self._methods and (size is None or size < 0):
            self._forbid("read")
        if size is None or size < 0:
            size = len(self._text) - self._pos
        start = self._pos
        end = min(len(self._text), self._pos + size)
        self._pos = end
        return self._text[start:end]

    def readline(self, size=-1):
        if self._pos >= len(self._text):
            return ""
        newline = self._text.find("\\n", self._pos)
        if newline == -1:
            end = len(self._text)
        else:
            end = newline + 1
        if size is not None and size >= 0:
            end = min(end, self._pos + size)
        start = self._pos
        self._pos = end
        return self._text[start:end]

    def readlines(self, hint=-1):
        if "readlines" in self._methods:
            self._forbid("readlines")
        lines = []
        total = 0
        while True:
            line = self.readline()
            if line == "":
                break
            lines.append(line)
            total += len(line)
            if hint is not None and hint > 0 and total >= hint:
                break
        return lines

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if line == "":
            raise StopIteration
        return line

def _bg_forbidden_read_gate(
    name,
    text,
    methods=None,
    *,
    _gate_cls=_bg_ForbiddenReadGate,
    _context=_bg_assignment_context,
):
    if methods is None:
        for gate in _context().get("behavioral_gates", []):
            if gate.get("name") == name and gate.get("kind") == "forbidden-read":
                methods = gate.get("options", {}).get("methods")
                break
    if methods is None:
        raise ValueError(
            f"BrowserGrad forbidden-read gate is not configured: {name}"
        )
    return _gate_cls(name, text, methods)

_bg_mod.assert_pass = _bg_assert_pass
_bg_mod.assert_fail = _bg_assert_fail
_bg_mod.assert_error = _bg_assert_error
_bg_mod.log = _bg_log
_bg_mod.emit_json = _bg_emit_json
_bg_mod.emit_image = _bg_emit_image
_bg_mod.oracle = _bg_oracle
_bg_mod.assignment_context = _bg_assignment_context
_bg_mod.StreamingGateViolation = _bg_StreamingGateViolation
_bg_mod.streaming_gate = _bg_streaming_gate
_bg_mod.ForbiddenReadViolation = _bg_ForbiddenReadViolation
_bg_mod.forbidden_read_gate = _bg_forbidden_read_gate

_bg_sys.modules["browsergrad"] = _bg_mod

# Clean up loader-local names so user globals stay tidy.
# Safe because every helper captured the names it needs via default args
# at definition time — they don't depend on module globals at call time.
del _bg_mod, _bg_json, _bg_importlib, _bg_os, _bg_sys, _bg_types, _bg_traceback, _bg_native_
del _bg_post_assertion, _bg_post_artifact
del _bg_assert_pass, _bg_assert_fail, _bg_assert_error
del _bg_log, _bg_emit_json, _bg_emit_image, _bg_oracle, _bg_assignment_context
del _bg_StreamingGateViolation, _bg_StreamingInput, _bg_StreamingOutput
del _bg_StreamingGate, _bg_streaming_gate
del _bg_ForbiddenReadViolation, _bg_ForbiddenReadGate, _bg_forbidden_read_gate
`;
