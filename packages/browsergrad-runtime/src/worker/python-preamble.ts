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

_bg_mod.assert_pass = _bg_assert_pass
_bg_mod.assert_fail = _bg_assert_fail
_bg_mod.assert_error = _bg_assert_error
_bg_mod.log = _bg_log
_bg_mod.emit_json = _bg_emit_json
_bg_mod.emit_image = _bg_emit_image
_bg_mod.oracle = _bg_oracle
_bg_mod.assignment_context = _bg_assignment_context

_bg_sys.modules["browsergrad"] = _bg_mod

# Clean up loader-local names so user globals stay tidy.
# Safe because every helper captured the names it needs via default args
# at definition time — they don't depend on module globals at call time.
del _bg_mod, _bg_json, _bg_importlib, _bg_os, _bg_sys, _bg_types, _bg_traceback, _bg_native_
del _bg_post_assertion, _bg_post_artifact
del _bg_assert_pass, _bg_assert_fail, _bg_assert_error
del _bg_log, _bg_emit_json, _bg_emit_image, _bg_oracle, _bg_assignment_context
`;
