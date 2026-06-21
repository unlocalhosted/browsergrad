/**
 * Verifies the runtime's PY_PREAMBLE installs a working `browsergrad`
 * module: when user code imports it and calls assert_pass / assert_fail /
 * emit_json / etc., the JS bridge receives correctly-shaped messages.
 *
 * Run via real Pyodide in node — the same engine the runtime worker uses
 * in the browser. This catches any bug in the Python preamble itself.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { loadPyodide } from "pyodide";
import {
  createAssignmentRunPlan,
  parseAssignmentProfile,
} from "../src/index";
import { PY_PREAMBLE } from "../src/worker/python-preamble";

interface PyodideAPI {
  runPythonAsync: (code: string) => Promise<unknown>;
  registerJsModule: (name: string, module: object) => void;
}

let pyodide: PyodideAPI;
let assertions: unknown[] = [];
let artifacts: unknown[] = [];

beforeAll(async () => {
  pyodide = (await loadPyodide({
    stdout: () => {},
    stderr: () => {},
  })) as unknown as PyodideAPI;

  pyodide.registerJsModule("_bg_native", {
    postAssertion: (json: string) => assertions.push(JSON.parse(json)),
    postArtifact: (json: string) => artifacts.push(JSON.parse(json)),
  });

  await pyodide.runPythonAsync(PY_PREAMBLE);
}, 120_000);

async function exec(code: string): Promise<void> {
  assertions = [];
  artifacts = [];
  await pyodide.runPythonAsync(code);
}

describe("browsergrad module — installed by PY_PREAMBLE", () => {
  it("is importable as `browsergrad`", async () => {
    await exec("import browsergrad as bg");
  });

  it("exposes the documented function set", async () => {
    await exec(`
import browsergrad as bg
assert callable(bg.assert_pass)
assert callable(bg.assert_fail)
assert callable(bg.assert_error)
assert callable(bg.log)
assert callable(bg.emit_json)
assert callable(bg.emit_image)
assert callable(bg.oracle)
assert callable(bg.assignment_context)
assert callable(bg.streaming_gate)
`);
  });
});

describe("assignment execution context", () => {
  it("parses launcher-provided assignment environment for Python rubrics", async () => {
    await exec(`
import os
import browsergrad as bg

os.environ["BROWSERGRAD_ASSIGNMENT_ID"] = "cs336-assignment1"
os.environ["BROWSERGRAD_ASSIGNMENT_ROOT"] = "/assignments/cs336-assignment1"
os.environ["BROWSERGRAD_FIXTURES_PATH"] = "/assignments/cs336-assignment1/fixtures"
os.environ["BROWSERGRAD_ALLOWED_TESTS_JSON"] = "[\\"test_train_bpe_tiny\\"]"
os.environ["BROWSERGRAD_BEHAVIORAL_GATES_JSON"] = "[{\\"name\\":\\"encode_iterable_streaming\\",\\"kind\\":\\"streaming\\",\\"options\\":{\\"max_chunks_before_first_yield\\":2}}]"

ctx = bg.assignment_context()
expected = {
    "id": "cs336-assignment1",
    "root": "/assignments/cs336-assignment1",
    "fixtures_path": "/assignments/cs336-assignment1/fixtures",
    "allowed_tests": ["test_train_bpe_tiny"],
    "behavioral_gates": [{
        "name": "encode_iterable_streaming",
        "kind": "streaming",
        "options": {"max_chunks_before_first_yield": 2},
    }],
}
if ctx == expected:
    bg.assert_pass("test_assignment_context")
else:
    bg.assert_fail("test_assignment_context", "wrong assignment context",
                   expected=expected, actual=ctx)
`);
    expect(assertions).toEqual([
      { kind: "pass", name: "test_assignment_context", durationMs: null },
    ]);
  });
});

describe("browser-safe streaming gates", () => {
  it("allows output before the configured input-read limit", async () => {
    await exec(`
import browsergrad as bg

gate = bg.streaming_gate(
    "encode_iterable_streaming",
    ["a", "b", "c"],
    max_chunks_before_first_yield=2,
)

def streaming_consumer(chunks):
    for chunk in chunks:
        yield chunk.upper()

out = gate.wrap_output(streaming_consumer(gate.input))
first = next(iter(out))
if first == "A" and gate.chunks_consumed == 1 and gate.first_output_yielded:
    bg.assert_pass("test_streaming_gate_allows_incremental_output")
else:
    bg.assert_fail(
        "test_streaming_gate_allows_incremental_output",
        "streaming gate state mismatch",
        expected={"first": "A", "chunks": 1, "yielded": True},
        actual={
            "first": first,
            "chunks": gate.chunks_consumed,
            "yielded": gate.first_output_yielded,
        },
    )
`);
    expect(assertions).toEqual([
      {
        kind: "pass",
        name: "test_streaming_gate_allows_incremental_output",
        durationMs: null,
      },
    ]);
  });

  it("raises a clear error when student code consumes input before first output", async () => {
    await expect(
      pyodide.runPythonAsync(`
import browsergrad as bg

gate = bg.streaming_gate(
    "encode_iterable_streaming",
    ["chunk-1", "chunk-2", "chunk-3"],
    max_chunks_before_first_yield=1,
)

def eager_consumer(chunks):
    return "".join(chunks)

eager_consumer(gate.input)
`),
    ).rejects.toThrow(
      "encode_iterable_streaming consumed input eagerly: read 2 chunks before first output",
    );
  });

  it("uses the active assignment streaming gate configuration by name", async () => {
    await expect(
      pyodide.runPythonAsync(`
import os
import browsergrad as bg

os.environ["BROWSERGRAD_BEHAVIORAL_GATES_JSON"] = "[{\\"name\\":\\"encode_iterable_streaming\\",\\"kind\\":\\"streaming\\",\\"options\\":{\\"max_chunks_before_first_yield\\":1}}]"
gate = bg.streaming_gate("encode_iterable_streaming", ["chunk-1", "chunk-2"])

def eager_consumer(chunks):
    return list(chunks)

eager_consumer(gate.input)
`),
    ).rejects.toThrow(
      "encode_iterable_streaming consumed input eagerly: read 2 chunks before first output",
    );
  });
});

describe("profile JS oracle bridge", () => {
  it("lets Python rubrics call a profile-declared JS oracle module", async () => {
    const parsed = parseAssignmentProfile({
      id: "oracle-bridge",
      version: "1.0.0",
      requires_browsergrad: "^0.1.0",
      runtime_packages: [],
      files: {
        root: "/assignments/oracle-bridge",
        rubric_path: "rubric.py",
      },
      allowed_tests: ["test_oracle_bridge"],
      oracles: [
        {
          name: "_bg_test_oracle",
          js_module: "/assets/test-oracle.js",
          export_name: "oracle",
        },
      ],
      gates: [],
      datasets: [],
    });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    const plan = createAssignmentRunPlan(parsed.profile, { capabilities: [] });
    const oracleName = plan.session.jsModules[0]?.name;
    expect(oracleName).toBe("_bg_test_oracle");
    if (!oracleName) return;

    pyodide.registerJsModule(oracleName, {
      score: (text: string) => text.length * 2,
    });

    await exec(`
import browsergrad as bg

oracle = bg.oracle("${oracleName}")
actual = oracle.score("abc")
if actual == 6:
    bg.assert_pass("test_oracle_bridge")
else:
    bg.assert_fail("test_oracle_bridge", "oracle returned wrong score",
                   expected=6, actual=actual)
`);
    expect(assertions).toEqual([
      { kind: "pass", name: "test_oracle_bridge", durationMs: null },
    ]);
  });

  it("raises a clear error when a rubric asks for an unregistered oracle", async () => {
    await expect(
      pyodide.runPythonAsync(`
import browsergrad as bg
bg.oracle("_bg_missing_oracle")
`),
    ).rejects.toThrow(
      "BrowserGrad oracle module is not registered: _bg_missing_oracle",
    );
  });
});

describe("assert_pass / assert_fail / assert_error", () => {
  it("assert_pass sends a `pass` assertion with name and optional duration", async () => {
    await exec(`
import browsergrad as bg
bg.assert_pass("test_one")
bg.assert_pass("test_two", duration_ms=12.5)
`);
    expect(assertions).toEqual([
      { kind: "pass", name: "test_one", durationMs: null },
      { kind: "pass", name: "test_two", durationMs: 12.5 },
    ]);
  });

  it("assert_fail captures expected and actual as repr strings", async () => {
    await exec(`
import browsergrad as bg
bg.assert_fail("sum_check", "wrong sum",
               expected=42, actual=41, duration_ms=2.0)
`);
    expect(assertions).toEqual([
      {
        kind: "fail",
        name: "sum_check",
        message: "wrong sum",
        expectedRepr: "42",
        actualRepr: "41",
        durationMs: 2.0,
      },
    ]);
  });

  it("assert_fail omits expected/actual repr when not passed", async () => {
    await exec(`
import browsergrad as bg
bg.assert_fail("plain_fail", "broke")
`);
    expect(assertions).toEqual([
      {
        kind: "fail",
        name: "plain_fail",
        message: "broke",
        expectedRepr: null,
        actualRepr: null,
        durationMs: null,
      },
    ]);
  });

  it("assert_error attaches a traceback when given an exception", async () => {
    await exec(`
import browsergrad as bg
try:
    raise RuntimeError("kaboom")
except RuntimeError as e:
    bg.assert_error("kaboomed", "caught it", exc=e)
`);
    expect(assertions).toHaveLength(1);
    const a = assertions[0] as { kind: string; name: string; message: string; traceback: string };
    expect(a.kind).toBe("error");
    expect(a.name).toBe("kaboomed");
    expect(a.message).toBe("caught it");
    expect(a.traceback).toContain("RuntimeError: kaboom");
  });
});

describe("log / emit_json / emit_image", () => {
  it("log emits an artifact with default level=info", async () => {
    await exec(`
import browsergrad as bg
bg.log("status", "training complete")
`);
    expect(artifacts).toEqual([
      { kind: "log", name: "status", level: "info", data: "training complete" },
    ]);
  });

  it("log accepts a custom level", async () => {
    await exec(`
import browsergrad as bg
bg.log("warn_event", "memory pressure", level="warn")
`);
    expect((artifacts[0] as { level: string }).level).toBe("warn");
  });

  it("emit_json round-trips dicts / lists / numbers", async () => {
    await exec(`
import browsergrad as bg
bg.emit_json("series", {"x": [1, 2, 3], "y": [4.0, 5.5, 6.25]})
`);
    expect(artifacts[0]).toEqual({
      kind: "json",
      name: "series",
      data: { x: [1, 2, 3], y: [4.0, 5.5, 6.25] },
    });
  });

  it("emit_image preserves mime and base64 data", async () => {
    await exec(`
import browsergrad as bg
bg.emit_image("plot_0", "image/png", "iVBORw0KGgo=")
`);
    expect(artifacts[0]).toEqual({
      kind: "image",
      name: "plot_0",
      mime: "image/png",
      dataBase64: "iVBORw0KGgo=",
    });
  });
});

describe("user-namespace hygiene", () => {
  it("PY_PREAMBLE doesn't leak loader-local names into globals", async () => {
    // None of the _bg_* helpers should be in globals() after the preamble.
    const result = await pyodide.runPythonAsync(`
[name for name in globals() if name.startswith("_bg_")]
`);
    // toJs is auto for PyProxy lists in Pyodide; we cast through unknown.
    const leaked = (result as { toJs?: () => unknown[] }).toJs?.() ?? result;
    expect(leaked).toEqual([]);
  });
});
