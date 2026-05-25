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
`);
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
