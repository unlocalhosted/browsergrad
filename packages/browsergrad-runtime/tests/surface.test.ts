import { describe, expect, it } from "vitest";
import {
  BrowsergradError,
  createSession,
  runAssignmentJavascriptProfile,
  runVerifiedAssignmentJavascriptProfile,
  type Artifact,
  type ArtifactImage,
  type ArtifactJson,
  type ArtifactLog,
  type Assertion,
  type AssertionFail,
  type AssertionPass,
  type ExecError,
  type ExecResult,
  type Session,
  type SessionOptions,
} from "../src/index";

/**
 * v0 tests cover what we can without a real browser:
 *   1. The public surface exists and types as expected.
 *   2. Argument validation rejects bad input.
 *
 * Worker + Pyodide integration tests come via @vitest/browser (planned).
 */

describe("public surface", () => {
  it("exports createSession as a function", () => {
    expect(typeof createSession).toBe("function");
  });

  it("exports profile-level JavaScript assignment runner", () => {
    expect(typeof runAssignmentJavascriptProfile).toBe("function");
  });

  it("exports verified profile-level JavaScript assignment runner", () => {
    expect(typeof runVerifiedAssignmentJavascriptProfile).toBe("function");
  });

  it("exports BrowsergradError as a constructable error subclass", () => {
    const err = new BrowsergradError("test");
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(BrowsergradError);
    expect(err.name).toBe("BrowsergradError");
    expect(err.message).toBe("test");
  });
});

describe("createSession argument validation", () => {
  it("rejects when pyodideIndexURL is missing", async () => {
    await expect(
      createSession({ pyodideIndexURL: "" }),
    ).rejects.toThrow(/pyodideIndexURL is required/);
  });

  it("rejects when Worker is unavailable (node env)", async () => {
    // node doesn't ship a Worker global; our factory checks for it.
    await expect(
      createSession({ pyodideIndexURL: "/pyodide/v0.26.4/" }),
    ).rejects.toThrow(/Worker is not available|pyodide/i);
  });
});

describe("type shape sanity", () => {
  it("ExecResult has the documented fields", () => {
    const sample: ExecResult = {
      ok: true,
      stdout: "",
      stderr: "",
      assertions: [],
      artifacts: [],
      error: null,
      durationMs: 0,
    };
    expect(sample.ok).toBe(true);
    expect(sample.assertions).toEqual([]);
    expect(sample.artifacts).toEqual([]);
  });

  it("ExecError shape supports all documented kinds", () => {
    const kinds: ExecError["kind"][] = [
      "syntax",
      "runtime",
      "timeout",
      "aborted",
      "interrupted",
      "worker-crash",
    ];
    expect(kinds).toHaveLength(6);
  });

  it("Assertion variants discriminate on kind", () => {
    const pass: AssertionPass = { kind: "pass", name: "t1" };
    const fail: AssertionFail = {
      kind: "fail",
      name: "t2",
      message: "expected x got y",
      expectedRepr: "3",
      actualRepr: "2",
    };
    const all: Assertion[] = [pass, fail];
    expect(all.filter((a) => a.kind === "pass")).toHaveLength(1);
    expect(all.filter((a) => a.kind === "fail")).toHaveLength(1);
  });

  it("Artifact variants discriminate on kind", () => {
    const log: ArtifactLog = { kind: "log", name: "boot", data: "ok" };
    const json: ArtifactJson = {
      kind: "json",
      name: "plot",
      data: { x: [1, 2], y: [3, 4] },
    };
    const image: ArtifactImage = {
      kind: "image",
      name: "filter_0",
      mime: "image/png",
      dataBase64: "iVBORw0KGgo=",
    };
    const all: Artifact[] = [log, json, image];
    expect(all).toHaveLength(3);
  });

  it("SessionOptions accepts the documented optional fields", () => {
    const opts: SessionOptions = {
      pyodideIndexURL: "/pyodide/",
      packages: ["numpy"],
      onPackageProgress: (e) => {
        expect(typeof e.package).toBe("string");
      },
      disableInterruptBuffer: false,
    };
    expect(opts.packages).toEqual(["numpy"]);
  });

  it("ExecOptions accepts the documented optional fields", () => {
    const _opts: import("../src/index").ExecOptions = {
      code: "print('hi')",
      timeoutMs: 1000,
      signal: new AbortController().signal,
      onStdout: () => {},
      onStderr: () => {},
      onAssertion: () => {},
      onArtifact: () => {},
    };
    expect(_opts.code).toBe("print('hi')");
  });

  it("Session interface is structurally compatible with the impl", () => {
    // If SessionImpl drifts from the public Session interface, this fails
    // at type-check time (and so this test won't compile).
    const _check: (s: Session) => Promise<unknown> = async (s) => {
      await s.fs.write("/x", "");
      await s.fs.read("/x");
      await s.fs.readBytes("/x");
      const result = await s.exec({ code: "" });
      expect(result.assertions).toBeDefined();
      expect(result.artifacts).toBeDefined();
      s.interrupt();
      const _can: boolean = s.canInterrupt;
      await s.clearNamespace();
      await s.dispose();
      return _can;
    };
    expect(typeof _check).toBe("function");
  });
});
