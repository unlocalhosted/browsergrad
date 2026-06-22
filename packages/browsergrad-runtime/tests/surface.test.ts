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
import type { ClientToWorker, WorkerToClient } from "../src/protocol";

class FakeWorker {
  readonly messages: ClientToWorker[] = [];
  readonly holdExec: boolean;
  interruptBuffer: Uint8Array | undefined;
  terminated = false;

  private readonly messageListeners = new Set<(event: MessageEvent<WorkerToClient>) => void>();

  constructor(options: { readonly holdExec?: boolean } = {}) {
    this.holdExec = options.holdExec ?? false;
  }

  addEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    if (type !== "message") return;
    this.messageListeners.add(listener as (event: MessageEvent<WorkerToClient>) => void);
  }

  removeEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    if (type !== "message") return;
    this.messageListeners.delete(listener as (event: MessageEvent<WorkerToClient>) => void);
  }

  postMessage(message: ClientToWorker): void {
    this.messages.push(message);
    if (message.kind === "init") {
      this.interruptBuffer = message.interruptBuffer;
      this.emit({ id: message.id, kind: "init:done" });
      return;
    }
    if (message.kind === "exec" && !this.holdExec) {
      this.finishExec(message.id);
    }
  }

  terminate(): void {
    this.terminated = true;
  }

  finishExec(id: number): void {
    this.emit({
      id,
      kind: "exec:done",
      ok: true,
      stdout: "",
      stderr: "",
      error: null,
      durationMs: 1,
    });
  }

  private emit(message: WorkerToClient): void {
    const event = { data: message } as MessageEvent<WorkerToClient>;
    for (const listener of this.messageListeners) listener(event);
  }
}

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

describe("Session worker hardening", () => {
  it("rejects concurrent exec calls on one worker session", async () => {
    const worker = new FakeWorker({ holdExec: true });
    const session = await createSession({
      pyodideIndexURL: "/pyodide/",
      worker: worker as unknown as Worker,
      disableInterruptBuffer: true,
    });

    const first = session.exec({ code: "print('first')" });
    await expect(session.exec({ code: "print('second')" })).rejects.toThrow(/already running/);

    const execMessage = worker.messages.find((message) => message.kind === "exec");
    expect(execMessage).toBeDefined();
    worker.finishExec(execMessage!.id);
    await expect(first).resolves.toMatchObject({ ok: true });
    await session.dispose();
  });

  it("only sets the interrupt byte for active execs and resets it after completion", async () => {
    if (typeof SharedArrayBuffer === "undefined") return;
    const worker = new FakeWorker({ holdExec: true });
    const session = await createSession({
      pyodideIndexURL: "/pyodide/",
      worker: worker as unknown as Worker,
    });
    const interruptBuffer = worker.interruptBuffer;
    expect(interruptBuffer).toBeDefined();

    session.interrupt();
    expect(interruptBuffer![0]).toBe(0);

    const first = session.exec({ code: "while True: pass" });
    expect(interruptBuffer![0]).toBe(0);
    session.interrupt();
    expect(interruptBuffer![0]).toBe(2);

    const execMessage = worker.messages.find((message) => message.kind === "exec");
    expect(execMessage).toBeDefined();
    worker.finishExec(execMessage!.id);
    await first;
    expect(interruptBuffer![0]).toBe(0);
    await session.dispose();
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
