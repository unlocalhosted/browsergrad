/**
 * Fast tests for client.ts message routing. Uses a fake Worker (just
 * implements postMessage / addEventListener / terminate) that returns
 * pre-canned responses on a fast timer. This validates the request/response
 * correlation, streaming-event routing (stdout/stderr/assertion/artifact),
 * AbortSignal cancellation, timeout, and dispose paths — without booting
 * Pyodide.
 */

import { describe, expect, it } from "vitest";
import { createSession, type Session } from "../src/index";

type Listener<E extends Event = Event> = (e: E) => void;

/**
 * A fake Worker that lets the test script reply with whatever it wants.
 * Implements only the surface the runtime touches: postMessage,
 * addEventListener(message|error), terminate.
 */
class FakeWorker {
  private listeners = new Map<string, Listener[]>();
  public terminated = false;
  public lastMessage: unknown = null;
  /** Set by the test — handles incoming postMessage by replying. */
  public handler: ((msg: { id: number; kind: string } & Record<string, unknown>) => void) | null = null;

  addEventListener(type: string, listener: Listener): void {
    const arr = this.listeners.get(type) ?? [];
    arr.push(listener);
    this.listeners.set(type, arr);
  }
  removeEventListener(type: string, listener: Listener): void {
    const arr = this.listeners.get(type);
    if (!arr) return;
    const idx = arr.indexOf(listener);
    if (idx >= 0) arr.splice(idx, 1);
  }
  postMessage(msg: unknown): void {
    this.lastMessage = msg;
    if (this.handler) {
      // Defer so that pending.set() finishes before we deliver the reply.
      setTimeout(() => this.handler!(msg as { id: number; kind: string } & Record<string, unknown>), 0);
    }
  }
  terminate(): void {
    this.terminated = true;
  }
  /** Dispatch a message event to all message listeners. */
  reply(data: unknown): void {
    for (const l of this.listeners.get("message") ?? []) {
      (l as Listener<MessageEvent>)({ data } as MessageEvent);
    }
  }
  errorOut(message: string): void {
    for (const l of this.listeners.get("error") ?? []) {
      (l as Listener<ErrorEvent>)({ message } as ErrorEvent);
    }
  }
}

/** Helper: create a session whose worker auto-acks init then runs `script`. */
async function makeSession(
  fake: FakeWorker,
  options: Partial<Parameters<typeof createSession>[0]> = {},
): Promise<Session> {
  // Auto-handle the init message so createSession's await resolves.
  fake.handler = (msg) => {
    if (msg.kind === "init") fake.reply({ id: msg.id, kind: "init:done" });
  };
  const session = await createSession({
    pyodideIndexURL: "/pyodide/",
    // Cast: FakeWorker is structurally a Worker for the runtime's usage.
    worker: fake as unknown as Worker,
    disableInterruptBuffer: true,
    ...options,
  });
  return session;
}

describe("client request/response correlation", () => {
  it("createSession awaits the init:done reply", async () => {
    const fake = new FakeWorker();
    const session = await makeSession(fake);
    expect(session).toBeDefined();
    await session.dispose();
  });

  it("exec resolves with exec:done payload", async () => {
    const fake = new FakeWorker();
    const session = await makeSession(fake);
    fake.handler = (msg) => {
      if (msg.kind === "exec") {
        fake.reply({
          id: msg.id,
          kind: "exec:done",
          ok: true,
          durationMs: 5,
          stdout: "hi\n",
          stderr: "",
          error: null,
        });
      }
    };
    const result = await session.exec({ code: "print('hi')" });
    expect(result.ok).toBe(true);
    expect(result.stdout).toBe("hi\n");
    expect(result.durationMs).toBe(5);
    await session.dispose();
  });

  it("fs.write then fs.read returns the stored content", async () => {
    const fake = new FakeWorker();
    const session = await makeSession(fake);
    fake.handler = (msg) => {
      if (msg.kind === "fs.write") fake.reply({ id: msg.id, kind: "fs.write:done" });
      else if (msg.kind === "fs.read") fake.reply({ id: msg.id, kind: "fs.read:done", content: "hello world" });
    };
    await session.fs.write("/main.py", "print('hi')");
    const content = await session.fs.read("/main.py");
    expect(content).toBe("hello world");
    await session.dispose();
  });
});

describe("streaming event routing", () => {
  it("onStdout receives chunks before exec resolves", async () => {
    const fake = new FakeWorker();
    const session = await makeSession(fake);
    fake.handler = (msg) => {
      if (msg.kind !== "exec") return;
      // Stream three chunks, then done.
      fake.reply({ id: msg.id, kind: "exec:stdout", chunk: "a" });
      fake.reply({ id: msg.id, kind: "exec:stdout", chunk: "b" });
      fake.reply({ id: msg.id, kind: "exec:stdout", chunk: "c" });
      fake.reply({
        id: msg.id,
        kind: "exec:done",
        ok: true,
        durationMs: 1,
        stdout: "abc",
        stderr: "",
        error: null,
      });
    };
    const chunks: string[] = [];
    const result = await session.exec({
      code: "x",
      onStdout: (c) => chunks.push(c),
    });
    expect(chunks).toEqual(["a", "b", "c"]);
    expect(result.stdout).toBe("abc");
    await session.dispose();
  });

  it("onAssertion captures structured assertions and stores them in result", async () => {
    const fake = new FakeWorker();
    const session = await makeSession(fake);
    fake.handler = (msg) => {
      if (msg.kind !== "exec") return;
      fake.reply({
        id: msg.id,
        kind: "exec:assertion",
        assertion: { kind: "pass", name: "t1" },
      });
      fake.reply({
        id: msg.id,
        kind: "exec:assertion",
        assertion: { kind: "fail", name: "t2", message: "nope" },
      });
      fake.reply({
        id: msg.id,
        kind: "exec:done",
        ok: true,
        durationMs: 1,
        stdout: "",
        stderr: "",
        error: null,
      });
    };
    const observed: Array<{ kind: string; name: string }> = [];
    const result = await session.exec({
      code: "x",
      onAssertion: (a) => observed.push({ kind: a.kind, name: a.name }),
    });
    expect(observed).toEqual([
      { kind: "pass", name: "t1" },
      { kind: "fail", name: "t2" },
    ]);
    expect(result.assertions).toHaveLength(2);
    await session.dispose();
  });

  it("onArtifact routes JSON / log / image events", async () => {
    const fake = new FakeWorker();
    const session = await makeSession(fake);
    fake.handler = (msg) => {
      if (msg.kind !== "exec") return;
      fake.reply({ id: msg.id, kind: "exec:artifact", artifact: { kind: "log", name: "x", data: "hi" } });
      fake.reply({ id: msg.id, kind: "exec:artifact", artifact: { kind: "json", name: "y", data: { a: 1 } } });
      fake.reply({
        id: msg.id, kind: "exec:done", ok: true, durationMs: 0,
        stdout: "", stderr: "", error: null,
      });
    };
    const seen: Array<string> = [];
    await session.exec({ code: "x", onArtifact: (a) => seen.push(a.kind) });
    expect(seen).toEqual(["log", "json"]);
    await session.dispose();
  });
});

describe("error paths", () => {
  it("exec returns ExecResult with error.kind='runtime' when worker reports a Python error", async () => {
    const fake = new FakeWorker();
    const session = await makeSession(fake);
    fake.handler = (msg) => {
      if (msg.kind !== "exec") return;
      fake.reply({
        id: msg.id,
        kind: "exec:done",
        ok: false,
        durationMs: 1,
        stdout: "",
        stderr: "Traceback...\n",
        error: { kind: "runtime", message: "ZeroDivisionError" },
      });
    };
    const result = await session.exec({ code: "1/0" });
    expect(result.ok).toBe(false);
    expect(result.error?.kind).toBe("runtime");
    expect(result.error?.message).toBe("ZeroDivisionError");
    await session.dispose();
  });

  it("timeoutMs returns an aborted-style ExecResult and disposes the worker", async () => {
    const fake = new FakeWorker();
    const session = await makeSession(fake);
    // Never reply to the exec — let timeoutMs fire.
    fake.handler = (msg) => {
      if (msg.kind === "exec") return;
    };
    const result = await session.exec({ code: "while True: pass", timeoutMs: 50 });
    expect(result.ok).toBe(false);
    expect(result.error?.kind).toBe("timeout");
    expect(fake.terminated).toBe(true);
  });

  it("AbortSignal terminates worker after grace period and returns aborted", async () => {
    const fake = new FakeWorker();
    const session = await makeSession(fake);
    fake.handler = (msg) => {
      if (msg.kind === "exec") return; // never reply
    };
    const ctrl = new AbortController();
    const execPromise = session.exec({ code: "while True: pass", signal: ctrl.signal });
    setTimeout(() => ctrl.abort(), 10);
    const result = await execPromise;
    expect(result.ok).toBe(false);
    expect(result.error?.kind).toBe("aborted");
  });
});

describe("dispose", () => {
  it("dispose terminates the worker and rejects in-flight requests", async () => {
    const fake = new FakeWorker();
    const session = await makeSession(fake);
    fake.handler = (msg) => {
      if (msg.kind === "exec") return; // never reply
    };
    const inflight = session.exec({ code: "x" });
    await session.dispose();
    expect(fake.terminated).toBe(true);
    // The in-flight exec returns a synthesized aborted/crash result.
    const result = await inflight;
    expect(result.ok).toBe(false);
  });

  it("dispose is idempotent", async () => {
    const fake = new FakeWorker();
    const session = await makeSession(fake);
    await session.dispose();
    await expect(session.dispose()).resolves.toBeUndefined();
  });

  it("operations after dispose throw BrowsergradError", async () => {
    const fake = new FakeWorker();
    const session = await makeSession(fake);
    await session.dispose();
    await expect(session.fs.write("/x", "y")).rejects.toThrow(/disposed/);
    await expect(session.exec({ code: "x" })).rejects.toThrow(/disposed/);
  });
});
