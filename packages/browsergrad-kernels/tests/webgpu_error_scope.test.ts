import { describe, expect, it, vi } from "vitest";

import {
  ScopedWebGpuIssueError,
  issueAsyncWithWebGpuErrorScopes,
  issueWithWebGpuErrorScopes,
} from "../src/webgpu_error_scope";
import { createDevice } from "../src/device";
import { materializeFloat32 } from "../src/runner";

describe("production WebGPU issue scopes", () => {
  it("pops async issue scopes before awaiting the operation", async () => {
    const events: string[] = [];
    const stack: GPUErrorFilter[] = [];
    let complete!: (value: { readonly root: number }) => void;
    const completion = new Promise<{ readonly root: number }>((resolve) => {
      complete = resolve;
    });
    const pending = issueAsyncWithWebGpuErrorScopes(
      fakeGpu({
        push(scope) {
          events.push(`push:${scope}`);
          stack.push(scope);
        },
        pop: async () => {
          events.push(`pop:${stack.pop()!}`);
          return null;
        },
      }),
      "$.async",
      () => completion,
    );
    expect(events).toEqual([
      "push:internal",
      "push:out-of-memory",
      "push:validation",
      "pop:validation",
      "pop:out-of-memory",
      "pop:internal",
    ]);
    complete({ root: 1 });
    await expect(pending).resolves.toEqual({ root: 1 });
  });

  it("classifies async scope failures and cleans the completed value", async () => {
    const cleanup = vi.fn();
    await expect(issueAsyncWithWebGpuErrorScopes(
      scopedErrorGpu("out-of-memory", "allocation failed"),
      "$.async",
      async () => ({ root: 1 }),
      { cleanup },
    )).rejects.toEqual(expect.objectContaining<Partial<ScopedWebGpuIssueError>>({
      kind: "out-of-memory",
      path: "$.async",
    }));
    expect(cleanup).toHaveBeenCalledWith({ root: 1 });
  });

  it("makes device loss authoritative for a pending async issue", async () => {
    let complete!: (value: { readonly root: number }) => void;
    const operation = new Promise<{ readonly root: number }>((resolve) => {
      complete = resolve;
    });
    const cleanup = vi.fn();
    const pending = issueAsyncWithWebGpuErrorScopes(
      fakeGpu({
        lost: Promise.resolve({
          reason: "destroyed",
          message: "gone",
        } as GPUDeviceLostInfo),
      }),
      "$.async",
      () => operation,
      { cleanup },
    );
    await expect(pending).rejects.toMatchObject({ kind: "device-lost" });
    complete({ root: 1 });
    await Promise.resolve();
    expect(cleanup).toHaveBeenCalledWith({ root: 1 });
  });

  it("calls every pop LIFO before awaiting delayed scopes and operation completion", async () => {
    const events: string[] = [];
    const scopeStack: GPUErrorFilter[] = [];
    const scopeSettlements: Array<(error: GPUError | null) => void> = [];
    let completeOperation!: () => void;
    const operationCompletion = new Promise<void>((resolve) => {
      completeOperation = resolve;
    });
    const gpu = fakeGpu({
      push(scope) {
        events.push(`push:${scope}`);
        scopeStack.push(scope);
      },
      pop() {
        const scope = scopeStack.pop()!;
        events.push(`pop:${scope}`);
        return new Promise<GPUError | null>((resolve) => scopeSettlements.push(resolve));
      },
    });

    const pending = issueWithWebGpuErrorScopes(
      gpu,
      "$.test",
      () => ({ operationCompletion }),
      { completion: ({ operationCompletion: completion }) => completion },
    );
    expect(events).toEqual([
      "push:internal",
      "push:out-of-memory",
      "push:validation",
      "pop:validation",
      "pop:out-of-memory",
      "pop:internal",
    ]);
    scopeSettlements.forEach((settle) => settle(null));
    let settled = false;
    void pending.finally(() => { settled = true; });
    await Promise.resolve();
    expect(settled).toBe(false);
    completeOperation();
    await expect(pending).resolves.toBeDefined();
  });

  it("makes device loss authoritative and cleans a synchronously issued value", async () => {
    const cleanup = vi.fn();
    const gpu = fakeGpu({
      lost: Promise.resolve({ reason: "destroyed", message: "lost" } as GPUDeviceLostInfo),
    });
    await expect(issueWithWebGpuErrorScopes(
      gpu,
      "$.test",
      () => ({ root: 1 }),
      { cleanup },
    )).rejects.toEqual(expect.objectContaining<Partial<ScopedWebGpuIssueError>>({
      name: "ScopedWebGpuIssueError",
      kind: "device-lost",
    }));
    expect(cleanup).toHaveBeenCalledWith({ root: 1 });
  });

  it.each([
    ["validation", "validation"],
    ["out-of-memory", "out-of-memory"],
    ["internal", "internal"],
  ] as const)("classifies a captured %s error and cleans the issued value", async (scope, kind) => {
    const cleanup = vi.fn();
    const gpu = scopedErrorGpu(scope, `${scope} failed`);
    await expect(issueWithWebGpuErrorScopes(
      gpu,
      "$.test",
      () => ({ root: 1 }),
      { cleanup },
    )).rejects.toEqual(expect.objectContaining<Partial<ScopedWebGpuIssueError>>({
      name: "ScopedWebGpuIssueError",
      kind,
      path: "$.test",
      message: expect.stringContaining(`${scope} failed`),
    }));
    expect(cleanup).toHaveBeenCalledOnce();
  });

  it("classifies pop rejection before operation failure and cleans once", async () => {
    const cleanup = vi.fn();
    const stack: GPUErrorFilter[] = [];
    const gpu = fakeGpu({
      push: (scope) => stack.push(scope),
      pop: async () => {
        const scope = stack.pop();
        if (scope === "validation") throw new Error("scope transport failed");
        return null;
      },
    });
    await expect(issueWithWebGpuErrorScopes(
      gpu,
      "$.test",
      () => ({ completion: Promise.reject(new Error("operation failed")) }),
      {
        completion: ({ completion }) => completion,
        cleanup,
      },
    )).rejects.toEqual(expect.objectContaining<Partial<ScopedWebGpuIssueError>>({
      kind: "error-scope",
      message: expect.stringContaining("scope transport failed"),
    }));
    expect(cleanup).toHaveBeenCalledOnce();
  });

  it("awaits completion for an undefined issued value", async () => {
    let rejectCompletion!: (error: Error) => void;
    const completion = new Promise<void>((_resolve, reject) => {
      rejectCompletion = reject;
    });
    const cleanup = vi.fn();
    const pending = issueWithWebGpuErrorScopes<void>(
      fakeGpu(),
      "$.test",
      () => undefined,
      { completion: () => completion, cleanup },
    );
    rejectCompletion(new Error("undefined operation failed"));
    await expect(pending).rejects.toEqual(expect.objectContaining<Partial<ScopedWebGpuIssueError>>({
      kind: "operation",
      message: "undefined operation failed",
    }));
    expect(cleanup).toHaveBeenCalledOnce();
    expect(cleanup).toHaveBeenCalledWith(undefined);
  });

  it("classifies a synchronous completion callback throw and cleans the issued value", async () => {
    const cleanup = vi.fn();
    await expect(issueWithWebGpuErrorScopes(
      fakeGpu(),
      "$.test",
      () => ({ root: 1 }),
      {
        completion: () => { throw new Error("completion callback failed"); },
        cleanup,
      },
    )).rejects.toEqual(expect.objectContaining<Partial<ScopedWebGpuIssueError>>({
      kind: "operation",
      message: "completion callback failed",
    }));
    expect(cleanup).toHaveBeenCalledOnce();
  });

  it("pops only successfully pushed scopes after a synchronous push failure", async () => {
    const events: string[] = [];
    const stack: GPUErrorFilter[] = [];
    const gpu = fakeGpu({
      push(scope) {
        events.push(`push:${scope}`);
        if (scope === "out-of-memory") throw new Error("push failed");
        stack.push(scope);
      },
      pop: async () => {
        const scope = stack.pop()!;
        events.push(`pop:${scope}`);
        return null;
      },
    });
    await expect(issueWithWebGpuErrorScopes(
      gpu,
      "$.test",
      () => ({ root: 1 }),
    )).rejects.toEqual(expect.objectContaining<Partial<ScopedWebGpuIssueError>>({
      kind: "operation",
      message: "push failed",
    }));
    expect(events).toEqual([
      "push:internal",
      "push:out-of-memory",
      "pop:internal",
    ]);
  });

  it("attempts every pop after a synchronous issue throw", async () => {
    const events: string[] = [];
    const stack: GPUErrorFilter[] = [];
    const gpu = fakeGpu({
      push(scope) {
        events.push(`push:${scope}`);
        stack.push(scope);
      },
      pop: async () => {
        const scope = stack.pop()!;
        events.push(`pop:${scope}`);
        return null;
      },
    });
    await expect(issueWithWebGpuErrorScopes(
      gpu,
      "$.test",
      () => { throw new Error("issue failed"); },
    )).rejects.toEqual(expect.objectContaining<Partial<ScopedWebGpuIssueError>>({
      kind: "operation",
      message: "issue failed",
    }));
    expect(events).toEqual([
      "push:internal",
      "push:out-of-memory",
      "push:validation",
      "pop:validation",
      "pop:out-of-memory",
      "pop:internal",
    ]);
  });

  it("makes delayed device loss beat pending scopes and completion", async () => {
    let loseDevice!: (info: GPUDeviceLostInfo) => void;
    const lost = new Promise<GPUDeviceLostInfo>((resolve) => { loseDevice = resolve; });
    const cleanup = vi.fn();
    const gpu = fakeGpu({
      lost,
      pop: () => new Promise<GPUError | null>(() => undefined),
    });
    const pending = issueWithWebGpuErrorScopes(
      gpu,
      "$.test",
      () => ({ completion: new Promise<void>(() => undefined) }),
      { completion: ({ completion }) => completion, cleanup },
    );
    loseDevice({ reason: "unknown", message: "late loss" } as GPUDeviceLostInfo);
    await expect(pending).rejects.toEqual(expect.objectContaining<Partial<ScopedWebGpuIssueError>>({
      kind: "device-lost",
      message: expect.stringContaining("late loss"),
    }));
    expect(cleanup).toHaveBeenCalledOnce();
  });

  it.each([
    ["map rejection", { mapFailure: "map failed" }, "operation"],
    ["validation scope", { failedScope: "validation", scopeMessage: "copy invalid" }, "validation"],
  ] as const)("cleans materialization readback after %s", async (_label, options, kind) => {
    vi.stubGlobal("GPUBufferUsage", { MAP_READ: 1, COPY_DST: 2 });
    vi.stubGlobal("GPUMapMode", { READ: 1 });
    try {
      const fake = materializeGpu(options);
      const device = await createDevice({ device: fake.device });
      await expect(materializeFloat32(device, fake.source, 16)).rejects.toEqual(
        expect.objectContaining<Partial<ScopedWebGpuIssueError>>({ kind }),
      );
      expect(fake.events).toEqual([
        "push:internal",
        "push:out-of-memory",
        "push:validation",
        "copy",
        "submit",
        "map",
        "pop:validation",
        "pop:out-of-memory",
        "pop:internal",
        "destroy",
      ]);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

function materializeGpu(options: Readonly<{
  mapFailure?: string;
  failedScope?: GPUErrorFilter;
  scopeMessage?: string;
}>): Readonly<{ device: GPUDevice; source: GPUBuffer; events: string[] }> {
  const events: string[] = [];
  const scopes: GPUErrorFilter[] = [];
  const readBuffer = {
    size: 16,
    usage: 3,
    mapAsync: () => {
      events.push("map");
      return options.mapFailure === undefined
        ? Promise.resolve()
        : Promise.reject(new Error(options.mapFailure));
    },
    getMappedRange: () => new ArrayBuffer(16),
    unmap: () => events.push("unmap"),
    destroy: () => events.push("destroy"),
  } as unknown as GPUBuffer;
  const source = {
    size: 16,
    usage: 0,
    destroy: () => events.push("source-destroy"),
  } as unknown as GPUBuffer;
  const device = {
    lost: new Promise<GPUDeviceLostInfo>(() => undefined),
    pushErrorScope: (scope: GPUErrorFilter) => {
      events.push(`push:${scope}`);
      scopes.push(scope);
    },
    popErrorScope: async () => {
      const scope = scopes.pop()!;
      events.push(`pop:${scope}`);
      return scope === options.failedScope
        ? { message: options.scopeMessage ?? "scope failed" } as GPUError
        : null;
    },
    createBuffer: () => readBuffer,
    createCommandEncoder: () => ({
      copyBufferToBuffer: () => events.push("copy"),
      finish: () => ({} as GPUCommandBuffer),
    }),
    queue: {
      submit: () => events.push("submit"),
    },
  } as unknown as GPUDevice;
  return { device, source, events };
}

function scopedErrorGpu(
  failedScope: GPUErrorFilter,
  failureMessage: string,
): Pick<GPUDevice, "pushErrorScope" | "popErrorScope" | "lost"> {
  const stack: GPUErrorFilter[] = [];
  return fakeGpu({
    push: (scope) => stack.push(scope),
    pop: async () => {
      const scope = stack.pop()!;
      return scope === failedScope
        ? { message: failureMessage } as GPUError
        : null;
    },
  });
}

function fakeGpu(overrides: Readonly<{
  push?: (scope: GPUErrorFilter) => void;
  pop?: () => Promise<GPUError | null>;
  lost?: Promise<GPUDeviceLostInfo>;
}> = {}): Pick<GPUDevice, "pushErrorScope" | "popErrorScope" | "lost"> {
  return {
    lost: overrides.lost ?? new Promise<GPUDeviceLostInfo>(() => undefined),
    pushErrorScope(scope) {
      overrides.push?.(scope);
      return undefined;
    },
    popErrorScope: overrides.pop ?? (async () => null),
  };
}
