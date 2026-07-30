import { describe, expect, it } from "vitest";

import {
  createVerifiedHostGraphArtifact,
  type HostGraphProgram,
} from "@unlocalhosted/browsergrad-semantic-core/graph";
import {
  parseWireU64,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
  SemanticHostGraphBrowserWorkerReportedError,
  executeSemanticHostGraphBrowserWorkerWithPlatform,
  type ExecuteSemanticHostGraphBrowserWorkerInput,
  type SemanticHostGraphWorkerLike,
  type SemanticHostGraphWorkerTransportPlatform,
} from "../src/semantic_host_graph_worker_transport";

const REQUEST_ID = "bg.host-graph.worker.0123456789abcdef0123456789abcdef";
const HASH = "a".repeat(64);
const wire = (value: number): WireU64 => parseWireU64(String(value));

type LaunchMessage =
  Parameters<SemanticHostGraphWorkerLike["postMessage"]>[0];

class FakeWorker implements SemanticHostGraphWorkerLike {
  readonly listeners = new Map<string, Set<(event: never) => void>>();
  terminated = 0;
  launch?: LaunchMessage;
  transfer?: readonly ArrayBuffer[];

  constructor(
    private readonly behavior: (
      worker: FakeWorker,
      message: LaunchMessage,
    ) => void,
  ) {}

  postMessage(
    message: LaunchMessage,
    transfer: readonly ArrayBuffer[],
  ): void {
    this.launch = message;
    this.transfer = transfer;
    this.behavior(this, message);
  }

  addEventListener(
    type: "message" | "error" | "messageerror",
    listener: ((event: { readonly data: unknown }) => void) |
      ((event: unknown) => void),
  ): void {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener as (event: never) => void);
    this.listeners.set(type, listeners);
  }

  removeEventListener(
    type: "message" | "error" | "messageerror",
    listener: ((event: { readonly data: unknown }) => void) |
      ((event: unknown) => void),
  ): void {
    this.listeners.get(type)?.delete(listener as (event: never) => void);
  }

  terminate(): void {
    this.terminated += 1;
  }

  emit(type: "message" | "error" | "messageerror", value: unknown): void {
    for (const listener of this.listeners.get(type) ?? []) {
      listener(
        (type === "message" ? { data: value } : value) as never,
      );
    }
  }
}

function platform(worker: FakeWorker): SemanticHostGraphWorkerTransportPlatform {
  return {
    createWorker: () => worker,
    nextRequestId: () => REQUEST_ID,
    setTimeout: (callback, delayMs) => setTimeout(callback, delayMs),
    clearTimeout: (handle) =>
      clearTimeout(handle as ReturnType<typeof setTimeout>),
  };
}

async function executionInput(
  bytes = new Uint8Array([1, 2, 3, 4]),
): Promise<ExecuteSemanticHostGraphBrowserWorkerInput> {
  const graph = (await createVerifiedHostGraphArtifact(rawCopyProgram(), {
    kernelArtifacts: [],
    layoutArtifacts: [],
  })).artifact;
  return {
    graphArtifact: graph,
    kernelArtifacts: [],
    layoutArtifacts: [],
    request: {
      inputs: [{ rank: wire(0), resourceId: "input", bytes }],
    },
  };
}

function rawCopyProgram(): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 3 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire(1),
    resources: [
      {
        resourceId: "input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "u8",
        byteLength: wire(4),
        alignmentBytes: 1,
      },
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "u8",
        byteLength: wire(4),
        alignmentBytes: 1,
      },
    ],
    nodes: [
      {
        nodeId: "copy",
        kind: "copy",
        dependsOn: [],
        sourceResourceId: "input",
        destinationResourceId: "output",
        mode: "whole-allocation-bytes-per-rank",
      },
      {
        nodeId: "materialize",
        kind: "materialize",
        dependsOn: ["copy"],
        resourceId: "output",
        mode: "host-readback-after-graph-success",
      },
    ],
  };
}

function success(
  requestId: string,
  bytes: Uint8Array,
): Record<string, unknown> {
  return {
    kind: "browsergrad-host-graph-worker-success",
    version: 1,
    protocol: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
    requestId,
    workerExecutionObserved: true,
    outputs: [{ rank: wire(0), resourceId: "output", bytes }],
    trace: {
      profile: "browsergrad.host-graph.webgpu@1",
      backendVersion: "1.30.0",
      graphSemanticHash: HASH,
      pipelineIdentityHash: HASH,
      backendSpecializationHash: HASH,
      failureModel: "fail-stop-no-partial-output-commit",
      executedNodeIds: ["copy", "materialize"],
      expandedStepCount: 2,
      dispatchStepCount: 0,
      copyStepCount: 1,
      materializationCount: 1,
      completedEventIds: [],
      completedRepeats: [],
      completedDynamicDispatches: [],
      completedConditionals: [],
      midGraphFeedbackCount: 0,
      midGraphFeedbackStageCount: 0,
      collectiveReductionStepCount: 0,
      collectiveReplicationStepCount: 0,
      wgslModuleHashes: [],
      plannedTransientGpuBytes: wire(8),
      plannedTransientHostBytes: wire(8),
      plannedTransientWorkingSetBytes: wire(16),
      maxTransientWorkingSetBytes: wire(1024),
      submitted: true,
      device: {
        features: [],
        limits: {
          maxBufferSize: 1024,
          maxStorageBufferBindingSize: 1024,
          maxComputeWorkgroupsPerDimension: 65_535,
          maxComputeInvocationsPerWorkgroup: 256,
          maxComputeWorkgroupSizeX: 256,
          maxBindingsPerBindGroup: 8,
          maxStorageBuffersPerShaderStage: 8,
          maxUniformBuffersPerShaderStage: 8,
        },
      },
    },
  };
}

describe("semantic host-graph browser Worker transport", () => {
  it("snapshots caller bytes, transfers private copies, and terminates after one terminal", async () => {
    const callerBytes = new Uint8Array([1, 2, 3, 4]);
    const worker = new FakeWorker((self, message) => {
      queueMicrotask(() => self.emit(
        "message",
        success(message.requestId, new Uint8Array(message.inputs[0]!.bytes)),
      ));
    });
    const pending = executeSemanticHostGraphBrowserWorkerWithPlatform(
      await executionInput(callerBytes),
      {},
      platform(worker),
    );
    callerBytes.fill(9);
    const result = await pending;

    expect(result.outputs[0]?.bytes).toEqual(new Uint8Array([1, 2, 3, 4]));
    expect(worker.launch?.inputs[0]?.bytes).toEqual(
      new Uint8Array([1, 2, 3, 4]),
    );
    expect(worker.transfer).toHaveLength(2);
    expect(result.transportTrace).toMatchObject({
      topology: "single-dedicated-browser-worker",
      acceptedTerminalMessages: 1,
      workerExecutionObserved: true,
      workerLifecycle: "one-shot-terminated",
      inputByteLength: 4,
      outputByteLength: 4,
    });
    expect(Object.isFrozen(result.backendTrace)).toBe(true);
    expect(Object.isFrozen(result.backendTrace.device.limits)).toBe(true);
    expect(worker.terminated).toBe(1);
    expect(worker.listeners.get("message")).toHaveLength(0);
  });

  it("rejects stale or extended terminal frames and terminates the Worker", async () => {
    for (const terminal of [
      success(
        "bg.host-graph.worker.ffffffffffffffffffffffffffffffff",
        new Uint8Array(4),
      ),
      { ...success(REQUEST_ID, new Uint8Array(4)), unexpected: true },
    ]) {
      const worker = new FakeWorker((self) => {
        queueMicrotask(() => self.emit("message", terminal));
      });
      await expect(executeSemanticHostGraphBrowserWorkerWithPlatform(
        await executionInput(),
        {},
        platform(worker),
      )).rejects.toMatchObject({
        code: "BG-WEBGPU-GRAPH-WORKER-TERMINAL",
        path: "$.terminal",
      });
      expect(worker.terminated).toBe(1);
    }
  });

  it("projects authenticated Worker failures without execution claims", async () => {
    const worker = new FakeWorker((self, message) => {
      queueMicrotask(() => self.emit("message", {
        kind: "browsergrad-host-graph-worker-failure",
        version: 1,
        protocol: SEMANTIC_HOST_GRAPH_BROWSER_WORKER_TRANSPORT_PROTOCOL,
        requestId: message.requestId,
        phase: "device-acquisition",
        failureCode: "BG-WEBGPU-GRAPH-WORKER-INTERNAL",
        failurePath: "$.worker",
        failureDetail: "adapter unavailable",
        workerExecutionObserved: false,
      }));
    });
    await expect(executeSemanticHostGraphBrowserWorkerWithPlatform(
      await executionInput(),
      {},
      platform(worker),
    )).rejects.toBeInstanceOf(SemanticHostGraphBrowserWorkerReportedError);
    expect(worker.terminated).toBe(1);
  });

  it("owns timeout and cancellation by terminating the one-shot Worker", async () => {
    const timeoutWorker = new FakeWorker(() => {});
    await expect(executeSemanticHostGraphBrowserWorkerWithPlatform(
      await executionInput(),
      { timeoutMs: 1 },
      platform(timeoutWorker),
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-WORKER-TIMEOUT",
    });
    expect(timeoutWorker.terminated).toBe(1);

    const controller = new AbortController();
    const cancelledWorker = new FakeWorker(() => {
      queueMicrotask(() => controller.abort());
    });
    await expect(executeSemanticHostGraphBrowserWorkerWithPlatform(
      await executionInput(),
      { signal: controller.signal },
      platform(cancelledWorker),
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-WORKER-CANCELLED",
    });
    expect(cancelledWorker.terminated).toBe(1);
  });

  it("rejects forged graph authority before creating a Worker", async () => {
    let created = false;
    const worker = new FakeWorker(() => {});
    const testPlatform = {
      ...platform(worker),
      createWorker: () => {
        created = true;
        return worker;
      },
    };
    const input = await executionInput();
    await expect(executeSemanticHostGraphBrowserWorkerWithPlatform(
      { ...input, graphArtifact: {} as never },
      {},
      testPlatform,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GRAPH-WORKER-INVALID",
      path: "$.input.graphArtifact",
    });
    expect(created).toBe(false);
  });
});
