import { beforeAll, describe, expect, it } from "vitest";
import {
  createDevice,
  defineCuda1DProgram,
  runCuda1DProgramWebGpu,
  simulateCuda1DProgram,
} from "../src/index";

interface DeviceCheck {
  readonly available: boolean;
  readonly reason?: string;
}

async function checkDevice(): Promise<DeviceCheck> {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { available: false, reason: "navigator.gpu undefined" };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { available: false, reason: "no GPU adapter" };
    return { available: true };
  } catch (error) {
    return {
      available: false,
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

describe("real WebGPU — CUDA-shaped program lowering", () => {
  let deviceCheck: DeviceCheck;

  beforeAll(async () => {
    deviceCheck = await checkDevice();
    if (!deviceCheck.available) {
      console.warn(`[skip] WebGPU not available: ${deviceCheck.reason}`);
    }
  });

  it("runs a SAXPY-shaped Cuda1DProgram through emitted WGSL", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const program = defineCuda1DProgram({
      name: "saxpy_webgpu",
      inputLength: 4,
      outputLength: 4,
      parameters: { a: 2 },
      launch: { blocks: 1, threadsPerBlock: 8 },
      body: [
        {
          op: "if",
          condition: {
            op: "lt",
            left: { op: "threadId" },
            right: { op: "outputLength" },
          },
          body: [
            {
              op: "write",
              index: { op: "threadId" },
              value: {
                op: "add",
                left: {
                  op: "mul",
                  left: { op: "param", name: "a" },
                  right: { op: "read", index: { op: "threadId" } },
                },
                right: { op: "outputRead", index: { op: "threadId" } },
              },
            },
          ],
        },
      ],
    });
    const runInput = {
      initialInput: [1, 2, 3, 4],
      initialOutput: [10, 20, 30, 40],
    };

    const expected = simulateCuda1DProgram(program, runInput).output;
    const actual = await runCuda1DProgramWebGpu(device, program, runInput);

    expect([...actual]).toEqual(expected);
  });
});
