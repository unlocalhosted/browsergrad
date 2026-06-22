import { describe, expect, it } from "vitest";
import { VALID_PROFILE } from "./assignment-fixtures";
import {
  browserGpuCapabilities,
  createAssignmentCapabilityEnvironment,
  evaluateAssignmentCapabilities,
  parseAssignmentProfile,
  requiredAssignmentCapabilities,
} from "../src/index";

describe("assignment capabilities", () => {
  it("derives browser GPU capability labels from detected feature facts", () => {
    expect(browserGpuCapabilities({
      webgpu: true,
      wgslKernel: true,
      cudaLiteCompiler: true,
      cudaCompatibleSubset: true,
      shaderF16: true,
      subgroups: true,
      performanceRubric: true,
      kernelVisualizer: true,
    })).toEqual([
      "cuda-compatible-subset",
      "cuda-lite-compiler",
      "kernel-visualizer",
      "performance-rubric",
      "shader-f16",
      "subgroups",
      "webgpu",
      "wgsl-kernel",
    ]);

    expect(browserGpuCapabilities({
      webgpu: false,
      wgslKernel: true,
      cudaLiteCompiler: true,
      shaderF16: true,
      subgroups: true,
      performanceRubric: true,
    })).toEqual(["performance-rubric"]);
  });

  it("evaluates required and alternative capability gates", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "flash_attention_path",
          kind: "capability",
          options: {
            requires: ["pyodide", "torch-compat"],
            any_of: [["webgpu", "wgsl-kernel"], ["triton-compatible"]],
            message: "FlashAttention labs need a browser or native kernel path.",
          },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(
      evaluateAssignmentCapabilities(result.profile, {
        capabilities: ["pyodide", "torch-compat", "webgpu"],
      }),
    ).toEqual({
      ok: false,
      satisfiedCapabilities: ["pyodide", "torch-compat"],
      missingCapabilities: ["triton-compatible", "wgsl-kernel"],
      capabilityModes: {},
      gates: [
        {
          name: "flash_attention_path",
          ok: false,
          status: "blocked",
          requires: ["pyodide", "torch-compat"],
          anyOf: [["webgpu", "wgsl-kernel"], ["triton-compatible"]],
          selectedAnyOf: [],
          selectedCapabilities: [],
          satisfiedCapabilities: ["pyodide", "torch-compat"],
          missingRequired: [],
          missingAnyOf: [["wgsl-kernel"], ["triton-compatible"]],
          message: "FlashAttention labs need a browser or native kernel path.",
        },
      ],
    });

    expect(
      evaluateAssignmentCapabilities(result.profile, {
        capabilities: ["pyodide", "torch-compat", "webgpu", "wgsl-kernel"],
      }).ok,
    ).toBe(true);
  });

  it("selects the strongest satisfied capability route for each gate", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "gpu_path",
          kind: "capability",
          options: {
            requires: ["webgpu"],
            any_of: [["cuda-compatible-subset"], ["wgsl-kernel"], ["native-cuda-external"]],
          },
        },
        {
          name: "native_only_path",
          kind: "capability",
          options: {
            any_of: [["native-cuda-external"]],
          },
        },
        {
          name: "missing_path",
          kind: "capability",
          options: {
            any_of: [["worker-mesh", "distributed-simulator"]],
          },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const evaluation = evaluateAssignmentCapabilities(result.profile, {
      capabilities: [
        "cuda-compatible-subset",
        "native-cuda-external",
        "webgpu",
        "wgsl-kernel",
      ],
      capabilityModes: {
        "cuda-compatible-subset": "simulated",
        "native-cuda-external": "external",
        webgpu: "browser",
        "wgsl-kernel": "browser",
      },
    });

    expect(evaluation.gates.map((gate) => ({
      name: gate.name,
      status: gate.status,
      selectedAnyOf: gate.selectedAnyOf,
      selectedCapabilities: gate.selectedCapabilities,
    }))).toEqual([
      {
        name: "gpu_path",
        status: "runnable",
        selectedAnyOf: ["wgsl-kernel"],
        selectedCapabilities: ["webgpu", "wgsl-kernel"],
      },
      {
        name: "native_only_path",
        status: "external-only",
        selectedAnyOf: ["native-cuda-external"],
        selectedCapabilities: ["native-cuda-external"],
      },
      {
        name: "missing_path",
        status: "blocked",
        selectedAnyOf: [],
        selectedCapabilities: [],
      },
    ]);
  });

  it("lists required capabilities deterministically and ignores behavioral gates", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "kernel_path",
          kind: "capability",
          options: {
            requires: ["pyodide", "torch-compat", "pyodide"],
            any_of: [["webgpu", "wgsl-kernel"], ["native-cuda-external"]],
          },
        },
        {
          name: "streaming_behavior",
          kind: "streaming",
          options: { max_chunks_before_first_yield: 2 },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(requiredAssignmentCapabilities(result.profile)).toEqual([
      "native-cuda-external",
      "pyodide",
      "torch-compat",
      "webgpu",
      "wgsl-kernel",
    ]);
  });

  it("normalizes browser, simulated, and external capability groups", () => {
    expect(
      createAssignmentCapabilityEnvironment({
        browserCapabilities: ["pyodide", "webgpu", "pyodide"],
        simulatedCapabilities: ["worker-mesh", "distributed-simulator"],
        externalCapabilities: ["native-cuda-external", "webgpu", "worker-mesh"],
      }),
    ).toEqual({
      capabilities: [
        "distributed-simulator",
        "native-cuda-external",
        "pyodide",
        "webgpu",
        "worker-mesh",
      ],
      capabilityModes: {
        "distributed-simulator": "simulated",
        "native-cuda-external": "external",
        pyodide: "browser",
        webgpu: "browser",
        "worker-mesh": "simulated",
      },
    });
  });
});
