import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  browserGpuCapabilities,
  createAssignmentCapabilityEnvironment,
  type AssignmentCapabilityEnvironmentInput,
  type BrowserGpuCapabilityInput,
} from "../src/index.js";

interface BrowserMappingCase {
  readonly id: string;
  readonly kind: "browser-mapping";
  readonly input: BrowserGpuCapabilityInput;
  readonly expected: readonly string[];
}

interface RoutePrecedenceCase {
  readonly id: string;
  readonly kind: "route-precedence";
  readonly input: AssignmentCapabilityEnvironmentInput;
  readonly expectedCapabilities: readonly string[];
  readonly expectedModes: Readonly<Record<string, "browser" | "simulated" | "external">>;
}

interface Fixture {
  readonly schemaVersion: 1;
  readonly adapterId: "runtime.generic-backend-labels.v0";
  readonly cases: readonly (BrowserMappingCase | RoutePrecedenceCase)[];
}

const fixture = JSON.parse(readFileSync(
  join(dirname(fileURLToPath(import.meta.url)), "fixtures/assignment-requirements.v0.json"),
  "utf8",
)) as Fixture;

describe("frozen legacy assignment requirement behavior", () => {
  for (const testCase of fixture.cases) {
    it(testCase.id, () => {
      if (testCase.kind === "browser-mapping") {
        expect(browserGpuCapabilities(testCase.input)).toEqual(testCase.expected);
        return;
      }

      const environment = createAssignmentCapabilityEnvironment(testCase.input);
      expect(environment.capabilities).toEqual(testCase.expectedCapabilities);
      expect(environment.capabilityModes).toEqual(testCase.expectedModes);
    });
  }

  it("proves all 256 browser mapping combinations", () => {
    const fields = [
      "webgpu",
      "wgslKernel",
      "cudaLiteCompiler",
      "cudaCompatibleSubset",
      "shaderF16",
      "subgroups",
      "performanceRubric",
      "kernelVisualizer",
    ] as const;
    for (let mask = 0; mask < 2 ** fields.length; mask += 1) {
      const input = Object.fromEntries(fields.map((field, bit) => [field, (mask & (1 << bit)) !== 0])) as unknown as Required<BrowserGpuCapabilityInput>;
      const expected: string[] = [];
      if (input.webgpu) {
        expected.push("webgpu");
        if (input.wgslKernel) expected.push("wgsl-kernel");
        if (input.cudaLiteCompiler) expected.push("cuda-lite-compiler");
        if (input.cudaCompatibleSubset) expected.push("cuda-compatible-subset");
        if (input.shaderF16) expected.push("shader-f16");
        if (input.subgroups) expected.push("subgroups");
      }
      if (input.performanceRubric) expected.push("performance-rubric");
      if (input.kernelVisualizer) expected.push("kernel-visualizer");
      expect(browserGpuCapabilities(input), `mask ${mask}`).toEqual(expected.sort());
    }
  });
});
