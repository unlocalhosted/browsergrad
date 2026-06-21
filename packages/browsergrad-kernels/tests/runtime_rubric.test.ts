import { describe, expect, it } from "vitest";
import {
  createAssignmentCapabilityEnvironment,
  createAssignmentRunPlan,
  parseAssignmentProfile,
  runAssignmentJavascriptRubric,
} from "../../browsergrad-runtime/src/index";
import {
  createBrowsergradKernelRubric,
  tensor,
} from "../src/index";

const PROFILE = {
  id: "gpu-puzzles-smoke",
  version: "1.0.0",
  requires_browsergrad: "^0.1.0",
  runtime_packages: [],
  files: {
    root: "/assignments/gpu-puzzles-smoke",
    rubric_path: "rubric.js",
    starter_path: "student.js",
  },
  allowed_tests: ["test_kernel_pass", "test_kernel_fail"],
  oracles: [],
  gates: [
    {
      name: "js_kernel_rubric",
      kind: "capability",
      options: { requires: ["javascript-rubric", "kernel-visualizer"] },
    },
  ],
  datasets: [],
};

describe("BrowserGrad JS rubric integration", () => {
  it("forwards kernel rubric tensor assertions through runtime JS rubrics", async () => {
    const parsed = parseAssignmentProfile(PROFILE);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    const plan = createAssignmentRunPlan(
      parsed.profile,
      createAssignmentCapabilityEnvironment({
        browserCapabilities: ["javascript-rubric", "kernel-visualizer"],
      }),
    );

    const result = await runAssignmentJavascriptRubric(
      plan,
      {
        files: {
          "/assignments/gpu-puzzles-smoke/rubric.js": "export async function run() {}",
          "/assignments/gpu-puzzles-smoke/student.js": "export const kernel = null;",
        },
      },
      async (ctx) => {
        const rubric = createBrowsergradKernelRubric(ctx);
        rubric.assertCloseTensor(
          "test_kernel_pass",
          tensor([2], new Float32Array([1, 2])),
          tensor([2], new Float32Array([1, 2])),
        );
        rubric.assertCloseTensor(
          "test_kernel_fail",
          tensor([2], new Float32Array([1, 2.25])),
          tensor([2], new Float32Array([1, 2])),
          { atol: 1e-3, maxPreview: 2 },
        );
      },
    );

    expect(result.assertions).toEqual([
      { kind: "pass", name: "test_kernel_pass" },
      {
        kind: "fail",
        name: "test_kernel_fail",
        message: "tensor values differ beyond tolerance",
        expectedRepr: "shape=[2] preview=[1,2] value[1]=2 atol=0.001 rtol=0.000001",
        actualRepr: "shape=[2] preview=[1,2.25] value[1]=2.25 maxAbsDiff=0.25",
      },
    ]);
  });
});
