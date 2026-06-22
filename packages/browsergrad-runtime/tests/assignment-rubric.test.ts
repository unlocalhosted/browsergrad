import { describe, expect, it } from "vitest";
import { VALID_PROFILE } from "./assignment-fixtures";
import {
  createAssignmentRunPlan,
  parseAssignmentProfile,
  runAssignmentJavascriptRubric,
  runAssignmentRubric,
} from "../src/index";

describe("assignment rubric runners", () => {
  it("rejects assignment rubric runs with failed preflight before mounting files", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "kernel_path",
          kind: "capability",
          options: { requires: ["pyodide", "webgpu"] },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const writes: Array<{ path: string; content: string | Uint8Array }> = [];

    await expect(
      runAssignmentRubric(
        {
          fs: {
            async write(path, content) {
              writes.push({ path, content });
            },
            async read() {
              return "";
            },
            async readBytes() {
              return new Uint8Array();
            },
          },
          async exec() {
            throw new Error("exec should not run");
          },
        },
        plan,
        {
          files: {
            "/assignments/cs336-assignment1/rubric.py": "print('rubric')",
          },
          datasets: {
            tiny: "fixture text",
          },
        },
      ),
    ).rejects.toThrow(
      "cannot create rubric exec request; missing assignment capabilities: webgpu",
    );
    expect(writes).toEqual([]);
  });

  it("runs a JavaScript rubric with mounted text, assignment context, oracles, and assertions", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      id: "gpu-puzzles",
      runtime_packages: [],
      files: {
        root: "/assignments/gpu-puzzles",
        rubric_path: "rubric.js",
        starter_path: "student.js",
        fixtures_path: "fixtures",
      },
      allowed_tests: ["test_js_rubric"],
      oracles: [
        {
          name: "_bg_gpu_oracle",
          js_module: "/assignments/gpu-puzzles/oracles/gpu-puzzles.js",
          export_name: "oracle",
        },
      ],
      gates: [
        {
          name: "js_rubric_runtime",
          kind: "capability",
          options: { requires: ["javascript-rubric"] },
        },
        {
          name: "streaming_contract",
          kind: "streaming",
          options: { max_chunks_before_first_yield: 2 },
        },
      ],
      datasets: [{ name: "tiny", url: "/fixtures/tiny.txt" }],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["javascript-rubric"],
    });

    const run = await runAssignmentJavascriptRubric(
      plan,
      {
        files: {
          "/assignments/gpu-puzzles/rubric.js": "export async function run() {}",
          "/assignments/gpu-puzzles/student.js": "export const answer = 42;",
        },
        datasets: {
          tiny: "fixture text",
        },
      },
      async (ctx) => {
        const oracle = ctx.oracle<{ score(input: string): number }>("_bg_gpu_oracle");
        const fixture = ctx.readText("/assignments/gpu-puzzles/fixtures/datasets/tiny.txt");
        ctx.emitJson("js_rubric_context", {
          id: ctx.id,
          root: ctx.root,
          fixturesPath: ctx.fixturesPath,
          allowedTests: ctx.allowedTests,
          behavioralGates: ctx.behavioralGates,
          rubricSource: ctx.readText("/assignments/gpu-puzzles/rubric.js"),
          studentSource: ctx.readText("/assignments/gpu-puzzles/student.js"),
        });
        if (fixture === "fixture text" && oracle.score(fixture) === 24) {
          ctx.assertPass("test_js_rubric");
        } else {
          ctx.assertFail("test_js_rubric", "JS rubric mismatch", {
            expected: "fixture text / 24",
            actual: `${fixture} / ${oracle.score(fixture)}`,
          });
        }
      },
      {
        oracles: {
          _bg_gpu_oracle: {
            score: (input: string) => input.length * 2,
          },
        },
      },
    );

    expect(run.mount).toEqual({
      writtenPaths: [
        "/assignments/gpu-puzzles/rubric.js",
        "/assignments/gpu-puzzles/student.js",
        "/assignments/gpu-puzzles/fixtures/datasets/tiny.txt",
      ],
      skippedOptionalPaths: [],
    });
    expect(run.assertions).toEqual([
      { kind: "pass", name: "test_js_rubric" },
    ]);
    expect(run.artifacts).toEqual([
      {
        kind: "json",
        name: "js_rubric_context",
        data: {
          id: "gpu-puzzles",
          root: "/assignments/gpu-puzzles",
          fixturesPath: "/assignments/gpu-puzzles/fixtures",
          allowedTests: ["test_js_rubric"],
          behavioralGates: [
            {
              name: "streaming_contract",
              kind: "streaming",
              options: { max_chunks_before_first_yield: 2 },
            },
          ],
          rubricSource: "export async function run() {}",
          studentSource: "export const answer = 42;",
        },
      },
    ]);
  });

  it("rejects JavaScript rubric runs with failed preflight before invoking the rubric", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      files: {
        ...VALID_PROFILE.files,
        rubric_path: "rubric.js",
      },
      gates: [
        {
          name: "js_rubric_runtime",
          kind: "capability",
          options: { requires: ["javascript-rubric", "webgpu"] },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["javascript-rubric"],
    });
    let invoked = false;

    await expect(
      runAssignmentJavascriptRubric(
        plan,
        {
          files: {
            "/assignments/cs336-assignment1/rubric.js": "export async function run() {}",
          },
          datasets: {
            tiny: "fixture text",
          },
        },
        () => {
          invoked = true;
        },
      ),
    ).rejects.toThrow(
      "cannot run JavaScript rubric; missing assignment capabilities: webgpu",
    );
    expect(invoked).toBe(false);
  });

  it("passes browser substrate objects to JavaScript rubrics", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      id: "gpu-puzzles",
      runtime_packages: [],
      files: {
        root: "/assignments/gpu-puzzles",
        rubric_path: "rubric.js",
      },
      allowed_tests: ["test_gpu_substrate"],
      gates: [
        {
          name: "webgpu_path",
          kind: "capability",
          options: { requires: ["javascript-rubric", "webgpu"] },
        },
      ],
      datasets: [],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["javascript-rubric", "webgpu"],
    });

    const run = await runAssignmentJavascriptRubric(
      plan,
      {
        files: {
          "/assignments/gpu-puzzles/rubric.js": "export async function run() {}",
        },
      },
      async (ctx) => {
        const webgpu = ctx.substrate<{ adapter: string }>("webgpu");
        if (webgpu.adapter === "mock-adapter") {
          ctx.assertPass("test_gpu_substrate");
        } else {
          ctx.assertFail("test_gpu_substrate", "wrong substrate");
        }
      },
      {
        substrates: {
          webgpu: { adapter: "mock-adapter" },
        },
      },
    );

    expect(run.assertions).toEqual([
      { kind: "pass", name: "test_gpu_substrate" },
    ]);
  });

  it("lets JavaScript rubrics read binary mounted content", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      id: "alignment-snapshots",
      runtime_packages: [],
      files: {
        root: "/assignments/alignment-snapshots",
        rubric_path: "rubric.js",
        fixtures_path: "fixtures",
      },
      allowed_tests: ["test_binary_snapshot"],
      gates: [
        {
          name: "js_snapshot_rubric",
          kind: "capability",
          options: { requires: ["javascript-rubric", "snapshot-oracle"] },
        },
      ],
      datasets: [{ name: "tiny-npz", url: "/fixtures/tiny.npz" }],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["javascript-rubric", "snapshot-oracle"],
    });
    const snapshotBytes = Uint8Array.of(0x50, 0x4b, 0x03, 0x04);

    const run = await runAssignmentJavascriptRubric(
      plan,
      {
        files: {
          "/assignments/alignment-snapshots/rubric.js": "export async function run() {}",
        },
        datasets: {
          "tiny-npz": snapshotBytes,
        },
      },
      (ctx) => {
        const bytes = ctx.readBytes(
          "/assignments/alignment-snapshots/fixtures/datasets/tiny.npz",
        );
        if (bytes[0] === 0x50 && bytes[1] === 0x4b) {
          ctx.assertPass("test_binary_snapshot");
        } else {
          ctx.assertFail("test_binary_snapshot", "wrong bytes");
        }
      },
    );

    expect(run.assertions).toEqual([
      { kind: "pass", name: "test_binary_snapshot" },
    ]);
  });

  it("throws a clear error when a JavaScript rubric requires a missing substrate", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      files: {
        root: "/assignments/gpu-puzzles",
        rubric_path: "rubric.js",
      },
      gates: [
        {
          name: "webgpu_path",
          kind: "capability",
          options: { requires: ["javascript-rubric", "webgpu"] },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["javascript-rubric", "webgpu"],
    });

    await expect(
      runAssignmentJavascriptRubric(
        plan,
        {
          files: {
            "/assignments/gpu-puzzles/rubric.js": "export async function run() {}",
          },
          datasets: {
            tiny: "fixture text",
          },
        },
        (ctx) => {
          ctx.substrate("webgpu");
        },
      ),
    ).rejects.toThrow("assignment JavaScript substrate is not registered: webgpu");
  });
});
