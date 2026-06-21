/**
 * End-to-end assignment runner smoke test.
 *
 * Boots real Pyodide in node and drives the public assignment APIs from a
 * profile through mounting, rubric execution, JS oracle access, assignment
 * context parsing, and structured assertion emission.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { loadPyodide } from "pyodide";
import {
  createAssignmentRunPlan,
  parseAssignmentProfile,
  runAssignmentRubric,
  type Artifact,
  type Assertion,
  type ExecOptions,
  type ExecResult,
  type SessionFS,
} from "../src/index";
import { PY_PREAMBLE } from "../src/worker/python-preamble";

interface PyodideAPI {
  runPythonAsync: (code: string) => Promise<unknown>;
  registerJsModule: (name: string, module: object) => void;
  FS: {
    writeFile: (
      path: string,
      data: string | Uint8Array,
      opts?: { encoding?: string },
    ) => void;
    readFile: (path: string, opts?: { encoding?: string }) => string | Uint8Array;
    mkdirTree: (path: string) => void;
    analyzePath: (path: string) => { exists: boolean };
  };
}

const PROFILE = {
  id: "assignment-smoke",
  version: "1.0.0",
  requires_browsergrad: "^0.1.0",
  runtime_packages: [],
  files: {
    root: "/assignments/assignment-smoke",
    rubric_path: "rubric.py",
    starter_path: "student.py",
    fixtures_path: "fixtures",
  },
  allowed_tests: ["test_assignment_smoke"],
  oracles: [
    {
      name: "_bg_smoke_oracle",
      js_module: "/assets/smoke-oracle.js",
      export_name: "oracle",
    },
  ],
  gates: [
    {
      name: "browser_runtime",
      kind: "capability",
      options: { requires: ["pyodide"] },
    },
    {
      name: "streaming_contract",
      kind: "streaming",
      options: { max_chunks_before_first_yield: 2 },
    },
  ],
  datasets: [{ name: "tiny", url: "/fixtures/tiny.txt" }],
};

const RUBRIC_SOURCE = `
import browsergrad as bg
import student

ctx = bg.assignment_context()
oracle = bg.oracle("_bg_smoke_oracle")
fixture = open(ctx["fixtures_path"] + "/datasets/tiny.txt").read().strip()
actual = {
    "id": ctx["id"],
    "root": ctx["root"],
    "allowed": ctx["allowed_tests"],
    "gate": ctx["behavioral_gates"][0],
    "fixture": fixture,
    "student": student.answer(),
    "score": oracle.score(fixture),
}
expected = {
    "id": "assignment-smoke",
    "root": "/assignments/assignment-smoke",
    "allowed": ["test_assignment_smoke"],
    "gate": {
        "name": "streaming_contract",
        "kind": "streaming",
        "options": {"max_chunks_before_first_yield": 2},
    },
    "fixture": "fixture text",
    "student": "fixture text",
    "score": 24,
}

if actual == expected:
    bg.assert_pass("test_assignment_smoke")
else:
    bg.assert_fail("test_assignment_smoke", "assignment smoke mismatch",
                   expected=expected, actual=actual)
`;

const STARTER_SOURCE = `
def answer():
    with open("/assignments/assignment-smoke/fixtures/datasets/tiny.txt") as f:
        return f.read().strip()
`;

let pyodide: PyodideAPI;
let assertions: Assertion[] = [];
let artifacts: Artifact[] = [];

beforeAll(async () => {
  pyodide = (await loadPyodide({
    stdout: () => {},
    stderr: () => {},
  })) as unknown as PyodideAPI;

  pyodide.registerJsModule("_bg_native", {
    postAssertion: (json: string) => assertions.push(JSON.parse(json) as Assertion),
    postArtifact: (json: string) => artifacts.push(JSON.parse(json) as Artifact),
  });

  await pyodide.runPythonAsync(PY_PREAMBLE);
}, 120_000);

describe("runAssignmentRubric", () => {
  it("mounts files and runs a rubric that calls JS oracles and assignment context", async () => {
    const parsed = parseAssignmentProfile(PROFILE);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    const plan = createAssignmentRunPlan(parsed.profile, {
      capabilities: ["pyodide"],
    });
    const oracleSpec = plan.session.jsModules[0];
    expect(oracleSpec?.name).toBe("_bg_smoke_oracle");
    if (!oracleSpec) return;

    pyodide.registerJsModule(oracleSpec.name, {
      score: (text: string) => text.length * 2,
    });

    assertions = [];
    artifacts = [];
    const result = await runAssignmentRubric(
      {
        fs: pyodideSessionFs(pyodide),
        exec: pyodideExec,
      },
      plan,
      {
        files: {
          [plan.files.rubricPath]: RUBRIC_SOURCE,
          [plan.files.starterPath!]: STARTER_SOURCE,
        },
        datasets: {
          tiny: "fixture text",
        },
      },
    );

    expect(result.mount).toEqual({
      writtenPaths: [
        "/assignments/assignment-smoke/rubric.py",
        "/assignments/assignment-smoke/student.py",
        "/assignments/assignment-smoke/fixtures/datasets/tiny.txt",
      ],
      skippedOptionalPaths: [],
    });
    expect(result.exec.ok).toBe(true);
    expect(result.exec.assertions).toEqual([
      { kind: "pass", name: "test_assignment_smoke", durationMs: null },
    ]);
  });

  it("mounts binary dataset contents for Python rubrics", async () => {
    const parsed = parseAssignmentProfile(PROFILE);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    const plan = createAssignmentRunPlan(parsed.profile, {
      capabilities: ["pyodide"],
    });

    assertions = [];
    artifacts = [];
    const fs = pyodideSessionFs(pyodide);
    const fixtureBytes = Uint8Array.of(0, 1, 255);
    const result = await runAssignmentRubric(
      {
        fs,
        exec: pyodideExec,
      },
      plan,
      {
        files: {
          [plan.files.rubricPath]: `
import browsergrad as bg
with open("/assignments/assignment-smoke/fixtures/datasets/tiny.txt", "rb") as f:
    data = f.read()
if data == bytes([0, 1, 255]):
    bg.assert_pass("test_binary_dataset")
else:
    bg.assert_fail("test_binary_dataset", "binary dataset mismatch", expected=[0, 1, 255], actual=list(data))
`,
        },
        datasets: {
          tiny: fixtureBytes,
        },
      },
    );

    expect(result.exec.ok).toBe(true);
    expect(result.exec.assertions).toEqual([
      { kind: "pass", name: "test_binary_dataset", durationMs: null },
    ]);
    await expect(
      fs.readBytes("/assignments/assignment-smoke/fixtures/datasets/tiny.txt"),
    ).resolves.toEqual(fixtureBytes);
  });
});

function pyodideSessionFs(py: PyodideAPI): SessionFS {
  return {
    async write(path, content) {
      const parent = path.substring(0, path.lastIndexOf("/"));
      if (parent && !py.FS.analyzePath(parent).exists) {
        py.FS.mkdirTree(parent);
      }
      if (typeof content === "string") {
        py.FS.writeFile(path, content, { encoding: "utf8" });
      } else {
        py.FS.writeFile(path, content);
      }
    },
    async read(path) {
      const content = py.FS.readFile(path, { encoding: "utf8" });
      if (typeof content !== "string") {
        throw new Error(`expected text content for ${path}`);
      }
      return content;
    },
    async readBytes(path) {
      const content = py.FS.readFile(path);
      if (typeof content === "string") {
        throw new Error(`expected binary content for ${path}`);
      }
      return content;
    },
  };
}

async function pyodideExec(options: ExecOptions): Promise<ExecResult> {
  const startedAt = performance.now();
  try {
    await pyodide.runPythonAsync(options.code);
    return {
      ok: true,
      stdout: "",
      stderr: "",
      assertions: [...assertions],
      artifacts: [...artifacts],
      error: null,
      durationMs: performance.now() - startedAt,
    };
  } catch (err) {
    return {
      ok: false,
      stdout: "",
      stderr: "",
      assertions: [...assertions],
      artifacts: [...artifacts],
      error: {
        kind: "runtime",
        message: err instanceof Error ? err.message : String(err),
      },
      durationMs: performance.now() - startedAt,
    };
  }
}
