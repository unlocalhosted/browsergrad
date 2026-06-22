/**
 * End-to-end assignment runner smoke test.
 *
 * Boots real Pyodide in node and drives the public assignment APIs from a
 * profile through mounting, rubric execution, JS oracle access, assignment
 * context parsing, and structured assertion emission.
 */

import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { loadPyodide } from "pyodide";
import {
  referenceFlashAttention,
  tensor,
  type Tensor,
} from "@unlocalhosted/browsergrad-kernels";
import {
  createDataCleaningReference,
  createHostedTrainingApiFixture,
  fitPowerLawScalingLaw,
  rl,
  simulation,
} from "@unlocalhosted/browsergrad-primitives";
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

const CS336_A3_PROFILE = JSON.parse(
  readFileSync(
    new URL("../../../docs/internal/cs336-assignment3-scaling.profile.json", import.meta.url),
    "utf8",
  ),
);

const CS336_A2_PROFILE = JSON.parse(
  readFileSync(
    new URL("../../../docs/internal/cs336-assignment2-systems.profile.json", import.meta.url),
    "utf8",
  ),
);

const CS336_A4_PROFILE = JSON.parse(
  readFileSync(
    new URL("../../../docs/internal/cs336-assignment4-data.profile.json", import.meta.url),
    "utf8",
  ),
);

const CS336_A5_PROFILE = JSON.parse(
  readFileSync(
    new URL("../../../docs/internal/cs336-assignment5-alignment.profile.json", import.meta.url),
    "utf8",
  ),
);

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

    expect(result.exec.error).toBeNull();
    expect(result.exec.ok).toBe(true);
    expect(result.exec.assertions).toEqual([
      { kind: "pass", name: "test_binary_dataset", durationMs: null },
    ]);
    await expect(
      fs.readBytes("/assignments/assignment-smoke/fixtures/datasets/tiny.txt"),
    ).resolves.toEqual(fixtureBytes);
  });

  it("runs the CS336 A2 profile with generic attention and distributed references", async () => {
    const parsed = parseAssignmentProfile(CS336_A2_PROFILE);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    const plan = createAssignmentRunPlan(parsed.profile, {
      capabilities: [
        "ddp-simulator",
        "distributed-simulator",
        "flash-attention-oracle",
        "fsdp-simulator",
        "js-oracles",
        "pyodide",
        "sharded-optimizer-simulator",
        "structured-assertions",
        "torch-compat",
        "worker-mesh",
      ],
    });
    const [attentionSpec, distributedSpec] = plan.session.jsModules;
    expect(attentionSpec?.name).toBe("_bg_attention_math");
    expect(distributedSpec?.name).toBe("_bg_distributed_training");
    if (!attentionSpec || !distributedSpec) return;

    pyodide.registerJsModule(attentionSpec.name, createAttentionMathBridge());
    pyodide.registerJsModule(distributedSpec.name, createDistributedTrainingBridge());

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
          [plan.files.rubricPath]: `
import browsergrad as bg
import json

ctx = bg.assignment_context()
attention = bg.oracle("_bg_attention_math")
distributed = bg.oracle("_bg_distributed_training")

forward = json.loads(attention.flash_forward_json(json.dumps({
    "q": {"shape": [1, 1, 2, 2], "data": [1, 0, 0, 1]},
    "k": {"shape": [1, 1, 2, 2], "data": [1, 0, 0, 1]},
    "v": {"shape": [1, 1, 2, 2], "data": [1, 2, 3, 4]},
})))
ddp = json.loads(distributed.ddp_average_json(json.dumps({
    "parameters": [{"name": "w"}],
    "rankGradients": [{"w": [1, 3]}, {"w": [3, 5]}],
})))

if (
    "test_flash_forward_pass_pytorch" in ctx["allowed_tests"]
    and abs(forward["output"][0] - 1.660476923) < 1e-6
    and abs(forward["logSumExp"][0] - 1.107940316) < 1e-6
    and ddp["synchronizedGradients"][0]["w"] == [2, 4]
):
    bg.assert_pass("test_cs336_a2_attention_distributed_references")
    bg.emit_json("cs336-a2-systems-summary", {
        "output0": forward["output"][0],
        "log_sum_exp0": forward["logSumExp"][0],
        "averaged_gradient": ddp["synchronizedGradients"][0]["w"],
    })
else:
    bg.assert_fail(
        "test_cs336_a2_attention_distributed_references",
        "CS336 A2 attention/distributed reference mismatch",
        expected={"output0": 1.660476923, "log_sum_exp0": 1.107940316, "averaged_gradient": [2, 4]},
        actual={"forward": forward, "ddp": ddp},
    )
`,
        },
        datasets: {
          "ddp-test-data": Uint8Array.of(1, 2, 3),
          "ddp-test-labels": Uint8Array.of(0, 1),
        },
      },
    );

    expect(result.exec.error).toBeNull();
    expect(result.exec.ok).toBe(true);
    expect(result.exec.assertions).toEqual([
      {
        kind: "pass",
        name: "test_cs336_a2_attention_distributed_references",
        durationMs: null,
      },
    ]);
    expect(result.exec.artifacts).toEqual([
      expect.objectContaining({
        kind: "json",
        name: "cs336-a2-systems-summary",
        data: expect.objectContaining({
          averaged_gradient: [2, 4],
        }),
      }),
    ]);
  });

  it("runs the CS336 A3 profile with a primitive-facade scaling oracle", async () => {
    const parsed = parseAssignmentProfile(CS336_A3_PROFILE);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    const plan = createAssignmentRunPlan(parsed.profile, {
      capabilities: [
        "hosted-api-mock",
        "http-client",
        "pyodide",
        "scaling-law-oracle",
        "scheduler-simulator",
        "server-fixture",
      ],
    });
    const oracleSpec = plan.session.jsModules[0];
    expect(oracleSpec?.name).toBe("_bg_scaling_api");
    if (!oracleSpec) return;

    const api = createHostedTrainingApiFixture({
      totalBudgetSeconds: 30,
      users: [{ userId: "student-001", apiKey: "a3-key" }],
    });
    pyodide.registerJsModule(oracleSpec.name, {
      budgetAfterSubmit() {
        const response = api.submitExperiment("a3-key", {
          model: "tiny-transformer",
          max_runtime_seconds: 10,
        });
        return response.budgetSummary.remainingSeconds;
      },
      duplicateSubmitStatus() {
        try {
          api.submitExperiment("a3-key", {
            model: "tiny-transformer",
            max_runtime_seconds: 10,
          });
        } catch (error) {
          return error instanceof Error ? error.message : "unknown";
        }
        return "no-error";
      },
      scalingPrediction() {
        const fit = fitPowerLawScalingLaw(
          [
            { compute: 1, loss: 4 },
            { compute: 4, loss: 2 },
            { compute: 16, loss: 1 },
          ],
          { x: "compute", y: "loss" },
        );
        return fit.predict(64);
      },
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
          [plan.files.rubricPath]: `
import browsergrad as bg

ctx = bg.assignment_context()
oracle = bg.oracle("_bg_scaling_api")

remaining = oracle.budgetAfterSubmit()
duplicate = oracle.duplicateSubmitStatus()
prediction = oracle.scalingPrediction()

if (
    "test_submit_jobs" in ctx["allowed_tests"]
    and remaining == 20
    and "experiment already exists" in duplicate
    and abs(prediction - 0.5) < 1e-9
):
    bg.assert_pass("test_cs336_a3_scaling_primitive_facade")
    bg.emit_json("cs336-a3-scaling-summary", {
        "remaining": remaining,
        "prediction": prediction,
    })
else:
    bg.assert_fail(
        "test_cs336_a3_scaling_primitive_facade",
        "CS336 A3 primitive facade oracle mismatch",
        expected={"remaining": 20, "duplicate": "experiment already exists", "prediction": 0.5},
        actual={"remaining": remaining, "duplicate": duplicate, "prediction": prediction},
    )
`,
        },
      },
    );

    expect(result.exec.ok).toBe(true);
    expect(result.exec.assertions).toEqual([
      {
        kind: "pass",
        name: "test_cs336_a3_scaling_primitive_facade",
        durationMs: null,
      },
    ]);
    expect(result.exec.artifacts).toEqual([
      expect.objectContaining({
        kind: "json",
        name: "cs336-a3-scaling-summary",
        data: { remaining: 20, prediction: 0.5 },
      }),
    ]);
  });

  it("runs the CS336 A4 profile with a generic data-cleaning reference", async () => {
    const parsed = parseAssignmentProfile(CS336_A4_PROFILE);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    const plan = createAssignmentRunPlan(parsed.profile, {
      capabilities: [
        "classifier-oracle",
        "dataset-fixture",
        "dedupe-oracle",
        "large-file-streaming",
        "near-dedupe-oracle",
        "pii-oracle",
        "pyodide",
        "quality-rule-oracle",
      ],
    });
    const oracleSpec = plan.session.jsModules[0];
    expect(oracleSpec?.name).toBe("_bg_data_cleaning");
    if (!oracleSpec) return;

    pyodide.registerJsModule(oracleSpec.name, createDataCleaningBridge());

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
          [plan.files.rubricPath]: `
import browsergrad as bg
import json

ctx = bg.assignment_context()
data = bg.oracle("_bg_data_cleaning")

html_text = data.extract_visible_text_from_html("<html><style>x</style><p>Hello&nbsp;data</p><script>bad()</script></html>")
masked = data.mask_pii_text("Email jane@example.com or call 415-555-1212 from 127.0.0.1")
kept = json.loads(data.exact_line_deduplicate_kept_json('["alpha", "beta", "alpha"]'))
near_duplicate_count = data.minhash_duplicate_count_json(
    '[{"id": "a", "text": "permission is hereby granted free of charge"},'
    ' {"id": "b", "text": "permission is hereby granted free of charge"}]'
)
quality_passed = data.gopher_quality_passed("high quality words " * 80)

if (
    "test_extract_text_from_html_bytes" in ctx["allowed_tests"]
    and html_text == "Hello data"
    and masked == "Email <EMAIL> or call <PHONE> from <IP>"
    and kept == ["alpha", "beta"]
    and near_duplicate_count == 1
    and quality_passed is True
):
    bg.assert_pass("test_cs336_a4_data_primitive_facade")
    bg.emit_json("cs336-a4-data-summary", {
        "html_text": html_text,
        "masked": masked,
        "kept": kept,
        "near_duplicate_count": near_duplicate_count,
    })
else:
    bg.assert_fail(
        "test_cs336_a4_data_primitive_facade",
        "CS336 A4 primitive facade oracle mismatch",
        expected={"html_text": "Hello data", "near_duplicate_count": 1},
        actual={
            "html_text": html_text,
            "masked": masked,
            "kept": kept,
            "near_duplicate_count": near_duplicate_count,
            "quality_passed": quality_passed,
        },
    )
`,
        },
        datasets: {
          "moby-html": "<p>Hello&nbsp;data</p>",
          "low-quality-cc": "short text",
        },
      },
    );

    expect(result.exec.error).toBeNull();
    expect(result.exec.ok).toBe(true);
    expect(result.exec.assertions).toEqual([
      {
        kind: "pass",
        name: "test_cs336_a4_data_primitive_facade",
        durationMs: null,
      },
    ]);
    expect(result.exec.artifacts).toEqual([
      expect.objectContaining({
        kind: "json",
        name: "cs336-a4-data-summary",
        data: expect.objectContaining({
          html_text: "Hello data",
          near_duplicate_count: 1,
        }),
      }),
    ]);
  });

  it("runs the CS336 A5 profile with a generic RL math reference", async () => {
    const parsed = parseAssignmentProfile(CS336_A5_PROFILE);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    const plan = createAssignmentRunPlan(parsed.profile, {
      capabilities: [
        "browser-math-only",
        "pyodide",
        "response-parser-oracle",
        "rl-loss-oracle",
        "snapshot-oracle",
        "tokenizer-oracle",
        "torch-compat",
      ],
    });
    const oracleSpec = plan.session.jsModules[0];
    expect(oracleSpec?.name).toBe("_bg_rl_math");
    if (!oracleSpec) return;

    pyodide.registerJsModule(oracleSpec.name, createRlMathBridge());

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
          [plan.files.rubricPath]: `
import browsergrad as bg
import json

ctx = bg.assignment_context()
math = bg.oracle("_bg_rl_math")

dpo_loss = math.dpo_loss_json(json.dumps({
    "beta": 0.1,
    "policyChosenLogProbability": -1.0,
    "policyRejectedLogProbability": -2.0,
    "referenceChosenLogProbability": -1.4,
    "referenceRejectedLogProbability": -1.8,
}))
mmlu = math.parse_mmlu_response_json(json.dumps({
    "options": ["A", "B", "C", "D"],
}), "The answer is (C).")
gsm8k = math.parse_gsm8k_response("First 12, then 1,234.")
advantages = json.loads(math.group_normalized_rewards_json(json.dumps({
    "rawRewards": [1, 3],
    "groupSize": 2,
    "baseline": "mean",
    "advantageNormalizer": "none",
})))["advantages"]

if (
    "test_per_instance_dpo_loss" in ctx["allowed_tests"]
    and abs(dpo_loss - 0.663597113) < 1e-6
    and mmlu == "C"
    and gsm8k == "1234"
    and advantages == [-1, 1]
):
    bg.assert_pass("test_cs336_a5_rl_math_reference")
    bg.emit_json("cs336-a5-rl-summary", {
        "dpo_loss": dpo_loss,
        "mmlu": mmlu,
        "gsm8k": gsm8k,
        "advantages": advantages,
    })
else:
    bg.assert_fail(
        "test_cs336_a5_rl_math_reference",
        "CS336 A5 RL math reference mismatch",
        expected={"dpo_loss": 0.663597113, "mmlu": "C", "gsm8k": "1234", "advantages": [-1, 1]},
        actual={"dpo_loss": dpo_loss, "mmlu": mmlu, "gsm8k": gsm8k, "advantages": advantages},
    )
`,
        },
        datasets: {
          "sft-sample": "{\"prompt\":\"2+2\",\"response\":\"4\"}\\n",
        },
      },
    );

    expect(result.exec.error).toBeNull();
    expect(result.exec.ok).toBe(true);
    expect(result.exec.assertions).toEqual([
      {
        kind: "pass",
        name: "test_cs336_a5_rl_math_reference",
        durationMs: null,
      },
    ]);
    expect(result.exec.artifacts).toEqual([
      expect.objectContaining({
        kind: "json",
        name: "cs336-a5-rl-summary",
        data: expect.objectContaining({
          mmlu: "C",
          gsm8k: "1234",
          advantages: [-1, 1],
        }),
      }),
    ]);
  });
});

function createDataCleaningBridge(): object {
  const reference = createDataCleaningReference();
  return {
    extract_visible_text_from_html(html: string) {
      return reference.extractVisibleTextFromHtml(html);
    },
    mask_pii_text(input: string) {
      return reference.maskPii(input).text;
    },
    exact_line_deduplicate_kept_json(linesJson: string) {
      const lines = JSON.parse(linesJson) as string[];
      return JSON.stringify(reference.exactLineDeduplicate(lines).keptLines);
    },
    minhash_duplicate_count_json(documentsJson: string) {
      const documents = JSON.parse(documentsJson) as {
        id: string;
        text: string;
      }[];
      return reference.minhashDeduplicateDocuments(documents).duplicates.length;
    },
    gopher_quality_passed(text: string) {
      return reference.evaluateGopherQuality(text).passed;
    },
  };
}

interface SerializedTensor {
  readonly shape: readonly number[];
  readonly data: readonly number[];
}

function createAttentionMathBridge(): object {
  return {
    flash_forward_json(inputJson: string) {
      const input = JSON.parse(inputJson) as {
        q: SerializedTensor;
        k: SerializedTensor;
        v: SerializedTensor;
      };
      const forward = referenceFlashAttention(
        serializedTensor(input.q),
        serializedTensor(input.k),
        serializedTensor(input.v),
      );
      return JSON.stringify({
        output: [...forward.output.data],
        logSumExp: [...forward.logSumExp.data],
      });
    },
  };
}

function serializedTensor(input: SerializedTensor): Tensor {
  return tensor(input.shape, new Float32Array(input.data));
}

function createDistributedTrainingBridge(): object {
  return {
    ddp_average_json(inputJson: string) {
      return JSON.stringify(
        simulation.simulateDdpGradientSynchronization(JSON.parse(inputJson)),
      );
    },
  };
}

function createRlMathBridge(): object {
  return {
    dpo_loss_json(inputJson: string) {
      return rl.computePerInstanceDpoLoss(JSON.parse(inputJson));
    },
    parse_mmlu_response_json(exampleJson: string, modelOutput: string) {
      return rl.parseMmluResponse(JSON.parse(exampleJson), modelOutput);
    },
    parse_gsm8k_response(modelOutput: string) {
      return rl.parseGsm8kResponse(modelOutput);
    },
    group_normalized_rewards_json(inputJson: string) {
      return JSON.stringify(rl.computeGroupNormalizedRewards(JSON.parse(inputJson)));
    },
  };
}

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
