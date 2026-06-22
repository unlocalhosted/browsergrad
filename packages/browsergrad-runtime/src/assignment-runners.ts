import { BrowsergradError } from "./types.js";
import type { Artifact, Assertion } from "./types.js";
import { collectAssignmentMountEntries, createAssignmentMountPlan, materializeAssignmentMountPlan } from "./assignment-mount.js";
import { uniqueSorted } from "./assignment-profile.js";
import {
  assignmentRubricKind,
  createAssignmentPreflightReport,
  createAssignmentRubricExecRequest,
} from "./assignment-run-plan.js";
import { createVerifiedAssignmentPlatformHandoff } from "./assignment-handoff.js";
import type {
  AssignmentCapabilityEnvironment,
  AssignmentJavascriptProfileRunResult,
  AssignmentJavascriptRubric,
  AssignmentJavascriptRubricContext,
  AssignmentJavascriptRubricRunOptions,
  AssignmentJavascriptRubricRunResult,
  AssignmentMountContents,
  AssignmentProfile,
  AssignmentRubricRunOptions,
  AssignmentRubricRunResult,
  AssignmentRubricSession,
  AssignmentRunPlan,
} from "./assignment-types.js";

export async function runAssignmentRubric(
  session: AssignmentRubricSession,
  plan: AssignmentRunPlan,
  contents: AssignmentMountContents,
  options: AssignmentRubricRunOptions = {},
): Promise<AssignmentRubricRunResult> {
  const request = createAssignmentRubricExecRequest(plan);
  const mountPlan = createAssignmentMountPlan(plan);
  const mount = await materializeAssignmentMountPlan(
    session.fs,
    mountPlan,
    contents,
  );
  const exec = await session.exec({ ...request, ...options });
  return { mount, exec };
}

export async function runAssignmentJavascriptRubric(
  plan: AssignmentRunPlan,
  contents: AssignmentMountContents,
  rubric: AssignmentJavascriptRubric,
  options: AssignmentJavascriptRubricRunOptions = {},
): Promise<AssignmentJavascriptRubricRunResult> {
  if (!plan.ok) {
    const missing = plan.capabilityEvaluation.missingCapabilities.join(", ");
    const reason = missing.length > 0
      ? `missing assignment capabilities: ${missing}`
      : "assignment capability preflight failed";
    throw new BrowsergradError(`cannot run JavaScript rubric; ${reason}`);
  }
  if (assignmentRubricKind(plan) !== "javascript") {
    throw new BrowsergradError(
      "runAssignmentJavascriptRubric requires a JavaScript rubric path",
    );
  }

  const mountPlan = createAssignmentMountPlan(plan);
  const entries = collectAssignmentMountEntries(mountPlan, contents);
  const textByPath = new Map(
    entries.writes.map((entry) => [entry.path, entry.content] as const),
  );
  const assertions: Assertion[] = [];
  const artifacts: Artifact[] = [];

  const context: AssignmentJavascriptRubricContext = {
    id: plan.id,
    root: plan.files.root,
    ...(plan.files.fixturesPath ? { fixturesPath: plan.files.fixturesPath } : {}),
    allowedTests: plan.execution.allowedTests,
    behavioralGates: plan.behavioralGates,
    readText(path) {
      const content = textByPath.get(path);
      if (content === undefined) {
        throw new BrowsergradError(`assignment text file is not mounted: ${path}`);
      }
      if (typeof content !== "string") {
        throw new BrowsergradError(`assignment text file is binary: ${path}`);
      }
      return content;
    },
    readBytes(path) {
      const content = textByPath.get(path);
      if (content === undefined) {
        throw new BrowsergradError(`assignment byte file is not mounted: ${path}`);
      }
      if (typeof content === "string") {
        return new TextEncoder().encode(content);
      }
      return new Uint8Array(content);
    },
    oracle<T = unknown>(name: string): T {
      const oracle = options.oracles?.[name];
      if (oracle === undefined) {
        throw new BrowsergradError(`assignment JavaScript oracle is not registered: ${name}`);
      }
      return oracle as T;
    },
    substrate<T = unknown>(name: string): T {
      const substrate = options.substrates?.[name];
      if (substrate === undefined) {
        throw new BrowsergradError(`assignment JavaScript substrate is not registered: ${name}`);
      }
      return substrate as T;
    },
    assertPass(name, durationMs) {
      assertions.push({
        kind: "pass",
        name,
        ...(durationMs !== undefined ? { durationMs } : {}),
      });
    },
    assertFail(name, message, details = {}) {
      assertions.push({
        kind: "fail",
        name,
        message,
        ...(details.expected !== undefined ? { expectedRepr: String(details.expected) } : {}),
        ...(details.actual !== undefined ? { actualRepr: String(details.actual) } : {}),
        ...(details.durationMs !== undefined ? { durationMs: details.durationMs } : {}),
      });
    },
    assertError(name, message, error) {
      assertions.push({
        kind: "error",
        name,
        message,
        ...(error !== undefined ? { traceback: error instanceof Error ? error.stack : String(error) } : {}),
      });
    },
    log(name, data, level = "info") {
      artifacts.push({ kind: "log", name, level, data });
    },
    emitJson(name, data) {
      artifacts.push({ kind: "json", name, data });
    },
    emitImage(name, mime, dataBase64) {
      artifacts.push({ kind: "image", name, mime, dataBase64 });
    },
  };

  await rubric(context);

  return {
    mount: {
      writtenPaths: entries.writes.map((entry) => entry.path),
      skippedOptionalPaths: entries.skippedOptionalPaths,
    },
    assertions,
    artifacts,
  };
}

export async function runAssignmentJavascriptProfile(
  profile: AssignmentProfile,
  environment: AssignmentCapabilityEnvironment,
  contents: AssignmentMountContents,
  rubric: AssignmentJavascriptRubric,
  options: AssignmentJavascriptRubricRunOptions = {},
): Promise<AssignmentJavascriptProfileRunResult> {
  const report = createAssignmentPreflightReport(profile, environment);
  if (report.runnerRoute.target !== "javascript") {
    throw new BrowsergradError(
      `cannot run JavaScript profile; runner route is ${report.runnerRoute.target}: ${report.runnerRoute.message}`,
    );
  }
  const missingOracles = missingAssignmentJavascriptOracles(
    profile,
    options.oracles,
  );
  if (missingOracles.length > 0) {
    throw new BrowsergradError(
      `cannot run JavaScript profile; missing declared JavaScript assignment oracle: ${missingOracles.join(", ")}`,
    );
  }
  const run = await runAssignmentJavascriptRubric(
    report.plan,
    contents,
    rubric,
    options,
  );
  return { report, run };
}

export async function runVerifiedAssignmentJavascriptProfile(
  profile: AssignmentProfile,
  environment: AssignmentCapabilityEnvironment,
  contents: AssignmentMountContents,
  rubric: AssignmentJavascriptRubric,
  options: AssignmentJavascriptRubricRunOptions = {},
): Promise<AssignmentJavascriptProfileRunResult> {
  const report = createAssignmentPreflightReport(profile, environment);
  const handoff = await createVerifiedAssignmentPlatformHandoff(
    profile,
    report,
    contents,
  );
  if (handoff.nextAction !== "run-javascript") {
    const details = handoff.messages.length > 0
      ? `: ${handoff.messages.join("; ")}`
      : "";
    throw new BrowsergradError(
      `cannot run verified JavaScript profile; ${handoff.summary}${details}`,
    );
  }

  return runAssignmentJavascriptProfile(
    profile,
    environment,
    contents,
    rubric,
    options,
  );
}

function missingAssignmentJavascriptOracles(
  profile: AssignmentProfile,
  oracles: Readonly<Record<string, unknown>> | undefined,
): string[] {
  const provided = new Set(Object.keys(oracles ?? {}));
  return uniqueSorted(
    profile.oracles
      .map((oracle) => oracle.name)
      .filter((name) => !provided.has(name)),
  );
}
