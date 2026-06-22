import { BrowsergradError } from "./types.js";
import type { Artifact, Assertion } from "./types.js";
import {
  joinAssignmentPath,
  profileOracleJsModules,
  uniqueSorted,
} from "./assignment-profile.js";
export { parseAssignmentProfile, profileOracleJsModules } from "./assignment-profile.js";
import {
  evaluateAssignmentCapabilities,
  requiredAssignmentCapabilities,
} from "./assignment-capabilities.js";
export {
  createAssignmentCapabilityCatalog,
  createAssignmentCapabilityEnvironment,
  evaluateAssignmentCapabilities,
  requiredAssignmentCapabilities,
} from "./assignment-capabilities.js";
import {
  collectAssignmentMountEntries,
  createAssignmentDatasetCachePlan,
  createAssignmentMountPlan,
  evaluateAssignmentMountContents,
  materializeAssignmentMountPlan,
  verifyAssignmentMountContentHashes,
} from "./assignment-mount.js";
export {
  createAssignmentDatasetCachePlan,
  createAssignmentMountPlan,
  createAssignmentMountPreflightReport,
  evaluateAssignmentMountContents,
  materializeAssignmentMountPlan,
  verifyAssignmentMountContentHashes,
} from "./assignment-mount.js";
import type {
  AssignmentProfile,
  AssignmentRubricKind,
  AssignmentCapabilityEnvironment,
  AssignmentRunReadiness,
  AssignmentRunnerRoute,
  AssignmentPreflightReport,
  AssignmentBenchmarkPreflightRow,
  AssignmentBenchmarkPreflightMatrix,
  AssignmentPlatformNextAction,
  AssignmentPlatformHandoff,
  AssignmentVerifiedPlatformHandoff,
  AssignmentPlatformIssueDraft,
  AssignmentVerifiedBenchmarkPreflightMatrix,
  AssignmentRunPlan,
  AssignmentRubricExecRequest,
  AssignmentExternalRunnerRequest,
  AssignmentMountContents,
  AssignmentMountContentEvaluation,
  AssignmentMountHashVerification,
  AssignmentRubricSession,
  AssignmentRubricRunOptions,
  AssignmentRubricRunResult,
  AssignmentJavascriptRubricContext,
  AssignmentJavascriptRubric,
  AssignmentJavascriptRubricRunOptions,
  AssignmentJavascriptRubricRunResult,
  AssignmentJavascriptProfileRunResult,
} from "./assignment-types.js";

export type {
  AssignmentProfile,
  AssignmentProfileMetadata,
  AssignmentProfileFiles,
  AssignmentProfileTimeouts,
  AssignmentOracleSpec,
  AssignmentGateKind,
  AssignmentRubricKind,
  AssignmentGateSpec,
  AssignmentDataset,
  AssignmentCapabilityEnvironment,
  AssignmentCapabilityMode,
  AssignmentCapabilityEnvironmentInput,
  AssignmentCapabilityGateEvaluation,
  AssignmentCapabilityEvaluation,
  AssignmentCapabilityCatalog,
  AssignmentCapabilityCatalogEntry,
  AssignmentCapabilityCatalogReference,
  AssignmentCapabilityCatalogAlternative,
  AssignmentRunReadinessStatus,
  AssignmentRunnerTarget,
  AssignmentRunReadiness,
  AssignmentRunnerRoute,
  AssignmentPreflightReport,
  AssignmentBenchmarkPreflightRow,
  AssignmentBenchmarkPreflightGateRow,
  AssignmentBenchmarkPreflightMatrix,
  AssignmentPlatformNextAction,
  AssignmentPlatformHandoff,
  AssignmentVerifiedPlatformHandoff,
  AssignmentPlatformIssueDraft,
  AssignmentVerifiedBenchmarkPreflightRow,
  AssignmentVerifiedBenchmarkPreflightMatrix,
  AssignmentRunPlan,
  AssignmentRunPlanSession,
  AssignmentRunPlanFiles,
  AssignmentRunPlanExecution,
  AssignmentRubricExecRequest,
  AssignmentExternalRunnerRequest,
  AssignmentMountPlan,
  AssignmentMountFileRole,
  AssignmentMountFile,
  AssignmentDatasetMount,
  AssignmentDatasetCacheStrategy,
  AssignmentDatasetCacheEntry,
  AssignmentDatasetCachePlan,
  AssignmentMountContent,
  AssignmentMountContents,
  AssignmentMountContentEvaluation,
  AssignmentMountHashStatus,
  AssignmentMountHashCheck,
  AssignmentMountHashVerification,
  AssignmentMountPreflightReport,
  AssignmentMaterializeResult,
  AssignmentRubricSession,
  AssignmentRubricRunOptions,
  AssignmentRubricRunResult,
  AssignmentJavascriptRubricContext,
  AssignmentJavascriptRubric,
  AssignmentJavascriptRubricRunOptions,
  AssignmentJavascriptRubricRunResult,
  AssignmentJavascriptProfileRunResult,
  AssignmentProfileParseResult,
} from "./assignment-types.js";
export function createAssignmentRunPlan(
  profile: AssignmentProfile,
  environment: AssignmentCapabilityEnvironment,
): AssignmentRunPlan {
  const capabilityEvaluation = evaluateAssignmentCapabilities(profile, environment);
  return {
    id: profile.id,
    profileVersion: profile.version,
    requiresBrowsergrad: profile.requires_browsergrad,
    ok: capabilityEvaluation.ok,
    session: {
      packages: profile.runtime_packages,
      jsModules: profileOracleJsModules(profile),
    },
    files: {
      root: profile.files.root,
      rubricPath: joinAssignmentPath(profile.files.root, profile.files.rubric_path),
      ...(profile.files.starter_path
        ? { starterPath: joinAssignmentPath(profile.files.root, profile.files.starter_path) }
        : {}),
      ...(profile.files.reference_path
        ? { referencePath: joinAssignmentPath(profile.files.root, profile.files.reference_path) }
        : {}),
      ...(profile.files.fixtures_path
        ? { fixturesPath: joinAssignmentPath(profile.files.root, profile.files.fixtures_path) }
        : {}),
    },
    execution: {
      allowedTests: profile.allowed_tests,
      ...(profile.timeouts.setup_ms !== undefined
        ? { setupTimeoutMs: profile.timeouts.setup_ms }
        : {}),
      ...(profile.timeouts.test_ms !== undefined
        ? { testTimeoutMs: profile.timeouts.test_ms }
        : {}),
      ...(profile.timeouts.worker_ms !== undefined
        ? { workerTimeoutMs: profile.timeouts.worker_ms }
        : {}),
    },
    datasets: profile.datasets,
    capabilityEvaluation,
    behavioralGates: profile.gates.filter((gate) => gate.kind !== "capability"),
  };
}






export function createAssignmentPreflightReport(
  profile: AssignmentProfile,
  environment: AssignmentCapabilityEnvironment,
): AssignmentPreflightReport {
  const plan = createAssignmentRunPlan(profile, environment);
  const rubricKind = assignmentRubricKind(plan);
  const readiness = assignmentRunReadiness(plan);
  const runnerRoute = assignmentRunnerRoute(plan);
  const mountPlan = createAssignmentMountPlan(plan);
  return {
    plan,
    rubricKind,
    readiness,
    runnerRoute,
    requiredCapabilities: requiredAssignmentCapabilities(profile),
    mountPlan,
    datasetCachePlan: createAssignmentDatasetCachePlan(mountPlan),
    ...(runnerRoute.target === "external"
      ? { externalRunnerRequest: createAssignmentExternalRunnerRequest(plan) }
      : {}),
  };
}

export function createAssignmentBenchmarkPreflightMatrix(
  profiles: readonly AssignmentProfile[],
  environment: AssignmentCapabilityEnvironment,
  contents: AssignmentMountContents = { files: {} },
): AssignmentBenchmarkPreflightMatrix {
  const rows = profiles.map((profile) =>
    createAssignmentBenchmarkPreflightRow(
      profile,
      createAssignmentPreflightReport(profile, environment),
      contents,
    )
  );

  return {
    ok: rows.every((row) => row.ok),
    rows,
  };
}

export function createAssignmentPlatformHandoff(
  profile: AssignmentProfile,
  report: AssignmentPreflightReport,
  contents: AssignmentMountContents = { files: {} },
): AssignmentPlatformHandoff {
  const content = evaluateAssignmentMountContents(report.mountPlan, contents);
  const nextAction = assignmentPlatformNextAction(report, content);
  const summary = assignmentPlatformSummary(report, nextAction);
  const launchable = nextAction === "run-pyodide" || nextAction === "run-javascript";

  return {
    id: profile.id,
    ...(profile.metadata?.title ? { title: profile.metadata.title } : {}),
    ...(profile.metadata?.course ? { course: profile.metadata.course } : {}),
    ...(profile.metadata?.source_url
      ? { sourceUrl: profile.metadata.source_url }
      : {}),
    readinessStatus: report.readiness.status,
    runnerTarget: report.runnerRoute.target,
    rubricKind: report.rubricKind,
    nextAction,
    summary,
    launchable,
    missingCapabilities: report.readiness.missingCapabilities,
    missingRequiredFiles: content.missingRequiredFiles,
    missingDatasets: content.missingDatasets,
    skippedOptionalPaths: content.skippedOptionalPaths,
    selectedCapabilities: report.readiness.selectedCapabilities,
    simulatedCapabilities: report.readiness.simulatedCapabilities,
    externalCapabilities: report.readiness.externalCapabilities,
    cacheStrategies: report.datasetCachePlan.datasets.map(
      (dataset) => dataset.strategy,
    ),
    externalRunnerRequired: Boolean(report.externalRunnerRequest),
    messages: assignmentPlatformMessages(report, content, nextAction),
  };
}

export async function createVerifiedAssignmentPlatformHandoff(
  profile: AssignmentProfile,
  report: AssignmentPreflightReport,
  contents: AssignmentMountContents = { files: {} },
): Promise<AssignmentVerifiedPlatformHandoff> {
  const handoff = createAssignmentPlatformHandoff(profile, report, contents);
  const hashes = await verifyAssignmentMountContentHashes(report.mountPlan, contents);
  const canHashBlockLaunch =
    handoff.nextAction === "run-pyodide" ||
    handoff.nextAction === "run-javascript" ||
    handoff.nextAction === "request-external-runner";
  if (hashes.ok || !canHashBlockLaunch) {
    return { ...handoff, hashOk: hashes.ok, hashChecks: hashes.checks };
  }

  return {
    ...handoff,
    nextAction: "verify-content",
    summary: "fix dataset hash declarations or contents before launch",
    launchable: false,
    hashOk: false,
    hashChecks: hashes.checks,
    messages: assignmentPlatformHashMessages(hashes),
  };
}

export function createAssignmentPlatformIssueDraft(
  profile: AssignmentProfile,
  handoff: AssignmentPlatformHandoff,
): AssignmentPlatformIssueDraft {
  const assignmentName = profile.metadata?.title ?? profile.id;
  return {
    title: `BrowserGrad handoff: ${assignmentName}`,
    labels: [
      "browsergrad-handoff",
      `next:${handoff.nextAction}`,
      `readiness:${handoff.readinessStatus}`,
      `runner:${handoff.runnerTarget}`,
    ],
    body: [
      "## BrowserGrad Handoff",
      "",
      `- Assignment: ${assignmentName}`,
      `- Profile: ${profile.id}`,
      ...(profile.metadata?.source_url
        ? [`- Source: ${profile.metadata.source_url}`]
        : []),
      `- Readiness: ${handoff.readinessStatus}`,
      `- Runner: ${handoff.runnerTarget}`,
      `- Next action: ${handoff.nextAction}`,
      `- Launchable: ${handoff.launchable ? "yes" : "no"}`,
      "",
      "## Messages",
      "",
      ...markdownListOrNone(handoff.messages),
      "",
      "## Missing Content",
      "",
      ...markdownListOrNone([
        ...handoff.missingRequiredFiles.map((path) => `Required file: ${path}`),
        ...handoff.missingDatasets.map((name) => `Dataset: ${name}`),
      ]),
      "",
      "## Capabilities",
      "",
      `- Selected: ${humanList(handoff.selectedCapabilities)}`,
      `- Simulated: ${humanList(handoff.simulatedCapabilities)}`,
      `- External: ${humanList(handoff.externalCapabilities)}`,
      ...platformIssueHashSection(handoff),
    ].join("\n"),
  };
}

export async function createVerifiedAssignmentBenchmarkPreflightMatrix(
  profiles: readonly AssignmentProfile[],
  environment: AssignmentCapabilityEnvironment,
  contents: AssignmentMountContents,
): Promise<AssignmentVerifiedBenchmarkPreflightMatrix> {
  const rows = await Promise.all(
    profiles.map(async (profile) => {
      const report = createAssignmentPreflightReport(profile, environment);
      const row = createAssignmentBenchmarkPreflightRow(profile, report, contents);
      const hashes = await verifyAssignmentMountContentHashes(
        report.mountPlan,
        contents,
      );

      return {
        ...row,
        ok: row.ok && hashes.ok,
        hashOk: hashes.ok,
        hashChecks: hashes.checks,
      };
    }),
  );

  return {
    ok: rows.every((row) => row.ok),
    rows,
  };
}

function assignmentPlatformNextAction(
  report: AssignmentPreflightReport,
  content: AssignmentMountContentEvaluation,
): AssignmentPlatformNextAction {
  if (report.readiness.status === "blocked" || report.runnerRoute.target === "blocked") {
    return "install-capabilities";
  }
  if (report.runnerRoute.target === "unsupported") return "unsupported";
  if (!content.ok) return "mount-content";
  if (report.runnerRoute.target === "external") return "request-external-runner";
  if (report.runnerRoute.target === "pyodide") return "run-pyodide";
  if (report.runnerRoute.target === "javascript") return "run-javascript";
  return "unsupported";
}

function assignmentPlatformSummary(
  report: AssignmentPreflightReport,
  nextAction: AssignmentPlatformNextAction,
): string {
  switch (nextAction) {
    case "install-capabilities":
      return report.readiness.summary;
    case "mount-content":
      return "mount required assignment files and datasets before launch";
    case "verify-content":
      return "fix dataset hash declarations or contents before launch";
    case "request-external-runner":
      return "assignment requires an external runner handoff";
    case "run-pyodide":
      return "assignment can launch in Pyodide";
    case "run-javascript":
      return "assignment can launch in JavaScript";
    case "unsupported":
      return report.runnerRoute.message;
  }
}

function assignmentPlatformMessages(
  report: AssignmentPreflightReport,
  content: AssignmentMountContentEvaluation,
  nextAction: AssignmentPlatformNextAction,
): string[] {
  const messages = [
    ...report.readiness.missingCapabilities.map(
      (capability) => `missing capability: ${capability}`,
    ),
    ...content.missingRequiredFiles.map((path) => `missing required file: ${path}`),
    ...content.missingDatasets.map((name) => `missing dataset: ${name}`),
  ];
  if (messages.length > 0) return messages;

  switch (nextAction) {
    case "request-external-runner":
      return ["ready for external runner handoff"];
    case "run-pyodide":
      return ["ready for Pyodide rubric runner"];
    case "run-javascript":
      return ["ready for JavaScript rubric runner"];
    case "install-capabilities":
    case "mount-content":
    case "verify-content":
    case "unsupported":
      return [report.runnerRoute.message];
  }
}

function assignmentPlatformHashMessages(
  hashes: AssignmentMountHashVerification,
): string[] {
  const messages = hashes.checks
    .filter((check) => !check.ok)
    .map((check) => `dataset hash ${check.status}: ${check.name}`);
  return messages.length > 0 ? messages : ["dataset hash verification failed"];
}

function markdownListOrNone(items: readonly string[]): string[] {
  return items.length > 0 ? items.map((item) => `- ${item}`) : ["- none"];
}

function humanList(items: readonly string[]): string {
  return items.length > 0 ? items.join(", ") : "none";
}

function platformIssueHashSection(
  handoff: AssignmentPlatformHandoff,
): string[] {
  if (!("hashChecks" in handoff)) return [];
  const verified = handoff as AssignmentVerifiedPlatformHandoff;
  return [
    "",
    "## Hash Checks",
    "",
    ...markdownListOrNone(
      verified.hashChecks.map((check) => {
        const parts = [`${check.name}: ${check.status}`];
        if (check.expected) parts.push(`expected ${check.expected}`);
        if (check.actual) parts.push(`actual ${check.actual}`);
        return parts.join(" — ");
      }),
    ),
  ];
}

function createAssignmentBenchmarkPreflightRow(
  profile: AssignmentProfile,
  report: AssignmentPreflightReport,
  contents: AssignmentMountContents,
): AssignmentBenchmarkPreflightRow {
  const content = evaluateAssignmentMountContents(report.mountPlan, contents);
  const ok =
    report.readiness.status !== "blocked" &&
    report.runnerRoute.target !== "unsupported" &&
    content.ok;

  return {
    id: profile.id,
    ...(profile.metadata?.title ? { title: profile.metadata.title } : {}),
    ...(profile.metadata?.course ? { course: profile.metadata.course } : {}),
    ...(profile.metadata?.source_url
      ? { sourceUrl: profile.metadata.source_url }
      : {}),
    ok,
    readinessStatus: report.readiness.status,
    runnerTarget: report.runnerRoute.target,
    rubricKind: report.rubricKind,
    requiredCapabilities: report.requiredCapabilities,
    selectedCapabilities: report.readiness.selectedCapabilities,
    missingCapabilities: report.readiness.missingCapabilities,
    simulatedCapabilities: report.readiness.simulatedCapabilities,
    externalCapabilities: report.readiness.externalCapabilities,
    contentOk: content.ok,
    missingRequiredFiles: content.missingRequiredFiles,
    missingDatasets: content.missingDatasets,
    skippedOptionalPaths: content.skippedOptionalPaths,
    cacheStrategies: report.datasetCachePlan.datasets.map(
      (dataset) => dataset.strategy,
    ),
    externalRunnerRequired: Boolean(report.externalRunnerRequest),
    gates: report.plan.capabilityEvaluation.gates.map((gate) => ({
      name: gate.name,
      ok: gate.ok,
      status: gate.status,
      requires: gate.requires,
      anyOf: gate.anyOf,
      selectedAnyOf: gate.selectedAnyOf,
      selectedCapabilities: gate.selectedCapabilities,
      missingRequired: gate.missingRequired,
      missingAnyOf: gate.missingAnyOf,
      ...(gate.message ? { message: gate.message } : {}),
    })),
  };
}

export function assignmentRunnerRoute(
  plan: AssignmentRunPlan,
): AssignmentRunnerRoute {
  const readiness = assignmentRunReadiness(plan);
  const rubricKind = assignmentRubricKind(plan);
  if (readiness.status === "blocked") {
    return {
      target: "blocked",
      readinessStatus: readiness.status,
      rubricKind,
      message: readiness.summary,
      missingCapabilities: readiness.missingCapabilities,
      selectedCapabilities: readiness.selectedCapabilities,
    };
  }
  if (readiness.status === "external-only") {
    return {
      target: "external",
      readinessStatus: readiness.status,
      rubricKind,
      message: readiness.summary,
      missingCapabilities: readiness.missingCapabilities,
      selectedCapabilities: readiness.selectedCapabilities,
    };
  }
  if (rubricKind === "python") {
    return {
      target: "pyodide",
      readinessStatus: readiness.status,
      rubricKind,
      message: "assignment routes to Pyodide rubric runner",
      missingCapabilities: readiness.missingCapabilities,
      selectedCapabilities: readiness.selectedCapabilities,
    };
  }
  if (rubricKind === "javascript") {
    return {
      target: "javascript",
      readinessStatus: readiness.status,
      rubricKind,
      message: "assignment routes to JavaScript rubric runner",
      missingCapabilities: readiness.missingCapabilities,
      selectedCapabilities: readiness.selectedCapabilities,
    };
  }
  return {
    target: "unsupported",
    readinessStatus: readiness.status,
    rubricKind,
    message: `assignment rubric kind is not supported by built-in runners: ${rubricKind}`,
    missingCapabilities: readiness.missingCapabilities,
    selectedCapabilities: readiness.selectedCapabilities,
  };
}

export function assignmentRunReadiness(
  plan: AssignmentRunPlan,
): AssignmentRunReadiness {
  const selectedCapabilities = plan.capabilityEvaluation.satisfiedCapabilities;
  const capabilityModes = plan.capabilityEvaluation.capabilityModes;
  const simulatedCapabilities = selectedCapabilities.filter(
    (capability) => capabilityModes[capability] === "simulated",
  );
  const externalCapabilities = selectedCapabilities.filter(
    (capability) => capabilityModes[capability] === "external",
  );

  if (!plan.ok) {
    return {
      status: "blocked",
      summary: "assignment cannot run until missing platform capabilities are available",
      missingCapabilities: plan.capabilityEvaluation.missingCapabilities,
      selectedCapabilities,
      simulatedCapabilities,
      externalCapabilities,
    };
  }

  if (externalCapabilities.length > 0) {
    return {
      status: "external-only",
      summary: "assignment requires external runner capabilities",
      missingCapabilities: [],
      selectedCapabilities,
      simulatedCapabilities,
      externalCapabilities,
    };
  }

  if (simulatedCapabilities.length > 0) {
    return {
      status: "simulated",
      summary: "assignment can run through simulated platform capabilities",
      missingCapabilities: [],
      selectedCapabilities,
      simulatedCapabilities,
      externalCapabilities,
    };
  }

  return {
    status: "runnable",
    summary: "assignment can run in the current platform",
    missingCapabilities: [],
    selectedCapabilities,
    simulatedCapabilities,
    externalCapabilities,
  };
}

export function createAssignmentRubricExecRequest(
  plan: AssignmentRunPlan,
): AssignmentRubricExecRequest {
  if (!plan.ok) {
    const missing = plan.capabilityEvaluation.missingCapabilities.join(", ");
    const reason = missing.length > 0
      ? `missing assignment capabilities: ${missing}`
      : "assignment capability preflight failed";
    throw new BrowsergradError(`cannot create rubric exec request; ${reason}`);
  }

  if (assignmentRubricKind(plan) !== "python") {
    throw new BrowsergradError(
      "createAssignmentRubricExecRequest requires a Python rubric path",
    );
  }
  const lines = [
    "import json, os, runpy, sys",
    `assignment_root = ${JSON.stringify(plan.files.root)}`,
    "if assignment_root not in sys.path:",
    "    sys.path.insert(0, assignment_root)",
    `os.environ["BROWSERGRAD_ASSIGNMENT_ID"] = ${JSON.stringify(plan.id)}`,
    `os.environ["BROWSERGRAD_ASSIGNMENT_ROOT"] = ${JSON.stringify(plan.files.root)}`,
    ...(plan.files.fixturesPath
      ? [`os.environ["BROWSERGRAD_FIXTURES_PATH"] = ${JSON.stringify(plan.files.fixturesPath)}`]
      : []),
    `os.environ["BROWSERGRAD_ALLOWED_TESTS_JSON"] = ${JSON.stringify(JSON.stringify(plan.execution.allowedTests))}`,
    `os.environ["BROWSERGRAD_BEHAVIORAL_GATES_JSON"] = ${JSON.stringify(JSON.stringify(plan.behavioralGates))}`,
    `runpy.run_path(${JSON.stringify(plan.files.rubricPath)}, run_name="__main__")`,
  ];
  const timeoutMs = assignmentRubricTimeoutMs(plan);
  return {
    code: lines.join("\n"),
    ...(timeoutMs !== undefined ? { timeoutMs } : {}),
  };
}

export function createAssignmentExternalRunnerRequest(
  plan: AssignmentRunPlan,
): AssignmentExternalRunnerRequest {
  const route = assignmentRunnerRoute(plan);
  if (route.target !== "external") {
    throw new BrowsergradError(
      `createAssignmentExternalRunnerRequest requires an external runner plan; got ${route.target}`,
    );
  }
  const readiness = assignmentRunReadiness(plan);
  const mountPlan = createAssignmentMountPlan(plan);
  return {
    id: plan.id,
    profileVersion: plan.profileVersion,
    requiresBrowsergrad: plan.requiresBrowsergrad,
    route,
    selectedCapabilities: readiness.selectedCapabilities,
    externalCapabilities: readiness.externalCapabilities,
    simulatedCapabilities: readiness.simulatedCapabilities,
    environment: createAssignmentExternalRunnerEnvironment(
      plan,
      route,
      readiness,
    ),
    files: plan.files,
    execution: plan.execution,
    mountPlan,
    datasetCachePlan: createAssignmentDatasetCachePlan(mountPlan),
    behavioralGates: plan.behavioralGates,
  };
}

function createAssignmentExternalRunnerEnvironment(
  plan: AssignmentRunPlan,
  route: AssignmentRunnerRoute,
  readiness: AssignmentRunReadiness,
): Readonly<Record<string, string>> {
  return {
    BROWSERGRAD_ALLOWED_TESTS_JSON: JSON.stringify(plan.execution.allowedTests),
    BROWSERGRAD_ASSIGNMENT_ID: plan.id,
    BROWSERGRAD_ASSIGNMENT_ROOT: plan.files.root,
    BROWSERGRAD_BEHAVIORAL_GATES_JSON: JSON.stringify(plan.behavioralGates),
    BROWSERGRAD_EXTERNAL_CAPABILITIES_JSON: JSON.stringify(
      readiness.externalCapabilities,
    ),
    ...(plan.files.fixturesPath
      ? { BROWSERGRAD_FIXTURES_PATH: plan.files.fixturesPath }
      : {}),
    BROWSERGRAD_RUNNER_READINESS: route.readinessStatus,
    BROWSERGRAD_RUNNER_TARGET: route.target,
    BROWSERGRAD_SELECTED_CAPABILITIES_JSON: JSON.stringify(
      readiness.selectedCapabilities,
    ),
    BROWSERGRAD_SIMULATED_CAPABILITIES_JSON: JSON.stringify(
      readiness.simulatedCapabilities,
    ),
  };
}

export function assignmentRubricKind(
  plan: AssignmentRunPlan,
): AssignmentRubricKind {
  const path = plan.files.rubricPath.toLowerCase();
  if (path.endsWith(".py")) return "python";
  if (path.endsWith(".js") || path.endsWith(".mjs")) return "javascript";
  return "unknown";
}








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

function assignmentRubricTimeoutMs(plan: AssignmentRunPlan): number | undefined {
  const candidates = [
    plan.execution.testTimeoutMs,
    plan.execution.workerTimeoutMs,
  ].filter((timeout): timeout is number => timeout !== undefined);
  if (candidates.length === 0) return undefined;
  return Math.min(...candidates);
}



















