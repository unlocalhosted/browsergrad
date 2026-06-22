import { evaluateAssignmentMountContents, verifyAssignmentMountContentHashes } from "./assignment-mount.js";
import { createAssignmentPreflightReport } from "./assignment-run-plan.js";
import type {
  AssignmentBenchmarkPreflightMatrix,
  AssignmentBenchmarkPreflightRow,
  AssignmentCapabilityEnvironment,
  AssignmentMountContentEvaluation,
  AssignmentMountContents,
  AssignmentMountHashVerification,
  AssignmentPlatformHandoff,
  AssignmentPlatformIssueDraft,
  AssignmentPlatformNextAction,
  AssignmentPreflightReport,
  AssignmentProfile,
  AssignmentVerifiedBenchmarkPreflightMatrix,
  AssignmentVerifiedPlatformHandoff,
} from "./assignment-types.js";

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
