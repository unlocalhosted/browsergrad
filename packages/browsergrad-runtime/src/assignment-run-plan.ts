import { BrowsergradError } from "./types.js";
import {
  evaluateAssignmentCapabilities,
  requiredAssignmentCapabilities,
} from "./assignment-capabilities.js";
import { createAssignmentDatasetCachePlan, createAssignmentMountPlan } from "./assignment-mount.js";
import { joinAssignmentPath, profileOracleJsModules } from "./assignment-profile.js";
import type {
  AssignmentCapabilityEnvironment,
  AssignmentExternalRunnerRequest,
  AssignmentPreflightReport,
  AssignmentProfile,
  AssignmentRubricExecRequest,
  AssignmentRubricKind,
  AssignmentRunPlan,
  AssignmentRunReadiness,
  AssignmentRunnerRoute,
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

function assignmentRubricTimeoutMs(plan: AssignmentRunPlan): number | undefined {
  const candidates = [
    plan.execution.testTimeoutMs,
    plan.execution.workerTimeoutMs,
  ].filter((timeout): timeout is number => timeout !== undefined);
  if (candidates.length === 0) return undefined;
  return Math.min(...candidates);
}
