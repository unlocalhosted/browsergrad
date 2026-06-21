import {
  type Artifact,
  type Assertion,
  BrowsergradError,
  type ExecOptions,
  type ExecResult,
  type PyodideJsModule,
  type SessionFS,
} from "./types.js";

export interface AssignmentProfile {
  readonly id: string;
  readonly version: string;
  readonly requires_browsergrad: string;
  readonly metadata?: AssignmentProfileMetadata;
  readonly runtime_packages: readonly string[];
  readonly files: AssignmentProfileFiles;
  readonly timeouts: AssignmentProfileTimeouts;
  readonly allowed_tests: readonly string[];
  readonly oracles: readonly AssignmentOracleSpec[];
  readonly gates: readonly AssignmentGateSpec[];
  readonly datasets: readonly AssignmentDataset[];
}

export interface AssignmentProfileMetadata {
  readonly title?: string;
  readonly course?: string;
  readonly source_url?: string;
  readonly lecture_urls: readonly string[];
  readonly tags: readonly string[];
}

export interface AssignmentProfileFiles {
  readonly root: string;
  readonly rubric_path: string;
  readonly starter_path?: string;
  readonly reference_path?: string;
  readonly fixtures_path?: string;
}

export interface AssignmentProfileTimeouts {
  readonly setup_ms?: number;
  readonly test_ms?: number;
  readonly worker_ms?: number;
}

export interface AssignmentOracleSpec {
  readonly name: string;
  readonly js_module: string;
  readonly export_name?: string;
}

export type AssignmentGateKind =
  | "capability"
  | "streaming"
  | "timeout"
  | "forbidden-read";

export type AssignmentRubricKind = "python" | "javascript" | "unknown";

export interface AssignmentGateSpec {
  readonly name: string;
  readonly kind: AssignmentGateKind;
  readonly options: Record<string, unknown>;
}

export interface AssignmentDataset {
  readonly name: string;
  readonly url: string;
  readonly hash?: string;
}

export interface AssignmentCapabilityEnvironment {
  readonly capabilities: readonly string[];
  readonly capabilityModes?: Readonly<Record<string, AssignmentCapabilityMode>>;
}

export type AssignmentCapabilityMode = "browser" | "simulated" | "external";

export interface AssignmentCapabilityGateEvaluation {
  readonly name: string;
  readonly ok: boolean;
  readonly status: AssignmentRunReadinessStatus;
  readonly requires: readonly string[];
  readonly anyOf: readonly (readonly string[])[];
  readonly selectedAnyOf: readonly string[];
  readonly selectedCapabilities: readonly string[];
  readonly satisfiedCapabilities: readonly string[];
  readonly missingRequired: readonly string[];
  readonly missingAnyOf: readonly (readonly string[])[];
  readonly message?: string;
}

export interface AssignmentCapabilityEvaluation {
  readonly ok: boolean;
  readonly satisfiedCapabilities: readonly string[];
  readonly missingCapabilities: readonly string[];
  readonly capabilityModes: Readonly<Record<string, AssignmentCapabilityMode>>;
  readonly gates: readonly AssignmentCapabilityGateEvaluation[];
}

export type AssignmentRunReadinessStatus =
  | "runnable"
  | "simulated"
  | "external-only"
  | "blocked";

export interface AssignmentRunReadiness {
  readonly status: AssignmentRunReadinessStatus;
  readonly summary: string;
  readonly missingCapabilities: readonly string[];
  readonly selectedCapabilities: readonly string[];
  readonly simulatedCapabilities: readonly string[];
  readonly externalCapabilities: readonly string[];
}

export interface AssignmentPreflightReport {
  readonly plan: AssignmentRunPlan;
  readonly rubricKind: AssignmentRubricKind;
  readonly readiness: AssignmentRunReadiness;
  readonly requiredCapabilities: readonly string[];
  readonly mountPlan: AssignmentMountPlan;
}

export interface AssignmentRunPlan {
  readonly id: string;
  readonly profileVersion: string;
  readonly requiresBrowsergrad: string;
  readonly ok: boolean;
  readonly session: AssignmentRunPlanSession;
  readonly files: AssignmentRunPlanFiles;
  readonly execution: AssignmentRunPlanExecution;
  readonly datasets: readonly AssignmentDataset[];
  readonly capabilityEvaluation: AssignmentCapabilityEvaluation;
  readonly behavioralGates: readonly AssignmentGateSpec[];
}

export interface AssignmentRunPlanSession {
  readonly packages: readonly string[];
  readonly jsModules: readonly PyodideJsModule[];
}

export interface AssignmentRunPlanFiles {
  readonly root: string;
  readonly rubricPath: string;
  readonly starterPath?: string;
  readonly referencePath?: string;
  readonly fixturesPath?: string;
}

export interface AssignmentRunPlanExecution {
  readonly allowedTests: readonly string[];
  readonly setupTimeoutMs?: number;
  readonly testTimeoutMs?: number;
  readonly workerTimeoutMs?: number;
}

export interface AssignmentRubricExecRequest {
  readonly code: string;
  readonly timeoutMs?: number;
}

export interface AssignmentMountPlan {
  readonly root: string;
  readonly files: readonly AssignmentMountFile[];
  readonly datasets: readonly AssignmentDatasetMount[];
}

export type AssignmentMountFileRole = "rubric" | "starter" | "reference";

export interface AssignmentMountFile {
  readonly role: AssignmentMountFileRole;
  readonly path: string;
  readonly required: boolean;
}

export interface AssignmentDatasetMount {
  readonly name: string;
  readonly url: string;
  readonly hash?: string;
  readonly mountPath: string;
}

export type AssignmentMountContent = string | Uint8Array;

export interface AssignmentMountContents {
  readonly files: Readonly<Record<string, AssignmentMountContent>>;
  readonly datasets?: Readonly<Record<string, AssignmentMountContent>>;
}

export interface AssignmentMountContentEvaluation {
  readonly ok: boolean;
  readonly writablePaths: readonly string[];
  readonly missingRequiredFiles: readonly string[];
  readonly missingDatasets: readonly string[];
  readonly skippedOptionalPaths: readonly string[];
}

export interface AssignmentMaterializeResult {
  readonly writtenPaths: readonly string[];
  readonly skippedOptionalPaths: readonly string[];
}

export interface AssignmentRubricSession {
  readonly fs: SessionFS;
  exec(options: ExecOptions): Promise<ExecResult>;
}

export type AssignmentRubricRunOptions = Omit<ExecOptions, "code">;

export interface AssignmentRubricRunResult {
  readonly mount: AssignmentMaterializeResult;
  readonly exec: ExecResult;
}

export interface AssignmentJavascriptRubricContext {
  readonly id: string;
  readonly root: string;
  readonly fixturesPath?: string;
  readonly allowedTests: readonly string[];
  readonly behavioralGates: readonly AssignmentGateSpec[];
  readText(path: string): string;
  readBytes(path: string): Uint8Array;
  oracle<T = unknown>(name: string): T;
  substrate<T = unknown>(name: string): T;
  assertPass(name: string, durationMs?: number): void;
  assertFail(
    name: string,
    message: string,
    details?: {
      readonly expected?: unknown;
      readonly actual?: unknown;
      readonly durationMs?: number;
    },
  ): void;
  assertError(name: string, message: string, error?: unknown): void;
  log(name: string, data: string, level?: "info" | "warn" | "error"): void;
  emitJson(name: string, data: unknown): void;
  emitImage(name: string, mime: string, dataBase64: string): void;
}

export type AssignmentJavascriptRubric = (
  context: AssignmentJavascriptRubricContext,
) => void | Promise<void>;

export interface AssignmentJavascriptRubricRunOptions {
  readonly oracles?: Readonly<Record<string, unknown>>;
  readonly substrates?: Readonly<Record<string, unknown>>;
}

export interface AssignmentJavascriptRubricRunResult {
  readonly mount: AssignmentMaterializeResult;
  readonly assertions: readonly Assertion[];
  readonly artifacts: readonly Artifact[];
}

export type AssignmentProfileParseResult =
  | { ok: true; profile: AssignmentProfile }
  | { ok: false; errors: readonly string[] };

const SEMVER_RE = /^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/;
const SEMVER_RANGE_RE = /^[\^~]?\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/;
const ID_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const GATE_KINDS = new Set<AssignmentGateKind>([
  "capability",
  "streaming",
  "timeout",
  "forbidden-read",
]);

export function parseAssignmentProfile(
  input: unknown,
): AssignmentProfileParseResult {
  const errors: string[] = [];
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    return { ok: false, errors: ["assignment profile must be a JSON object"] };
  }
  const obj = input as Record<string, unknown>;

  const id = readString(obj, "id", errors, ID_RE);
  const version = readString(obj, "version", errors, SEMVER_RE);
  const requires_browsergrad = readString(
    obj,
    "requires_browsergrad",
    errors,
    SEMVER_RANGE_RE,
  );
  const metadata = readMetadata(obj.metadata, errors);
  const files = readFiles(obj.files, errors);
  const timeouts = readTimeouts(obj.timeouts, errors);
  const runtime_packages = readStringArray(
    obj.runtime_packages,
    "runtime_packages",
    errors,
    64,
  );
  const allowed_tests = readStringArray(
    obj.allowed_tests,
    "allowed_tests",
    errors,
    256,
  );
  const oracles = readOracles(obj.oracles, errors);
  const gates = readGates(obj.gates, errors);
  const datasets = readDatasets(obj.datasets, errors);

  if (errors.length > 0) return { ok: false, errors };

  return {
    ok: true,
    profile: {
      id: id!,
      version: version!,
      requires_browsergrad: requires_browsergrad!,
      ...(metadata ? { metadata } : {}),
      runtime_packages,
      files: files!,
      timeouts,
      allowed_tests,
      oracles,
      gates,
      datasets,
    },
  };
}

function readMetadata(
  value: unknown,
  errors: string[],
): AssignmentProfileMetadata | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    errors.push("metadata: must be an object when present");
    return undefined;
  }
  const obj = value as Record<string, unknown>;
  const title = readOptionalString(obj, "title", errors);
  const course = readOptionalString(obj, "course", errors);
  const source_url = readOptionalString(obj, "source_url", errors);
  const lecture_urls = readStringArray(
    obj.lecture_urls,
    "metadata.lecture_urls",
    errors,
    32,
  );
  const tags = readStringArray(obj.tags, "metadata.tags", errors, 64);
  return {
    ...(title ? { title } : {}),
    ...(course ? { course } : {}),
    ...(source_url ? { source_url } : {}),
    lecture_urls,
    tags,
  };
}

export function profileOracleJsModules(
  profile: AssignmentProfile,
): PyodideJsModule[] {
  return profile.oracles.map((oracle) => ({
    name: oracle.name,
    importURL: oracle.js_module,
    ...(oracle.export_name ? { exportName: oracle.export_name } : {}),
  }));
}

export function evaluateAssignmentCapabilities(
  profile: AssignmentProfile,
  environment: AssignmentCapabilityEnvironment,
): AssignmentCapabilityEvaluation {
  const available = new Set(environment.capabilities);
  const capabilityModes = normalizeCapabilityModes(environment.capabilityModes);
  const gates = profile.gates
    .filter((gate) => gate.kind === "capability")
    .map((gate) => evaluateCapabilityGate(gate, available, capabilityModes));
  const satisfiedCapabilities = uniqueSorted(
    gates.flatMap((gate) => gate.satisfiedCapabilities),
  );

  return {
    ok: gates.every((gate) => gate.ok),
    satisfiedCapabilities,
    missingCapabilities: uniqueSorted(
      gates.flatMap((gate) => [
        ...gate.missingRequired,
        ...gate.missingAnyOf.flat(),
      ]),
    ),
    capabilityModes,
    gates,
  };
}

export function requiredAssignmentCapabilities(
  profile: AssignmentProfile,
): string[] {
  return uniqueSorted(
    profile.gates
      .filter((gate) => gate.kind === "capability")
      .flatMap((gate) => {
        const requires = readCapabilityStringList(gate.options.requires);
        const anyOf = readCapabilityAlternatives(gate.options.any_of);
        return [...requires, ...anyOf.flat()];
      }),
  );
}

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
  return {
    plan,
    rubricKind: assignmentRubricKind(plan),
    readiness: assignmentRunReadiness(plan),
    requiredCapabilities: requiredAssignmentCapabilities(profile),
    mountPlan: createAssignmentMountPlan(plan),
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
  return {
    code: lines.join("\n"),
    ...(plan.execution.testTimeoutMs !== undefined
      ? { timeoutMs: plan.execution.testTimeoutMs }
      : {}),
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

export function createAssignmentMountPlan(
  plan: AssignmentRunPlan,
): AssignmentMountPlan {
  return {
    root: plan.files.root,
    files: [
      {
        role: "rubric",
        path: plan.files.rubricPath,
        required: true,
      },
      ...(plan.files.starterPath
        ? [{
            role: "starter" as const,
            path: plan.files.starterPath,
            required: false,
          }]
        : []),
      ...(plan.files.referencePath
        ? [{
            role: "reference" as const,
            path: plan.files.referencePath,
            required: false,
          }]
        : []),
    ],
    datasets: plan.datasets.map((dataset) => ({
      name: dataset.name,
      url: dataset.url,
      ...(dataset.hash ? { hash: dataset.hash } : {}),
      mountPath: joinAssignmentPath(
        plan.files.fixturesPath ?? joinAssignmentPath(plan.files.root, "fixtures"),
        `datasets/${datasetFileName(dataset)}`,
      ),
    })),
  };
}

export async function materializeAssignmentMountPlan(
  fs: SessionFS,
  plan: AssignmentMountPlan,
  contents: AssignmentMountContents,
): Promise<AssignmentMaterializeResult> {
  const entries = collectAssignmentMountEntries(plan, contents);

  const writtenPaths: string[] = [];
  for (const entry of entries.writes) {
    await fs.write(entry.path, entry.content);
    writtenPaths.push(entry.path);
  }

  return { writtenPaths, skippedOptionalPaths: entries.skippedOptionalPaths };
}

export function evaluateAssignmentMountContents(
  plan: AssignmentMountPlan,
  contents: AssignmentMountContents,
): AssignmentMountContentEvaluation {
  const evaluation = inspectAssignmentMountContents(plan, contents);
  return {
    ok: evaluation.missingRequiredFiles.length === 0 &&
      evaluation.missingDatasets.length === 0,
    writablePaths: evaluation.writes.map((entry) => entry.path),
    missingRequiredFiles: evaluation.missingRequiredFiles,
    missingDatasets: evaluation.missingDatasets,
    skippedOptionalPaths: evaluation.skippedOptionalPaths,
  };
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

function collectAssignmentMountEntries(
  plan: AssignmentMountPlan,
  contents: AssignmentMountContents,
): {
  readonly writes: readonly { readonly path: string; readonly content: AssignmentMountContent }[];
  readonly skippedOptionalPaths: readonly string[];
} {
  const evaluation = inspectAssignmentMountContents(plan, contents);
  if (evaluation.missingRequiredFiles.length > 0) {
    throw new BrowsergradError(
      `missing required assignment file content: ${evaluation.missingRequiredFiles[0]}`,
    );
  }
  if (evaluation.missingDatasets.length > 0) {
    throw new BrowsergradError(
      `missing assignment dataset content: ${evaluation.missingDatasets[0]}`,
    );
  }
  return {
    writes: evaluation.writes,
    skippedOptionalPaths: evaluation.skippedOptionalPaths,
  };
}

function inspectAssignmentMountContents(
  plan: AssignmentMountPlan,
  contents: AssignmentMountContents,
): {
  readonly writes: readonly { readonly path: string; readonly content: AssignmentMountContent }[];
  readonly missingRequiredFiles: readonly string[];
  readonly missingDatasets: readonly string[];
  readonly skippedOptionalPaths: readonly string[];
} {
  const skippedOptionalPaths: string[] = [];
  const missingRequiredFiles: string[] = [];
  const missingDatasets: string[] = [];
  const fileWrites: Array<{ path: string; content: AssignmentMountContent }> = [];
  const datasetWrites: Array<{ path: string; content: AssignmentMountContent }> = [];

  for (const file of plan.files) {
    const content = contents.files[file.path];
    if (content === undefined) {
      if (file.required) {
        missingRequiredFiles.push(file.path);
        continue;
      }
      skippedOptionalPaths.push(file.path);
      continue;
    }
    fileWrites.push({ path: file.path, content });
  }

  for (const dataset of plan.datasets) {
    const content = contents.datasets?.[dataset.name];
    if (content === undefined) {
      missingDatasets.push(dataset.name);
      continue;
    }
    datasetWrites.push({ path: dataset.mountPath, content });
  }

  return {
    writes: [...fileWrites, ...datasetWrites],
    missingRequiredFiles,
    missingDatasets,
    skippedOptionalPaths,
  };
}

function readString(
  obj: Record<string, unknown>,
  key: string,
  errors: string[],
  pattern?: RegExp,
): string | undefined {
  const value = obj[key];
  if (typeof value !== "string") {
    errors.push(`${key}: required string field missing or not a string`);
    return undefined;
  }
  if (pattern && !pattern.test(value)) {
    errors.push(`${key}: value ${JSON.stringify(value)} does not match ${pattern.source}`);
  }
  return value;
}

function readOptionalString(
  obj: Record<string, unknown>,
  key: string,
  errors: string[],
): string | undefined {
  const value = obj[key];
  if (value === undefined) return undefined;
  if (typeof value !== "string") {
    errors.push(`${key}: must be a string when present`);
    return undefined;
  }
  return value;
}

function readFiles(
  value: unknown,
  errors: string[],
): AssignmentProfileFiles | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    errors.push("files: required object field missing or not an object");
    return undefined;
  }
  const obj = value as Record<string, unknown>;
  const root = readString(obj, "root", errors);
  const rubric_path = readString(obj, "rubric_path", errors);
  const starter_path = readOptionalString(obj, "starter_path", errors);
  const reference_path = readOptionalString(obj, "reference_path", errors);
  const fixtures_path = readOptionalString(obj, "fixtures_path", errors);
  if (!root || !rubric_path) return undefined;
  return {
    root,
    rubric_path,
    ...(starter_path ? { starter_path } : {}),
    ...(reference_path ? { reference_path } : {}),
    ...(fixtures_path ? { fixtures_path } : {}),
  };
}

function readTimeouts(
  value: unknown,
  errors: string[],
): AssignmentProfileTimeouts {
  if (value === undefined) return {};
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    errors.push("timeouts: must be an object when present");
    return {};
  }
  const obj = value as Record<string, unknown>;
  const out: Record<string, number> = {};
  for (const key of ["setup_ms", "test_ms", "worker_ms"]) {
    const timeout = obj[key];
    if (timeout === undefined) continue;
    if (
      typeof timeout !== "number" ||
      !Number.isInteger(timeout) ||
      timeout < 0
    ) {
      errors.push(`timeouts.${key}: must be a non-negative integer`);
      continue;
    }
    out[key] = timeout;
  }
  return out;
}

function evaluateCapabilityGate(
  gate: AssignmentGateSpec,
  available: ReadonlySet<string>,
  capabilityModes: Readonly<Record<string, AssignmentCapabilityMode>>,
): AssignmentCapabilityGateEvaluation {
  const requires = readCapabilityStringList(gate.options.requires);
  const anyOf = readCapabilityAlternatives(gate.options.any_of);
  const missingRequired = requires.filter((capability) => !available.has(capability));
  const satisfiedAnyOf = bestSatisfiedCapabilityGroup(anyOf, available, capabilityModes);
  const missingAnyOf =
    anyOf.length === 0 || satisfiedAnyOf.length > 0
      ? []
      : anyOf.map((group) => group.filter((capability) => !available.has(capability)));
  const message =
    typeof gate.options.message === "string" ? gate.options.message : undefined;
  const ok = missingRequired.length === 0 && missingAnyOf.length === 0;
  const selectedCapabilities = ok
    ? uniqueSorted([...requires, ...satisfiedAnyOf])
    : [];
  const satisfiedCapabilities = ok
    ? selectedCapabilities
    : uniqueSorted(requires.filter((capability) => available.has(capability)));

  return {
    name: gate.name,
    ok,
    status: capabilityGateStatus(selectedCapabilities, capabilityModes, ok),
    requires,
    anyOf,
    selectedAnyOf: ok ? satisfiedAnyOf : [],
    selectedCapabilities,
    satisfiedCapabilities,
    missingRequired,
    missingAnyOf,
    ...(message ? { message } : {}),
  };
}

function capabilityGateStatus(
  selectedCapabilities: readonly string[],
  capabilityModes: Readonly<Record<string, AssignmentCapabilityMode>>,
  ok: boolean,
): AssignmentRunReadinessStatus {
  if (!ok) return "blocked";
  if (selectedCapabilities.some((capability) => capabilityModes[capability] === "external")) {
    return "external-only";
  }
  if (selectedCapabilities.some((capability) => capabilityModes[capability] === "simulated")) {
    return "simulated";
  }
  return "runnable";
}

function bestSatisfiedCapabilityGroup(
  groups: readonly (readonly string[])[],
  available: ReadonlySet<string>,
  capabilityModes: Readonly<Record<string, AssignmentCapabilityMode>>,
): readonly string[] {
  return groups
    .filter((group) => group.every((capability) => available.has(capability)))
    .sort((a, b) => {
      const modeDiff =
        capabilityGroupModeRank(a, capabilityModes) -
        capabilityGroupModeRank(b, capabilityModes);
      if (modeDiff !== 0) return modeDiff;
      return a.length - b.length;
    })[0] ?? [];
}

function capabilityGroupModeRank(
  group: readonly string[],
  capabilityModes: Readonly<Record<string, AssignmentCapabilityMode>>,
): number {
  return Math.max(
    0,
    ...group.map((capability) => capabilityModeRank(capabilityModes[capability])),
  );
}

function capabilityModeRank(mode: AssignmentCapabilityMode | undefined): number {
  if (mode === "external") return 2;
  if (mode === "simulated") return 1;
  return 0;
}

function normalizeCapabilityModes(
  value: AssignmentCapabilityEnvironment["capabilityModes"],
): Readonly<Record<string, AssignmentCapabilityMode>> {
  if (!value) return {};
  const out: Record<string, AssignmentCapabilityMode> = {};
  for (const [capability, mode] of Object.entries(value)) {
    if (mode === "browser" || mode === "simulated" || mode === "external") {
      out[capability] = mode;
    }
  }
  return out;
}

function readCapabilityStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

function validateCapabilityGateOptions(
  options: Record<string, unknown>,
  path: string,
  errors: string[],
): void {
  validateStringListOption(options.requires, `${path}.requires`, errors);

  const anyOf = options.any_of;
  if (anyOf !== undefined) {
    if (!Array.isArray(anyOf)) {
      errors.push(`${path}.any_of: must be an array of string arrays when present`);
    } else {
      for (let i = 0; i < anyOf.length; i++) {
        const group = anyOf[i];
        if (!Array.isArray(group)) {
          errors.push(`${path}.any_of[${i}]: must be a string array`);
          continue;
        }
        validateStringListItems(group, `${path}.any_of[${i}]`, errors);
      }
    }
  }

  const message = options.message;
  if (message !== undefined && typeof message !== "string") {
    errors.push(`${path}.message: must be a string when present`);
  }
}

function validateStringListOption(
  value: unknown,
  path: string,
  errors: string[],
): void {
  if (value === undefined) return;
  if (!Array.isArray(value)) {
    errors.push(`${path}: must be a string array when present`);
    return;
  }
  validateStringListItems(value, path, errors);
}

function validateStringListItems(
  value: readonly unknown[],
  path: string,
  errors: string[],
): void {
  for (let i = 0; i < value.length; i++) {
    if (typeof value[i] !== "string") {
      errors.push(`${path}[${i}]: must be a string`);
    }
  }
}

function readCapabilityAlternatives(value: unknown): string[][] {
  if (!Array.isArray(value)) return [];
  const groups: string[][] = [];
  for (const item of value) {
    if (!Array.isArray(item)) continue;
    groups.push(readCapabilityStringList(item));
  }
  return groups;
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort();
}

function joinAssignmentPath(root: string, path: string): string {
  if (path.startsWith("/")) return path;
  return `${root.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

function datasetFileName(dataset: AssignmentDataset): string {
  const withoutQuery = dataset.url.split(/[?#]/, 1)[0] ?? "";
  const lastSegment = withoutQuery.split("/").filter(Boolean).at(-1);
  return lastSegment || dataset.name;
}

function readStringArray(
  value: unknown,
  key: string,
  errors: string[],
  maxLength: number,
): string[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) {
    errors.push(`${key}: must be a string array when present`);
    return [];
  }
  if (value.length > maxLength) {
    errors.push(`${key}: at most ${maxLength} entries (got ${value.length})`);
  }
  const out: string[] = [];
  for (let i = 0; i < value.length; i++) {
    const item = value[i];
    if (typeof item !== "string") {
      errors.push(`${key}[${i}]: must be a string`);
    } else {
      out.push(item);
    }
  }
  return out;
}

function readOracles(
  value: unknown,
  errors: string[],
): AssignmentOracleSpec[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) {
    errors.push("oracles: must be an array when present");
    return [];
  }
  const out: AssignmentOracleSpec[] = [];
  for (let i = 0; i < value.length; i++) {
    const item = value[i];
    if (typeof item !== "object" || item === null || Array.isArray(item)) {
      errors.push(`oracles[${i}]: must be an object`);
      continue;
    }
    const obj = item as Record<string, unknown>;
    const name = readString(obj, "name", errors);
    const js_module = readString(obj, "js_module", errors);
    const export_name = readOptionalString(obj, "export_name", errors);
    if (name && js_module) {
      out.push({
        name,
        js_module,
        ...(export_name ? { export_name } : {}),
      });
    }
  }
  return out;
}

function readGates(
  value: unknown,
  errors: string[],
): AssignmentGateSpec[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) {
    errors.push("gates: must be an array when present");
    return [];
  }
  const out: AssignmentGateSpec[] = [];
  for (let i = 0; i < value.length; i++) {
    const item = value[i];
    if (typeof item !== "object" || item === null || Array.isArray(item)) {
      errors.push(`gates[${i}]: must be an object`);
      continue;
    }
    const obj = item as Record<string, unknown>;
    const name = readString(obj, "name", errors);
    const kindValue = obj.kind;
    if (typeof kindValue !== "string" || !GATE_KINDS.has(kindValue as AssignmentGateKind)) {
      errors.push(`gates[${i}].kind: must be one of capability, streaming, timeout, forbidden-read`);
      continue;
    }
    const options = obj.options;
    if (
      options !== undefined &&
      (typeof options !== "object" || options === null || Array.isArray(options))
    ) {
      errors.push(`gates[${i}].options: must be an object when present`);
      continue;
    }
    const normalizedOptions = options === undefined ? {} : { ...(options as Record<string, unknown>) };
    if (kindValue === "capability") {
      validateCapabilityGateOptions(
        normalizedOptions,
        `gates[${i}].options`,
        errors,
      );
    }
    if (name) {
      out.push({
        name,
        kind: kindValue as AssignmentGateKind,
        options: normalizedOptions,
      });
    }
  }
  return out;
}

function readDatasets(
  value: unknown,
  errors: string[],
): AssignmentDataset[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) {
    errors.push("datasets: must be an array when present");
    return [];
  }
  if (value.length > 32) {
    errors.push(`datasets: at most 32 entries (got ${value.length})`);
  }
  const out: AssignmentDataset[] = [];
  for (let i = 0; i < value.length; i++) {
    const item = value[i];
    if (typeof item !== "object" || item === null || Array.isArray(item)) {
      errors.push(`datasets[${i}]: must be an object`);
      continue;
    }
    const obj = item as Record<string, unknown>;
    const name = readString(obj, "name", errors);
    const url = readString(obj, "url", errors);
    const hash = readOptionalString(obj, "hash", errors);
    if (name && url) {
      out.push({
        name,
        url,
        ...(hash ? { hash } : {}),
      });
    }
  }
  return out;
}
