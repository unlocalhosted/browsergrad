import {
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
}

export interface AssignmentCapabilityGateEvaluation {
  readonly name: string;
  readonly ok: boolean;
  readonly requires: readonly string[];
  readonly anyOf: readonly (readonly string[])[];
  readonly missingRequired: readonly string[];
  readonly missingAnyOf: readonly (readonly string[])[];
  readonly message?: string;
}

export interface AssignmentCapabilityEvaluation {
  readonly ok: boolean;
  readonly missingCapabilities: readonly string[];
  readonly gates: readonly AssignmentCapabilityGateEvaluation[];
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

export interface AssignmentMountContents {
  readonly files: Readonly<Record<string, string>>;
  readonly datasets?: Readonly<Record<string, string>>;
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
  const gates = profile.gates
    .filter((gate) => gate.kind === "capability")
    .map((gate) => evaluateCapabilityGate(gate, available));

  return {
    ok: gates.every((gate) => gate.ok),
    missingCapabilities: uniqueSorted(
      gates.flatMap((gate) => [
        ...gate.missingRequired,
        ...gate.missingAnyOf.flat(),
      ]),
    ),
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
  const skippedOptionalPaths: string[] = [];
  const fileWrites: Array<{ path: string; content: string }> = [];
  const datasetWrites: Array<{ path: string; content: string }> = [];

  for (const file of plan.files) {
    const content = contents.files[file.path];
    if (content === undefined) {
      if (file.required) {
        throw new BrowsergradError(`missing required assignment file content: ${file.path}`);
      }
      skippedOptionalPaths.push(file.path);
      continue;
    }
    fileWrites.push({ path: file.path, content });
  }

  for (const dataset of plan.datasets) {
    const content = contents.datasets?.[dataset.name];
    if (content === undefined) {
      throw new BrowsergradError(`missing assignment dataset content: ${dataset.name}`);
    }
    datasetWrites.push({ path: dataset.mountPath, content });
  }

  const writtenPaths: string[] = [];
  for (const entry of [...fileWrites, ...datasetWrites]) {
    await fs.write(entry.path, entry.content);
    writtenPaths.push(entry.path);
  }

  return { writtenPaths, skippedOptionalPaths };
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
): AssignmentCapabilityGateEvaluation {
  const requires = readCapabilityStringList(gate.options.requires);
  const anyOf = readCapabilityAlternatives(gate.options.any_of);
  const missingRequired = requires.filter((capability) => !available.has(capability));
  const missingAnyOf =
    anyOf.length === 0 || anyOf.some((group) => group.every((capability) => available.has(capability)))
      ? []
      : anyOf.map((group) => group.filter((capability) => !available.has(capability)));
  const message =
    typeof gate.options.message === "string" ? gate.options.message : undefined;

  return {
    name: gate.name,
    ok: missingRequired.length === 0 && missingAnyOf.length === 0,
    requires,
    anyOf,
    missingRequired,
    missingAnyOf,
    ...(message ? { message } : {}),
  };
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
