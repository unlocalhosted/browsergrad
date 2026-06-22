import type {
  Artifact,
  Assertion,
  ExecOptions,
  ExecResult,
  PyodideJsModule,
  SessionFS,
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

export interface AssignmentCapabilityEnvironmentInput {
  readonly browserCapabilities?: readonly string[];
  readonly simulatedCapabilities?: readonly string[];
  readonly externalCapabilities?: readonly string[];
}

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

export interface AssignmentCapabilityCatalog {
  readonly capabilities: readonly AssignmentCapabilityCatalogEntry[];
}

export interface AssignmentCapabilityCatalogEntry {
  readonly capability: string;
  readonly profiles: readonly string[];
  readonly requiredBy: readonly AssignmentCapabilityCatalogReference[];
  readonly alternativeIn: readonly AssignmentCapabilityCatalogAlternative[];
}

export interface AssignmentCapabilityCatalogReference {
  readonly profileId: string;
  readonly gate: string;
}

export interface AssignmentCapabilityCatalogAlternative
  extends AssignmentCapabilityCatalogReference {
  readonly group: readonly string[];
}

export type AssignmentRunReadinessStatus =
  | "runnable"
  | "simulated"
  | "external-only"
  | "blocked";

export type AssignmentRunnerTarget =
  | "pyodide"
  | "javascript"
  | "external"
  | "unsupported"
  | "blocked";

export interface AssignmentRunReadiness {
  readonly status: AssignmentRunReadinessStatus;
  readonly summary: string;
  readonly missingCapabilities: readonly string[];
  readonly selectedCapabilities: readonly string[];
  readonly simulatedCapabilities: readonly string[];
  readonly externalCapabilities: readonly string[];
}

export interface AssignmentRunnerRoute {
  readonly target: AssignmentRunnerTarget;
  readonly readinessStatus: AssignmentRunReadinessStatus;
  readonly rubricKind: AssignmentRubricKind;
  readonly message: string;
  readonly missingCapabilities: readonly string[];
  readonly selectedCapabilities: readonly string[];
}

export interface AssignmentPreflightReport {
  readonly plan: AssignmentRunPlan;
  readonly rubricKind: AssignmentRubricKind;
  readonly readiness: AssignmentRunReadiness;
  readonly runnerRoute: AssignmentRunnerRoute;
  readonly requiredCapabilities: readonly string[];
  readonly mountPlan: AssignmentMountPlan;
  readonly datasetCachePlan: AssignmentDatasetCachePlan;
  readonly externalRunnerRequest?: AssignmentExternalRunnerRequest;
}

export interface AssignmentBenchmarkPreflightRow {
  readonly id: string;
  readonly title?: string;
  readonly course?: string;
  readonly sourceUrl?: string;
  readonly ok: boolean;
  readonly readinessStatus: AssignmentRunReadinessStatus;
  readonly runnerTarget: AssignmentRunnerTarget;
  readonly rubricKind: AssignmentRubricKind;
  readonly requiredCapabilities: readonly string[];
  readonly selectedCapabilities: readonly string[];
  readonly missingCapabilities: readonly string[];
  readonly simulatedCapabilities: readonly string[];
  readonly externalCapabilities: readonly string[];
  readonly contentOk: boolean;
  readonly missingRequiredFiles: readonly string[];
  readonly missingDatasets: readonly string[];
  readonly skippedOptionalPaths: readonly string[];
  readonly cacheStrategies: readonly AssignmentDatasetCacheStrategy[];
  readonly externalRunnerRequired: boolean;
  readonly gates: readonly AssignmentBenchmarkPreflightGateRow[];
}

export interface AssignmentBenchmarkPreflightGateRow {
  readonly name: string;
  readonly ok: boolean;
  readonly status: AssignmentRunReadinessStatus;
  readonly requires: readonly string[];
  readonly anyOf: readonly (readonly string[])[];
  readonly selectedAnyOf: readonly string[];
  readonly selectedCapabilities: readonly string[];
  readonly missingRequired: readonly string[];
  readonly missingAnyOf: readonly (readonly string[])[];
  readonly message?: string;
}

export interface AssignmentBenchmarkPreflightMatrix {
  readonly ok: boolean;
  readonly rows: readonly AssignmentBenchmarkPreflightRow[];
}

export type AssignmentPlatformNextAction =
  | "install-capabilities"
  | "mount-content"
  | "verify-content"
  | "run-pyodide"
  | "run-javascript"
  | "request-external-runner"
  | "unsupported";

export interface AssignmentPlatformHandoff {
  readonly id: string;
  readonly title?: string;
  readonly course?: string;
  readonly sourceUrl?: string;
  readonly readinessStatus: AssignmentRunReadinessStatus;
  readonly runnerTarget: AssignmentRunnerTarget;
  readonly rubricKind: AssignmentRubricKind;
  readonly nextAction: AssignmentPlatformNextAction;
  readonly summary: string;
  readonly launchable: boolean;
  readonly missingCapabilities: readonly string[];
  readonly missingRequiredFiles: readonly string[];
  readonly missingDatasets: readonly string[];
  readonly skippedOptionalPaths: readonly string[];
  readonly selectedCapabilities: readonly string[];
  readonly simulatedCapabilities: readonly string[];
  readonly externalCapabilities: readonly string[];
  readonly cacheStrategies: readonly AssignmentDatasetCacheStrategy[];
  readonly externalRunnerRequired: boolean;
  readonly messages: readonly string[];
}

export interface AssignmentVerifiedPlatformHandoff
  extends AssignmentPlatformHandoff {
  readonly hashOk: boolean;
  readonly hashChecks: readonly AssignmentMountHashCheck[];
}

export interface AssignmentPlatformIssueDraft {
  readonly title: string;
  readonly body: string;
  readonly labels: readonly string[];
}

export interface AssignmentVerifiedBenchmarkPreflightRow
  extends AssignmentBenchmarkPreflightRow {
  readonly hashOk: boolean;
  readonly hashChecks: readonly AssignmentMountHashCheck[];
}

export interface AssignmentVerifiedBenchmarkPreflightMatrix {
  readonly ok: boolean;
  readonly rows: readonly AssignmentVerifiedBenchmarkPreflightRow[];
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

export interface AssignmentExternalRunnerRequest {
  readonly id: string;
  readonly profileVersion: string;
  readonly requiresBrowsergrad: string;
  readonly route: AssignmentRunnerRoute;
  readonly selectedCapabilities: readonly string[];
  readonly externalCapabilities: readonly string[];
  readonly simulatedCapabilities: readonly string[];
  readonly environment: Readonly<Record<string, string>>;
  readonly files: AssignmentRunPlanFiles;
  readonly execution: AssignmentRunPlanExecution;
  readonly mountPlan: AssignmentMountPlan;
  readonly datasetCachePlan: AssignmentDatasetCachePlan;
  readonly behavioralGates: readonly AssignmentGateSpec[];
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

export type AssignmentDatasetCacheStrategy =
  | "content-addressed"
  | "source-addressed"
  | "invalid-hash"
  | "unsupported-hash";

export interface AssignmentDatasetCacheEntry {
  readonly name: string;
  readonly url: string;
  readonly hash?: string;
  readonly mountPath: string;
  readonly strategy: AssignmentDatasetCacheStrategy;
  readonly cacheKey: string;
  readonly cachePath: string;
}

export interface AssignmentDatasetCachePlan {
  readonly root: string;
  readonly datasets: readonly AssignmentDatasetCacheEntry[];
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

export type AssignmentMountHashStatus =
  | "match"
  | "mismatch"
  | "missing"
  | "invalid"
  | "unsupported";

export interface AssignmentMountHashCheck {
  readonly name: string;
  readonly path: string;
  readonly algorithm: string;
  readonly expected: string;
  readonly actual?: string;
  readonly ok: boolean;
  readonly status: AssignmentMountHashStatus;
}

export interface AssignmentMountHashVerification {
  readonly ok: boolean;
  readonly checks: readonly AssignmentMountHashCheck[];
}

export interface AssignmentMountPreflightReport {
  readonly ok: boolean;
  readonly content: AssignmentMountContentEvaluation;
  readonly hashes: AssignmentMountHashVerification;
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

export interface AssignmentJavascriptProfileRunResult {
  readonly report: AssignmentPreflightReport;
  readonly run: AssignmentJavascriptRubricRunResult;
}

export type AssignmentProfileParseResult =
  | { ok: true; profile: AssignmentProfile }
  | { ok: false; errors: readonly string[] };
