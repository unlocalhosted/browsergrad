export const COMPILER_VIEW_COPY_EVIDENCE_PREFIX: string;

export interface CompilerViewCopyEvidenceLogOptions {
  readonly expectedSourceRevision: string;
  readonly gitHead: string;
  readonly relevantStatus: string;
  readonly producerVersions: Readonly<Record<string, string>>;
}

export interface CompilerViewCopyPreparedCaseManifest {
  readonly caseId: string;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly specializationHash: string;
  readonly bindingProjectionHash: string;
  readonly compileIdentityHash: string;
  readonly wgslModuleHash: string;
  readonly programName: string;
  readonly sourceHash: string;
  readonly initialDestinationHash: string;
  readonly expectedSourceHash: string;
  readonly expectedDestinationHash: string;
  readonly logicalShape: readonly number[];
  readonly logicalInvocationCount: readonly number[];
  readonly plannedWorkgroupCount: readonly number[];
  readonly expectedReadElements: number;
  readonly expectedFilledElements: number;
  readonly caseArtifactHash: string;
}

export interface CompilerViewCopyExpectedEvidence {
  readonly preparedCases: readonly CompilerViewCopyPreparedCaseManifest[];
  readonly preparedBackendArtifactHash: string;
  readonly caseSetHash: string;
  readonly artifactHash: string;
}

export function verifyCompilerViewCopyBindingsEvidenceLog(
  log: string,
  options: CompilerViewCopyEvidenceLogOptions,
): Promise<Record<string, unknown>>;

export function deriveCompilerViewCopyBindingsExpectedEvidence(
  sourceRevision: string,
  producerVersions: Readonly<Record<string, string>>,
): Promise<CompilerViewCopyExpectedEvidence>;

export function loadCompilerViewCopyProducerVersions(
  repositoryRoot: URL,
): Readonly<Record<string, string>>;

export function readBoundedCompilerViewCopyEvidenceLog(logPath: string): string;
