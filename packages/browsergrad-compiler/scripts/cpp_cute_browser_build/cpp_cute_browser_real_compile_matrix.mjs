import { isAbsolute, resolve } from "node:path";

import {
  persistCppCuteBrowserRealCompileEvidence,
  runCppCuteBrowserRealCompile,
} from "./cpp_cute_browser_real_compile_runner.mjs";

const ERROR_CODE =
  "BG-COMPILER-CPP-CUTE-BROWSER-REAL-COMPILE-MATRIX";
const CASE_IDS = Object.freeze(["rank2", "rank3"]);

export class CppCuteBrowserRealCompileMatrixError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserRealCompileMatrixError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

export function parseCppCuteBrowserRealCompileMatrixArguments(argv) {
  const values = new Map();
  let separatorSeen = false;
  for (const [index, argument] of argv.entries()) {
    if (argument === "--") {
      if (separatorSeen) invalid(`$.argv[${index}]`, "duplicate argument separator");
      separatorSeen = true;
      continue;
    }
    if (argument === "--require-compiled" ||
        argument === "--allow-untrusted-diagnostic-wasm") {
      if (values.has(argument)) {
        invalid(`$.argv[${index}]`, `duplicate ${argument}`);
      }
      values.set(argument, true);
      continue;
    }
    const separator = argument.indexOf("=");
    if (!argument.startsWith("--") || separator < 3) {
      invalid(`$.argv[${index}]`, "expected one supported --name=value option");
    }
    const name = argument.slice(2, separator);
    if (name !== "wasm" && name !== "pack-root" &&
        name !== "evidence-output") {
      invalid(`$.argv[${index}]`, `unsupported option --${name}`);
    }
    if (values.has(name)) invalid(`$.argv[${index}]`, `duplicate --${name}`);
    const value = argument.slice(separator + 1);
    if (!isAbsolute(value)) {
      invalid(`$.argv[${index}]`, `--${name} requires one absolute path`);
    }
    values.set(name, value);
  }
  const wasmPath = values.get("wasm");
  const packRoot = values.get("pack-root");
  const evidenceOutput = values.get("evidence-output");
  if (typeof wasmPath !== "string" ||
      typeof packRoot !== "string" ||
      typeof evidenceOutput !== "string") {
    invalid("$.argv", "wasm, pack-root, and evidence-output are required");
  }
  const requireCompiled = values.get("--require-compiled") === true;
  const allowUntrustedDiagnosticWasm =
    values.get("--allow-untrusted-diagnostic-wasm") === true;
  if (requireCompiled && allowUntrustedDiagnosticWasm) {
    invalid(
      "$.argv",
      "strict and untrusted-diagnostic matrix modes are mutually exclusive",
    );
  }
  return Object.freeze({
    wasmPath,
    packRoot,
    evidenceOutput,
    requireCompiled,
    allowUntrustedDiagnosticWasm,
  });
}

export function prepareCppCuteBrowserRealCompileMatrix(observations) {
  if (!Array.isArray(observations) ||
      observations.length !== CASE_IDS.length) {
    invalid("$.observations", "expected exactly rank2 and rank3 observations");
  }
  const cases = CASE_IDS.map((caseId, index) => {
    const observation = observations[index];
    if (typeof observation !== "object" || observation === null ||
        observation.schema !==
          "browsergrad.compiler.cpp-cute.browser-real-compile-observation" ||
        observation.version !== 2 ||
        observation.outcome !== "compiled" ||
        observation.source?.caseId !== caseId ||
        observation.execution?.artifactOutcome !== "accepted" ||
        observation.semanticCandidate?.sourceCoordinateRank !== index + 2 ||
        observation.semanticCandidate?.destinationCoordinateRank !== index + 2 ||
        observation.semanticCandidate?.sharedViewCopySemanticsPrepared !== true ||
        observation.inputs?.packagePinnedHeaderPacksMatched !== true ||
        observation.execution?.rawWasmVerified !== true ||
        observation.execution?.exactInterfaceConformanceObserved !== true ||
        observation.execution?.verifierWorkerExecutionObserved !== true ||
        observation.execution?.workerExecutionObserved !== true ||
        observation.workerExecutionObserved !== true ||
        observation.headerDistributionLicenseApproved !== false ||
        observation.producerTrusted !== false ||
        observation.loweringAuthorityMinted !== false ||
        observation.backendExecutionAuthorized !== false ||
        observation.releaseReady !== false) {
      invalid(`$.observations[${index}]`, `invalid compiled ${caseId} evidence`);
    }
    return observation;
  });
  const wasmIdentities = new Set(cases.map((entry) =>
    `${entry.inputs.wasmSha256}:${entry.inputs.totalExternalByteLength}`));
  const assetSets = new Set(cases.map((entry) => entry.inputs.assetSetSha256));
  const workerEvidence = new Set(cases.map((entry) =>
    entry.execution.evidenceId));
  const artifacts = new Set(cases.map((entry) => entry.execution.artifactId));
  const candidates = new Set(cases.map((entry) =>
    entry.semanticCandidate.candidateId));
  if (wasmIdentities.size !== 1 ||
      assetSets.size !== 1 ||
      workerEvidence.size !== CASE_IDS.length ||
      artifacts.size !== CASE_IDS.length ||
      candidates.size !== CASE_IDS.length) {
    invalid("$.observations", "matrix identities are inconsistent or reused");
  }
  const pinnedReproducibleWasmMatched = cases.every(
    (entry) =>
      entry.inputs.pinnedReproducibleWasmMatched === true &&
      entry.inputs.untrustedDiagnosticWasm === false &&
      entry.inputs.wasmAuthority === "package-pinned-two-clean-build-output",
  );
  const untrustedDiagnosticWasm = cases.every(
    (entry) =>
      entry.inputs.pinnedReproducibleWasmMatched === false &&
      entry.inputs.untrustedDiagnosticWasm === true &&
      entry.inputs.wasmAuthority ===
        "untrusted-diagnostic-local-byte-observation-only",
  );
  if (!pinnedReproducibleWasmMatched && !untrustedDiagnosticWasm) {
    invalid("$.observations", "matrix mixes incompatible Wasm authority tiers");
  }
  return Object.freeze({
    schema:
      "browsergrad.compiler.cpp-cute.browser-real-compile-matrix-observation",
    version: 1,
    authority: "local-real-browser-worker-matrix-observation-only",
    caseCount: 2,
    cases: Object.freeze(cases),
    claims: Object.freeze({
      unchangedCpp17CuteRank2Compiled: true,
      unchangedCpp17CuteRank3Compiled: true,
      canonicalGate2LayoutFixturesMatched: true,
      packagePinnedHeaderPacksMatched: true,
      pinnedReproducibleWasmMatched,
      untrustedDiagnosticWasm,
      headerDistributionLicenseApproved: false,
      producerTrusted: false,
      loweringAuthorityMinted: false,
      backendExecutionAuthorized: false,
      releaseReady: false,
      workerExecutionObserved: true,
    }),
  });
}

export async function runCppCuteBrowserRealCompileMatrix(
  argv = process.argv.slice(2),
) {
  const options = parseCppCuteBrowserRealCompileMatrixArguments(argv);
  const observations = [];
  // Each compiler Worker reserves a bounded large memory. Keep the two cases
  // sequential so a coverage matrix cannot double peak browser memory.
  for (const caseId of CASE_IDS) {
    const caseArguments = [
      `--wasm=${options.wasmPath}`,
      `--pack-root=${options.packRoot}`,
      `--case=${caseId}`,
      ...(options.requireCompiled ? ["--require-compiled"] : []),
      ...(options.allowUntrustedDiagnosticWasm
        ? ["--allow-untrusted-diagnostic-wasm"]
        : []),
    ];
    observations.push(await runCppCuteBrowserRealCompile(caseArguments));
  }
  const matrix = prepareCppCuteBrowserRealCompileMatrix(observations);
  const written = await persistCppCuteBrowserRealCompileEvidence(
    options.evidenceOutput,
    matrix,
  );
  process.stdout.write(`Browser compile matrix evidence: ${written.outputPath}\n`);
  return matrix;
}

function invalid(path, message, options) {
  throw new CppCuteBrowserRealCompileMatrixError(path, message, options);
}

if (process.argv[1] !== undefined &&
    import.meta.url === new URL(`file://${resolve(process.argv[1])}`).href) {
  runCppCuteBrowserRealCompileMatrix().catch((error) => {
    process.stderr.write(
      `${error instanceof Error ? error.stack ?? error.message : String(error)}\n`,
    );
    process.exitCode = 1;
  });
}
