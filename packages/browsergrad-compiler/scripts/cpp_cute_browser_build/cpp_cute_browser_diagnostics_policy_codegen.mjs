import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const moduleDirectory = dirname(fileURLToPath(import.meta.url));
export const CPP_CUTE_DIAGNOSTICS_POLICY_INCLUDE_PATH = resolve(
  moduleDirectory,
  "extractor",
  "BrowserGradCppCuteDiagnosticsPolicy.inc",
);

/** @param {unknown} value @param {string} path */
function record(value, path) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${path} must be an object`);
  }
  return value;
}

/** @param {unknown} value @param {string} path */
function array(value, path) {
  if (!Array.isArray(value)) throw new TypeError(`${path} must be an array`);
  return value;
}

/** @param {unknown} value @param {string} path */
function ascii(value, path) {
  if (typeof value !== "string" || value.length === 0 || !/^[\x20-\x7e]+$/u.test(value)) {
    throw new TypeError(`${path} must be non-empty printable ASCII`);
  }
  return value;
}

/** @param {unknown} value @param {string} path */
function uint32(value, path) {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new TypeError(`${path} must be an unsigned 32-bit integer`);
  }
  return value;
}

/** @param {unknown} actual @param {unknown} expected @param {string} path */
function exact(actual, expected, path) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new TypeError(`${path} differs from the closed diagnostic-normalization policy`);
  }
}

/** @param {string} value */
function cppString(value) {
  return JSON.stringify(value);
}

/** @param {unknown} value */
function canonicalJson(value) {
  if (value === null || typeof value === "boolean" || typeof value === "string") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value) || Object.is(value, -0)) {
      throw new TypeError("diagnostic policy contains a noncanonical number");
    }
    return String(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  const object = record(value, "$canonical");
  const keys = Object.keys(object).sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`).join(",")}}`;
}

/** @param {unknown} body */
function diagnosticManifestId(body) {
  const canonical = canonicalJson({
    domain: "browsergrad.compiler.cpp-cute.diagnostic-normalization-manifest-id.v1",
    body,
  });
  return `bg.cpp.diagnostic-normalization.sha256.${createHash("sha256").update(canonical).digest("hex")}`;
}

const STAGES = new Map([
  ["preprocessor", ["kPreprocessor", "kPreprocessing", "kPreprocessing"]],
  ["parser", ["kParser", "kParsing", "kParsing"]],
  ["sema-name-lookup", ["kSemaNameLookup", "kNameLookup", "kNameLookup"]],
  ["sema-overload-resolution", ["kSemaOverloadResolution", "kOverloadResolution", "kOverloadResolution"]],
  ["sema-template-instantiation", ["kSemaTemplateInstantiation", "kTemplateInstantiation", "kTemplateInstantiation"]],
  ["sema-cuda", ["kSemaCuda", "kCudaSema", "kCudaSema"]],
  ["artifact-extractor", ["kArtifactExtractor", "kArtifactExtraction", "kArtifactExtraction"]],
]);
const SEVERITIES = new Map([
  ["ignored", ["kIgnored", "kNone"]],
  ["remark", ["kRemark", "kRemark"]],
  ["note", ["kNote", "kNote"]],
  ["warning", ["kWarning", "kWarning"]],
  ["error", ["kError", "kError"]],
  ["fatal", ["kFatal", "kFatal"]],
]);
const CUSTOM_CODES = new Map([
  ["browsergrad.cpp-cute:temporal-macro-forbidden", "kTemporalMacroForbidden"],
  ["browsergrad.cpp-cute:temporal-macro-mutation-forbidden", "kTemporalMacroMutationForbidden"],
  ["browsergrad.cpp-cute:diagnostic-resource-limit", "kDiagnosticResourceLimit"],
  ["browsergrad.cpp-cute:diagnostic-normalization-failed", "kDiagnosticNormalizationFailed"],
  ["browsergrad.cpp-cute:semantic-extraction-failed", "kSemanticExtractionFailed"],
  ["browsergrad.cpp-cute:host-device-surface-divergence", "kHostDeviceSurfaceDivergence"],
]);
const CATEGORIES = new Map([
  ["preprocessing", "kPreprocessing"],
  ["parsing", "kParsing"],
  ["name-lookup", "kNameLookup"],
  ["overload-resolution", "kOverloadResolution"],
  ["template-instantiation", "kTemplateInstantiation"],
  ["cuda-sema", "kCudaSema"],
  ["artifact-extraction", "kArtifactExtraction"],
  ["policy", "kPolicy"],
  ["resource-limit", "kResourceLimit"],
]);

/**
 * Deterministically renders the native policy from the exact package resource.
 * The function rejects policy drift instead of silently generating a different
 * native contract.
 *
 * @param {unknown} normalization
 */
export function renderCppCuteDiagnosticsPolicyInclude(normalization) {
  const root = record(normalization, "$normalization");
  exact(root.schema, "browsergrad.compiler.cpp-cute.diagnostic-normalization", "$normalization.schema");
  exact(root.version, { major: 1, minor: 0 }, "$normalization.version");
  const body = record(root.body, "$normalization.body");
  exact(root.manifestId, diagnosticManifestId(body), "$normalization.manifestId");
  exact(body.policyId, "browsergrad.compiler.cpp-cute.clang-diagnostic-normalization@1", "$normalization.body.policyId");
  exact(body.artifactBinding, {
    schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
    major: 3,
    diagnosticShape: "browsergrad.compiler.cpp-cute.frontend-diagnostic@3",
    fixItsRepresented: false,
  }, "$normalization.body.artifactBinding");
  const clang = record(body.clang, "$normalization.body.clang");
  exact(clang, { compilerId: "clang", version: "22.1.8" }, "$normalization.body.clang");
  const classification = record(body.classification, "$normalization.body.classification");
  exact(classification, {
    rawDiagnosticIdRange: { minimum: 0, maximum: 4_294_967_295 },
    rawDiagnosticIdEncoding: "unsigned-decimal-no-leading-zeroes",
    clangDiagnosticCodeFormat: "clang:diag-<raw-diagnostic-id>",
    categorySource: "producer-stage-default-with-closed-custom-overrides",
    producerStageSource: "active-frontend-callback-boundary",
    messageInference: "forbidden",
    diagnosticGroupInference: "forbidden",
  }, "$normalization.body.classification");

  const stable = record(body.stableDiagnosticId, "$normalization.body.stableDiagnosticId");
  exact(stable.canonicalProjectionKeys, [
    "domain", "compilationContractHash", "ownerPassId", "diagnostic",
  ], "$normalization.body.stableDiagnosticId.canonicalProjectionKeys");
  exact(stable.serializedDiagnosticFields, [
    "phase", "severity", "code", "renderedMessage", "location", "subject", "parentDiagnosticId",
  ], "$normalization.body.stableDiagnosticId.serializedDiagnosticFields");
  exact(stable.duplicatePolicy, "collapse-exact-canonical-projection-within-owner-pass", "$normalization.body.stableDiagnosticId.duplicatePolicy");

  const location = record(body.sourceLocationPolicy, "$normalization.body.sourceLocationPolicy");
  const text = record(body.renderedTextPolicy, "$normalization.body.renderedTextPolicy");
  const notes = record(body.notePolicy, "$normalization.body.notePolicy");
  const fixIts = record(body.fixItPolicy, "$normalization.body.fixItPolicy");
  const diagnosticSet = record(body.diagnosticSetPolicy, "$normalization.body.diagnosticSetPolicy");
  exact(notes.parentPolicy, "direct-child-of-emitted-root-only", "$normalization.body.notePolicy.parentPolicy");
  exact(notes.normalizationOrder, "root-before-direct-note-children", "$normalization.body.notePolicy.normalizationOrder");
  exact(fixIts.artifactV3Disposition, "bounded-validate-then-omit", "$normalization.body.fixItPolicy.artifactV3Disposition");
  exact(diagnosticSet.duplicateAccounting, "exact-duplicates-consume-no-additional-count-or-note-budget", "$normalization.body.diagnosticSetPolicy.duplicateAccounting");
  exact(diagnosticSet.terminalFailure, "invalid-or-resource-limit-poisons-normalizer", "$normalization.body.diagnosticSetPolicy.terminalFailure");

  const stageEntries = array(body.stageMappings, "$normalization.body.stageMappings");
  if (stageEntries.length !== STAGES.size) throw new TypeError("stage registry length drifted");
  const stageRecords = stageEntries.map((value, index) => {
    const path = `$normalization.body.stageMappings[${index}]`;
    const entry = record(value, path);
    const stage = ascii(entry.producerStage, `${path}.producerStage`);
    const mapping = STAGES.get(stage);
    if (mapping === undefined || entry.artifactPhase !== stagePhase(stage)) {
      throw new TypeError(`${path} is not an exact stage mapping`);
    }
    return `    {RawDiagnosticStage::${mapping[0]}, DiagnosticPhase::${mapping[1]}, DiagnosticCategory::${mapping[2]}},`;
  }).join("\n");

  const severityEntries = array(body.severityMappings, "$normalization.body.severityMappings");
  if (severityEntries.length !== SEVERITIES.size) throw new TypeError("severity registry length drifted");
  const severityRecords = severityEntries.map((value, index) => {
    const path = `$normalization.body.severityMappings[${index}]`;
    const entry = record(value, path);
    const level = ascii(entry.clangLevel, `${path}.clangLevel`);
    const mapping = SEVERITIES.get(level);
    if (mapping === undefined) throw new TypeError(`${path} has an unknown Clang level`);
    const emit = entry.disposition === "emit";
    if (!emit && entry.disposition !== "omit-before-normalization") {
      throw new TypeError(`${path}.disposition is not closed`);
    }
    exact(entry.artifactSeverity, level === "ignored" ? null : level, `${path}.artifactSeverity`);
    return `    {RawDiagnosticSeverity::${mapping[0]}, DiagnosticSeverity::${mapping[1]}, ${emit}, ${entry.blocking === true}, ${entry.parentRequired === true}},`;
  }).join("\n");

  const customEntries = array(body.customMappings, "$normalization.body.customMappings");
  if (customEntries.length !== CUSTOM_CODES.size) throw new TypeError("custom diagnostic registry length drifted");
  const customRecords = customEntries.map((value, index) => {
    const path = `$normalization.body.customMappings[${index}]`;
    const entry = record(value, path);
    const code = ascii(entry.customCode, `${path}.customCode`);
    const custom = CUSTOM_CODES.get(code);
    const stage = STAGES.get(ascii(entry.producerStage, `${path}.producerStage`));
    const severity = SEVERITIES.get(ascii(entry.artifactSeverity, `${path}.artifactSeverity`));
    const category = CATEGORIES.get(ascii(entry.category, `${path}.category`));
    if (custom === undefined || stage === undefined || severity === undefined || category === undefined || entry.blocking !== true) {
      throw new TypeError(`${path} is not an exact custom diagnostic mapping`);
    }
    return `    {CustomDiagnosticCode::${custom}, RawDiagnosticStage::${stage[0]}, DiagnosticSeverity::${severity[1]}, DiagnosticCategory::${category}, ${cppString(code)}},`;
  }).join("\n");

  return [
    "// Generated by cpp_cute_browser_diagnostics_policy_codegen.mjs.",
    "// Source: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE. DO NOT EDIT.",
    "",
    `constexpr std::string_view kDiagnosticNormalizationManifestId = ${cppString(ascii(root.manifestId, "$normalization.manifestId"))};`,
    `constexpr std::string_view kDiagnosticStableIdDomain = ${cppString(ascii(stable.domain, "$normalization.body.stableDiagnosticId.domain"))};`,
    `constexpr std::string_view kDiagnosticStableIdPrefix = ${cppString(ascii(stable.outputPrefix, "$normalization.body.stableDiagnosticId.outputPrefix"))};`,
    "constexpr std::string_view kClangDiagnosticCodePrefix = \"clang:diag-\";",
    `constexpr std::uint32_t kMaximumRenderedMessageBytes = ${uint32(text.maxRenderedMessageBytes, "$normalization.body.renderedTextPolicy.maxRenderedMessageBytes")}U;`,
    `constexpr std::uint32_t kMaximumRelatedLocationCount = ${uint32(location.maxRelatedLocations, "$normalization.body.sourceLocationPolicy.maxRelatedLocations")}U;`,
    `constexpr std::uint32_t kMaximumRelatedMessageBytes = ${uint32(location.maxRenderedBytesPerRelatedLocation, "$normalization.body.sourceLocationPolicy.maxRenderedBytesPerRelatedLocation")}U;`,
    `constexpr std::uint32_t kMaximumAggregateRelatedMessageBytes = ${uint32(location.maxAggregateRenderedRelatedLocationBytes, "$normalization.body.sourceLocationPolicy.maxAggregateRenderedRelatedLocationBytes")}U;`,
    `constexpr std::uint32_t kMaximumNotesPerRoot = ${uint32(notes.maxNotesPerPrimaryDiagnostic, "$normalization.body.notePolicy.maxNotesPerPrimaryDiagnostic")}U;`,
    `constexpr std::uint32_t kMaximumRenderedNoteBytes = ${uint32(notes.maxRenderedBytesPerNote, "$normalization.body.notePolicy.maxRenderedBytesPerNote")}U;`,
    `constexpr std::uint32_t kMaximumAggregateNoteBytes = ${uint32(notes.maxAggregateRenderedNoteBytesPerPrimaryDiagnostic, "$normalization.body.notePolicy.maxAggregateRenderedNoteBytesPerPrimaryDiagnostic")}U;`,
    `constexpr std::uint32_t kMaximumFixItsPerDiagnostic = ${uint32(fixIts.maxFixItsPerDiagnostic, "$normalization.body.fixItPolicy.maxFixItsPerDiagnostic")}U;`,
    `constexpr std::uint32_t kMaximumReplacementBytesPerFixIt = ${uint32(fixIts.maxReplacementBytesPerFixIt, "$normalization.body.fixItPolicy.maxReplacementBytesPerFixIt")}U;`,
    `constexpr std::uint32_t kMaximumAggregateReplacementBytes = ${uint32(fixIts.maxAggregateReplacementBytesPerDiagnostic, "$normalization.body.fixItPolicy.maxAggregateReplacementBytesPerDiagnostic")}U;`,
    `constexpr std::uint32_t kMaximumUniqueDiagnostics = ${uint32(diagnosticSet.maxUniqueDiagnosticsPerSession, "$normalization.body.diagnosticSetPolicy.maxUniqueDiagnosticsPerSession")}U;`,
    `constexpr std::uint32_t kMaximumRetainedNormalizedBytes = ${uint32(diagnosticSet.maxRetainedNormalizedBytesPerSession, "$normalization.body.diagnosticSetPolicy.maxRetainedNormalizedBytesPerSession")}U;`,
    "",
    `constexpr std::array<GeneratedStagePolicy, ${stageEntries.length}U> kGeneratedStagePolicies = {{`,
    stageRecords,
    "}};",
    "",
    `constexpr std::array<GeneratedSeverityPolicy, ${severityEntries.length}U> kGeneratedSeverityPolicies = {{`,
    severityRecords,
    "}};",
    "",
    `constexpr std::array<GeneratedCustomPolicy, ${customEntries.length}U> kGeneratedCustomPolicies = {{`,
    customRecords,
    "}};",
    "",
  ].join("\n");
}

/** @param {string} stage */
function stagePhase(stage) {
  return ({
    preprocessor: "preprocessing",
    parser: "parsing",
    "sema-name-lookup": "name-lookup",
    "sema-overload-resolution": "overload-resolution",
    "sema-template-instantiation": "template-instantiation",
    "sema-cuda": "cuda-sema",
    "artifact-extractor": "artifact-extraction",
  })[stage];
}

/** @param {unknown} normalization @param {Uint8Array} actualBytes */
export function cppCuteDiagnosticsPolicyIncludeMatches(normalization, actualBytes) {
  const expected = Buffer.from(renderCppCuteDiagnosticsPolicyInclude(normalization), "utf8");
  return expected.length === actualBytes.length && expected.equals(actualBytes);
}

async function main() {
  const mode = process.argv[2] ?? "--check";
  if (mode !== "--check" && mode !== "--write") {
    throw new TypeError("usage: node cpp_cute_browser_diagnostics_policy_codegen.mjs [--check|--write]");
  }
  let resourceModule;
  try {
    resourceModule = await import("../../dist/resources/cpp_cute_diagnostic_normalization_v1.js");
  } catch (error) {
    throw new Error("build compiler package before running diagnostics policy codegen", { cause: error });
  }
  const resource = resourceModule.CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE;
  const expected = renderCppCuteDiagnosticsPolicyInclude(resource);
  if (mode === "--write") {
    writeFileSync(CPP_CUTE_DIAGNOSTICS_POLICY_INCLUDE_PATH, expected, "utf8");
    return;
  }
  const actual = readFileSync(CPP_CUTE_DIAGNOSTICS_POLICY_INCLUDE_PATH);
  if (!cppCuteDiagnosticsPolicyIncludeMatches(resource, actual)) {
    throw new Error("generated native diagnostics policy differs from normalization resource");
  }
}

const invokedPath = process.argv[1] === undefined ? null : pathToFileURL(resolve(process.argv[1])).href;
if (invokedPath === import.meta.url) await main();
