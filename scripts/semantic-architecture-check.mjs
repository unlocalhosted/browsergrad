#!/usr/bin/env node

import fs from "node:fs";
import { createHash } from "node:crypto";
import { builtinModules, createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const defaultRepoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function option(name) {
  const value = process.argv.find((arg) => arg.startsWith(`${name}=`));
  return value?.slice(name.length + 1);
}

const cliRepoRoot = option("--repo-root");
const repoRoot = cliRepoRoot === undefined ? defaultRepoRoot : path.resolve(cliRepoRoot);
const jsonOutput = process.argv.includes("--json");
const NODE_BUILTINS = new Set(builtinModules.map((specifier) => specifier.replace(/^node:/u, "")));
const REQUIRED_FREEZES = new Map([
  ["compiler.pointer-scalar-memory.v0", "compiler-pointer-scalar-memory"],
  ["compiler.cute-static-layout.v0", "cute-static-layout"],
  ["compiler.cute-source-normalizers.v0", "cute-source-normalizers"],
  ["grad.view-bf16-compat.v0", "grad-view-bf16"],
  ["kernels.tensor-gpu-plan.v0", "tensor-gpu-plan"],
  ["jit.core-custom-ops.v0", "jit-op-custom"],
  ["runtime.generic-backend-labels.v0", "runtime-assignment-requirements"],
]);

export function runSemanticArchitectureCheck(root = repoRoot) {
  const failures = [];
  const manifestPath = path.join(root, "architecture/semantic-freeze.json");
  const manifest = readJson(manifestPath, failures);
  if (manifest === undefined) return failures;

  const compilerPackage = path.join(root, "packages/browsergrad-compiler/package.json");
  const compilerRequire = createRequire(compilerPackage);
  let ts;
  try {
    ts = compilerRequire("typescript");
  } catch (error) {
    failures.push(`cannot load TypeScript through ${relative(root, compilerPackage)}: ${errorMessage(error)}`);
    return failures;
  }

  validateManifest(root, manifest, failures);
  checkSharedSemanticFixtureContracts(root, failures);
  checkWorkspaceDependencies(root, failures);
  checkWorkspaceImports(root, ts, failures);
  checkGeneratedPython(root, failures);
  checkJitFrameworkOperationContracts(root, manifest, failures);

  const adapters = Array.isArray(manifest.adapters) ? manifest.adapters : [];
  for (const adapter of adapters) {
    if (!isRecord(adapter) || !isRecord(adapter.freeze)) continue;
    switch (adapter.freeze.kind) {
      case "compiler-pointer-scalar-memory":
        checkCompilerPointerScalarMemory(root, ts, adapter.freeze, failures);
        break;
      case "cute-static-layout":
        checkCuteStaticLayout(root, ts, adapter.freeze, failures);
        break;
      case "cute-source-normalizers":
        checkCuteSourceNormalizers(root, adapter.freeze, failures);
        break;
      case "grad-view-bf16":
        checkGradViewBf16(root, adapter.freeze, failures);
        break;
      case "tensor-gpu-plan":
        checkTensorGpuPlan(root, ts, adapter.freeze, failures);
        break;
      case "jit-op-custom":
        checkJitCustomOps(root, ts, adapter.freeze, failures);
        break;
      case "runtime-assignment-requirements":
        checkRuntimeAssignmentRequirements(root, ts, adapter.freeze, failures);
        break;
      default:
        failures.push(`adapter ${stringValue(adapter.id)} has unknown freeze kind ${stringValue(adapter.freeze.kind)}`);
    }
  }

  return failures;
}

export function checkWorkspaceImportSpecifier(packageName, file, specifier) {
  const failures = [];
  if (/^@unlocalhosted\/browsergrad-[^/]+\/(?:src|dist)(?:\/|$)/u.test(specifier)) {
    failures.push(`${file} deep-imports implementation path ${specifier}`);
  }
  if (specifier === "@unlocalhosted/browsergrad-semantic-core") {
    failures.push(`${file} imports the forbidden semantic-core root; use a declared narrow subpath`);
  }
  if (packageName === "@unlocalhosted/browsergrad-kernels" && specifier.startsWith("@unlocalhosted/browsergrad-compiler")) {
    failures.push(`${file} imports compiler from kernels`);
  }
  if (
    packageName === "@unlocalhosted/browsergrad-kernels" &&
    specifier.startsWith("@unlocalhosted/browsergrad-semantic-core") &&
    !new Set([
      "@unlocalhosted/browsergrad-semantic-core/schema",
      "@unlocalhosted/browsergrad-semantic-core/layout",
      "@unlocalhosted/browsergrad-semantic-core/kernel",
      "@unlocalhosted/browsergrad-semantic-core/schedule",
    ]).has(specifier)
  ) {
    failures.push(`${file} imports ${specifier}; kernels may import semantic-core schema/layout/kernel/schedule protocols only`);
  }
  if (packageName === "@unlocalhosted/browsergrad-compiler" && /^@unlocalhosted\/browsergrad-(?:jit|grad)(?:\/|$)/u.test(specifier)) {
    failures.push(`${file} imports framework internals from compiler`);
  }
  if (
    packageName === "@unlocalhosted/browsergrad-compiler"
    && file.startsWith("packages/browsergrad-compiler/src/")
    && isNodeBuiltinSpecifier(specifier)
  ) {
    failures.push(`${file} imports Node built-in ${specifier}; compiler production source must remain browser-safe`);
  }
  if (packageName === "@unlocalhosted/browsergrad-jit" && specifier.startsWith("@unlocalhosted/browsergrad-compiler")) {
    failures.push(`${file} imports compiler from JIT`);
  }
  if (packageName === "@unlocalhosted/browsergrad-semantic-core" && specifier.startsWith("@unlocalhosted/browsergrad-")) {
    failures.push(`${file} imports BrowserGrad package ${specifier} from semantic-core`);
  }
  if (
    packageName === "@unlocalhosted/browsergrad-runtime" &&
    specifier.startsWith("@unlocalhosted/browsergrad-semantic-core") &&
    !new Set([
      "@unlocalhosted/browsergrad-semantic-core/capability",
      "@unlocalhosted/browsergrad-semantic-core/diagnostic",
      "@unlocalhosted/browsergrad-semantic-core/requirement",
    ]).has(specifier)
  ) {
    failures.push(`${file} imports ${specifier}; runtime may import semantic-core diagnostic/capability/requirement protocols only`);
  }
  return failures;
}

function isNodeBuiltinSpecifier(specifier) {
  const normalized = specifier.replace(/^node:/u, "");
  const root = normalized.split("/", 1)[0];
  return NODE_BUILTINS.has(normalized) || (root !== undefined && NODE_BUILTINS.has(root));
}

export function countPythonCustomConstructors(source) {
  return pythonCallFacts(source).customConstructors;
}

export function extractPythonCustomLabels(source) {
  return [...pythonCallFacts(source).labels].sort();
}

export function extractPythonCustomLabelFields(source) {
  const facts = pythonCallFacts(source);
  return {
    name: [...facts.labelFields.name].sort(),
    op: [...facts.labelFields.op].sort(),
    dynamicName: [...facts.dynamicLabelFields.name].sort(),
    dynamicOp: [...facts.dynamicLabelFields.op].sort(),
  };
}

export function validateSemanticFreezeManifest(root, manifest) {
  const failures = [];
  validateManifest(path.resolve(root), manifest, failures);
  return failures;
}

export function checkFrozenCuteSourceNormalizerFiles(
  sources,
  freeze,
) {
  const failures = [];
  const expectedFiles = Object.keys(freeze.files ?? {}).sort();
  const actualFiles = Object.keys(sources ?? {}).sort();
  compareStringSets("CuTe source-normalizer files", actualFiles, expectedFiles, failures);
  for (const file of expectedFiles) {
    const source = sources?.[file];
    if (typeof source !== "string") continue;
    const actual = createHash("sha256").update(source, "utf8").digest("hex");
    const expected = freeze.files?.[file];
    if (actual !== expected) {
      failures.push(`${file} CuTe source normalizer changed; expected SHA-256 ${stringValue(expected)}, got ${actual}`);
    }
  }
  return failures;
}

export function validateSharedSemanticFixtureContracts(root, manifest) {
  const failures = [];
  validateSharedSemanticFixtureManifest(path.resolve(root), manifest, failures);
  return failures;
}

export function extractModuleSpecifiers(ts, source, filename = "source.ts") {
  const sourceFile = ts.createSourceFile(filename, source, ts.ScriptTarget.Latest, true);
  return moduleSpecifiers(ts, sourceFile);
}

export function buildAssignmentRequirementUsage(
  root,
  profileDirectory = "docs/internal",
  profileSuffix = ".profile.json",
) {
  const failures = [];
  const directory = path.resolve(root, profileDirectory);
  const usage = assignmentRequirementUsage(directory, profileSuffix, failures, path.resolve(root));
  if (failures.length > 0) throw new Error(failures.join("\n"));
  return usage;
}

export function checkFrozenCompilerPointerScalarMemorySource(
  ts,
  source,
  publicBarrelSource,
  freeze,
  filename = "semantic_ir_types.ts",
  publicBarrelFilename = "index.ts",
) {
  const failures = [];
  const sourceFile = ts.createSourceFile(filename, source, ts.ScriptTarget.Latest, true);
  const declarations = new Map();
  const addressSpaces = [];

  for (const statement of sourceFile.statements) {
    if (statement.name?.text) declarations.set(statement.name.text, [...(declarations.get(statement.name.text) ?? []), statement]);
    if (ts.isTypeAliasDeclaration(statement) && statement.name.text === "SemanticAddressSpace") {
      if (!ts.isUnionTypeNode(statement.type) || statement.type.types.some((type) => !ts.isLiteralTypeNode(type) || !ts.isStringLiteral(type.literal))) {
        failures.push("SemanticAddressSpace must remain a closed string-literal union");
      } else {
        addressSpaces.push(...statement.type.types.map((type) => type.literal.text));
      }
    }
  }

  compareStringSets("SemanticAddressSpace values", addressSpaces, freeze.addressSpaces, failures);
  checkExactInterfaceShapes(ts, sourceFile, declarations, freeze.interfaces, "compiler pointer/scalar interface", failures);
  checkExactTaggedUnionShapes(ts, sourceFile, declarations, "SemanticExpression", freeze.expressionVariants, "compiler pointer/scalar expression", failures);
  checkExactTaggedUnionShapes(ts, sourceFile, declarations, "SemanticKernelIrOperation", freeze.operationVariants, "compiler pointer/scalar operation", failures);

  const publicBarrel = ts.createSourceFile(publicBarrelFilename, publicBarrelSource, ts.ScriptTarget.Latest, true);
  const publicExports = new Set();
  for (const statement of publicBarrel.statements) {
    if (!ts.isExportDeclaration(statement) || statement.exportClause === undefined || !ts.isNamedExports(statement.exportClause)) continue;
    for (const element of statement.exportClause.elements) publicExports.add(element.name.text);
  }
  for (const name of freeze.publicExports ?? []) {
    if (!publicExports.has(name)) failures.push(`compiler pointer/scalar public export ${name} is missing`);
  }

  return failures;
}

export function validateCompilerPointerBehaviorFixture(fixture, freeze, filename = "pointer-scalar-memory.v0.json") {
  const failures = [];
  if (!isRecord(fixture) || fixture.schemaVersion !== 1 || fixture.adapterId !== "compiler.pointer-scalar-memory.v0" || !Array.isArray(fixture.cases)) {
    failures.push(`${filename} must be a schemaVersion 1 compiler.pointer-scalar-memory.v0 fixture`);
    return failures;
  }
  const ids = fixture.cases.map((entry) => isRecord(entry) ? entry.id : undefined);
  if (ids.some((id) => typeof id !== "string") || new Set(ids).size !== ids.length) {
    failures.push(`${filename} case IDs must be unique strings`);
    return failures;
  }
  compareStringSets("compiler pointer/scalar behavior fixture IDs", ids, freeze.behaviorFixtureIds, failures);
  return failures;
}

export function checkFrozenRuntimeAssignmentRequirementsSource(
  ts,
  capabilitySource,
  typesSource,
  freeze,
  capabilityFilename = "assignment-capabilities.ts",
  typesFilename = "assignment-types.ts",
) {
  const failures = [];
  const capabilityFile = ts.createSourceFile(capabilityFilename, capabilitySource, ts.ScriptTarget.Latest, true);
  const typesFile = ts.createSourceFile(typesFilename, typesSource, ts.ScriptTarget.Latest, true);
  const inputs = capabilityFile.statements.filter((statement) => ts.isInterfaceDeclaration(statement) && statement.name.text === "BrowserGpuCapabilityInput");
  const input = inputs[0];
  if (!input || !ts.isInterfaceDeclaration(input) || inputs.length !== 1) {
    failures.push(`BrowserGpuCapabilityInput must have exactly one interface declaration; got ${inputs.length}`);
  } else {
    const fields = [];
    for (const member of input.members) {
      if (!ts.isPropertySignature(member) || member.type === undefined) {
        failures.push("BrowserGpuCapabilityInput may contain property signatures only");
        continue;
      }
      const name = member.name.getText(capabilityFile);
      fields.push(name);
      if (member.questionToken === undefined || member.type.kind !== ts.SyntaxKind.BooleanKeyword) {
        failures.push(`BrowserGpuCapabilityInput.${name} must remain optional boolean`);
      }
      if (member.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ReadonlyKeyword) !== true) {
        failures.push(`BrowserGpuCapabilityInput.${name} must remain readonly`);
      }
    }
    compareStringSets("BrowserGpuCapabilityInput fields", fields, freeze.inputFields, failures);
  }

  const functionDeclarations = capabilityFile.statements.filter((statement) => ts.isFunctionDeclaration(statement) && statement.name?.text === "browserGpuCapabilities");
  if (functionDeclarations.length !== 1 || functionDeclarations[0].body === undefined) {
    failures.push(`browserGpuCapabilities must have exactly one function declaration; got ${functionDeclarations.length}`);
  } else {
    const mappings = [];
    for (const statement of functionDeclarations[0].body.statements) {
      if (!ts.isIfStatement(statement)) continue;
      const requirementId = pushedStringLiteral(ts, statement.thenStatement);
      if (requirementId === undefined) continue;
      const condition = normalizeTypeText(statement.expression.getText(capabilityFile));
      const fields = [...condition.matchAll(/\binput\.([A-Za-z][A-Za-z0-9]*)\b/gu)].map((match) => match[1]);
      const field = [...fields].reverse().find((name) => name !== "webgpu") ?? fields[0];
      mappings.push({ field, requirementId, condition });
    }
    let pushCalls = 0;
    const visit = (node) => {
      if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) && node.expression.expression.getText(capabilityFile) === "capabilities" && node.expression.name.text === "push") pushCalls += 1;
      ts.forEachChild(node, visit);
    };
    visit(functionDeclarations[0].body);
    if (pushCalls !== mappings.length) failures.push(`browserGpuCapabilities contains ${pushCalls} push calls but only ${mappings.length} frozen conditional mappings`);
    compareRecordLists("browserGpuCapabilities mappings", mappings, freeze.browserMappings, "requirementId", failures);
  }

  checkClosedStringUnion(ts, typesFile, "AssignmentCapabilityMode", freeze.assignmentModes, failures);
  checkClosedStringUnion(ts, typesFile, "AssignmentRunReadinessStatus", freeze.readinessStatuses, failures);
  checkClosedStringUnion(ts, typesFile, "AssignmentRunnerTarget", freeze.runnerTargets, failures);
  return failures;
}

export function validatePlatformVocabularySnapshot(root, vocabulary, profileIds, browserMappings) {
  const failures = [];
  validatePlatformVocabulary(path.resolve(root), vocabulary, profileIds, browserMappings, failures);
  return failures;
}

export function extractPythonDefinitionTokenDigests(source) {
  const definitions = pythonDefinitions(source);
  const digests = {};
  for (const definition of definitions) {
    const tokens = normalizedPythonDefinitionTokens(definition.source);
    const encoded = JSON.stringify(tokens.map((token) => [token.kind, token.value]));
    const digest = createHash("sha256").update(encoded, "utf8").digest("hex");
    (digests[definition.qualifiedName] ??= []).push(digest);
  }
  return sortRecord(digests);
}

export function checkFrozenGradCompatibilitySources(tensorSource, torchCompatSource, freeze) {
  const failures = [];
  const actualDefinitions = {
    "tensor.py": extractPythonDefinitionTokenDigests(tensorSource),
    "_torch_compat_real.py": extractPythonDefinitionTokenDigests(torchCompatSource),
  };
  for (const [file, expectedDefinitions] of Object.entries(freeze.definitionTokenDigests ?? {})) {
    const fileDefinitions = actualDefinitions[file];
    if (!isRecord(fileDefinitions)) {
      failures.push(`Grad compatibility freeze names unsupported source ${file}`);
      continue;
    }
    for (const [qualifiedName, expectedDigests] of Object.entries(expectedDefinitions ?? {})) {
      const actualDigests = fileDefinitions[qualifiedName];
      if (JSON.stringify(actualDigests) !== JSON.stringify(expectedDigests)) {
        failures.push(`Grad compatibility definition ${file}:${qualifiedName} changed; expected ${JSON.stringify(expectedDigests)}, got ${JSON.stringify(actualDigests)}`);
      }
    }
  }

  const torchTokens = pythonAttributeStringAssignments(torchCompatSource, "torch_mod");
  if (JSON.stringify(sortRecord(torchTokens)) !== JSON.stringify(sortRecord(freeze.torchDtypeTokens ?? {}))) {
    failures.push(`Grad torch dtype tokens changed; expected ${JSON.stringify(sortRecord(freeze.torchDtypeTokens ?? {}))}, got ${JSON.stringify(sortRecord(torchTokens))}`);
  }
  return failures;
}

export function validateGradCompatibilityInventory(inventory, fixture, freeze, filename = "grad-compatibility-inventory.json") {
  const failures = [];
  const exactKeys = (label, value, expected) => {
    if (!isRecord(value)) {
      failures.push(`${label} must be an object`);
      return;
    }
    compareStringSets(`${label} keys`, Object.keys(value), expected, failures);
  };

  exactKeys(filename, inventory, ["schemaVersion", "inventoryId", "owner", "evidenceTier", "environment", "executionContext", "dtypeResolution", "behaviors"]);
  if (!isRecord(inventory) || inventory.schemaVersion !== 2 || inventory.inventoryId !== "browsergrad.grad.compatibility.v1" || inventory.owner !== "@unlocalhosted/browsergrad-grad" || inventory.evidenceTier !== "pyodide-numpy-reference") {
    failures.push(`${filename} must identify the schemaVersion 2 BrowserGrad Grad Pyodide/NumPy compatibility inventory`);
    return failures;
  }
  exactKeys(`${filename}.environment`, inventory.environment, ["pyodide", "numpy"]);
  exactKeys(`${filename}.executionContext`, inventory.executionContext, ["residency", "backendDecision", "transformDecision", "exportDecision"]);
  if (!isRecord(inventory.executionContext) || inventory.executionContext.residency !== "pyodide-wasm-linear-memory" || inventory.executionContext.backendDecision !== "numpy-reference-only" || inventory.executionContext.transformDecision !== "not-applicable-eager" || inventory.executionContext.exportDecision !== "not-applicable-eager") {
    failures.push(`${filename}.executionContext changed`);
  }
  exactKeys(`${filename}.dtypeResolution`, inventory.dtypeResolution, ["defaultTensorStorageDtype", "defaultTorchIntegerStorageDtype", "unknownStringPolicy", "nonStringPolicy", "storageByteWidths", "unsupportedDtypes", "aliases", "torchTokens"]);
  if (!isRecord(inventory.dtypeResolution)) return failures;
  if (inventory.dtypeResolution.defaultTensorStorageDtype !== "float32" || inventory.dtypeResolution.defaultTorchIntegerStorageDtype !== "int64" || inventory.dtypeResolution.unknownStringPolicy !== "delegate-to-numpy-dtype" || inventory.dtypeResolution.nonStringPolicy !== "delegate-to-numpy-dtype") {
    failures.push(`${filename}.dtypeResolution policies changed`);
  }
  const expectedStorageByteWidths = { bool: 1, float16: 2, float32: 4, float64: 8, int8: 1, int16: 2, int32: 4, int64: 8, uint8: 1 };
  const expectedUnsupportedDtypes = {
    bf16: "rejected-no-real-bfloat16-storage-or-conversion",
    bfloat16: "rejected-no-real-bfloat16-storage-or-conversion",
  };
  if (JSON.stringify(sortRecord(inventory.dtypeResolution.storageByteWidths ?? {})) !== JSON.stringify(sortRecord(expectedStorageByteWidths))) failures.push(`${filename}.dtypeResolution.storageByteWidths changed`);
  if (JSON.stringify(sortRecord(inventory.dtypeResolution.unsupportedDtypes ?? {})) !== JSON.stringify(sortRecord(expectedUnsupportedDtypes))) failures.push(`${filename}.dtypeResolution.unsupportedDtypes changed`);
  for (const field of ["aliases", "torchTokens"]) {
    const values = inventory.dtypeResolution[field];
    if (!isRecord(values) || Object.keys(values).length === 0 || Object.values(values).some((value) => typeof value !== "string" || value.length === 0)) {
      failures.push(`${filename}.dtypeResolution.${field} must be a non-empty string map`);
    }
  }
  if (JSON.stringify(sortRecord(inventory.dtypeResolution.aliases ?? {})) !== JSON.stringify(sortRecord(freeze.dtypeAliases ?? {}))) {
    failures.push(`${filename}.dtypeResolution.aliases differs from the frozen source alias map`);
  }
  if (JSON.stringify(sortRecord(inventory.dtypeResolution.torchTokens ?? {})) !== JSON.stringify(sortRecord(freeze.torchDtypeTokens ?? {}))) {
    failures.push(`${filename}.dtypeResolution.torchTokens differs from the frozen source assignment map`);
  }

  if (!Array.isArray(inventory.behaviors) || inventory.behaviors.length === 0) {
    failures.push(`${filename}.behaviors must be a non-empty array`);
    return failures;
  }
  const behaviorKeys = ["id", "surface", "class", "observationStatus", "targetConformance", "referenceContract", "dtypeEffect", "condition", "failurePolicy", "aliasing", "contiguity", "materialization", "autograd", "truth", "sourceDefinitions", "fixtureCaseIds"];
  const classes = new Set(["conversion", "dtype-refusal", "dtype-resolution", "interop", "materialization", "view"]);
  const observationStatuses = new Set(["verified"]);
  const targetConformance = new Set(["browsergrad-defined", "compatibility-debt", "pytorch-compatible"]);
  const referenceContracts = new Set(["browsergrad-grad-explicit", "numpy-array-protocol", "pytorch-shaped-compatibility"]);
  const dtypeEffects = new Set(["distinct-unsupported-token", "none-rejected", "preserves", "preserves-float32-otherwise-coerces-float32", "resolved-target", "surface-dependent"]);
  const conditions = new Set(["all-inputs", "index-kind-dependent", "input-dtype-and-index-kind-dependent", "input-dtype-and-layout-dependent", "input-dtype-dependent", "input-layout-dependent", "no-requested-dtype", "requested-bf16-token", "requested-dtype-dependent", "requested-dtype-differs-from-storage-dtype", "requested-dtype-equals-storage-dtype", "surface-and-input-dependent", "torch-bfloat16-token", "unrecognized-string"]);
  const failurePolicies = new Set(["delegate-invalid-broadcast-to-numpy", "delegate-invalid-dimension-to-numpy", "delegate-invalid-index-to-numpy", "delegate-invalid-input-to-python-or-numpy", "delegate-invalid-permutation-to-numpy", "delegate-invalid-shape-to-numpy", "no-dedicated-failure-path", "reject-invalid-dtype-after-numpy-delegation", "reject-invalid-expand-shape-before-execution", "reject-unsupported-bfloat16-before-allocation", "unrecognized-dtype-treated-as-device-noop"]);
  const aliasing = new Set(["conditional", "must-alias", "must-not-alias", "not-applicable", "same-object"]);
  const contiguity = new Set(["contiguous", "input-dependent", "not-applicable", "preserved"]);
  const materialization = new Set(["always", "conditional", "none", "not-applicable"]);
  const autograd = new Set(["constructor-policy", "detached", "graph-edge", "identity-or-graph-edge", "not-applicable", "preserved"]);
  const behaviorIds = [];
  const referencedFixtureIds = [];
  const referencedSourceDefinitions = [];
  for (const [index, behavior] of inventory.behaviors.entries()) {
    const label = `${filename}.behaviors[${index}]`;
    exactKeys(label, behavior, behaviorKeys);
    if (!isRecord(behavior)) continue;
    for (const field of ["id", "surface", "truth"]) {
      if (typeof behavior[field] !== "string" || behavior[field].trim() === "") failures.push(`${label}.${field} must be a non-empty string`);
    }
    if (!classes.has(behavior.class)) failures.push(`${label}.class is not registered`);
    if (!observationStatuses.has(behavior.observationStatus)) failures.push(`${label}.observationStatus is not registered`);
    if (!targetConformance.has(behavior.targetConformance)) failures.push(`${label}.targetConformance is not registered`);
    if (!referenceContracts.has(behavior.referenceContract)) failures.push(`${label}.referenceContract is not registered`);
    if (!dtypeEffects.has(behavior.dtypeEffect)) failures.push(`${label}.dtypeEffect is not registered`);
    if (!conditions.has(behavior.condition)) failures.push(`${label}.condition is not registered`);
    if (!failurePolicies.has(behavior.failurePolicy)) failures.push(`${label}.failurePolicy is not registered`);
    if (!aliasing.has(behavior.aliasing)) failures.push(`${label}.aliasing is not registered`);
    if (!contiguity.has(behavior.contiguity)) failures.push(`${label}.contiguity is not registered`);
    if (!materialization.has(behavior.materialization)) failures.push(`${label}.materialization is not registered`);
    if (!autograd.has(behavior.autograd)) failures.push(`${label}.autograd is not registered`);
    if (!Array.isArray(behavior.sourceDefinitions) || behavior.sourceDefinitions.length === 0 || behavior.sourceDefinitions.some((entry) => typeof entry !== "string" || !/^(?:tensor|_torch_compat_real)\.py:[A-Za-z_][A-Za-z0-9_.]*$/u.test(entry))) {
      failures.push(`${label}.sourceDefinitions must name Grad Python definitions`);
    } else {
      referencedSourceDefinitions.push(...behavior.sourceDefinitions);
    }
    if (!Array.isArray(behavior.fixtureCaseIds) || behavior.fixtureCaseIds.length === 0 || behavior.fixtureCaseIds.some((entry) => typeof entry !== "string")) {
      failures.push(`${label}.fixtureCaseIds must contain strings`);
    } else {
      referencedFixtureIds.push(...behavior.fixtureCaseIds);
    }
    if (typeof behavior.id === "string") behaviorIds.push(behavior.id);
  }
  if (new Set(behaviorIds).size !== behaviorIds.length) failures.push(`${filename} behavior IDs must be unique`);
  compareStringSets("Grad compatibility behavior IDs", behaviorIds, freeze.behaviorIds, failures);
  const frozenSourceDefinitions = Object.entries(freeze.definitionTokenDigests ?? {}).flatMap(([file, definitions]) =>
    Object.keys(definitions ?? {}).map((qualifiedName) => `${file}:${qualifiedName}`));
  const frozenTorchTokens = Object.keys(freeze.torchDtypeTokens ?? {}).map((name) => `_torch_compat_real.py:torch_mod.${name}`);
  const knownSourceDefinitions = new Set([...frozenSourceDefinitions, ...frozenTorchTokens]);
  for (const reference of new Set(referencedSourceDefinitions)) {
    if (!knownSourceDefinitions.has(reference)) failures.push(`Grad compatibility inventory references unfrozen source definition ${reference}`);
  }
  compareStringSets(
    "Grad compatibility source definition references",
    [...new Set(referencedSourceDefinitions.filter((reference) => !reference.includes(":torch_mod.")))],
    frozenSourceDefinitions,
    failures,
  );

  validateNamedFixture(fixture, "grad.view-bf16-compat.v0", freeze.behaviorFixtureIds, stringValue(freeze.behaviorFixtureFile), "Grad compatibility", failures);
  const fixtureIds = isRecord(fixture) && Array.isArray(fixture.cases)
    ? fixture.cases.filter(isRecord).map((entry) => entry.id).filter((id) => typeof id === "string")
    : [];
  compareStringSets("Grad compatibility inventory fixture references", referencedFixtureIds, fixtureIds, failures);
  if (isRecord(fixture)) {
    exactKeys("Grad compatibility fixture", fixture, ["schemaVersion", "adapterId", "environment", "cases"]);
    if (JSON.stringify(fixture.environment) !== JSON.stringify(inventory.environment)) failures.push("Grad compatibility fixture environment differs from inventory environment");
    const fixtureCases = Array.isArray(fixture.cases) ? fixture.cases : [];
    for (const [index, testCase] of fixtureCases.entries()) {
      exactKeys(`Grad compatibility fixture cases[${index}]`, testCase, ["id", "expected"]);
      if (!isRecord(testCase) || !isRecord(testCase.expected) || Object.keys(testCase.expected).length === 0) failures.push(`Grad compatibility fixture cases[${index}].expected must be a non-empty object`);
    }
  }
  return failures;
}

export function validateJitOpaqueOperationInventory(inventory, fixture, freeze, filename = "jit-opaque-operation-inventory.json") {
  const failures = [];
  const exactKeys = (label, value, expected) => {
    if (!isRecord(value)) {
      failures.push(`${label} must be an object`);
      return;
    }
    compareStringSets(`${label} keys`, Object.keys(value), expected, failures);
  };
  const nonEmptyString = (label, value) => {
    if (typeof value !== "string" || value.trim() === "") failures.push(`${label} must be a non-empty string`);
  };

  exactKeys(filename, inventory, ["schemaVersion", "inventoryId", "owner", "irOpcode", "observationStatus", "executionContext", "policies", "constructorSites", "decisionSourceDefinitions", "operations", "evidenceDefinitions"]);
  if (!isRecord(inventory) || inventory.schemaVersion !== 1 || inventory.inventoryId !== "browsergrad.jit.opaque-operations.v0" || inventory.owner !== "@unlocalhosted/browsergrad-jit" || inventory.irOpcode !== "CUSTOM" || inventory.observationStatus !== "verified-current-behavior") {
    failures.push(`${filename} must identify the schemaVersion 1 BrowserGrad JIT CUSTOM inventory`);
    return failures;
  }

  exactKeys(`${filename}.executionContext`, inventory.executionContext, ["pythonRuntime", "pyodideVersion", "cpuReference", "numpyVersion", "legacyWebgpu", "tensorGpuPlanDefault", "tensorGpuPlanAllowCustom", "tensorGpuPlanExecution"]);
  const expectedExecutionContext = {
    pythonRuntime: "pyodide",
    pyodideVersion: "0.26.4",
    cpuReference: "numpy-wasm",
    numpyVersion: "1.26.4",
    legacyWebgpu: "registered-bridge-with-root-materialization",
    tensorGpuPlanDefault: "refuses-custom",
    tensorGpuPlanAllowCustom: "admits-custom-plan-structure-only",
    tensorGpuPlanExecution: "refuses-custom",
  };
  if (JSON.stringify(inventory.executionContext) !== JSON.stringify(expectedExecutionContext)) failures.push(`${filename}.executionContext changed`);

  const policyKeys = ["id", "category", "cpuRealization", "closureAutograd", "symbolicVjp", "functionalGrad", "vmap", "onnxExport", "tensorGpuPlanDefault", "tensorGpuPlanAllowCustom", "tensorGpuPlanExecution", "legacyWebgpu", "residency", "realizedResultValidation", "failurePolicy", "targetConformance"];
  const policyEnums = {
    category: new Set(["constructor-only", "explicit-accelerator-kernel", "legacy-numpy-callback"]),
    cpuRealization: new Set(["numpy-callback", "refused-missing-fn"]),
    closureAutograd: new Set(["disconnected-requires-grad-false", "implemented", "nearest-implemented-bilinear-refused"]),
    symbolicVjp: new Set(["refused-no-custom-rule"]),
    functionalGrad: new Set(["refused-missing-symbolic-vjp", "returns-zero-for-disconnected-input"]),
    vmap: new Set(["refused-custom"]),
    onnxExport: new Set(["refused-custom"]),
    tensorGpuPlanDefault: new Set(["refused-custom"]),
    tensorGpuPlanAllowCustom: new Set(["admitted-plan-structure-only"]),
    tensorGpuPlanExecution: new Set(["refused-custom"]),
    legacyWebgpu: new Set(["refused-name-field", "refused-unsupported-op-field", "supported-op-dispatch"]),
    residency: new Set(["host-materialized", "legacy-webgpu-root-materialized-to-host", "not-realizable"]),
    realizedResultValidation: new Set(["materialized-by-declared-shape-and-dtype", "ndarray-only-shape-and-dtype-unchecked", "not-realizable"]),
    failurePolicy: new Set(["all-realizers-refuse", "bilinear-backward-raises-not-implemented", "cpu-callback-only", "cpu-path-refuses", "silent-autograd-disconnection"]),
    targetConformance: new Set(["compatibility-debt", "operation-specific"]),
  };
  const policyIds = [];
  const policyById = new Map();
  if (!Array.isArray(inventory.policies) || inventory.policies.length === 0) failures.push(`${filename}.policies must be a non-empty array`);
  for (const [index, policy] of (Array.isArray(inventory.policies) ? inventory.policies : []).entries()) {
    const label = `${filename}.policies[${index}]`;
    exactKeys(label, policy, policyKeys);
    if (!isRecord(policy)) continue;
    nonEmptyString(`${label}.id`, policy.id);
    for (const [field, values] of Object.entries(policyEnums)) {
      if (!values.has(policy[field])) failures.push(`${label}.${field} is not registered`);
    }
    if (typeof policy.id === "string") {
      policyIds.push(policy.id);
      policyById.set(policy.id, policy);
    }
  }
  if (new Set(policyIds).size !== policyIds.length) failures.push(`${filename} policy IDs must be unique`);
  compareStringSets("JIT opaque-operation policy IDs", policyIds, freeze.policyIds, failures);

  const siteKeys = ["id", "file", "definition", "constructorCount", "labelBinding", "operationIds"];
  const siteIds = [];
  const siteById = new Map();
  const siteOperationIds = [];
  const sourceDefinitions = [];
  const labelBindings = new Set(["dynamic-reviewed-name", "static-name", "static-op-in-arg-record"]);
  const frozenConstructorCounts = Object.values(freeze.constructorCounts ?? {});
  const expectedConstructorCount = frozenConstructorCounts.every((count) => Number.isInteger(count) && count >= 0)
    ? frozenConstructorCounts.reduce((total, count) => total + count, 0)
    : -1;
  if (expectedConstructorCount < 1) failures.push("JIT opaque-operation frozen constructor counts must contain positive integer total");
  if (!Array.isArray(inventory.constructorSites) || inventory.constructorSites.length !== expectedConstructorCount) {
    failures.push(`${filename}.constructorSites must contain the ${expectedConstructorCount} exact frozen CUSTOM constructor calls`);
  }
  for (const [index, site] of (Array.isArray(inventory.constructorSites) ? inventory.constructorSites : []).entries()) {
    const label = `${filename}.constructorSites[${index}]`;
    exactKeys(label, site, siteKeys);
    if (!isRecord(site)) continue;
    for (const field of ["id", "file", "definition"]) nonEmptyString(`${label}.${field}`, site[field]);
    if (site.constructorCount !== 1) failures.push(`${label}.constructorCount must be exactly 1; each record names one constructor call`);
    if (!labelBindings.has(site.labelBinding)) failures.push(`${label}.labelBinding is not registered`);
    if (!Array.isArray(site.operationIds) || site.operationIds.length === 0 || site.operationIds.some((entry) => typeof entry !== "string")) failures.push(`${label}.operationIds must contain strings`);
    if (typeof site.id === "string") {
      siteIds.push(site.id);
      siteById.set(site.id, site);
    }
    if (typeof site.file === "string" && typeof site.definition === "string") sourceDefinitions.push(`${site.file}:${site.definition}`);
    if (Array.isArray(site.operationIds)) siteOperationIds.push(...site.operationIds.filter((entry) => typeof entry === "string"));
  }
  if (new Set(siteIds).size !== siteIds.length) failures.push(`${filename} constructor site IDs must be unique`);
  compareStringSets("JIT opaque-operation constructor site IDs", siteIds, freeze.constructorSiteIds, failures);
  const frozenDefinitions = Object.entries(freeze.definitionTokenDigests ?? {}).flatMap(([file, definitions]) => Object.keys(definitions ?? {}).map((definition) => `${file}:${definition}`));
  for (const reference of sourceDefinitions) {
    if (!new Set(frozenDefinitions).has(reference)) failures.push(`${filename} references unfrozen constructor definition ${reference}`);
  }
  if (!Array.isArray(inventory.decisionSourceDefinitions) || inventory.decisionSourceDefinitions.length === 0 || inventory.decisionSourceDefinitions.some((entry) => typeof entry !== "string")) {
    failures.push(`${filename}.decisionSourceDefinitions must contain strings`);
  } else {
    if (new Set(inventory.decisionSourceDefinitions).size !== inventory.decisionSourceDefinitions.length) failures.push(`${filename}.decisionSourceDefinitions must be unique`);
    compareStringSets("JIT opaque-operation decision source definitions", inventory.decisionSourceDefinitions, freeze.decisionSourceDefinitions, failures);
    for (const reference of inventory.decisionSourceDefinitions) {
      if (!new Set(frozenDefinitions).has(reference)) failures.push(`${filename} references unfrozen decision definition ${reference}`);
    }
  }

  const operationKeys = ["id", "label", "labelField", "constructorSite", "policy", "constructorReachability", "effectCondition", "effectClass", "replayContract", "inputArity", "shapeRule", "declaredDtypeRule", "realizedDtypeRule", "webgpuRoute", "currentFailure", "targetConformance", "evidence"];
  const operationIds = [];
  const operationLabels = [];
  const policyReferences = [];
  const evidenceReferences = [];
  const effectClasses = new Set(["accelerator-kernel", "captured-state", "constructor-only", "host-shape-probe", "module-stateful", "pure", "rng-and-captured-state", "user-authored-kernel"]);
  const realizedDtypeRules = new Set(["bridge-materialized-as-declared", "callback-result-unvalidated", "not-realizable"]);
  const targetConformance = new Set(["compatibility-debt", "intentional-extension"]);
  if (!Array.isArray(inventory.operations) || inventory.operations.length === 0) failures.push(`${filename}.operations must be a non-empty array`);
  for (const [index, operation] of (Array.isArray(inventory.operations) ? inventory.operations : []).entries()) {
    const label = `${filename}.operations[${index}]`;
    exactKeys(label, operation, operationKeys);
    if (!isRecord(operation)) continue;
    for (const field of ["id", "label", "constructorSite", "policy", "constructorReachability", "effectCondition", "replayContract", "inputArity", "shapeRule", "declaredDtypeRule", "realizedDtypeRule", "webgpuRoute", "currentFailure"]) nonEmptyString(`${label}.${field}`, operation[field]);
    if (!new Set(["name", "op"]).has(operation.labelField)) failures.push(`${label}.labelField must be name or op`);
    if (!effectClasses.has(operation.effectClass)) failures.push(`${label}.effectClass is not registered`);
    if (!realizedDtypeRules.has(operation.realizedDtypeRule)) failures.push(`${label}.realizedDtypeRule is not registered`);
    if (!targetConformance.has(operation.targetConformance)) failures.push(`${label}.targetConformance is not registered`);
    if (!Array.isArray(operation.evidence) || operation.evidence.length === 0 || operation.evidence.some((entry) => typeof entry !== "string")) failures.push(`${label}.evidence must contain strings`);
    const site = siteById.get(operation.constructorSite);
    if (site === undefined) failures.push(`${label}.constructorSite is not registered`);
    else {
      if (!site.operationIds.includes(operation.id)) failures.push(`${label} is absent from its constructor site's operationIds`);
      const expectedField = site.labelBinding.includes("name") ? "name" : "op";
      if (operation.labelField !== expectedField) failures.push(`${label}.labelField disagrees with ${site.id}.labelBinding`);
    }
    if (!new Set(policyIds).has(operation.policy)) failures.push(`${label}.policy is not registered`);
    else {
      const policy = policyById.get(operation.policy);
      if (policy?.targetConformance !== "operation-specific" && policy?.targetConformance !== operation.targetConformance) failures.push(`${label}.targetConformance disagrees with policy ${operation.policy}`);
    }
    if (typeof operation.id === "string") operationIds.push(operation.id);
    if (typeof operation.label === "string") operationLabels.push(operation.label);
    if (typeof operation.policy === "string") policyReferences.push(operation.policy);
    if (Array.isArray(operation.evidence)) evidenceReferences.push(...operation.evidence.filter((entry) => typeof entry === "string"));
  }
  if (new Set(operationIds).size !== operationIds.length) failures.push(`${filename} operation IDs must be unique`);
  if (new Set(operationLabels).size !== operationLabels.length) failures.push(`${filename} operation labels must be unique`);
  compareStringSets("JIT opaque-operation IDs", operationIds, freeze.operationIds, failures);
  compareStringSets("JIT opaque-operation labels", operationLabels, freeze.labels, failures);
  compareStringSets("JIT constructor-site operation coverage", siteOperationIds, operationIds, failures);
  compareStringSets("JIT opaque-operation policy coverage", [...new Set(policyReferences)], policyIds, failures);

  const evidenceKeys = ["id", "kind", "path"];
  const evidenceIds = [];
  if (!Array.isArray(inventory.evidenceDefinitions) || inventory.evidenceDefinitions.length === 0) failures.push(`${filename}.evidenceDefinitions must be a non-empty array`);
  for (const [index, evidence] of (Array.isArray(inventory.evidenceDefinitions) ? inventory.evidenceDefinitions : []).entries()) {
    const label = `${filename}.evidenceDefinitions[${index}]`;
    exactKeys(label, evidence, evidenceKeys);
    if (!isRecord(evidence)) continue;
    for (const field of evidenceKeys) nonEmptyString(`${label}.${field}`, evidence[field]);
    if (typeof evidence.id === "string") evidenceIds.push(evidence.id);
  }
  if (new Set(evidenceIds).size !== evidenceIds.length) failures.push(`${filename} evidence IDs must be unique`);
  for (const reference of new Set(evidenceReferences)) {
    if (!new Set(evidenceIds).has(reference)) failures.push(`${filename} references unknown evidence ${reference}`);
  }
  compareStringSets("JIT opaque-operation evidence coverage", [...new Set(evidenceReferences)], evidenceIds, failures);

  validateNamedFixture(fixture, "jit.core-custom-ops.v0", freeze.behaviorFixtureIds, stringValue(freeze.behaviorFixtureFile), "JIT opaque-operation", failures);
  if (isRecord(fixture)) {
    exactKeys("JIT opaque-operation fixture", fixture, ["schemaVersion", "adapterId", "cases"]);
    for (const [index, testCase] of (Array.isArray(fixture.cases) ? fixture.cases : []).entries()) {
      exactKeys(`JIT opaque-operation fixture cases[${index}]`, testCase, ["id", "expected"]);
      if (!isRecord(testCase) || !isRecord(testCase.expected) || Object.keys(testCase.expected).length === 0) failures.push(`JIT opaque-operation fixture cases[${index}].expected must be a non-empty object`);
    }
  }
  return failures;
}

function checkJitFrameworkOperationContracts(root, manifest, failures) {
  const registryFile = path.join(
    root,
    "packages/browsergrad-jit/src/python/framework-operation-contracts.v1.json",
  );
  const registry = readJson(registryFile, failures);
  if (!isRecord(registry)) return;
  if (fs.statSync(registryFile).size < 1 || fs.statSync(registryFile).size > 64 * 1024) {
    failures.push("JIT framework-operation registry must contain 1..65536 bytes");
  }
  compareExactKeys("JIT framework-operation registry", registry, ["schema", "version", "operations"], failures);
  if (registry.schema !== "browsergrad.jit.framework-operation-contracts") {
    failures.push("JIT framework-operation registry schema changed");
  }
  if (!isRecord(registry.version)) {
    failures.push("JIT framework-operation registry version must be an object");
  } else {
    compareExactKeys("JIT framework-operation registry version", registry.version, ["major", "minor"], failures);
    if (registry.version.major !== 1 || registry.version.minor !== 0) {
      failures.push("JIT framework-operation registry version must be exactly 1.0");
    }
  }
  if (!Array.isArray(registry.operations) || registry.operations.length === 0) {
    failures.push("JIT framework-operation registry operations must be a nonempty array");
    return;
  }

  const operationFields = [
    "contractId",
    "publicSurface",
    "opcode",
    "semanticState",
    "shapeContract",
    "dtypeContract",
    "decisions",
    "retiredOpaqueOperationId",
  ];
  const decisionFields = [
    "cpu",
    "closureAutograd",
    "symbolicVjp",
    "functionalGrad",
    "vmap",
    "onnxExport",
    "tensorPlan",
    "webgpu",
    "residency",
    "materialization",
  ];
  const allowed = {
    semanticState: new Set(["typed"]),
    shapeContract: new Set(["canonical-general-einstein-contraction", "class-axis-index-loss-with-batched-reduction", "preserve-batched-lower-triangular", "preserve-batched-upper-triangular", "preserve-single-axis-inclusive-scan", "preserve-single-axis-reverse", "preserve-source-with-broadcast-bool-mask", "preserve-unary-input", "same-rank-index-shaped-gather", "same-rank-unique-index-overwrite-scatter", "same-shape-axis-ordering", "same-shape-elementwise-loss-with-batched-reduction", "selected-axis-becomes-exact-k", "selected-axis-times-repeat-count", "static-broadcast-with-existing-dim-minus-one", "static-product-reduction", "static-variance-reduction", "tile-multipliers-with-left-rank-padding", "trailing-dimension-constant-padding", "variadic-existing-axis-concatenation-with-legacy-empty", "variadic-new-axis-stacking"]),
    dtypeContract: new Set(["dimensioned-tensor-promotion-with-fp32-half-accumulator", "preserve-floating-input", "preserve-floating-input-require-int64-target-and-optional-matching-weight", "preserve-input", "preserve-input-require-bool-mask", "preserve-real-numeric-input", "preserve-source-require-int64-index", "preserve-supported-input-with-exact-fill", "preserve-target-require-int64-index-matching-source", "promote-floating-inputs-with-fp32-half-accumulator", "promote-integral-default-or-explicit-scan-dtype", "values-preserve-input-indices-int64", "pytorch-dimensioned-tensor-promotion"]),
    cpu: new Set(["supported-numpy-dtype-preserving", "supported-numpy-owning-bounded-bce-reduction", "supported-numpy-owning-bounded-nll-reduction", "supported-numpy-owning-kl-div-reduction", "supported-numpy-owning-stable-bce-with-logits-reduction", "supported-numpy-owning-concatenation-copy", "supported-numpy-owning-constant-pad-copy", "supported-numpy-owning-copy", "supported-numpy-owning-copy-with-range-check", "supported-numpy-owning-greedy-einsum", "supported-numpy-owning-loss-reduction", "supported-numpy-owning-partial-topk-indices", "supported-numpy-owning-scan-copy", "supported-numpy-owning-sort-gather", "supported-numpy-owning-stable-sort-indices", "supported-numpy-owning-stack-copy", "supported-numpy-owning-topk-gather", "supported-numpy-owning-unique-overwrite-scatter"]),
    closureAutograd: new Set(["supported-centered-variance-rule", "supported-clamped-bce-derivatives-for-both-inputs", "supported-native-kl-div-derivatives-for-both-inputs", "supported-selected-class-negative-weight-gradient", "supported-stable-bce-with-logits-derivatives-for-both-inputs", "supported-cos-derivative", "supported-deterministic-scatter-add", "supported-general-einsum-vjp", "supported-idempotent-triangular-selection", "supported-inclusive-bound-mask", "supported-involutive-flip", "supported-mask-complement-selection", "not-applicable-discrete-indices", "supported-negative-sin-derivative", "supported-permutation-scatter", "supported-piecewise-difference-for-both-inputs", "supported-unique-overwrite-scatter", "supported-opposite-direction-inclusive-scan-for-floating-source-and-output", "supported-selected-axis-block-sum", "supported-sign-derivative", "supported-signed-difference-for-both-inputs", "supported-static-axis-index", "supported-static-axis-split", "supported-static-interior-slice", "supported-tile-block-sum", "supported-unbroadcast-sum", "supported-zero-aware-product-rule", "supported-zero-derivative"]),
    symbolicVjp: new Set(["supported-centered-variance-rule", "supported-clamped-bce-derivatives-for-both-inputs", "supported-native-kl-div-derivatives-for-both-inputs", "supported-selected-class-negative-weight-gradient", "supported-stable-bce-with-logits-derivatives-for-both-inputs", "supported-cos-derivative", "supported-deterministic-scatter-add", "supported-general-einsum-vjp", "supported-idempotent-triangular-selection", "supported-inclusive-bound-mask", "supported-involutive-flip", "supported-mask-complement-selection", "not-applicable-discrete-indices", "supported-negative-sin-derivative", "supported-permutation-scatter", "supported-piecewise-difference-for-both-inputs", "supported-unique-overwrite-scatter", "supported-opposite-direction-inclusive-scan-for-floating-source-and-output", "supported-selected-axis-block-sum", "supported-sign-derivative", "supported-signed-difference-for-both-inputs", "supported-static-axis-index", "supported-static-axis-split", "supported-static-interior-slice", "supported-tile-block-sum", "supported-unbroadcast-sum", "supported-zero-aware-product-rule", "supported-zero-derivative"]),
    functionalGrad: new Set(["not-applicable-discrete-output", "supported-for-both-floating-inputs-via-symbolic-vjp", "supported-for-floating-input-via-symbolic-vjp", "supported-for-floating-output-via-symbolic-vjp", "supported-for-floating-source-and-output-via-symbolic-vjp", "supported-for-floating-target-and-source-via-symbolic-vjp", "supported-via-symbolic-vjp"]),
    vmap: new Set(["supported-leading-batch-axis", "supported-leading-batch-axis-preserve-matrix-axes", "supported-leading-batch-axis-preserving-pad", "supported-leading-batch-axis-with-axis-shift", "supported-leading-batch-axis-with-axis-shift-and-captured-broadcast", "supported-leading-batch-axis-with-class-axis-shift-and-captured-weight", "supported-leading-batch-axis-with-einsum-captured-broadcast", "supported-leading-batch-axis-with-index-axis-shift", "supported-leading-batch-axis-with-mask-broadcast", "supported-leading-batch-axis-with-per-example-reduction", "supported-leading-batch-axis-with-scan-axis-shift", "supported-leading-batch-axis-with-scatter-captured-broadcast", "supported-leading-batch-axis-with-unit-repeat"]),
    onnxExport: new Set(["refused-runtime-probability-domain-cannot-fail-closed", "supported-opset17-kl-div-float16-float32-float64", "supported-opset17-negative-log-likelihood-loss-unmapped-profile", "supported-opset17-stable-bce-with-logits-float16-float32-float64", "supported-opset17-clip-export-dtypes", "supported-opset17-concat-with-casts-float32-int32-int64-bool", "supported-opset17-cumsum-with-cast-float32-int32-int64", "supported-opset17-direct-unary-export-dtypes", "supported-opset17-expand", "supported-opset17-full-axis-topk-gather-float32-int32-int64", "supported-opset17-gather-elements-float32-int32-int64-bool", "supported-opset17-pad-float32-int32-int64", "supported-opset17-piecewise-smooth-l1-float16-float32-float64", "supported-opset17-reduce-prod-float32-int32-int64", "supported-opset17-resolved-einsum-numeric-dtypes", "supported-opset17-scatter-elements-float32-int32-int64-bool", "supported-opset17-selected-k-topk-gather-float32-int32-int64", "supported-opset17-slice-float32-int32-int64-bool", "supported-opset17-sub-abs-reduce-float16-float32-float64", "supported-opset17-tile-float32-int32-int64-bool", "supported-opset17-trilu-float32-int32-int64-bool", "supported-opset17-unsqueeze-concat-with-casts-float32-int32-int64-bool", "supported-opset17-unsqueeze-tile-reshape-float32-int32-int64-bool", "supported-opset17-variance-decomposition-float32", "supported-opset17-where-float32-int32-int64-bool"]),
    tensorPlan: new Set(["refused-negative-stride-profile", "refused-no-canonical-contraction-lowering", "refused-no-canonical-loss-reduction-lowering", "refused-no-canonical-pad-lowering", "refused-no-canonical-scatter-overwrite-lowering", "refused-no-canonical-selected-axis-replication-profile", "refused-no-canonical-sort-lowering", "refused-no-canonical-topk-lowering", "refused-no-canonical-tile-layout-profile", "refused-no-canonical-variadic-copy-lowering", "refused-no-deterministic-index-lowering", "refused-no-portable-lowering", "refused-no-portable-masked-selection", "refused-no-portable-triangular-selection", "supported-primitive"]),
    webgpu: new Set(["profile-nonempty-f32-rank-at-most-4", "refused-negative-stride-profile", "refused-no-canonical-selected-axis-replication-profile", "refused-no-canonical-tile-layout-profile", "refused-no-deterministic-index-kernel", "refused-no-tensor-plan-kernel"]),
    residency: new Set(["host-materialized", "supported-materializing-and-resident"]),
    materialization: new Set(["cpu-owning-array", "cpu-owning-copy"]),
  };
  allowed.shapeContract.add(
    "class-axis-logits-loss-with-index-or-probability-target-and-batched-reduction",
  );
  allowed.dtypeContract.add(
    "preserve-floating-input-require-index-or-matching-floating-target-and-optional-matching-weight",
  );
  allowed.cpu.add("supported-numpy-owning-stable-cross-entropy-reduction");
  allowed.closureAutograd.add(
    "supported-stable-logits-and-probability-target-gradients",
  );
  allowed.symbolicVjp.add(
    "supported-stable-logits-and-probability-target-gradients",
  );
  allowed.functionalGrad.add(
    "supported-for-floating-input-and-probability-target-via-symbolic-vjp",
  );
  allowed.vmap.add(
    "supported-leading-batch-axis-with-target-mode-class-axis-shift-and-captured-weight",
  );
  allowed.onnxExport.add(
    "supported-opset17-softmax-cross-entropy-loss-unmapped-index-profile",
  );
  allowed.shapeContract.add(
    "preserve-input-with-elementwise-bernoulli-mask",
  );
  allowed.dtypeContract.add(
    "preserve-input-with-floating-stochastic-profile",
  );
  allowed.cpu.add("supported-numpy-owning-keyed-inverted-dropout");
  allowed.closureAutograd.add("supported-keyed-mask-replay");
  allowed.symbolicVjp.add("supported-keyed-mask-replay");
  allowed.vmap.add(
    "supported-deterministic-drop-all-refuses-stochastic-without-randomness-policy",
  );
  allowed.onnxExport.add(
    "refused-training-dropout-in-inference-export",
  );
  allowed.tensorPlan.add("refused-no-canonical-keyed-rng-lowering");
  allowed.shapeContract.add(
    "preserve-rank-2-or-3-channel-normalized-input",
  );
  allowed.dtypeContract.add(
    "exact-float32-input-affine-and-state",
  );
  allowed.cpu.add("supported-numpy-owning-batch-normalization");
  allowed.closureAutograd.add(
    "supported-batch-stat-or-running-stat-derivatives",
  );
  allowed.symbolicVjp.add(
    "supported-batch-stat-or-running-stat-derivatives",
  );
  allowed.vmap.add(
    "refused-state-and-batch-axis-contract-undefined",
  );
  allowed.onnxExport.add(
    "refused-no-stable-running-stat-export-profile",
  );
  allowed.tensorPlan.add(
    "refused-no-canonical-batch-normalization-lowering",
  );
  allowed.shapeContract.add(
    "resize-last-two-spatial-axes-of-rank-4-input",
  );
  allowed.dtypeContract.add("preserve-floating-input");
  allowed.cpu.add(
    "supported-numpy-owning-nearest-or-bilinear-resample",
  );
  allowed.closureAutograd.add(
    "supported-transpose-of-nearest-or-bilinear-resample",
  );
  allowed.symbolicVjp.add(
    "supported-transpose-of-nearest-or-bilinear-resample",
  );
  allowed.vmap.add(
    "supported-leading-batch-axis-preserve-spatial-axes",
  );
  allowed.onnxExport.add(
    "supported-opset17-resize-nearest-or-linear-floating",
  );
  allowed.tensorPlan.add(
    "refused-no-canonical-spatial-resample-lowering",
  );
  allowed.shapeContract.add(
    "dense-rank4-batched-multihead-attention-forward",
  );
  allowed.dtypeContract.add("exact-float32-query-key-value");
  allowed.cpu.add("supported-numpy-owning-stable-attention-forward");
  allowed.closureAutograd.add("refused-attention-vjp-not-defined");
  allowed.symbolicVjp.add("refused-attention-vjp-not-defined");
  allowed.functionalGrad.add("refused-attention-vjp-not-defined");
  allowed.vmap.add("refused-no-attention-batching-contract");
  allowed.onnxExport.add("refused-no-canonical-attention-export");
  allowed.tensorPlan.add("refused-attention-side-table-not-integrated");
  allowed.webgpu.add("supported-legacy-row-wise-online-softmax-f32");
  allowed.residency.add("legacy-webgpu-root-materialized-to-host");
  allowed.materialization.add(
    "cpu-owning-copy-or-legacy-webgpu-root-copy",
  );
  const irSource = fs.readFileSync(
    path.join(root, "packages/browsergrad-jit/src/python/_ir.py"),
    "utf8",
  );
  const declaredOpcodes = new Set(
    [...irSource.matchAll(/^OP_[A-Z0-9_]+\s*=\s*"([A-Z0-9_]+)"/gmu)].map((match) => match[1]),
  );
  const validatorSource = fs.readFileSync(
    path.join(root, "packages/browsergrad-jit/src/python/_framework_contracts.py"),
    "utf8",
  );
  const contractIds = [];
  const opcodes = [];
  const retiredIds = [];
  for (const [index, operation] of registry.operations.entries()) {
    const label = `JIT framework-operation registry operations[${index}]`;
    if (!isRecord(operation)) {
      failures.push(`${label} must be an object`);
      continue;
    }
    compareExactKeys(label, operation, operationFields, failures);
    for (const field of [
      "contractId",
      "publicSurface",
      "opcode",
      "semanticState",
      "shapeContract",
      "dtypeContract",
      "retiredOpaqueOperationId",
    ]) {
      if (typeof operation[field] !== "string" || operation[field].length === 0 || operation[field].trim() !== operation[field]) {
        failures.push(`${label}.${field} must be a nonempty canonical string`);
      }
    }
    if (typeof operation.contractId === "string") {
      contractIds.push(operation.contractId);
      if (!operation.contractId.startsWith("browsergrad.jit.framework.")) {
        failures.push(`${label}.contractId is outside the BrowserGrad JIT namespace`);
      }
      if (!validatorSource.includes(`"${operation.contractId}"`)) {
        failures.push(`${label}.contractId has no package-owned executable validator binding`);
      }
    }
    if (typeof operation.opcode === "string") {
      opcodes.push(operation.opcode);
      if (operation.opcode === "CUSTOM" || !declaredOpcodes.has(operation.opcode)) {
        failures.push(`${label}.opcode must name a declared non-CUSTOM UOp`);
      }
    }
    if (typeof operation.retiredOpaqueOperationId === "string") {
      retiredIds.push(operation.retiredOpaqueOperationId);
      if (!operation.retiredOpaqueOperationId.startsWith("jit.custom.")) {
        failures.push(`${label}.retiredOpaqueOperationId is outside the frozen namespace`);
      }
    }
    for (const field of ["semanticState", "shapeContract", "dtypeContract"]) {
      if (!allowed[field].has(operation[field])) failures.push(`${label}.${field} is not registered`);
    }
    if (!isRecord(operation.decisions)) {
      failures.push(`${label}.decisions must be an object`);
    } else {
      compareExactKeys(`${label}.decisions`, operation.decisions, decisionFields, failures);
      for (const field of decisionFields) {
        if (!allowed[field].has(operation.decisions[field])) {
          failures.push(`${label}.decisions.${field} is not registered`);
        }
      }
    }
  }
  for (const [label, values] of [
    ["contract IDs", contractIds],
    ["opcodes", opcodes],
    ["retired opaque-operation IDs", retiredIds],
  ]) {
    if (new Set(values).size !== values.length) failures.push(`JIT framework-operation ${label} must be unique`);
  }
  const sortedContractIds = [...contractIds].sort((left, right) => left.localeCompare(right));
  if (JSON.stringify(contractIds) !== JSON.stringify(sortedContractIds)) {
    failures.push("JIT framework-operation contracts must be ordered by contractId");
  }

  const opaqueInventory = readJson(
    path.join(root, "architecture/jit-opaque-operation-inventory.json"),
    failures,
  );
  const currentOpaqueIds = isRecord(opaqueInventory) && Array.isArray(opaqueInventory.operations)
    ? opaqueInventory.operations.map((operation) => isRecord(operation) ? operation.id : undefined)
      .filter((id) => typeof id === "string")
    : [];
  for (const retiredId of retiredIds) {
    if (currentOpaqueIds.includes(retiredId)) {
      failures.push(`JIT typed framework operation ${retiredId} remains in the opaque inventory`);
    }
  }
  const jitFreeze = Array.isArray(manifest.adapters)
    ? manifest.adapters.find((adapter) => isRecord(adapter) && adapter.id === "jit.core-custom-ops.v0")?.freeze
    : undefined;
  const originalIds = isRecord(jitFreeze) ? jitFreeze.originalOperationIds : undefined;
  const removedUnsupportedIds = isRecord(jitFreeze)
    ? jitFreeze.removedUnsupportedSurfaceOperationIds
    : undefined;
  if (!Array.isArray(originalIds) || originalIds.length === 0 || originalIds.some((id) => typeof id !== "string")) {
    failures.push("JIT opaque-operation freeze must retain the original operation ID partition");
  } else if (
    !Array.isArray(removedUnsupportedIds)
    || removedUnsupportedIds.some(
      (id) => typeof id !== "string" || !id.startsWith("jit.custom."),
    )
    || new Set(removedUnsupportedIds).size !== removedUnsupportedIds.length
  ) {
    failures.push(
      "JIT opaque-operation freeze must retain unique removed unsupported-surface IDs",
    );
  } else {
    for (const removedId of removedUnsupportedIds) {
      if (currentOpaqueIds.includes(removedId)) {
        failures.push(
          `JIT removed unsupported surface ${removedId} remains in the opaque inventory`,
        );
      }
      if (retiredIds.includes(removedId)) {
        failures.push(
          `JIT removed unsupported surface ${removedId} is also classified as typed`,
        );
      }
    }
    compareStringSets(
      "JIT original opaque-operation partition",
      [...currentOpaqueIds, ...retiredIds, ...removedUnsupportedIds],
      originalIds,
      failures,
    );
  }
}

function validateManifest(root, manifest, failures) {
  if (!isRecord(manifest) || manifest.schemaVersion !== 1) {
    failures.push("architecture/semantic-freeze.json must have schemaVersion 1");
    return;
  }
  if (!Array.isArray(manifest.adapters) || manifest.adapters.length === 0) {
    failures.push("architecture/semantic-freeze.json must contain adapters");
    return;
  }
  const decisionDirectory = isRecord(manifest.policy) && typeof manifest.policy.architectureDecisionDirectory === "string"
    ? manifest.policy.architectureDecisionDirectory
    : undefined;
  if (decisionDirectory === undefined || manifest.policy.baselineChangesRequireDecision !== true) {
    failures.push("architecture/semantic-freeze.json policy must require decisions and name the decision directory");
  }
  const ids = new Set();
  for (const [index, adapter] of manifest.adapters.entries()) {
    const prefix = `architecture/semantic-freeze.json adapters[${index}]`;
    if (!isRecord(adapter)) {
      failures.push(`${prefix} must be an object`);
      continue;
    }
    for (const field of ["id", "owner", "newUseProhibition", "retirementGate", "compatibilityRemovalVersion"]) {
      if (typeof adapter[field] !== "string" || adapter[field].trim() === "") failures.push(`${prefix}.${field} must be a non-empty string`);
    }
    if (!Array.isArray(adapter.permittedCallers) || adapter.permittedCallers.length === 0 || adapter.permittedCallers.some((entry) => typeof entry !== "string" || entry.trim() === "")) {
      failures.push(`${prefix}.permittedCallers must contain non-empty strings`);
    }
    if (typeof adapter.id === "string") {
      if (ids.has(adapter.id)) failures.push(`${prefix}.id duplicates ${adapter.id}`);
      ids.add(adapter.id);
    }
    if (isRecord(adapter.freeze)) {
      if (typeof adapter.baselineDecision !== "string" || adapter.baselineDecision.trim() === "") {
        failures.push(`${prefix}.baselineDecision is required for a frozen adapter`);
      } else if (decisionDirectory !== undefined) {
        const decision = path.resolve(root, adapter.baselineDecision);
        const decisionsRoot = path.resolve(root, decisionDirectory);
        if (!decision.startsWith(`${decisionsRoot}${path.sep}`) || !fs.existsSync(decision)) {
          failures.push(`${prefix}.baselineDecision must reference an existing file under ${relative(root, decisionsRoot)}`);
        }
      }
      if (adapter.freeze.kind === "jit-op-custom") {
        const removedIds = adapter.freeze.removedUnsupportedSurfaceOperationIds;
        if (
          !Array.isArray(removedIds)
          || removedIds.some(
            (id) => typeof id !== "string" || !id.startsWith("jit.custom."),
          )
          || new Set(removedIds).size !== removedIds.length
        ) {
          failures.push(
            `${prefix}.freeze.removedUnsupportedSurfaceOperationIds must contain unique frozen IDs`,
          );
        } else {
          const currentIds = Array.isArray(adapter.freeze.operationIds)
            ? adapter.freeze.operationIds
            : [];
          const originalIds = Array.isArray(adapter.freeze.originalOperationIds)
            ? adapter.freeze.originalOperationIds
            : [];
          for (const removedId of removedIds) {
            if (currentIds.includes(removedId)) {
              failures.push(
                `${prefix}.freeze removed unsupported surface ${removedId} remains current opaque`,
              );
            }
            if (!originalIds.includes(removedId)) {
              failures.push(
                `${prefix}.freeze removed unsupported surface ${removedId} is outside the original partition`,
              );
            }
          }
        }
      }
    }
    if (typeof adapter.compatibilityRemovalVersion === "string" && !/^@unlocalhosted\/browsergrad-[a-z-]+@[0-9]+\.[0-9]+\.[0-9]+$/u.test(adapter.compatibilityRemovalVersion)) {
      failures.push(`${prefix}.compatibilityRemovalVersion must name an exact BrowserGrad package version`);
    }
  }
  for (const [id, kind] of REQUIRED_FREEZES) {
    const adapter = manifest.adapters.find((entry) => isRecord(entry) && entry.id === id);
    if (!isRecord(adapter) || !isRecord(adapter.freeze) || adapter.freeze.kind !== kind) {
      failures.push(`required freeze ${id} (${kind}) is missing`);
    }
  }
}

function checkSharedSemanticFixtureContracts(root, failures) {
  const manifestFile = path.join(root, "architecture/semantic-fixture-contracts.json");
  const manifest = readJson(manifestFile, failures);
  if (manifest !== undefined) validateSharedSemanticFixtureManifest(root, manifest, failures);
}

function validateSharedSemanticFixtureManifest(root, manifest, failures) {
  if (!isRecord(manifest) || manifest.schemaVersion !== 1 || !Array.isArray(manifest.contracts) || manifest.contracts.length === 0) {
    failures.push("architecture/semantic-fixture-contracts.json must be schemaVersion 1 with nonempty contracts");
    return;
  }
  const ids = new Set();
  for (const [index, contract] of manifest.contracts.entries()) {
    const label = `architecture/semantic-fixture-contracts.json contracts[${index}]`;
    if (!isRecord(contract)) {
      failures.push(`${label} must be an object`);
      continue;
    }
    const expectedFields = [
      "caseIds",
      "contentSha256",
      "excludedRoutingFields",
      "fixtureFile",
      "id",
      "owner",
      "packageExport",
      "schema",
      "version",
    ];
    compareExactKeys(label, contract, expectedFields, failures);
    for (const field of ["id", "owner", "fixtureFile", "packageExport", "schema", "contentSha256"]) {
      if (typeof contract[field] !== "string" || contract[field].length === 0) failures.push(`${label}.${field} must be a nonempty string`);
    }
    if (typeof contract.id === "string") {
      if (ids.has(contract.id)) failures.push(`${label}.id duplicates ${contract.id}`);
      ids.add(contract.id);
    }
    if (!Array.isArray(contract.caseIds) || contract.caseIds.length === 0 || contract.caseIds.some((id) => typeof id !== "string" || id.length === 0)) {
      failures.push(`${label}.caseIds must contain nonempty strings`);
    }
    if (!Array.isArray(contract.excludedRoutingFields) || contract.excludedRoutingFields.length === 0 || contract.excludedRoutingFields.some((field) => typeof field !== "string" || field.length === 0)) {
      failures.push(`${label}.excludedRoutingFields must contain nonempty strings`);
    }
    if (!isRecord(contract.version) || contract.version.major !== 1 || contract.version.minor !== 0 || Object.keys(contract.version).sort().join(",") !== "major,minor") {
      failures.push(`${label}.version must be the closed version 1.0 object`);
    }
    if (typeof contract.fixtureFile !== "string") continue;
    const fixtureFile = path.resolve(root, contract.fixtureFile);
    if (!fixtureFile.startsWith(`${path.resolve(root)}${path.sep}`) || !fs.existsSync(fixtureFile)) {
      failures.push(`${label}.fixtureFile is missing or outside the repository`);
      continue;
    }
    const contentSha256 = createHash("sha256").update(fs.readFileSync(fixtureFile)).digest("hex");
    if (contentSha256 !== contract.contentSha256) {
      failures.push(`${relative(root, fixtureFile)} content changed; expected SHA-256 ${stringValue(contract.contentSha256)}, got ${contentSha256}`);
    }
    const fixture = readJson(fixtureFile, failures);
    if (!isRecord(fixture)) continue;
    compareExactKeys(relative(root, fixtureFile), fixture, ["cases", "schema", "version"], failures);
    if (fixture.schema !== contract.schema || JSON.stringify(fixture.version) !== JSON.stringify(contract.version)) {
      failures.push(`${relative(root, fixtureFile)} schema/version differs from its contract`);
    }
    const fixtureCases = Array.isArray(fixture.cases) ? fixture.cases : [];
    const fixtureCaseIds = fixtureCases.map((entry) => isRecord(entry) ? entry.id : undefined);
    if (JSON.stringify(fixtureCaseIds) !== JSON.stringify(contract.caseIds)) {
      failures.push(`${relative(root, fixtureFile)} case order/coverage changed; expected ${JSON.stringify(contract.caseIds)}, got ${JSON.stringify(fixtureCaseIds)}`);
    }
    for (const [caseIndex, fixtureCase] of fixtureCases.entries()) {
      const caseLabel = `${relative(root, fixtureFile)} cases[${caseIndex}]`;
      if (!isRecord(fixtureCase)) {
        failures.push(`${caseLabel} must be an object`);
        continue;
      }
      compareExactKeys(caseLabel, fixtureCase, [
        "expectedOutputWords",
        "id",
        "kernelSemanticHash",
        "layoutSemanticHash",
        "outputShape",
        "request",
        "sourceWords",
      ], failures);
      if (!isRecord(fixtureCase.request)) {
        failures.push(`${caseLabel}.request must be an object`);
        continue;
      }
      compareExactKeys(`${caseLabel}.request`, fixtureCase.request, ["axes", "dtype", "inputShape", "kind"], failures);
      for (const routingField of Array.isArray(contract.excludedRoutingFields) ? contract.excludedRoutingFields : []) {
        if (typeof routingField === "string" && routingField in fixtureCase.request) {
          failures.push(`${caseLabel}.request contains excluded routing field ${routingField}`);
        }
      }
      for (const hashField of ["layoutSemanticHash", "kernelSemanticHash"]) {
        if (typeof fixtureCase[hashField] !== "string" || !/^[0-9a-f]{64}$/u.test(fixtureCase[hashField])) {
          failures.push(`${caseLabel}.${hashField} must be a full SHA-256 digest`);
        }
      }
    }
    if (typeof contract.owner !== "string" || typeof contract.packageExport !== "string") continue;
    const packageJson = findWorkspacePackageJson(root, contract.owner);
    if (packageJson === undefined) {
      failures.push(`${label}.owner does not name a workspace package`);
      continue;
    }
    const packageDirectory = path.dirname(packageJson);
    const packageManifest = JSON.parse(fs.readFileSync(packageJson, "utf8"));
    const expectedExportTarget = `./${relative(packageDirectory, fixtureFile)}`;
    if (packageManifest.exports?.[contract.packageExport] !== expectedExportTarget) {
      failures.push(`${label}.packageExport must map ${contract.packageExport} to ${expectedExportTarget}`);
    }
  }
}

function findWorkspacePackageJson(root, packageName) {
  const packagesRoot = path.join(root, "packages");
  if (!fs.existsSync(packagesRoot)) return undefined;
  for (const directory of fs.readdirSync(packagesRoot, { withFileTypes: true })) {
    if (!directory.isDirectory()) continue;
    const packageJson = path.join(packagesRoot, directory.name, "package.json");
    if (fs.existsSync(packageJson) && JSON.parse(fs.readFileSync(packageJson, "utf8")).name === packageName) return packageJson;
  }
  return undefined;
}

function compareExactKeys(label, value, expected, failures) {
  const actualKeys = Object.keys(value).sort();
  const expectedKeys = [...expected].sort();
  if (JSON.stringify(actualKeys) !== JSON.stringify(expectedKeys)) {
    failures.push(`${label} fields changed; expected ${JSON.stringify(expectedKeys)}, got ${JSON.stringify(actualKeys)}`);
  }
}

function checkWorkspaceDependencies(root, failures) {
  const packagesRoot = path.join(root, "packages");
  const packages = fs.existsSync(packagesRoot)
    ? fs.readdirSync(packagesRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => path.join(packagesRoot, entry.name, "package.json"))
      .filter((file) => fs.existsSync(file))
      .map((file) => ({ file, manifest: JSON.parse(fs.readFileSync(file, "utf8")) }))
    : [];
  const byName = new Map(packages.map((entry) => [entry.manifest.name, entry]));
  const graph = new Map();

  for (const entry of packages) {
    const name = entry.manifest.name;
    if (typeof name !== "string") continue;
    const runtimeDependencies = {
      ...(entry.manifest.dependencies ?? {}),
      ...(entry.manifest.optionalDependencies ?? {}),
    };
    const architectureDependencies = {
      ...runtimeDependencies,
      ...(entry.manifest.peerDependencies ?? {}),
      ...(entry.manifest.devDependencies ?? {}),
    };
    const localDependencies = Object.keys(runtimeDependencies).filter((dependency) => byName.has(dependency));
    graph.set(name, Object.keys(architectureDependencies).filter((dependency) => byName.has(dependency)));
    if (entry.manifest.private === true) continue;
    for (const dependency of localDependencies) {
      const target = byName.get(dependency);
      if (target?.manifest.private === true) {
        failures.push(`${relative(root, entry.file)} is public but depends on private workspace package ${dependency}`);
      }
    }
  }

  for (const cycle of dependencyCycles(graph)) failures.push(`workspace package dependency cycle: ${cycle.join(" -> ")}`);
}

function checkWorkspaceImports(root, ts, failures) {
  const packagesRoot = path.join(root, "packages");
  if (!fs.existsSync(packagesRoot)) return;
  for (const directory of fs.readdirSync(packagesRoot, { withFileTypes: true }).filter((entry) => entry.isDirectory())) {
    const packageRoot = path.join(packagesRoot, directory.name);
    const packageJson = path.join(packageRoot, "package.json");
    const srcRoot = path.join(packageRoot, "src");
    if (!fs.existsSync(packageJson) || !fs.existsSync(srcRoot)) continue;
    const packageName = JSON.parse(fs.readFileSync(packageJson, "utf8")).name;
    if (typeof packageName !== "string") continue;
    for (const file of walk(srcRoot, (candidate) => /\.(?:ts|mts|js|mjs)$/u.test(candidate) && !candidate.endsWith(".generated.ts"))) {
      const source = fs.readFileSync(file, "utf8");
      const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true);
      for (const specifier of moduleSpecifiers(ts, sourceFile)) {
        const relativeFile = relative(root, file);
        failures.push(...checkWorkspaceImportSpecifier(packageName, relativeFile, specifier));
        if (specifier.startsWith(".")) {
          const target = path.resolve(path.dirname(file), specifier.replace(/\.(?:js|mjs)$/u, ""));
          const sourcePackage = workspacePackageDirectory(root, file);
          const targetPackage = workspacePackageDirectory(root, target);
          if (sourcePackage !== undefined && targetPackage !== undefined && sourcePackage !== targetPackage) {
            failures.push(`${relativeFile} uses relative cross-package import ${specifier}; use a package export`);
          }
        }
      }
    }
  }
}

function checkCuteStaticLayout(root, ts, freeze, failures) {
  const definitionFile = resolveManifestPath(root, freeze.definitionFile, failures);
  if (definitionFile === undefined) return;
  const source = fs.readFileSync(definitionFile, "utf8");
  failures.push(...checkFrozenCuteStaticLayoutSource(ts, source, freeze, definitionFile));
  const permittedImporters = new Set(freeze.permittedProductionImporters ?? []);
  const compilerSrc = path.join(root, "packages/browsergrad-compiler/src");
  for (const file of walk(compilerSrc, (candidate) => candidate.endsWith(".ts") && !candidate.endsWith(".generated.ts"))) {
    const text = fs.readFileSync(file, "utf8");
    const parsed = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
    for (const statement of parsed.statements) {
      if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier)) continue;
      if (!/(?:^|\/)cute_static_layout\.js$/u.test(statement.moduleSpecifier.text)) continue;
      const importer = relative(root, file);
      if (!permittedImporters.has(importer)) failures.push(`${importer} is a new production cute_static_layout caller`);
    }
  }

  const parserFile = path.join(root, "packages/browsergrad-compiler/src/parser.ts");
  const parserSource = fs.readFileSync(parserFile, "utf8");
  for (const [token, maximum] of Object.entries(freeze.maxParserTokenCounts ?? {})) {
    const count = countToken(parserSource, token);
    if (count > maximum) failures.push(`packages/browsergrad-compiler/src/parser.ts uses ${token} ${count} times; frozen maximum is ${maximum}`);
  }
}

function checkCuteSourceNormalizers(root, freeze, failures) {
  const scriptsRoot = path.join(root, "scripts");
  const expectedExceptionFiles = [...(freeze.exceptionFiles ?? [])].sort();
  const actualExceptionFiles = walk(
    scriptsRoot,
    (candidate) => /^cuda-lite-source-normalizer-(?:cute-.+|wgmma)\.mjs$/u.test(path.basename(candidate)),
  ).map((file) => relative(root, file)).sort();
  compareStringSets(
    "CuTe source-normalizer exception files",
    actualExceptionFiles,
    expectedExceptionFiles,
    failures,
  );

  const sources = {};
  for (const file of Object.keys(freeze.files ?? {})) {
    const resolved = resolveManifestFile(root, file, "files", failures);
    if (resolved !== undefined) sources[file] = fs.readFileSync(resolved, "utf8");
  }
  failures.push(...checkFrozenCuteSourceNormalizerFiles(sources, freeze));
}

export function checkFrozenCuteStaticLayoutSource(ts, source, freeze, filename = "cute_static_layout.ts") {
  const failures = [];
  const sourceFile = ts.createSourceFile(filename, source, ts.ScriptTarget.Latest, true);
  const exportedNames = [];
  let layoutFields = [];
  let layoutKind;
  let queries = [];
  let layoutDeclarations = 0;
  let queryDeclarations = 0;

  for (const statement of sourceFile.statements) {
    const exported = statement.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword) === true;
    const isDefault = statement.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.DefaultKeyword) === true;
    if (isDefault) failures.push("cute_static_layout must not add a default export");
    if (exported && statement.name?.text) exportedNames.push(statement.name.text);
    if (exported && ts.isVariableStatement(statement)) {
      for (const declaration of statement.declarationList.declarations) {
        if (ts.isIdentifier(declaration.name)) exportedNames.push(declaration.name.text);
      }
    }
    if (ts.isExportDeclaration(statement)) failures.push("cute_static_layout must not add export-list or re-export declarations");
    if (ts.isInterfaceDeclaration(statement) && statement.name.text === "CuteStaticRank1Layout") {
      layoutDeclarations += 1;
      if ((statement.heritageClauses?.length ?? 0) > 0) failures.push("CuteStaticRank1Layout must not extend another interface");
      if (statement.members.some((member) => !ts.isPropertySignature(member))) failures.push("CuteStaticRank1Layout may contain property signatures only");
      if (statement.members.some((member) => ts.isPropertySignature(member) && member.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ReadonlyKeyword) !== true)) {
        failures.push("CuteStaticRank1Layout properties must remain readonly");
      }
      layoutFields = statement.members.filter(ts.isPropertySignature).map((member) => member.name.getText(sourceFile)).sort();
      const kindProperty = statement.members.find((member) => ts.isPropertySignature(member) && member.name.getText(sourceFile) === "kind");
      if (kindProperty?.type && ts.isLiteralTypeNode(kindProperty.type) && ts.isStringLiteral(kindProperty.type.literal)) layoutKind = kindProperty.type.literal.text;
    }
    if (ts.isTypeAliasDeclaration(statement) && statement.name.text === "CuteStaticLayoutQuery") {
      queryDeclarations += 1;
      if (!ts.isUnionTypeNode(statement.type) || statement.type.types.some((type) => !ts.isLiteralTypeNode(type) || !ts.isStringLiteral(type.literal))) {
        failures.push("CuteStaticLayoutQuery must remain a closed string-literal union");
      } else {
        queries = statement.type.types.map((type) => type.literal.text).sort();
      }
    }
  }

  if (layoutDeclarations !== 1) failures.push(`CuteStaticRank1Layout must have exactly one declaration; got ${layoutDeclarations}`);
  if (queryDeclarations !== 1) failures.push(`CuteStaticLayoutQuery must have exactly one declaration; got ${queryDeclarations}`);

  compareStringSets("cute_static_layout exports", exportedNames, freeze.exports, failures);
  compareStringSets("cute_static_layout fields", layoutFields, freeze.layoutFields, failures);
  compareStringSets("cute_static_layout queries", queries, freeze.queries, failures);
  if (layoutKind !== freeze.layoutKind) failures.push(`cute_static_layout kind changed: expected ${stringValue(freeze.layoutKind)}, got ${stringValue(layoutKind)}`);

  return failures;
}

function checkTensorGpuPlan(root, ts, freeze, failures) {
  const definitionFile = resolveManifestPath(root, freeze.definitionFile, failures);
  if (definitionFile === undefined) return;
  const source = fs.readFileSync(definitionFile, "utf8");
  failures.push(...checkFrozenTensorGpuPlanSource(ts, source, freeze, definitionFile));
}

function checkCompilerPointerScalarMemory(root, ts, freeze, failures) {
  const definitionFile = resolveManifestPath(root, freeze.definitionFile, failures);
  const publicBarrel = resolveManifestFile(root, freeze.publicBarrel, "publicBarrel", failures);
  const behaviorFixture = resolveManifestFile(root, freeze.behaviorFixtureFile, "behaviorFixtureFile", failures);
  if (definitionFile === undefined || publicBarrel === undefined || behaviorFixture === undefined) return;

  failures.push(...checkFrozenCompilerPointerScalarMemorySource(
    ts,
    fs.readFileSync(definitionFile, "utf8"),
    fs.readFileSync(publicBarrel, "utf8"),
    freeze,
    definitionFile,
    publicBarrel,
  ));

  const fixture = readJson(behaviorFixture, failures);
  failures.push(...validateCompilerPointerBehaviorFixture(fixture, freeze, relative(root, behaviorFixture)));
}

function checkExactInterfaceShapes(ts, sourceFile, declarations, expectedInterfaces, label, failures) {
  for (const [name, expected] of Object.entries(expectedInterfaces ?? {})) {
    const matches = (declarations.get(name) ?? []).filter(ts.isInterfaceDeclaration);
    if (matches.length !== 1) {
      failures.push(`${label} ${name} must have exactly one declaration; got ${matches.length}`);
      continue;
    }
    const declaration = matches[0];
    if ((declaration.heritageClauses?.length ?? 0) > 0) failures.push(`${label} ${name} must not extend another interface`);
    const actual = exactPropertyShape(ts, sourceFile, declaration.members, `${label} ${name}`, failures);
    compareExactShape(`${label} ${name}`, actual, expected, failures);
  }
}

function checkExactTaggedUnionShapes(ts, sourceFile, declarations, aliasName, expectedVariants, label, failures) {
  const aliases = (declarations.get(aliasName) ?? []).filter(ts.isTypeAliasDeclaration);
  if (aliases.length !== 1) {
    failures.push(`${label} owner ${aliasName} must have exactly one declaration; got ${aliases.length}`);
    return;
  }
  const type = aliases[0].type;
  if (!ts.isUnionTypeNode(type)) {
    failures.push(`${label} owner ${aliasName} must remain a union`);
    return;
  }
  const byKind = new Map();
  for (const member of type.types) {
    if (!ts.isTypeLiteralNode(member)) continue;
    const kind = member.members.find((entry) => ts.isPropertySignature(entry) && entry.name.getText(sourceFile) === "kind");
    if (!kind || !ts.isPropertySignature(kind) || kind.type === undefined || !ts.isLiteralTypeNode(kind.type) || !ts.isStringLiteral(kind.type.literal)) continue;
    byKind.set(kind.type.literal.text, [...(byKind.get(kind.type.literal.text) ?? []), member]);
  }
  for (const [kind, expected] of Object.entries(expectedVariants ?? {})) {
    const matches = byKind.get(kind) ?? [];
    if (matches.length !== 1) {
      failures.push(`${label} ${kind} must have exactly one variant; got ${matches.length}`);
      continue;
    }
    const actual = exactPropertyShape(ts, sourceFile, matches[0].members, `${label} ${kind}`, failures);
    compareExactShape(`${label} ${kind}`, actual, expected, failures);
  }
}

function exactPropertyShape(ts, sourceFile, members, label, failures) {
  const properties = {};
  for (const member of members) {
    if (!ts.isPropertySignature(member) || member.type === undefined) {
      failures.push(`${label} may contain property signatures only`);
      continue;
    }
    const property = member.name.getText(sourceFile);
    if (member.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ReadonlyKeyword) !== true) {
      failures.push(`${label}.${property} must remain readonly`);
    }
    const name = `${property}${member.questionToken === undefined ? "" : "?"}`;
    properties[name] = normalizeTypeText(member.type.getText(sourceFile));
  }
  return properties;
}

function compareExactShape(label, actual, expected, failures) {
  const normalizedExpected = Object.fromEntries(Object.entries(expected ?? {}).map(([key, value]) => [key, normalizeTypeText(value)]));
  if (JSON.stringify(sortRecord(actual)) !== JSON.stringify(sortRecord(normalizedExpected))) {
    failures.push(`${label} changed; compatibility schema is frozen`);
  }
}

export function checkFrozenTensorGpuPlanSource(ts, source, freeze, filename = "tensor_plan.ts") {
  const failures = [];
  const sourceFile = ts.createSourceFile(filename, source, ts.ScriptTarget.Latest, true);
  let operations = [];
  const interfaces = {};
  let operationDeclarations = 0;
  const interfaceDeclarations = {};

  for (const statement of sourceFile.statements) {
    if (ts.isTypeAliasDeclaration(statement) && statement.name.text === "TensorPlanOp") {
      operationDeclarations += 1;
      if (!ts.isUnionTypeNode(statement.type) || statement.type.types.some((type) => !ts.isLiteralTypeNode(type) || !ts.isStringLiteral(type.literal))) {
        failures.push("TensorPlanOp must remain a closed string-literal union");
      } else {
        operations = statement.type.types.map((type) => type.literal.text).sort();
      }
    }
    if (ts.isInterfaceDeclaration(statement) && isRecord(freeze.interfaces) && Object.hasOwn(freeze.interfaces, statement.name.text)) {
      interfaceDeclarations[statement.name.text] = (interfaceDeclarations[statement.name.text] ?? 0) + 1;
      if ((statement.heritageClauses?.length ?? 0) > 0) failures.push(`tensor plan interface ${statement.name.text} must not extend another interface`);
      if (statement.members.some((member) => !ts.isPropertySignature(member))) failures.push(`tensor plan interface ${statement.name.text} may contain property signatures only`);
      const properties = {};
      for (const member of statement.members) {
        if (!ts.isPropertySignature(member) || member.type === undefined) continue;
        if (member.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ReadonlyKeyword) !== true) {
          failures.push(`tensor plan interface ${statement.name.text}.${member.name.getText(sourceFile)} must remain readonly`);
        }
        const name = `${member.name.getText(sourceFile)}${member.questionToken === undefined ? "" : "?"}`;
        properties[name] = normalizeTypeText(member.type.getText(sourceFile));
      }
      interfaces[statement.name.text] = properties;
    }
  }

  if (operationDeclarations !== 1) failures.push(`TensorPlanOp must have exactly one declaration; got ${operationDeclarations}`);

  compareStringSets("TensorPlanOp operations", operations, freeze.operations, failures);
  for (const [name, expected] of Object.entries(freeze.interfaces ?? {})) {
    if (interfaceDeclarations[name] !== 1) failures.push(`tensor plan interface ${name} must have exactly one declaration; got ${interfaceDeclarations[name] ?? 0}`);
    const actual = interfaces[name];
    if (!isRecord(actual)) {
      failures.push(`tensor plan interface ${name} is missing`);
      continue;
    }
    const normalizedExpected = Object.fromEntries(Object.entries(expected).map(([key, value]) => [key, normalizeTypeText(value)]));
    if (JSON.stringify(sortRecord(actual)) !== JSON.stringify(sortRecord(normalizedExpected))) {
      failures.push(`tensor plan interface ${name} changed; compatibility plan schema is frozen`);
    }
  }
  return failures;
}

function checkJitCustomOps(root, ts, freeze, failures) {
  const inventoryFile = resolveManifestFile(root, freeze.inventoryFile, "inventoryFile", failures);
  const behaviorFixtureFile = resolveManifestFile(root, freeze.behaviorFixtureFile, "behaviorFixtureFile", failures);
  const behaviorTestFile = resolveManifestFile(root, freeze.behaviorTestFile, "behaviorTestFile", failures);
  if ([inventoryFile, behaviorFixtureFile, behaviorTestFile].some((value) => value === undefined)) return;

  const expectedCounts = freeze.constructorCounts ?? {};
  const actualCounts = {};
  const labels = new Set();
  const aliases = new Set();
  const labelFieldsByFile = {};
  const pythonRoot = path.join(root, "packages/browsergrad-jit/src/python");

  for (const file of walk(pythonRoot, (candidate) => candidate.endsWith(".py"))) {
    const source = fs.readFileSync(file, "utf8");
    const facts = pythonCallFacts(source);
    if (facts.customConstructors > 0) {
      const rel = relative(root, file);
      actualCounts[rel] = facts.customConstructors;
      labelFieldsByFile[rel] = customLabelFieldsRecord(facts);
    }
    for (const label of facts.labels) labels.add(label);
    for (const alias of facts.aliases) aliases.add(`${relative(root, file)}:${alias}`);
  }

  const indexFile = path.join(pythonRoot, "index.ts");
  const indexSource = fs.readFileSync(indexFile, "utf8");
  const indexFacts = pythonCallFacts(indexSource);
  if (indexFacts.customConstructors > 0) {
    const rel = relative(root, indexFile);
    actualCounts[rel] = indexFacts.customConstructors;
    labelFieldsByFile[rel] = customLabelFieldsRecord(indexFacts);
  }
  for (const label of indexFacts.labels) labels.add(label);
  for (const alias of indexFacts.aliases) aliases.add(`${relative(root, indexFile)}:${alias}`);

  if (JSON.stringify(sortRecord(actualCounts)) !== JSON.stringify(sortRecord(expectedCounts))) {
    failures.push(`JIT OP_CUSTOM constructor sites changed; expected ${JSON.stringify(sortRecord(expectedCounts))}, got ${JSON.stringify(sortRecord(actualCounts))}`);
  }

  for (const [file, tokenCounts] of Object.entries(freeze.exactTokenCounts ?? {})) {
    const source = fs.readFileSync(path.join(root, file), "utf8");
    const tokens = pythonTokens(source);
    for (const [token, expected] of Object.entries(tokenCounts)) {
      const count = tokens.filter((entry) => entry.kind === "identifier" && entry.value === token).length;
      if (count !== expected) failures.push(`${file} uses ${token} ${count} times; frozen exact count is ${expected}`);
    }
  }

  for (const alias of [...aliases].sort()) failures.push(`JIT OP_CUSTOM alias is forbidden: ${alias}`);

  compareStringSets("JIT OP_CUSTOM executable labels", labels, freeze.labels, failures);
  if (JSON.stringify(sortRecord(labelFieldsByFile)) !== JSON.stringify(sortRecord(freeze.labelFieldsByFile ?? {}))) {
    failures.push(`JIT OP_CUSTOM label fields changed; expected ${JSON.stringify(sortRecord(freeze.labelFieldsByFile ?? {}))}, got ${JSON.stringify(sortRecord(labelFieldsByFile))}`);
  }

  for (const [file, expectedDefinitions] of Object.entries(freeze.definitionTokenDigests ?? {})) {
    const source = fs.readFileSync(path.join(root, file), "utf8");
    const actualDefinitions = extractPythonDefinitionTokenDigests(source);
    const selected = {};
    for (const definition of Object.keys(expectedDefinitions ?? {})) {
      if (actualDefinitions[definition] !== undefined) selected[definition] = actualDefinitions[definition];
    }
    if (JSON.stringify(sortRecord(selected)) !== JSON.stringify(sortRecord(expectedDefinitions ?? {}))) {
      failures.push(`${file} JIT opaque-operation definitions changed`);
    }
  }

  const inventory = readJson(inventoryFile, failures);
  const fixture = readJson(behaviorFixtureFile, failures);
  failures.push(...validateJitOpaqueOperationInventory(inventory, fixture, freeze, relative(root, inventoryFile)));

  if (isRecord(inventory) && Array.isArray(inventory.evidenceDefinitions)) {
    for (const evidence of inventory.evidenceDefinitions) {
      if (!isRecord(evidence) || typeof evidence.path !== "string") continue;
      resolveManifestFile(root, evidence.path, `JIT evidence ${stringValue(evidence.id)}`, failures);
    }
  }
  for (const [file, expected] of [
    [inventoryFile, freeze.inventorySha256],
    [behaviorFixtureFile, freeze.behaviorFixtureSha256],
    [behaviorTestFile, freeze.behaviorTestSha256],
  ]) {
    const actual = createHash("sha256").update(fs.readFileSync(file)).digest("hex");
    if (actual !== expected) failures.push(`${relative(root, file)} content changed; expected SHA-256 ${stringValue(expected)}, got ${actual}`);
  }
}

function customLabelFieldsRecord(facts) {
  return {
    name: [...facts.labelFields.name].sort(),
    op: [...facts.labelFields.op].sort(),
    dynamicName: [...facts.dynamicLabelFields.name].sort(),
    dynamicOp: [...facts.dynamicLabelFields.op].sort(),
  };
}

function checkGradViewBf16(root, freeze, failures) {
  const tensorFile = resolveManifestFile(root, freeze.tensorFile, "tensorFile", failures);
  const torchCompatFile = resolveManifestFile(root, freeze.torchCompatFile, "torchCompatFile", failures);
  const inventoryFile = resolveManifestFile(root, freeze.inventoryFile, "inventoryFile", failures);
  const behaviorFixtureFile = resolveManifestFile(root, freeze.behaviorFixtureFile, "behaviorFixtureFile", failures);
  const behaviorTestFile = resolveManifestFile(root, freeze.behaviorTestFile, "behaviorTestFile", failures);
  if ([tensorFile, torchCompatFile, inventoryFile, behaviorFixtureFile, behaviorTestFile].some((value) => value === undefined)) return;

  failures.push(...checkFrozenGradCompatibilitySources(
    fs.readFileSync(tensorFile, "utf8"),
    fs.readFileSync(torchCompatFile, "utf8"),
    freeze,
  ));
  const inventory = readJson(inventoryFile, failures);
  const fixture = readJson(behaviorFixtureFile, failures);
  failures.push(...validateGradCompatibilityInventory(inventory, fixture, freeze, relative(root, inventoryFile)));

  for (const [file, expected] of [
    [inventoryFile, freeze.inventorySha256],
    [behaviorFixtureFile, freeze.behaviorFixtureSha256],
    [behaviorTestFile, freeze.behaviorTestSha256],
  ]) {
    const actual = createHash("sha256").update(fs.readFileSync(file)).digest("hex");
    if (actual !== expected) failures.push(`${relative(root, file)} content changed; expected SHA-256 ${stringValue(expected)}, got ${actual}`);
  }
}

function checkRuntimeAssignmentRequirements(root, ts, freeze, failures) {
  const definitionFile = resolveManifestPath(root, freeze.definitionFile, failures);
  const typesFile = resolveManifestFile(root, freeze.typesFile, "typesFile", failures);
  const vocabularyFile = resolveManifestFile(root, freeze.vocabularyFile, "vocabularyFile", failures);
  const behaviorFixture = resolveManifestFile(root, freeze.behaviorFixtureFile, "behaviorFixtureFile", failures);
  const usageInventoryFile = resolveManifestFile(root, freeze.usageInventoryFile, "usageInventoryFile", failures);
  const profileDirectory = resolveManifestFile(root, freeze.profileDirectory, "profileDirectory", failures);
  if ([definitionFile, typesFile, vocabularyFile, behaviorFixture, usageInventoryFile, profileDirectory].some((value) => value === undefined)) return;

  failures.push(...checkFrozenRuntimeAssignmentRequirementsSource(
    ts,
    fs.readFileSync(definitionFile, "utf8"),
    fs.readFileSync(typesFile, "utf8"),
    freeze,
    definitionFile,
    typesFile,
  ));

  const fixture = readJson(behaviorFixture, failures);
  validateNamedFixture(
    fixture,
    "runtime.generic-backend-labels.v0",
    freeze.behaviorFixtureIds,
    relative(root, behaviorFixture),
    "runtime assignment requirement",
    failures,
  );
  const fixtureSha256 = createHash("sha256").update(fs.readFileSync(behaviorFixture)).digest("hex");
  if (fixtureSha256 !== freeze.behaviorFixtureSha256) {
    failures.push(`${relative(root, behaviorFixture)} content changed; expected SHA-256 ${stringValue(freeze.behaviorFixtureSha256)}, got ${fixtureSha256}`);
  }

  const usage = assignmentRequirementUsage(profileDirectory, freeze.profileSuffix, failures, root);
  const recordedUsage = readJson(usageInventoryFile, failures);
  if (JSON.stringify(recordedUsage) !== JSON.stringify(usage)) {
    failures.push(`${relative(root, usageInventoryFile)} is stale; run pnpm architecture:generate-requirements`);
  }
  const profileIds = usage.requirements.map((entry) => entry.requirementId);
  const vocabulary = readJson(vocabularyFile, failures);
  validatePlatformVocabulary(root, vocabulary, profileIds, freeze.browserMappings, failures);
}

function pushedStringLiteral(ts, statement) {
  const candidate = ts.isBlock(statement) && statement.statements.length === 1 ? statement.statements[0] : statement;
  if (!ts.isExpressionStatement(candidate) || !ts.isCallExpression(candidate.expression)) return undefined;
  const call = candidate.expression;
  if (!ts.isPropertyAccessExpression(call.expression) || call.expression.expression.getText() !== "capabilities" || call.expression.name.text !== "push") return undefined;
  return call.arguments.length === 1 && ts.isStringLiteral(call.arguments[0]) ? call.arguments[0].text : undefined;
}

function checkClosedStringUnion(ts, sourceFile, name, expected, failures) {
  const declarations = sourceFile.statements.filter((statement) => ts.isTypeAliasDeclaration(statement) && statement.name.text === name);
  if (declarations.length !== 1) {
    failures.push(`${name} must have exactly one type alias declaration; got ${declarations.length}`);
    return;
  }
  const type = declarations[0].type;
  if (!ts.isUnionTypeNode(type) || type.types.some((entry) => !ts.isLiteralTypeNode(entry) || !ts.isStringLiteral(entry.literal))) {
    failures.push(`${name} must remain a closed string-literal union`);
    return;
  }
  compareStringSets(`${name} values`, type.types.map((entry) => entry.literal.text), expected, failures);
}

function compareRecordLists(label, actual, expected, key, failures) {
  const normalize = (values) => [...(values ?? [])]
    .map((entry) => sortRecord(Object.fromEntries(Object.entries(entry).map(([name, value]) => [name, typeof value === "string" ? normalizeTypeText(value) : value]))))
    .sort((left, right) => String(left[key]).localeCompare(String(right[key])));
  const left = normalize(actual);
  const right = normalize(expected);
  if (JSON.stringify(left) !== JSON.stringify(right)) failures.push(`${label} changed; expected ${JSON.stringify(right)}, got ${JSON.stringify(left)}`);
}

function validateNamedFixture(fixture, adapterId, expectedIds, filename, label, failures) {
  if (!isRecord(fixture) || fixture.schemaVersion !== 1 || fixture.adapterId !== adapterId || !Array.isArray(fixture.cases)) {
    failures.push(`${filename} must be a schemaVersion 1 ${adapterId} fixture`);
    return;
  }
  const ids = fixture.cases.map((entry) => isRecord(entry) ? entry.id : undefined);
  if (ids.some((id) => typeof id !== "string") || new Set(ids).size !== ids.length) {
    failures.push(`${filename} case IDs must be unique strings`);
    return;
  }
  compareStringSets(`${label} behavior fixture IDs`, ids, expectedIds, failures);
}

function assignmentRequirementUsage(directory, suffix, failures, root) {
  if (typeof suffix !== "string" || suffix.length === 0) {
    failures.push("runtime assignment requirement profileSuffix must be a non-empty string");
    return { schemaVersion: 1, sourcePattern: `${relative(root, directory)}/*`, requirements: [] };
  }
  const usages = new Map();
  const add = (requirementId, usage) => usages.set(requirementId, [...(usages.get(requirementId) ?? []), usage]);
  for (const entry of fs.readdirSync(directory, { withFileTypes: true }).filter((candidate) => candidate.isFile() && candidate.name.endsWith(suffix))) {
    const file = path.join(directory, entry.name);
    const profile = readJson(file, failures);
    if (!isRecord(profile) || !Array.isArray(profile.gates)) continue;
    const profileId = typeof profile.id === "string" ? profile.id : entry.name.slice(0, -suffix.length);
    for (const gate of profile.gates) {
      if (!isRecord(gate) || gate.kind !== "capability" || !isRecord(gate.options)) continue;
      const gateName = typeof gate.name === "string" ? gate.name : "<unnamed>";
      const requires = gate.options.requires;
      if (requires !== undefined && !Array.isArray(requires)) failures.push(`${relative(root, file)} capability requires must be an array`);
      for (const value of Array.isArray(requires) ? requires : []) {
        if (typeof value === "string") add(value, { profileId, sourcePath: relative(root, file), gate: gateName, relation: "requires" });
        else failures.push(`${relative(root, file)} capability requirement must be a string`);
      }
      const alternatives = gate.options.any_of;
      if (alternatives !== undefined && !Array.isArray(alternatives)) failures.push(`${relative(root, file)} capability any_of must be an array`);
      for (const group of Array.isArray(alternatives) ? alternatives : []) {
        if (!Array.isArray(group)) {
          failures.push(`${relative(root, file)} capability alternative must be an array`);
          continue;
        }
        for (const value of group) {
          if (typeof value === "string") add(value, { profileId, sourcePath: relative(root, file), gate: gateName, relation: "any-of", group });
          else failures.push(`${relative(root, file)} capability alternative requirement must be a string`);
        }
      }
    }
  }
  return {
    schemaVersion: 1,
    sourcePattern: `${relative(root, directory)}/*${suffix}`,
    requirements: [...usages.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([requirementId, entries]) => ({
      requirementId,
      usages: entries.sort((left, right) =>
        left.sourcePath.localeCompare(right.sourcePath) || left.gate.localeCompare(right.gate) || left.relation.localeCompare(right.relation) ||
        JSON.stringify(left.group ?? []).localeCompare(JSON.stringify(right.group ?? []))),
    })),
  };
}

function validatePlatformVocabulary(root, vocabulary, profileIds, browserMappings, failures) {
  const filename = "architecture/platform-vocabulary.json";
  if (!isRecord(vocabulary) || vocabulary.schemaVersion !== 1) {
    failures.push(`${filename} must have schemaVersion 1`);
    return;
  }
  checkExactRecordKeys(filename, "top-level", vocabulary, [
    "schemaVersion", "identifierPolicies", "diagnosticStages", "requirementKinds",
    "semanticCapabilities", "backends", "legacyAssignmentRequirements",
  ], failures);
  compareStringSets(`${filename} diagnostic stages`, vocabulary.diagnosticStages, [
    "preprocess", "frontend", "semantic-lowering", "verification", "scheduling",
    "backend-lowering", "device-validation", "execution", "evidence",
  ], failures);
  compareStringSets(`${filename} requirement kinds`, vocabulary.requirementKinds, [
    "semantic-feature", "runtime-facility", "device-feature", "oracle", "simulator", "fixture", "external-service", "policy",
  ], failures);
  if (!isRecord(vocabulary.identifierPolicies)) {
    failures.push(`${filename} identifierPolicies are required`);
    return;
  }
  checkExactRecordKeys(filename, "identifierPolicies", vocabulary.identifierPolicies, ["canonical", "legacyAssignment"], failures);
  if (vocabulary.identifierPolicies.canonical !== "^[a-z][a-z0-9]*(?:\\.[a-z][a-z0-9-]*)+$") {
    failures.push(`${filename} canonical identifier policy changed`);
  }
  if (vocabulary.identifierPolicies.legacyAssignment !== "^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$") {
    failures.push(`${filename} legacy assignment identifier policy changed`);
  }
  const canonical = safeRegex(vocabulary.identifierPolicies.canonical, `${filename} canonical identifier policy`, failures);
  const legacy = safeRegex(vocabulary.identifierPolicies.legacyAssignment, `${filename} legacy assignment identifier policy`, failures);
  const capabilities = validateVocabularyRecords(root, vocabulary.semanticCapabilities, "capabilityId", canonical, filename, failures);
  const backends = validateVocabularyRecords(root, vocabulary.backends, "backendId", canonical, filename, failures);
  const requirements = validateVocabularyRecords(root, vocabulary.legacyAssignmentRequirements, "requirementId", legacy, filename, failures);

  const allowedPreservation = new Set(["observable-equivalent", "portable-relegalized", "schedule-preserving", "native-facility"]);
  for (const entry of vocabulary.semanticCapabilities ?? []) {
    if (!isRecord(entry)) continue;
    checkExactRecordKeys(filename, `semantic capability ${stringValue(entry.capabilityId)}`, entry, [
      "capabilityId", "semanticVersion", "operationVersion", "preservationLevels", "owner", "evidence",
    ], failures);
    if (!Array.isArray(entry.preservationLevels) || entry.preservationLevels.length === 0 || entry.preservationLevels.some((value) => !allowedPreservation.has(value))) {
      failures.push(`${filename} semantic capability ${stringValue(entry.capabilityId)} has invalid preservationLevels`);
    }
    if (Object.hasOwn(entry, "state") || Object.hasOwn(entry, "outcome") || Object.hasOwn(entry, "passed")) {
      failures.push(`${filename} static semantic capability ${stringValue(entry.capabilityId)} contains runtime/evidence state`);
    }
    if (typeof entry.operationVersion !== "string" || entry.operationVersion.trim() === "" || !Array.isArray(entry.evidence) || entry.evidence.length === 0) {
      failures.push(`${filename} semantic capability ${stringValue(entry.capabilityId)} requires operationVersion and evidence`);
    }
  }
  const allowedExecutionTiers = new Set(["semantic-reference", "webgpu-core", "webgpu-enhanced", "native-companion", "simulation"]);
  for (const entry of vocabulary.backends ?? []) {
    if (!isRecord(entry)) continue;
    checkExactRecordKeys(filename, `backend ${stringValue(entry.backendId)}`, entry, [
      "backendId", "semanticVersion", "owner", "executionTiers", "evidence",
    ], failures);
    if (!Array.isArray(entry.executionTiers) || entry.executionTiers.length === 0 || entry.executionTiers.some((value) => !allowedExecutionTiers.has(value))) {
      failures.push(`${filename} backend ${stringValue(entry.backendId)} has invalid executionTiers`);
    }
    if (!Array.isArray(entry.evidence) || entry.evidence.length === 0) failures.push(`${filename} backend ${stringValue(entry.backendId)} requires evidence`);
  }
  const kinds = new Set(vocabulary.requirementKinds ?? []);
  for (const entry of vocabulary.legacyAssignmentRequirements ?? []) {
    if (!isRecord(entry)) continue;
    checkExactRecordKeys(filename, `legacy requirement ${stringValue(entry.requirementId)}`, entry, [
      "requirementId", "semanticVersion", "kind", "owner", "lifecycle", "meaning", "capabilityId?",
    ], failures);
    if (!kinds.has(entry.kind) || entry.lifecycle !== "legacy" || typeof entry.meaning !== "string" || entry.meaning.trim() === "") {
      failures.push(`${filename} legacy requirement ${stringValue(entry.requirementId)} has invalid classification/lifecycle/meaning`);
    }
    if (entry.capabilityId !== undefined) {
      if (entry.kind !== "semantic-feature") failures.push(`${filename} legacy requirement ${stringValue(entry.requirementId)} may link a capability only when kind is semantic-feature`);
      if (typeof entry.capabilityId !== "string" || !capabilities.includes(entry.capabilityId)) failures.push(`${filename} legacy requirement ${stringValue(entry.requirementId)} links unknown capability ${stringValue(entry.capabilityId)}`);
    }
  }
  const browserIds = (browserMappings ?? []).map((entry) => entry.requirementId);
  compareStringSets(`${filename} registered legacy assignment requirements`, requirements, [...new Set([...profileIds, ...browserIds])], failures);
  const collisions = capabilities.filter((id) => requirements.includes(id));
  if (collisions.length > 0) failures.push(`${filename} capability and requirement IDs collide: ${collisions.join(", ")}`);
  if (backends.some((id) => capabilities.includes(id))) failures.push(`${filename} backend and capability IDs must use distinct namespaces`);
}

function checkExactRecordKeys(filename, label, record, fields, failures) {
  const required = fields.filter((field) => !field.endsWith("?"));
  const allowed = new Set(fields.map((field) => field.replace(/\?$/u, "")));
  const actual = Object.keys(record);
  const missing = required.filter((field) => !Object.hasOwn(record, field));
  const unknown = actual.filter((field) => !allowed.has(field));
  if (missing.length > 0 || unknown.length > 0) {
    failures.push(`${filename} ${label} record keys changed; missing ${JSON.stringify(missing)}, unknown ${JSON.stringify(unknown)}`);
  }
}

function validateVocabularyRecords(root, values, idField, pattern, filename, failures) {
  if (!Array.isArray(values)) {
    failures.push(`${filename} ${idField} records must be an array`);
    return [];
  }
  const ids = [];
  for (const entry of values) {
    if (!isRecord(entry) || typeof entry[idField] !== "string") {
      failures.push(`${filename} ${idField} record is invalid`);
      continue;
    }
    const id = entry[idField];
    ids.push(id);
    if (pattern !== undefined && !pattern.test(id)) failures.push(`${filename} ${idField} ${id} violates its identifier policy`);
    if (typeof entry.semanticVersion !== "string" || !/^\d+\.\d+\.\d+$/u.test(entry.semanticVersion)) failures.push(`${filename} ${id} must have an exact semanticVersion`);
    if (typeof entry.owner !== "string" || entry.owner.trim() === "") failures.push(`${filename} ${id} must have an owner`);
    if (entry.evidence !== undefined && !Array.isArray(entry.evidence)) failures.push(`${filename} ${id} evidence must be an array`);
    for (const evidence of Array.isArray(entry.evidence) ? entry.evidence : []) {
      if (typeof evidence !== "string" || !fs.existsSync(path.resolve(root, evidence))) failures.push(`${filename} ${id} evidence path is missing: ${stringValue(evidence)}`);
    }
  }
  if (new Set(ids).size !== ids.length) failures.push(`${filename} ${idField} values must be unique`);
  return ids.sort();
}

function safeRegex(value, label, failures) {
  if (typeof value !== "string") {
    failures.push(`${label} must be a string`);
    return undefined;
  }
  try {
    return new RegExp(value, "u");
  } catch (error) {
    failures.push(`${label} is invalid: ${errorMessage(error)}`);
    return undefined;
  }
}

function checkGeneratedPython(root, failures) {
  for (const packageName of ["browsergrad-grad", "browsergrad-jit"]) {
    const pythonRoot = path.join(root, "packages", packageName, "src/python");
    if (!fs.existsSync(pythonRoot)) continue;
    for (const generatedFile of walk(pythonRoot, (candidate) => candidate.endsWith(".generated.ts"))) {
      const generated = fs.readFileSync(generatedFile, "utf8");
      const sourceMarker = generated.match(/^\/\/ @generated[^\n]*Source: src\/python\/(.+)$/mu)?.[1];
      const base64 = generated.match(/const b64 = "([A-Za-z0-9+/=]+)";/u)?.[1];
      if (sourceMarker === undefined || base64 === undefined) {
        failures.push(`${relative(root, generatedFile)} has an unrecognized generated Python envelope`);
        continue;
      }
      const expected = readGeneratedPythonSource(pythonRoot, sourceMarker, failures, root);
      if (expected === undefined) continue;
      const actual = Buffer.from(base64, "base64").toString("utf8");
      if (actual !== expected) failures.push(`${relative(root, generatedFile)} is stale; run package codegen`);
    }
  }
}

function readGeneratedPythonSource(pythonRoot, marker, failures, root) {
  const chunked = marker.match(/^(.+)\/\{([^}]+)\}\.py$/u);
  if (chunked) {
    const [, directory, names] = chunked;
    const parts = names.split(",").map((name) => path.join(pythonRoot, directory, `${name}.py`));
    for (const file of parts) {
      if (!fs.existsSync(file)) {
        failures.push(`${relative(root, file)} referenced by generated bundle is missing`);
        return undefined;
      }
    }
    return parts.map((file) => fs.readFileSync(file, "utf8")).join("\n");
  }
  const sourceFile = path.join(pythonRoot, marker);
  if (!fs.existsSync(sourceFile)) {
    failures.push(`${relative(root, sourceFile)} referenced by generated bundle is missing`);
    return undefined;
  }
  return fs.readFileSync(sourceFile, "utf8");
}

function pythonCallFacts(source) {
  const tokens = pythonTokens(source);
  let customConstructors = 0;
  const labels = new Set();
  const aliases = new Set();
  const labelFields = { name: new Set(), op: new Set() };
  const dynamicLabelFields = { name: new Set(), op: new Set() };
  const dictLabelFacts = new Map();

  for (let index = 0; index < tokens.length - 2; index += 1) {
    if (tokens[index].kind !== "identifier" || tokens[index + 1]?.value !== "=") continue;
    const rightIndex = skipParentheses(tokens, index + 2);
    if (tokens[rightIndex]?.value === "OP_CUSTOM" && tokens[index].value !== "op") aliases.add(tokens[index].value);
    const previous = tokens[index - 1]?.value;
    if (tokens[index + 2]?.value === "{" && previous !== "(" && previous !== ",") {
      const close = matchingDelimiter(tokens, index + 2, "{", "}");
      if (close === undefined) continue;
      const facts = labelBindingsInTokens(tokens.slice(index + 3, close));
      if (facts.name.size > 0 || facts.op.size > 0 || facts.dynamicName.size > 0 || facts.dynamicOp.size > 0) {
        const combined = dictLabelFacts.get(tokens[index].value) ?? emptyLabelBindings();
        for (const field of ["name", "op", "dynamicName", "dynamicOp"]) {
          for (const value of facts[field]) combined[field].add(value);
        }
        dictLabelFacts.set(tokens[index].value, combined);
      }
    }
  }

  for (let index = 0; index < tokens.length - 1; index += 1) {
    const token = tokens[index];
    if (token.kind !== "identifier" || tokens[index + 1]?.value !== "(") continue;
    if (token.value === "UOp") {
      const close = matchingParen(tokens, index + 1);
      if (close === undefined) continue;
      const body = tokens.slice(index + 2, close);
      const opValue = uopOperationValue(body);
      const isCustom = opValue === "OP_CUSTOM" || (opValue !== undefined && aliases.has(opValue));
      if (isCustom) {
        customConstructors += 1;
        mergeLabelBindings(labelBindingsInTokens(body), labels, labelFields, dynamicLabelFields);
        for (let bodyIndex = 0; bodyIndex < body.length - 2; bodyIndex += 1) {
          if (body[bodyIndex].value !== "arg" || body[bodyIndex + 1]?.value !== "=") continue;
          const variable = body[bodyIndex + 2];
          if (variable?.kind !== "identifier") continue;
          const facts = dictLabelFacts.get(variable.value);
          if (facts !== undefined) mergeLabelBindings(facts, labels, labelFields, dynamicLabelFields);
        }
      }
      index = close;
      continue;
    }
    if (token.value === "_custom_elementwise_loss" && tokens[index - 1]?.value !== "def") {
      const close = matchingParen(tokens, index + 1);
      if (close === undefined) continue;
      let depth = 0;
      for (const entry of tokens.slice(index + 2, close)) {
        if (entry.value === "(" || entry.value === "[" || entry.value === "{") depth += 1;
        if (entry.value === ")" || entry.value === "]" || entry.value === "}") depth -= 1;
        if (depth === 0 && entry.kind === "string") {
          labels.add(entry.value);
          labelFields.name.add(entry.value);
        }
      }
      index = close;
    }
  }
  return { customConstructors, labels, aliases, labelFields, dynamicLabelFields };
}

function uopOperationValue(body) {
  let depth = 0;
  let firstTopLevel;
  for (let index = 0; index < body.length; index += 1) {
    const token = body[index];
    if (token.value === "(" || token.value === "[" || token.value === "{") depth += 1;
    if (token.value === ")" || token.value === "]" || token.value === "}") depth -= 1;
    if (depth !== 0) continue;
    if (firstTopLevel === undefined && token.value !== ",") firstTopLevel = token.value;
    if (token.value !== "op" || body[index + 1]?.value !== "=") continue;
    return body[skipParentheses(body, index + 2)]?.value;
  }
  return firstTopLevel;
}

function labelBindingsInTokens(tokens) {
  const facts = emptyLabelBindings();
  for (let index = 0; index < tokens.length - 2; index += 1) {
    const key = tokens[index];
    if (key.kind !== "string" || (key.value !== "name" && key.value !== "op") || tokens[index + 1]?.value !== ":") continue;
    const value = tokens[index + 2];
    const staticKey = key.value;
    const dynamicKey = key.value === "name" ? "dynamicName" : "dynamicOp";
    if (value?.kind === "string") facts[staticKey].add(value.value);
    else if (value?.kind === "identifier") facts[dynamicKey].add(value.value);
  }
  return facts;
}

function emptyLabelBindings() {
  return {
    name: new Set(),
    op: new Set(),
    dynamicName: new Set(),
    dynamicOp: new Set(),
  };
}

function mergeLabelBindings(facts, labels, labelFields, dynamicLabelFields) {
  for (const field of ["name", "op"]) {
    for (const value of facts[field]) {
      labels.add(value);
      labelFields[field].add(value);
    }
  }
  for (const value of facts.dynamicName) dynamicLabelFields.name.add(value);
  for (const value of facts.dynamicOp) dynamicLabelFields.op.add(value);
}

function skipParentheses(tokens, start) {
  let index = start;
  while (tokens[index]?.value === "(") index += 1;
  return index;
}

function pythonTokens(source) {
  const tokens = [];
  let index = 0;
  while (index < source.length) {
    const char = source[index];
    if (/\s/u.test(char)) {
      index += 1;
      continue;
    }
    if (char === "#") {
      while (index < source.length && source[index] !== "\n") index += 1;
      continue;
    }
    if (char === "'" || char === '"') {
      const parsed = readPythonString(source, index);
      tokens.push({ kind: "string", value: parsed.value });
      index = parsed.end;
      continue;
    }
    if (/[A-Za-z_]/u.test(char)) {
      const start = index;
      index += 1;
      while (index < source.length && /[A-Za-z0-9_]/u.test(source[index])) index += 1;
      tokens.push({ kind: "identifier", value: source.slice(start, index) });
      continue;
    }
    tokens.push({ kind: "punct", value: char });
    index += 1;
  }
  return tokens;
}

function pythonDefinitions(source) {
  const lines = source.split(/\r?\n/u);
  const definitions = [];
  const stack = [];
  const significant = (line) => line.trim() !== "" && !line.trimStart().startsWith("#");
  const indentation = (line) => {
    const prefix = line.match(/^[\t ]*/u)?.[0] ?? "";
    return [...prefix].reduce((total, char) => total + (char === "\t" ? 8 - (total % 8) : 1), 0);
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (!significant(line)) continue;
    const indent = indentation(line);
    while (stack.length > 0 && indent <= stack.at(-1).indent) stack.pop();
    const match = line.match(/^\s*(?:async\s+)?(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b/u);
    if (match === null) continue;
    const qualifiedName = [...stack.map((entry) => entry.name), match[2]].join(".");
    let end = lines.length;
    for (let candidate = index + 1; candidate < lines.length; candidate += 1) {
      if (!significant(lines[candidate])) continue;
      if (indentation(lines[candidate]) <= indent) {
        end = candidate;
        break;
      }
    }
    let start = index;
    while (start > 0 && indentation(lines[start - 1]) === indent && lines[start - 1].trimStart().startsWith("@")) start -= 1;
    definitions.push({
      qualifiedName,
      source: lines.slice(start, end).join("\n"),
    });
    stack.push({ indent, name: match[2] });
  }
  return definitions;
}

function normalizedPythonDefinitionTokens(source) {
  const tokens = pythonTokens(source);
  const defIndex = tokens.findIndex((token) => token.kind === "identifier" && token.value === "def");
  if (defIndex < 0) return tokens;
  let depth = 0;
  for (let index = defIndex + 1; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (token.value === "(" || token.value === "[" || token.value === "{") depth += 1;
    if (token.value === ")" || token.value === "]" || token.value === "}") depth -= 1;
    if (depth === 0 && token.value === ":" && tokens[index + 1]?.kind === "string") {
      return [...tokens.slice(0, index + 1), ...tokens.slice(index + 2)];
    }
  }
  return tokens;
}

function pythonAttributeStringAssignments(source, objectName) {
  const tokens = pythonTokens(source);
  const assignments = {};
  for (let index = 0; index < tokens.length - 4; index += 1) {
    if (tokens[index].kind !== "identifier" || tokens[index].value !== objectName || tokens[index + 1]?.value !== "." || tokens[index + 2]?.kind !== "identifier" || tokens[index + 3]?.value !== "=" || tokens[index + 4]?.kind !== "string") continue;
    assignments[tokens[index + 2].value] = tokens[index + 4].value;
  }
  return assignments;
}

function readPythonString(source, start) {
  const quote = source[start];
  const triple = source.slice(start, start + 3) === quote.repeat(3);
  let index = start + (triple ? 3 : 1);
  let value = "";
  while (index < source.length) {
    if (triple ? source.slice(index, index + 3) === quote.repeat(3) : source[index] === quote) {
      return { value, end: index + (triple ? 3 : 1) };
    }
    if (source[index] === "\\" && index + 1 < source.length) {
      value += source[index + 1];
      index += 2;
      continue;
    }
    value += source[index];
    index += 1;
  }
  return { value, end: source.length };
}

function matchingParen(tokens, openIndex) {
  let depth = 0;
  for (let index = openIndex; index < tokens.length; index += 1) {
    if (tokens[index].value === "(") depth += 1;
    if (tokens[index].value === ")") {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return undefined;
}

function matchingDelimiter(tokens, openIndex, open, close) {
  let depth = 0;
  for (let index = openIndex; index < tokens.length; index += 1) {
    if (tokens[index].value === open) depth += 1;
    if (tokens[index].value === close) {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return undefined;
}

function moduleSpecifiers(ts, sourceFile) {
  const specifiers = [];
  function visit(node) {
    if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier)) {
      specifiers.push(node.moduleSpecifier.text);
    } else if (ts.isExportDeclaration(node) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
      specifiers.push(node.moduleSpecifier.text);
    } else if (
      ts.isImportEqualsDeclaration(node) &&
      ts.isExternalModuleReference(node.moduleReference) &&
      node.moduleReference.expression &&
      ts.isStringLiteral(node.moduleReference.expression)
    ) {
      specifiers.push(node.moduleReference.expression.text);
    } else if (
      ts.isCallExpression(node) &&
      node.arguments.length > 0 &&
      ts.isStringLiteral(node.arguments[0]) &&
      (node.expression.kind === ts.SyntaxKind.ImportKeyword || (ts.isIdentifier(node.expression) && node.expression.text === "require"))
    ) {
      specifiers.push(node.arguments[0].text);
    }
    ts.forEachChild(node, visit);
  }
  visit(sourceFile);
  return [...new Set(specifiers)];
}

function workspacePackageDirectory(root, file) {
  const rel = relative(root, path.resolve(file));
  const parts = rel.split("/");
  return parts[0] === "packages" && parts[1] !== undefined ? parts[1] : undefined;
}

function dependencyCycles(graph) {
  const visiting = new Set();
  const visited = new Set();
  const stack = [];
  const cycles = new Set();
  function visit(node) {
    if (visiting.has(node)) {
      const start = stack.indexOf(node);
      const cycle = [...stack.slice(start), node];
      cycles.add(cycle.join(" -> "));
      return;
    }
    if (visited.has(node)) return;
    visiting.add(node);
    stack.push(node);
    for (const target of graph.get(node) ?? []) visit(target);
    stack.pop();
    visiting.delete(node);
    visited.add(node);
  }
  for (const node of [...graph.keys()].sort()) visit(node);
  return [...cycles].sort().map((cycle) => cycle.split(" -> "));
}

function compareStringSets(label, actual, expected, failures) {
  const left = [...(actual ?? [])].sort();
  const right = [...(expected ?? [])].sort();
  if (JSON.stringify(left) !== JSON.stringify(right)) failures.push(`${label} changed; expected ${JSON.stringify(right)}, got ${JSON.stringify(left)}`);
}

function walk(directory, predicate) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) return walk(full, predicate);
    return predicate(full) ? [full] : [];
  });
}

function resolveManifestPath(root, value, failures) {
  return resolveManifestFile(root, value, "definitionFile", failures);
}

function resolveManifestFile(root, value, field, failures) {
  if (typeof value !== "string") {
    failures.push(`freeze ${field} must be a string`);
    return undefined;
  }
  const file = path.resolve(root, value);
  if (!file.startsWith(`${path.resolve(root)}${path.sep}`) || !fs.existsSync(file)) {
    failures.push(`freeze ${field} is missing or outside the repository: ${value}`);
    return undefined;
  }
  return file;
}

function readJson(file, failures) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    failures.push(`${relative(path.dirname(path.dirname(file)), file)} cannot be read: ${errorMessage(error)}`);
    return undefined;
  }
}

function normalizeTypeText(value) {
  return String(value).replace(/\s+/gu, " ").trim();
}

function countToken(source, token) {
  const escaped = token.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
  return [...source.matchAll(new RegExp(`\\b${escaped}\\b`, "gu"))].length;
}

function sortRecord(record) {
  return Object.fromEntries(Object.entries(record).sort(([left], [right]) => left.localeCompare(right)));
}

function relative(root, file) {
  return path.relative(root, file).split(path.sep).join("/");
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function stringValue(value) {
  return typeof value === "string" ? value : JSON.stringify(value);
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

if (path.resolve(process.argv[1] ?? "") === fileURLToPath(import.meta.url)) {
  const failures = runSemanticArchitectureCheck(repoRoot);
  if (jsonOutput) {
    process.stdout.write(`${JSON.stringify({ ok: failures.length === 0, failures }, null, 2)}\n`);
  } else if (failures.length === 0) {
    process.stdout.write("Semantic architecture check passed.\n");
  } else {
    process.stderr.write("Semantic architecture check failed:\n");
    for (const failure of failures) process.stderr.write(`  - ${failure}\n`);
  }
  if (failures.length > 0) process.exitCode = 1;
}
