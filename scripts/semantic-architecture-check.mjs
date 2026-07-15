#!/usr/bin/env node

import fs from "node:fs";
import { createRequire } from "node:module";
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
const REQUIRED_FREEZES = new Map([
  ["compiler.pointer-scalar-memory.v0", "compiler-pointer-scalar-memory"],
  ["compiler.cute-static-layout.v0", "cute-static-layout"],
  ["kernels.tensor-gpu-plan.v0", "tensor-gpu-plan"],
  ["jit.core-custom-ops.v0", "jit-op-custom"],
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
  checkWorkspaceDependencies(root, failures);
  checkWorkspaceImports(root, ts, failures);
  checkGeneratedPython(root, failures);

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
      case "tensor-gpu-plan":
        checkTensorGpuPlan(root, ts, adapter.freeze, failures);
        break;
      case "jit-op-custom":
        checkJitCustomOps(root, ts, adapter.freeze, failures);
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
  if (packageName === "@unlocalhosted/browsergrad-compiler" && /^@unlocalhosted\/browsergrad-(?:jit|grad)(?:\/|$)/u.test(specifier)) {
    failures.push(`${file} imports framework internals from compiler`);
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
    specifier !== "@unlocalhosted/browsergrad-semantic-core/capability"
  ) {
    failures.push(`${file} imports ${specifier}; runtime may import semantic-core/capability only`);
  }
  return failures;
}

export function countPythonCustomConstructors(source) {
  return pythonCallFacts(source).customConstructors;
}

export function extractPythonCustomLabels(source) {
  return [...pythonCallFacts(source).labels].sort();
}

export function validateSemanticFreezeManifest(root, manifest) {
  const failures = [];
  validateManifest(path.resolve(root), manifest, failures);
  return failures;
}

export function extractModuleSpecifiers(ts, source, filename = "source.ts") {
  const sourceFile = ts.createSourceFile(filename, source, ts.ScriptTarget.Latest, true);
  return moduleSpecifiers(ts, sourceFile);
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
  const expectedCounts = freeze.constructorCounts ?? {};
  const actualCounts = {};
  const labels = new Set();
  const aliases = new Set();
  const pythonRoot = path.join(root, "packages/browsergrad-jit/src/python");

  for (const file of walk(pythonRoot, (candidate) => candidate.endsWith(".py"))) {
    const source = fs.readFileSync(file, "utf8");
    const facts = pythonCallFacts(source);
    if (facts.customConstructors > 0) actualCounts[relative(root, file)] = facts.customConstructors;
    for (const label of facts.labels) labels.add(label);
    for (const alias of facts.aliases) aliases.add(`${relative(root, file)}:${alias}`);
  }

  const indexFile = path.join(pythonRoot, "index.ts");
  const indexSource = fs.readFileSync(indexFile, "utf8");
  const indexFacts = pythonCallFacts(indexSource);
  if (indexFacts.customConstructors > 0) actualCounts[relative(root, indexFile)] = indexFacts.customConstructors;
  for (const label of indexFacts.labels) labels.add(label);
  for (const alias of indexFacts.aliases) aliases.add(`${relative(root, indexFile)}:${alias}`);

  if (JSON.stringify(sortRecord(actualCounts)) !== JSON.stringify(sortRecord(expectedCounts))) {
    failures.push(`JIT OP_CUSTOM constructor sites changed; expected ${JSON.stringify(sortRecord(expectedCounts))}, got ${JSON.stringify(sortRecord(actualCounts))}`);
  }

  for (const [file, tokenCounts] of Object.entries(freeze.maxTokenCounts ?? {})) {
    const source = fs.readFileSync(path.join(root, file), "utf8");
    const tokens = pythonTokens(source);
    for (const [token, maximum] of Object.entries(tokenCounts)) {
      const count = tokens.filter((entry) => entry.kind === "identifier" && entry.value === token).length;
      if (count > maximum) failures.push(`${file} uses ${token} ${count} times; frozen maximum is ${maximum}`);
    }
  }

  for (const alias of [...aliases].sort()) failures.push(`JIT OP_CUSTOM alias is forbidden: ${alias}`);

  const allowedLabels = new Set(freeze.labels ?? []);
  for (const label of [...labels].sort()) {
    if (!allowedLabels.has(label)) failures.push(`JIT OP_CUSTOM path introduces unregistered label ${label}`);
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
  const dictLabels = new Map();

  for (let index = 0; index < tokens.length - 2; index += 1) {
    if (tokens[index].kind !== "identifier" || tokens[index + 1]?.value !== "=") continue;
    const rightIndex = skipParentheses(tokens, index + 2);
    if (tokens[rightIndex]?.value === "OP_CUSTOM" && tokens[index].value !== "op") aliases.add(tokens[index].value);
    if (tokens[index + 2]?.value === "{") {
      const close = matchingDelimiter(tokens, index + 2, "{", "}");
      if (close === undefined) continue;
      const values = labelsInTokens(tokens.slice(index + 3, close));
      if (values.size > 0) dictLabels.set(tokens[index].value, values);
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
        for (const label of labelsInTokens(body)) labels.add(label);
        for (let bodyIndex = 0; bodyIndex < body.length - 2; bodyIndex += 1) {
          if (body[bodyIndex].value !== "arg" || body[bodyIndex + 1]?.value !== "=") continue;
          const variable = body[bodyIndex + 2];
          if (variable?.kind !== "identifier") continue;
          for (const label of dictLabels.get(variable.value) ?? []) labels.add(label);
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
        if (depth === 0 && entry.kind === "string") labels.add(entry.value);
      }
      index = close;
    }
  }
  return { customConstructors, labels, aliases };
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

function labelsInTokens(tokens) {
  const labels = new Set();
  for (let index = 0; index < tokens.length - 2; index += 1) {
    const key = tokens[index];
    if (key.kind === "string" && (key.value === "name" || key.value === "op") && tokens[index + 1]?.value === ":" && tokens[index + 2]?.kind === "string") {
      labels.add(tokens[index + 2].value);
    }
  }
  return labels;
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
