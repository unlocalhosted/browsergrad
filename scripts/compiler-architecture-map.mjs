#!/usr/bin/env node
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const compilerRoot = path.join(repoRoot, "packages/browsergrad-compiler");
const compilerRequire = createRequire(path.join(compilerRoot, "package.json"));
const ts = compilerRequire("typescript");
const compilerSrc = path.join(compilerRoot, "src");
const compilerTests = path.join(compilerRoot, "tests");
const args = new Set(process.argv.slice(2));
const option = (name, fallback) => {
  const value = process.argv.find((arg) => arg.startsWith(`${name}=`));
  return value === undefined ? fallback : value.slice(name.length + 1);
};
const minLines = Number.parseInt(option("--min-lines", "1000"), 10);
const maxSourceLines = Number.parseInt(option("--max-source-lines", "5500"), 10);
const maxTestLines = Number.parseInt(option("--max-test-lines", "4500"), 10);
const check = args.has("--check");
const json = args.has("--json");
const includeTests = args.has("--include-tests") || check;

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return walk(full);
    return /\.(?:ts|mts|mjs)$/u.test(entry.name) ? [full] : [];
  });
}

function rel(file) {
  return path.relative(repoRoot, file);
}

function importsOf(file, source) {
  const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true);
  const imports = [];
  for (const statement of sourceFile.statements) {
    if (ts.isImportDeclaration(statement) && ts.isStringLiteral(statement.moduleSpecifier)) {
      const clause = statement.importClause;
      const bindings = clause?.namedBindings;
      const hasRuntimeNamedImport = bindings === undefined ||
        !ts.isNamedImports(bindings) ||
        bindings.elements.some((element) => !element.isTypeOnly);
      imports.push({
        specifier: statement.moduleSpecifier.text,
        runtime: clause?.isTypeOnly !== true && (clause?.name !== undefined || hasRuntimeNamedImport),
      });
      continue;
    }
    if (ts.isExportDeclaration(statement) && statement.moduleSpecifier && ts.isStringLiteral(statement.moduleSpecifier)) {
      imports.push({
        specifier: statement.moduleSpecifier.text,
        runtime: statement.isTypeOnly !== true,
      });
    }
  }
  return imports;
}

function resolveLocalImport(file, specifier) {
  if (!specifier.startsWith(".")) return undefined;
  const base = path.resolve(path.dirname(file), specifier.replace(/\.js$/u, ""));
  const candidates = [`${base}.ts`, `${base}.mts`, path.join(base, "index.ts")];
  return candidates.find((candidate) => fs.existsSync(candidate));
}

function exportsOf(source) {
  return [...source.matchAll(/^export\s+(?:async\s+)?(?:function|class|interface|type|const)\s+(\w+)/gmu)]
    .map((match) => match[1])
    .filter((name) => name !== undefined);
}

function functionsOf(source) {
  return [...source.matchAll(/^(?:export\s+)?(?:async\s+)?function\s+(\w+)/gmu)]
    .map((match) => match[1])
    .filter((name) => name !== undefined);
}

function featureBucket(file) {
  const normalized = rel(file);
  if (/\/tests\//u.test(normalized)) return "tests";
  const feature = normalized.match(/\/features\/([^/]+)/u)?.[1];
  if (feature) return `feature:${feature}`;
  if (/\/semantic_/u.test(normalized) || normalized.endsWith("/semantic_ir.ts")) return "semantic-ir";
  if (/\/wgsl/u.test(normalized)) return "wgsl-backend";
  if (/\/reference/u.test(normalized)) return "reference";
  if (/analyzer|parser|lexer|diagnostics|cpp_cute_/u.test(normalized)) return "frontend";
  if (/runtime|dynamic_launch|peer_copy|webgpu_orchestration|runner/u.test(normalized)) return "runtime-orchestration";
  return "support";
}

function moduleRows(roots) {
  const files = roots.flatMap((root) => walk(root));
  const byFile = new Map(files.map((file) => [file, true]));
  return files.map((file) => {
    const source = fs.readFileSync(file, "utf8");
    const imports = importsOf(file, source);
    const localImports = imports
      .filter((entry) => entry.runtime)
      .map((entry) => resolveLocalImport(file, entry.specifier))
      .filter((candidate) => candidate !== undefined && byFile.has(candidate));
    return {
      file,
      relativeFile: rel(file),
      lines: source.split("\n").length,
      imports: imports.length,
      localImports: localImports.map((candidate) => rel(candidate)),
      exports: exportsOf(source).length,
      functions: functionsOf(source).length,
      bucket: featureBucket(file),
      source,
    };
  }).sort((left, right) => right.lines - left.lines);
}

function dependencyCycles(rows) {
  const byRelativeFile = new Map(rows.map((row) => [row.relativeFile, row]));
  const indexByFile = new Map();
  const lowLinkByFile = new Map();
  const stack = [];
  const onStack = new Set();
  const cycles = [];
  let index = 0;

  function visit(file) {
    indexByFile.set(file, index);
    lowLinkByFile.set(file, index);
    index += 1;
    stack.push(file);
    onStack.add(file);
    const row = byRelativeFile.get(file);
    for (const target of row?.localImports ?? []) {
      if (!indexByFile.has(target)) {
        visit(target);
        lowLinkByFile.set(file, Math.min(lowLinkByFile.get(file), lowLinkByFile.get(target)));
      } else if (onStack.has(target)) {
        lowLinkByFile.set(file, Math.min(lowLinkByFile.get(file), indexByFile.get(target)));
      }
    }
    if (lowLinkByFile.get(file) !== indexByFile.get(file)) return;
    const cycle = [];
    while (true) {
      const item = stack.pop();
      if (item === undefined) break;
      onStack.delete(item);
      cycle.push(item);
      if (item === file) break;
    }
    if (cycle.length > 1) cycles.push(cycle.sort());
  }

  for (const row of rows) {
    if (!indexByFile.has(row.relativeFile)) visit(row.relativeFile);
  }
  return cycles.sort((left, right) => left.join("|").localeCompare(right.join("|")));
}

function bucketSummary(rows) {
  const buckets = new Map();
  for (const row of rows) {
    const summary = buckets.get(row.bucket) ?? { files: 0, lines: 0 };
    summary.files += 1;
    summary.lines += row.lines;
    buckets.set(row.bucket, summary);
  }
  return [...buckets.entries()]
    .map(([bucket, summary]) => ({ bucket, ...summary }))
    .sort((left, right) => right.lines - left.lines);
}

function legacyBackendLeaks(rows) {
  const leaks = [];
  const forbiddenFiles = new Set(["ir_usage.ts", "kernel_ir_atomic_usage.ts", "kernel_ir_usage.ts", "reference.ts", "wgsl.ts"]);
  const forbiddenSymbols = ["KernelIrModule", "lowerCudaLiteToKernelIr", "lowerAnalyzedCudaLiteToKernelIr", "emitKernelIrWgsl"];
  for (const row of rows) {
    const sourceRelativeFile = path.relative(compilerSrc, row.file);
    if (forbiddenFiles.has(sourceRelativeFile)) leaks.push(`${row.relativeFile} is a removed AST backend module`);
    for (const symbol of forbiddenSymbols) {
      if (new RegExp(`\\b${symbol}\\b`, "u").test(row.source)) leaks.push(`${row.relativeFile} references removed ${symbol}`);
    }
  }
  return leaks;
}

function semanticIrRepresentationLeaks(rows) {
  const leaks = [];
  const representation = rows.find((row) => path.relative(compilerSrc, row.file) === "semantic_ir_types.ts");
  const representationFile = representation === undefined
    ? undefined
    : ts.createSourceFile(representation.file, representation.source, ts.ScriptTarget.Latest, true);
  const typeNames = new Set((representationFile?.statements ?? [])
    .filter((statement) => ts.isInterfaceDeclaration(statement) || ts.isTypeAliasDeclaration(statement))
    .map((statement) => statement.name.text));
  if (representation === undefined) leaks.push("semantic_ir_types.ts representation module is missing");
  for (const row of rows) {
    const sourceRelativeFile = path.relative(compilerSrc, row.file);
    const sourceFile = ts.createSourceFile(row.file, row.source, ts.ScriptTarget.Latest, true);
    if (sourceRelativeFile === "semantic_ir_types.ts") {
      for (const statement of sourceFile.statements) {
        const validImport = ts.isImportDeclaration(statement) && statement.importClause?.isTypeOnly === true;
        if (!validImport && !ts.isInterfaceDeclaration(statement) && !ts.isTypeAliasDeclaration(statement)) {
          leaks.push(`${row.relativeFile} contains runtime implementation in the IR representation module`);
        }
      }
      continue;
    }
    if (sourceRelativeFile === "semantic_ir.ts") continue;
    for (const statement of sourceFile.statements) {
      if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier) || statement.moduleSpecifier.text !== "./semantic_ir.js") continue;
      const clause = statement.importClause;
      const bindings = clause?.namedBindings;
      if (!bindings || !ts.isNamedImports(bindings)) continue;
      for (const element of bindings.elements) {
        const imported = element.propertyName?.text ?? element.name.text;
        if (clause?.isTypeOnly === true || element.isTypeOnly || typeNames.has(imported)) {
          leaks.push(`${row.relativeFile} imports IR type ${imported} from semantic_ir.ts instead of semantic_ir_types.ts`);
        }
      }
    }
  }
  return leaks;
}

function cppCuteFrontendLegacyLeaks(rows) {
  const forbidden = new Set([
    "analyzer.ts",
    "cute_static_layout.ts",
    "parser.ts",
    "semantic_ir.ts",
    "semantic_ir_types.ts",
    "semantic_ir_walk.ts",
    "semantic_layout_bindings.ts",
    "semantic_layout_lowering.ts",
    "semantic_view_copy_bindings.ts",
    "semantic_view_copy_lowering.ts",
  ]);
  const leaks = [];
  for (const row of rows) {
    const sourceRelativeFile = path.relative(compilerSrc, row.file);
    // Every real C++/CuTe module belongs to this guarded sibling architecture,
    // including later lowering/producer seams whose names are not "frontend".
    if (!sourceRelativeFile.startsWith("cpp_cute_")) continue;
    const importedTargets = importsOf(row.file, row.source)
      .map((entry) => resolveLocalImport(row.file, entry.specifier))
      .filter((target) => target !== undefined);
    for (const target of importedTargets) {
      const targetRelativeFile = path.relative(compilerSrc, target);
      if (forbidden.has(targetRelativeFile)) {
        leaks.push(`${row.relativeFile} imports frozen CUDA-lite path ${targetRelativeFile}`);
      }
    }
  }
  return leaks;
}

const sourceRows = moduleRows([compilerSrc]);
const testRows = includeTests ? moduleRows([compilerTests]) : [];
const allRows = [...sourceRows, ...testRows];
const cycles = dependencyCycles(allRows);
const report = {
  source: sourceRows.map(({ source, ...row }) => row),
  tests: testRows.map(({ source, ...row }) => row),
  buckets: bucketSummary(allRows),
  cycles,
  legacyBackendLeaks: legacyBackendLeaks(sourceRows),
  semanticIrRepresentationLeaks: semanticIrRepresentationLeaks(sourceRows),
  cppCuteFrontendLegacyLeaks: cppCuteFrontendLegacyLeaks(sourceRows),
  limits: { maxSourceLines, maxTestLines },
};

if (json) {
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} else {
  console.log("Compiler Architecture Map");
  console.log(`root: ${rel(compilerSrc)}`);
  console.log("");
  console.log("Large modules");
  for (const row of allRows.filter((entry) => entry.lines >= minLines)) {
    console.log(`${String(row.lines).padStart(5)}  ${row.relativeFile}  bucket=${row.bucket} imports=${row.imports} local=${row.localImports.length} exports=${row.exports} funcs=${row.functions}`);
  }
  console.log("");
  console.log("Buckets");
  for (const summary of report.buckets) {
    console.log(`${String(summary.lines).padStart(5)}  ${summary.bucket}  files=${summary.files}`);
  }
  console.log("");
  console.log(`Dependency cycles: ${cycles.length}`);
  for (const cycle of cycles) console.log(`  ${cycle.join(" -> ")}`);
  console.log(`Legacy backend leaks: ${report.legacyBackendLeaks.length}`);
  for (const leak of report.legacyBackendLeaks) console.log(`  ${leak}`);
  console.log(`Semantic IR representation leaks: ${report.semanticIrRepresentationLeaks.length}`);
  for (const leak of report.semanticIrRepresentationLeaks) console.log(`  ${leak}`);
  console.log(`C++/CuTe frontend legacy leaks: ${report.cppCuteFrontendLegacyLeaks.length}`);
  for (const leak of report.cppCuteFrontendLegacyLeaks) console.log(`  ${leak}`);
}

if (check) {
  const failures = [
    ...sourceRows
      .filter((row) => row.lines > maxSourceLines)
      .map((row) => `${row.relativeFile} has ${row.lines} lines (limit ${maxSourceLines})`),
    ...testRows
      .filter((row) => row.lines > maxTestLines)
      .map((row) => `${row.relativeFile} has ${row.lines} lines (limit ${maxTestLines})`),
    ...cycles.map((cycle) => `dependency cycle: ${cycle.join(" -> ")}`),
    ...report.legacyBackendLeaks,
    ...report.semanticIrRepresentationLeaks,
    ...report.cppCuteFrontendLegacyLeaks,
  ];
  if (failures.length > 0) {
    console.error("Compiler architecture check failed:");
    for (const failure of failures) console.error(`  ${failure}`);
    process.exitCode = 1;
  }
}
