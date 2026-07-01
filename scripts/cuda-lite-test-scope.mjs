#!/usr/bin/env node
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { findRepoRoot } from "./cuda-lite-webgpu-cli.mjs";
import {
  atomicScopeCases,
  controlScopeCases,
  runtimeScopeCases,
  storageScopeCases,
  textureScopeCases,
} from "./cuda-lite-webgpu-smoke-cases.mjs";

const COMPILER = "@unlocalhosted/browsergrad-compiler";
const KERNELS = "@unlocalhosted/browsergrad-kernels";
const scriptPath = fileURLToPath(import.meta.url);

const scopeAliases = new Map([
  ["types", ["compiler-types"]],
  ["compiler", ["compiler-types", "compiler-unit"]],
  ["unit", ["compiler-unit"]],
  ["parser", ["compiler-types", "compiler-unit"]],
  ["analyzer", ["compiler-types", "compiler-unit"]],
  ["reference", ["compiler-types", "compiler-unit"]],
  ["runtime", ["compiler-types", "compiler-unit", "runtime-orchestration"]],
  ["dynamic", ["compiler-types", "compiler-unit", "runtime-orchestration"]],
  ["launch", ["compiler-types", "compiler-unit", "runtime-orchestration"]],
  ["orchestration", ["compiler-types", "compiler-unit", "runtime-orchestration"]],
  ["wgsl", ["compiler-types", "wgsl-shared-emitter", "wgsl-storage-pointer", "wgsl-atomics", "wgsl-control-cooperative", "wgsl-texture-surface"]],
  ["wgsl-shared", ["compiler-types", "wgsl-shared-emitter"]],
  ["wgsl-storage", ["compiler-types", "wgsl-storage-pointer"]],
  ["storage", ["compiler-types", "wgsl-storage-pointer"]],
  ["pointer", ["compiler-types", "wgsl-storage-pointer"]],
  ["vector", ["compiler-types", "wgsl-storage-pointer"]],
  ["atomic", ["compiler-types", "wgsl-atomics"]],
  ["atomics", ["compiler-types", "wgsl-atomics"]],
  ["control", ["compiler-types", "wgsl-control-cooperative"]],
  ["cooperative", ["compiler-types", "wgsl-control-cooperative"]],
  ["barrier", ["compiler-types", "wgsl-control-cooperative"]],
  ["texture", ["compiler-types", "wgsl-texture-surface"]],
  ["surface", ["compiler-types", "wgsl-texture-surface"]],
  ["webgpu-fixtures", ["webgpu-fixture-scripts"]],
  ["real-world", ["real-world-verifier-scripts"]],
  ["corpus-fixtures", ["corpus-execution-fixture-files"]],
  ["corpus-registry", ["corpus-execution-fixture-registry"]],
  ["source-normalizer", ["source-normalizer-scripts"]],
  ["synthetic-input", ["synthetic-input-scripts"]],
  ["tool-runner", ["tool-runner"]],
  ["bugbash", ["bugbash-status"]],
  ["status", ["bugbash-status"]],
  ["benchmark", ["benchmark-scripts"]],
  ["bench", ["benchmark-scripts"]],
  ["kernels", ["kernels-wgsl-bridge"]],
  ["wgsl-program", ["kernels-wgsl-bridge"]],
  ["webgpu-smoke", ["webgpu-smoke-scripts"]],
  ["smoke", ["webgpu-smoke-scripts"]],
  ["package-script", ["package-script-smoke"]],
]);

export const scopeRules = [
  {
    id: "compiler-types",
    reason: "compiler TS source changed",
    matches: [
      /^packages\/browsergrad-compiler\/src\/.+\.ts$/u,
      /^packages\/browsergrad-compiler\/tests\/.+\.ts$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "typecheck"),
    ],
  },
  {
    id: "compiler-unit",
    reason: "parser/analyzer/reference/core compiler changed",
    matches: [
      /^packages\/browsergrad-compiler\/src\/(?:analyzer|parser|reference|types|index|webgpu_orchestration)\.ts$/u,
      /^packages\/browsergrad-compiler\/tests\/compiler\.test\.ts$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "exec", "vitest", "run", "tests/compiler.test.ts"),
    ],
  },
  {
    id: "runtime-orchestration",
    reason: "runtime orchestration/dynamic launch path changed",
    matches: [
      /^packages\/browsergrad-compiler\/src\/(?:webgpu_orchestration|dynamic_launch|runtime_elision|runtime_plan|launch|webgpu_inputs)\.ts$/u,
    ],
    commands: [
      e2eCases(...runtimeScopeCases),
    ],
  },
  {
    id: "wgsl-shared-emitter",
    reason: "WGSL shared emitter/orchestration path changed",
    matches: [
      /^packages\/browsergrad-compiler\/src\/(?:ir_usage|wgsl|wgsl_(?:context|declarations|feature_usage|ir_analysis|module|names|support_helpers))\.ts$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:wgsl-modules"),
      command("pnpm", "--filter", COMPILER, "run", "e2e:webgpu:smoke"),
    ],
  },
  {
    id: "wgsl-storage-pointer",
    reason: "WGSL storage/pointer/value path changed",
    matches: [
      /^packages\/browsergrad-compiler\/src\/wgsl(?:_storage|_storage_views|_device_pointers|_pointer_helpers|_pointer_usage|_value_conversion|_pool_access)?\.ts$/u,
      /^packages\/browsergrad-compiler\/tests\/wgsl_modules\.test\.ts$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:wgsl-modules"),
      e2eCases(...storageScopeCases),
    ],
  },
  {
    id: "wgsl-atomics",
    reason: "WGSL atomic path changed",
    matches: [
      /^packages\/browsergrad-compiler\/src\/wgsl_atomics?\.ts$/u,
      /^packages\/browsergrad-compiler\/src\/wgsl_atomic_helpers\.ts$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:wgsl-modules"),
      e2eCases(...atomicScopeCases),
    ],
  },
  {
    id: "wgsl-control-cooperative",
    reason: "WGSL active-lane/cooperative path changed",
    matches: [
      /^packages\/browsergrad-compiler\/src\/wgsl_(?:control_analysis|cooperative)\.ts$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:wgsl-modules"),
      e2eCases(...controlScopeCases),
    ],
  },
  {
    id: "wgsl-texture-surface",
    reason: "WGSL texture/surface path changed",
    matches: [
      /^packages\/browsergrad-compiler\/src\/wgsl_texture_surface\.ts$/u,
    ],
    commands: [
      e2eCases(...textureScopeCases),
      command(
        "pnpm",
        "--filter",
        COMPILER,
        "run",
        "e2e:webgpu:hot-case:gate",
        "--",
        "--cases",
        "texture-surface:roundtrip",
        "--repeat",
        "2",
        "--expect-warm-speedup-min",
        "1.01",
      ),
    ],
  },
  {
    id: "webgpu-fixture-scripts",
    reason: "WebGPU fixture/runner helper changed",
    matches: [
      /^scripts\/cuda-lite-webgpu-(?:cli|fixtures|report)\.mjs$/u,
      /^scripts\/test-cuda-lite-webgpu-fixtures\.mjs$/u,
      /^scripts\/e2e-cuda-lite-webgpu\.mjs$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:webgpu-fixtures"),
      command("pnpm", "--filter", COMPILER, "run", "e2e:webgpu:smoke"),
    ],
  },
  {
    id: "kernels-wgsl-bridge",
    reason: "kernels WGSL bridge changed",
    matches: [
      /^packages\/browsergrad-kernels\/src\/wgsl_program\.ts$/u,
    ],
    commands: [
      command("pnpm", "--filter", KERNELS, "run", "typecheck"),
      command("pnpm", "--filter", COMPILER, "run", "e2e:webgpu:smoke"),
    ],
  },
  {
    id: "real-world-verifier-scripts",
    reason: "real-world verifier wiring changed",
    matches: [
      /^scripts\/verify-real-world-cuda\.mjs$/u,
      /^scripts\/test-verify-real-world-cuda-cli\.mjs$/u,
      /^scripts\/verify-release\.mjs$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:verify-real-world-cli"),
      command("pnpm", "--filter", COMPILER, "run", "e2e:webgpu:fast"),
    ],
  },
  {
    id: "corpus-execution-fixture-files",
    reason: "corpus execution fixture definitions changed",
    matches: [
      /^scripts\/cuda-lite-corpus-fixtures(?:-extra(?:-2)?)?\.mjs$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:webgpu-fixtures"),
    ],
  },
  {
    id: "corpus-execution-fixture-registry",
    reason: "corpus execution fixture registry/baseline changed",
    matches: [
      /^scripts\/cuda-lite-corpus-registry\.mjs$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:webgpu-fixtures"),
      command("pnpm", "--filter", COMPILER, "run", "e2e:webgpu:fixtures"),
    ],
  },
  {
    id: "source-normalizer-scripts",
    reason: "source normalizer changed",
    matches: [
      /^scripts\/cuda-lite-source-normalizer/u,
      /^scripts\/test-cuda-lite-source-normalizer/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:source-normalizer"),
    ],
  },
  {
    id: "synthetic-input-scripts",
    reason: "synthetic input changed",
    matches: [
      /^scripts\/cuda-lite-synthetic-input\.mjs$/u,
      /^scripts\/test-cuda-lite-synthetic-input\.mjs$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:synthetic-input:run"),
    ],
  },
  {
    id: "tool-runner",
    reason: "CUDA-lite tool runner changed",
    matches: [
      /^scripts\/run-cuda-lite-tool\.mjs$/u,
      /^scripts\/run-cuda-lite-last-failures\.mjs$/u,
      /^scripts\/cuda-lite-tool-lock\.mjs$/u,
      /^scripts\/test-cuda-lite-tool-lock\.mjs$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:tool-lock"),
    ],
  },
  {
    id: "bugbash-status",
    reason: "compiler bugbash status/progress tracking changed",
    matches: [
      /^docs\/internal\/compiler-bugbash-progress\.md$/u,
      /^scripts\/cuda-lite-bugbash-status\.mjs$/u,
      /^scripts\/test-cuda-lite-bugbash-status\.mjs$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:bugbash-status"),
      command("pnpm", "--filter", COMPILER, "run", "bugbash:status"),
    ],
  },
  {
    id: "benchmark-scripts",
    reason: "CUDA-lite benchmark script changed",
    matches: [
      /^scripts\/benchmark-cuda-lite-(?:compiler|webgpu)\.mjs$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "bench", "--", "--skip-build", "--runs", "1", "--warmup", "0"),
      command("pnpm", "--filter", COMPILER, "run", "bench:browser", "--", "--skip-build", "--require-webgpu", "--runs", "1", "--warmup", "0", "--length", "256"),
    ],
  },
  {
    id: "webgpu-smoke-scripts",
    reason: "WebGPU smoke case list/runner changed",
    matches: [
      /^scripts\/cuda-lite-webgpu-smoke-cases\.mjs$/u,
      /^scripts\/run-cuda-lite-webgpu-smoke\.mjs$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:test-scope"),
      command("pnpm", "--filter", COMPILER, "run", "e2e:webgpu:smoke"),
    ],
  },
  {
    id: "package-script-smoke",
    reason: "compiler package scripts changed",
    matches: [
      /^packages\/browsergrad-compiler\/package\.json$/u,
      /^scripts\/cuda-lite-test-scope\.mjs$/u,
      /^scripts\/test-cuda-lite-test-scope\.mjs$/u,
    ],
    commands: [
      command("pnpm", "--filter", COMPILER, "run", "test:test-scope"),
    ],
  },
];

export function planForChangedFiles(files, options = {}) {
  const selected = selectScopesForFiles(files, options.scopes);
  const commands = selected.flatMap((item) => item.commands);
  const corpusFixtureCases = shouldIncludeChangedCorpusFixtures(options.scopes)
    ? corpusFixtureCasesForFiles(files, options.root, {
      base: options.base,
      changeMode: options.changeMode,
    })
    : [];
  if (corpusFixtureCases.length > 0) {
    commands.push(e2eCases(...corpusFixtureCases));
  }
  if (options.includeCompileCorpus && files.some((file) => file.startsWith("packages/browsergrad-compiler/src/"))) {
    commands.push(command("pnpm", "--filter", COMPILER, "run", "e2e:webgpu:compile"));
  }
  if (options.includeFastCorpus && files.some((file) => file.startsWith("packages/browsergrad-compiler/src/"))) {
    commands.push(command("pnpm", "--filter", COMPILER, "run", "e2e:webgpu:fast"));
  }
  return { files, scopes: selected, commands: dedupeCommands(commands) };
}

function shouldIncludeChangedCorpusFixtures(scopes = []) {
  if (scopes.length === 0) return true;
  const ids = requestedScopeIds(scopes);
  return ids.has("corpus-execution-fixture-files") || ids.has("corpus-execution-fixture-registry");
}

function selectScopesForFiles(files, requestedScopes = []) {
  if (requestedScopes.length === 0) {
    const selected = [];
    for (const rule of scopeRules) {
      const matchedFiles = files.filter((file) => rule.matches.some((pattern) => pattern.test(file)));
      if (matchedFiles.length === 0) continue;
      selected.push(scopeSelection(rule, matchedFiles));
    }
    return selected;
  }
  const ids = requestedScopeIds(requestedScopes);
  return scopeRules
    .filter((rule) => ids.has(rule.id))
    .map((rule) => {
      const matchedFiles = files.filter((file) => rule.matches.some((pattern) => pattern.test(file)));
      return scopeSelection(rule, matchedFiles, true);
    });
}

function requestedScopeIds(scopes) {
  const ids = new Set();
  for (const scope of scopes) {
    const key = scope.trim();
    if (!key) continue;
    const expanded = scopeAliases.get(key) ?? [key];
    for (const id of expanded) {
      if (!scopeRules.some((rule) => rule.id === id)) {
        throw new Error(`unknown scope: ${scope}`);
      }
      ids.add(id);
    }
  }
  return ids;
}

function scopeSelection(rule, files, requested = false) {
  return {
    id: rule.id,
    reason: rule.reason,
    files,
    ...(requested ? { requested: true } : {}),
    commands: rule.commands,
  };
}

export function changedFiles(root, base = "HEAD", mode = "all") {
  if (mode === "staged") {
    return runGitLines(root, ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "--"]).sort();
  }
  const trackedArgs = mode === "unstaged"
    ? ["diff", "--name-only", "--diff-filter=ACMR", "--"]
    : ["diff", "--name-only", "--diff-filter=ACMR", base, "--"];
  const tracked = runGitLines(root, trackedArgs);
  const untracked = mode === "staged" ? [] : runGitLines(root, ["ls-files", "--others", "--exclude-standard"]);
  return [...new Set([...tracked, ...untracked])].sort();
}

export function corpusFixtureCasesForFiles(files, root = process.cwd(), options = {}) {
  const caseNames = [];
  for (const file of files) {
    if (!/^scripts\/cuda-lite-corpus-fixtures(?:-extra(?:-2)?)?\.mjs$/u.test(file)) continue;
    const source = fs.readFileSync(path.join(root, file), "utf8");
    const changedLines = changedLineNumbersForFile(root, file, options);
    const scopedCases = changedLines.length > 0
      ? fixtureCaseNamesForChangedLines(source, changedLines)
      : fixtureCaseNames(source);
    for (const name of scopedCases) {
      if (!caseNames.includes(name)) caseNames.push(name);
    }
  }
  return caseNames;
}

export function fixtureCaseNamesForChangedLines(source, changedLines) {
  const cases = fixtureCaseLineRanges(source);
  if (changedLines.length === 0 || cases.length === 0) return fixtureCaseNames(source);
  const selected = [];
  for (const line of changedLines) {
    const matched = cases.find((item) => line >= item.startLine && line <= item.endLine)
      ?? cases.find((item) => line < item.startLine);
    if (matched && !selected.includes(matched.caseName)) selected.push(matched.caseName);
  }
  return selected.length > 0 ? selected : fixtureCaseNames(source);
}

function fixtureCaseNames(source) {
  const caseNames = [];
  for (const match of source.matchAll(/\bcaseName:\s*"([^"]+)"/gu)) {
    if (!caseNames.includes(match[1])) caseNames.push(match[1]);
  }
  return caseNames;
}

function fixtureCaseLineRanges(source) {
  const lines = source.split(/\r?\n/u);
  const cases = [];
  for (let index = 0; index < lines.length; index++) {
    const match = /\bcaseName:\s*"([^"]+)"/u.exec(lines[index]);
    if (!match) continue;
    cases.push({
      caseName: match[1],
      startLine: index + 1,
      endLine: lines.length,
    });
  }
  for (let index = 0; index < cases.length - 1; index++) {
    cases[index].endLine = cases[index + 1].startLine - 1;
  }
  return cases;
}

function changedLineNumbersForFile(root, file, options = {}) {
  const mode = options.changeMode ?? "all";
  if (mode === "staged") return changedLineNumbersFromDiff(root, ["diff", "--cached", "--unified=0", "--", file]);
  if (mode === "unstaged") return changedLineNumbersFromDiff(root, ["diff", "--unified=0", "--", file]);
  const lines = changedLineNumbersFromDiff(root, ["diff", "--unified=0", options.base ?? "HEAD", "--", file]);
  return lines.length > 0 ? lines : changedLineNumbersFromDiff(root, ["diff", "--unified=0", "--", file]);
}

function changedLineNumbersFromDiff(root, args) {
  const result = spawnSync("git", args, {
    cwd: root,
    encoding: "utf8",
  });
  if (result.error || result.status !== 0) return [];
  const lines = [];
  for (const line of result.stdout.split(/\r?\n/u)) {
    const match = /^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@/u.exec(line);
    if (!match) continue;
    const start = Number(match[1]);
    const count = match[2] === undefined ? 1 : Number(match[2]);
    for (let offset = 0; offset < Math.max(count, 1); offset++) lines.push(start + offset);
  }
  return [...new Set(lines)].sort((left, right) => left - right);
}

function runGitLines(root, args) {
  const result = spawnSync("git", args, {
    cwd: root,
    encoding: "utf8",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(result.stderr || result.stdout);
  return result.stdout.split(/\r?\n/u).map((line) => line.trim()).filter(Boolean);
}

function e2eCases(...cases) {
  return command(
    "pnpm",
    "--filter",
    COMPILER,
    "run",
    "e2e:webgpu:case",
    "--",
    "--cases",
    cases.join(","),
  );
}

function command(bin, ...args) {
  return { bin, args };
}

function dedupeCommands(commands) {
  const merged = mergeE2eCaseCommands(commands);
  const seen = new Set();
  const out = [];
  for (const item of merged) {
    const key = commandToString(item);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out;
}

function mergeE2eCaseCommands(commands) {
  const cases = [];
  let inserted = false;
  const out = [];
  for (const item of commands) {
    const itemCases = e2eCommandCases(item);
    if (!itemCases) {
      out.push(item);
      continue;
    }
    for (const caseName of itemCases) {
      if (!cases.includes(caseName)) cases.push(caseName);
    }
    if (!inserted) {
      out.push({ ...item, args: replaceE2eCases(item.args, cases) });
      inserted = true;
    }
  }
  if (!inserted) return out;
  return out.map((item) => e2eCommandCases(item) ? { ...item, args: replaceE2eCases(item.args, cases) } : item);
}

function e2eCommandCases(commandItem) {
  const args = commandItem.args;
  if (commandItem.bin !== "pnpm") return undefined;
  const runIndex = args.indexOf("run");
  if (runIndex < 0 || args[runIndex + 1] !== "e2e:webgpu:case") return undefined;
  const casesIndex = args.indexOf("--cases");
  if (casesIndex < 0 || typeof args[casesIndex + 1] !== "string") return undefined;
  return args[casesIndex + 1].split(",").map((item) => item.trim()).filter(Boolean);
}

function replaceE2eCases(args, cases) {
  const out = [...args];
  const casesIndex = out.indexOf("--cases");
  if (casesIndex >= 0) {
    out[casesIndex + 1] = cases.join(",");
    return out;
  }
  return [...out, "--", "--cases", cases.join(",")];
}

export function commandToString(item) {
  return [item.bin, ...item.args].map(shellToken).join(" ");
}

function shellToken(value) {
  return /^[A-Za-z0-9_/:@.,=+-]+$/u.test(value) ? value : JSON.stringify(value);
}

function parseArgs(argv) {
  const options = {
    base: "HEAD",
    run: false,
    json: false,
    includeCompileCorpus: false,
    includeFastCorpus: false,
    timingJson: undefined,
    changeMode: "all",
    files: [],
    scopes: [],
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--") continue;
    else if (arg === "--run") options.run = true;
    else if (arg === "--json") options.json = true;
    else if (arg === "--include-compile-corpus") options.includeCompileCorpus = true;
    else if (arg === "--include-fast-corpus") options.includeFastCorpus = true;
    else if (arg === "--staged") options.changeMode = "staged";
    else if (arg === "--unstaged") options.changeMode = "unstaged";
    else if (arg === "--timing-json") options.timingJson = argv[++index];
    else if (arg.startsWith("--timing-json=")) options.timingJson = arg.slice("--timing-json=".length);
    else if (arg === "--base") options.base = argv[++index] ?? options.base;
    else if (arg.startsWith("--base=")) options.base = arg.slice("--base=".length);
    else if (arg === "--files") {
      options.files.push(...(argv[++index] ?? "").split(",").map((item) => item.trim()).filter(Boolean));
    } else if (arg.startsWith("--files=")) {
      options.files.push(...arg.slice("--files=".length).split(",").map((item) => item.trim()).filter(Boolean));
    } else if (arg === "--scope") {
      options.scopes.push(...(argv[++index] ?? "").split(",").map((item) => item.trim()).filter(Boolean));
    } else if (arg.startsWith("--scope=")) {
      options.scopes.push(...arg.slice("--scope=".length).split(",").map((item) => item.trim()).filter(Boolean));
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }
  return options;
}

async function main() {
  const root = findRepoRoot(path.dirname(scriptPath));
  const options = parseArgs(process.argv.slice(2));
  const files = options.files.length > 0 ? options.files : changedFiles(root, options.base, options.changeMode);
  const plan = planForChangedFiles(files, {
    includeCompileCorpus: options.includeCompileCorpus,
    includeFastCorpus: options.includeFastCorpus,
    base: options.base,
    changeMode: options.changeMode,
    root,
    scopes: options.scopes,
  });
  if (options.json) {
    console.log(JSON.stringify({
      ...plan,
      commands: plan.commands.map(commandToString),
    }, null, 2));
  } else {
    printPlan(plan, options);
  }
  if (options.run) runPlan(root, plan, options);
}

function printPlan(plan, options) {
  if (plan.files.length === 0) {
    console.log("No changed files detected.");
    return;
  }
  console.log(`Changed files: ${plan.files.length}`);
  if (plan.scopes.length === 0) {
    console.log("No CUDA-lite compiler scope matched. No targeted test command selected.");
    return;
  }
  for (const scope of plan.scopes) {
    console.log(`- ${scope.id}: ${scope.reason}${scope.requested ? " (requested)" : ""}`);
    for (const file of scope.files) console.log(`  ${file}`);
  }
  console.log(options.run ? "Running:" : "Commands:");
  for (const item of plan.commands) console.log(`  ${commandToString(item)}`);
}

function runPlan(root, plan, options = {}) {
  const timings = [];
  for (const item of plan.commands) {
    const startMs = Date.now();
    const result = spawnSync(item.bin, item.args, { cwd: root, stdio: "inherit" });
    const endMs = Date.now();
    timings.push({
      command: commandToString(item),
      status: result.status,
      signal: result.signal,
      durationMs: endMs - startMs,
    });
    if (result.error) throw result.error;
    if (result.status !== 0) {
      writeTimingJson(root, options.timingJson, plan, timings, result.status ?? 1);
      process.exit(result.status ?? 1);
    }
  }
  writeTimingJson(root, options.timingJson, plan, timings, 0);
}

function writeTimingJson(root, timingJson, plan, timings, exitCode) {
  if (!timingJson) return;
  const out = {
    tool: "browsergrad-cuda-lite-test-scope",
    exitCode,
    files: plan.files,
    scopes: plan.scopes.map((scope) => ({
      id: scope.id,
      reason: scope.reason,
      files: scope.files,
      ...(scope.requested ? { requested: true } : {}),
    })),
    totalDurationMs: timings.reduce((sum, item) => sum + item.durationMs, 0),
    commands: timings,
  };
  const target = path.resolve(root, timingJson);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, `${JSON.stringify(out, null, 2)}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
