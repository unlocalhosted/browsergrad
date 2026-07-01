import fs from "node:fs";
import path from "node:path";

export function findRepoRoot(start) {
  let current = path.resolve(start);
  while (true) {
    if (fs.existsSync(path.join(current, "pnpm-workspace.yaml"))) return current;
    const parent = path.dirname(current);
    if (parent === current) return path.resolve(start);
    current = parent;
  }
}

export function parseBundle(value) {
  if (value === "src" || value === "dist") return value;
  throw new Error("--bundle expects src or dist");
}

export function parseAutoCorpusSmokeMode(value) {
  if (value === "reference" || value === "dispatch") return value;
  throw new Error("--auto-corpus-smoke-mode expects reference or dispatch");
}

export function parseAutoCorpusSmokeProfile(value) {
  if (value === "full" || value === "fast") return value;
  throw new Error("--auto-corpus-smoke-profile expects full or fast");
}

export function parseAutoCorpusSmokeFeatures(value) {
  if (value === "") return [];
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

export function parseAutoCorpusSmokeShard(value) {
  if (value === undefined || value === "" || value === "1/1") return { index: 1, count: 1 };
  const match = /^(\d+)\/(\d+)$/u.exec(value);
  if (!match) throw new Error("--auto-corpus-smoke-shard expects N/M, e.g. 1/4");
  const index = Number(match[1]);
  const count = Number(match[2]);
  if (!Number.isInteger(index) || !Number.isInteger(count) || count < 1 || index < 1 || index > count) {
    throw new Error("--auto-corpus-smoke-shard expects 1 <= N <= M");
  }
  return { index, count };
}

export function applyAutoCorpusSmokeShard(fixtures, shard) {
  if (shard.count <= 1) return fixtures;
  return fixtures.filter((_, index) => index % shard.count === shard.index - 1);
}

export function parseNonNegativeInteger(value, flag) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) throw new Error(`${flag} expects a non-negative integer`);
  return parsed;
}

export function parseFlagArgs(argv) {
  const parsed = new Map();
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--") continue;
    if (!arg?.startsWith("--")) continue;
    const equalsIndex = arg.indexOf("=");
    if (equalsIndex > 2) {
      parsed.set(arg.slice(0, equalsIndex), arg.slice(equalsIndex + 1));
      continue;
    }
    const value = argv[index + 1];
    parsed.set(arg, value?.startsWith("--") || value === undefined ? "true" : value);
    if (value && !value.startsWith("--")) index++;
  }
  return parsed;
}

export function effectiveAutoCorpusSmokeLimit(autoCorpusSmokeLimit, caseFilters, targetedFallbackLimit = 64) {
  if (autoCorpusSmokeLimit > 0) return autoCorpusSmokeLimit;
  return caseFilters.some((filter) => filter.startsWith("auto-corpus:")) ? targetedFallbackLimit : 0;
}

export function parseCaseFilters(argv) {
  const values = [];
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--case" || arg === "--cases" || arg === "--only") {
      const value = argv[index + 1];
      if (value && !value.startsWith("--")) {
        values.push(value);
        index++;
      }
      continue;
    }
    if (arg?.startsWith("--case=")) values.push(arg.slice("--case=".length));
    if (arg?.startsWith("--cases=")) values.push(arg.slice("--cases=".length));
    if (arg?.startsWith("--only=")) values.push(arg.slice("--only=".length));
  }
  return values.join(",")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function moduleAliases(root, bundleName) {
  const compilerEntry = bundleName === "dist"
    ? path.join(root, "packages/browsergrad-compiler/dist/index.js")
    : path.join(root, "packages/browsergrad-compiler/src/index.ts");
  const kernelsEntry = bundleName === "dist"
    ? path.join(root, "packages/browsergrad-kernels/dist/index.js")
    : path.join(root, "packages/browsergrad-kernels/src/index.ts");
  return {
    "@unlocalhosted/browsergrad-kernels": kernelsEntry,
    "@unlocalhosted/browsergrad-compiler": compilerEntry,
  };
}
