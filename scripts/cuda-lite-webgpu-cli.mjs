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

export function parseAutoCorpusSmokeFeatures(value) {
  if (value === "") return [];
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

export function parseNonNegativeInteger(value, flag) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) throw new Error(`${flag} expects a non-negative integer`);
  return parsed;
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
