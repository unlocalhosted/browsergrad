#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const compilerSrc = path.join(repoRoot, "packages/browsergrad-compiler/src");
const minLines = Number.parseInt(process.argv.find((arg) => arg.startsWith("--min-lines="))?.split("=")[1] ?? "1000", 10);

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

function importsOf(source) {
  return [...source.matchAll(/^import[\s\S]*?from\s+["'](.+?)["'];/gm)].map((match) => match[1]);
}

function exportsOf(source) {
  return [...source.matchAll(/^export\s+(?:async\s+)?(?:function|class|interface|type|const)\s+(\w+)/gm)].map((match) => match[1]);
}

function functionsOf(source) {
  return [...source.matchAll(/^(?:export\s+)?(?:async\s+)?function\s+(\w+)/gm)].map((match) => match[1]);
}

function featureBucket(file) {
  const normalized = rel(file);
  if (normalized.includes("/features/")) return "vertical-feature";
  if (/semantic_/u.test(normalized) || normalized.endsWith("/semantic_ir.ts")) return "semantic-ir";
  if (/wgsl/u.test(normalized)) return "wgsl-backend";
  if (/reference/u.test(normalized)) return "reference";
  if (/analyzer|parser|lexer|diagnostics/u.test(normalized)) return "frontend";
  if (/runtime|dynamic_launch|peer_copy|webgpu_orchestration|runner/u.test(normalized)) return "runtime-orchestration";
  return "support";
}

const rows = walk(compilerSrc).map((file) => {
  const source = fs.readFileSync(file, "utf8");
  return {
    file: rel(file),
    lines: source.split("\n").length,
    imports: importsOf(source).length,
    exports: exportsOf(source).length,
    functions: functionsOf(source).length,
    bucket: featureBucket(file),
  };
}).sort((left, right) => right.lines - left.lines);

const buckets = new Map();
for (const row of rows) {
  const bucket = buckets.get(row.bucket) ?? { files: 0, lines: 0 };
  bucket.files += 1;
  bucket.lines += row.lines;
  buckets.set(row.bucket, bucket);
}

console.log("Compiler Architecture Map");
console.log(`root: ${rel(compilerSrc)}`);
console.log("");
console.log("Large modules");
for (const row of rows.filter((entry) => entry.lines >= minLines)) {
  console.log(`${String(row.lines).padStart(5)}  ${row.file}  bucket=${row.bucket} imports=${row.imports} exports=${row.exports} funcs=${row.functions}`);
}
console.log("");
console.log("Buckets");
for (const [bucket, summary] of [...buckets.entries()].sort((left, right) => right[1].lines - left[1].lines)) {
  console.log(`${String(summary.lines).padStart(5)}  ${bucket}  files=${summary.files}`);
}
