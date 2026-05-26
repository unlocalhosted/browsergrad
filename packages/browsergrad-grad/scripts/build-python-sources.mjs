#!/usr/bin/env node
/**
 * Codegen: src/python/*.py → src/python/*.generated.ts
 *
 * Each `.py` file becomes a TypeScript module exporting its content as a
 * string constant. The string is base64-encoded inside the generated file
 * to bypass every quoting/escaping concern (the same idiom used by
 * `pythonStringLiteral` in src/install.ts).
 *
 * Run via `pnpm codegen`. The build script runs this first, so generated
 * files are always fresh in CI. The generated files are committed so
 * fresh clones work without running codegen first.
 *
 * To add a new Python module: drop a `.py` file in src/python/, add an
 * entry to MODULES below, then run `pnpm codegen`.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PYTHON_DIR = resolve(__dirname, "..", "src", "python");

/** Each entry: [py filename, generated.ts filename, exported const name]. */
const MODULES = [
  ["tensor.py", "tensor.generated.ts", "TENSOR_PY"],
  ["functional.py", "functional.generated.ts", "FUNCTIONAL_PY"],
  ["optim.py", "optim.generated.ts", "OPTIM_PY"],
  ["torch_compat.py", "torch_compat.generated.ts", "TORCH_COMPAT_PY"],
  ["_torch_compat_real.py", "_torch_compat_real.generated.ts", "TORCH_COMPAT_REAL_PY"],
  ["_torch_compat_limited.py", "_torch_compat_limited.generated.ts", "TORCH_COMPAT_LIMITED_PY"],
  ["_torch_compat_impossible.py", "_torch_compat_impossible.generated.ts", "TORCH_COMPAT_IMPOSSIBLE_PY"],
  ["utils_data.py", "utils_data.generated.ts", "UTILS_DATA_PY"],
];

/**
 * Concatenation-based chunked modules: source is a directory of .py chunk
 * files, assembled in the order specified by `order` into one Python string.
 * Each chunk maps 1:1 to a class family (linear, conv, norm, ...). Adding
 * a new chunk family means adding the file AND adding its name to `order`
 * — chunks not in `order` are silently absent.
 */
const CHUNKED_MODULES = [
  {
    chunksDir: "nn_chunks",
    out: "nn.generated.ts",
    constName: "NN_PY",
    order: [
      "module",
      "linear",
      "conv",
      "norm",
      "pool",
      "dropout",
      "activation",
      "embedding",
      "recurrent",
      "attention",
      "loss",
      "init",
    ],
  },
];

function toBase64(content) {
  // Latin-1 round-trip via Buffer so non-ASCII bytes are preserved.
  return Buffer.from(content, "utf8").toString("base64");
}

function emit(constName, b64, sourcePath) {
  return `// @generated — do not edit by hand. Source: src/python/${sourcePath}
// Regenerate via \`pnpm codegen\`.

export const ${constName}: string = ((): string => {
  const b64 = "${b64}";
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder("utf-8").decode(bytes);
})();
`;
}

// Fail-fast pre-check: every input file must exist before we emit anything.
// Without this, a missing chunk would either ENOENT mid-loop (after some
// generated files were already updated, leaving the repo in a half-emitted
// state) or — worse for CHUNKED_MODULES — produce a silently truncated
// concatenation that fails at Pyodide import with a cryptic Python
// traceback instead of here with a clear file path.
const missing = [];
for (const [py] of MODULES) {
  if (!existsSync(resolve(PYTHON_DIR, py))) missing.push(py);
}
for (const { chunksDir, order } of CHUNKED_MODULES) {
  for (const chunk of order) {
    const chunkPath = resolve(PYTHON_DIR, chunksDir, `${chunk}.py`);
    if (!existsSync(chunkPath)) missing.push(`${chunksDir}/${chunk}.py`);
  }
}
if (missing.length) {
  process.stderr.write(
    `codegen: refusing to emit — missing source files:\n` +
      missing.map((m) => `  - ${m}\n`).join("") +
      `Check MODULES / CHUNKED_MODULES in scripts/build-python-sources.mjs.\n`,
  );
  process.exit(1);
}

let wrote = 0;
for (const [py, gen, constName] of MODULES) {
  const pyPath = resolve(PYTHON_DIR, py);
  const genPath = resolve(PYTHON_DIR, gen);
  const content = readFileSync(pyPath, "utf8");
  const b64 = toBase64(content);
  const tsSource = emit(constName, b64, py);
  writeFileSync(genPath, tsSource);
  wrote += 1;
  process.stdout.write(`  ${py} → ${gen} (${content.length} chars)\n`);
}

for (const { chunksDir, out, constName, order } of CHUNKED_MODULES) {
  const parts = [];
  for (const chunk of order) {
    const chunkPath = resolve(PYTHON_DIR, chunksDir, `${chunk}.py`);
    parts.push(readFileSync(chunkPath, "utf8"));
  }
  const content = parts.join("\n");
  const b64 = toBase64(content);
  const genPath = resolve(PYTHON_DIR, out);
  const tsSource = emit(constName, b64, `${chunksDir}/{${order.join(",")}}.py`);
  writeFileSync(genPath, tsSource);
  wrote += 1;
  process.stdout.write(`  ${chunksDir}/ (${order.length} chunks) → ${out} (${content.length} chars)\n`);
}

process.stdout.write(`codegen: wrote ${wrote} files\n`);
