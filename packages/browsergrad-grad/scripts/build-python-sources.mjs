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

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PYTHON_DIR = resolve(__dirname, "..", "src", "python");

/** Each entry: [py filename, generated.ts filename, exported const name]. */
const MODULES = [
  ["tensor.py", "tensor.generated.ts", "TENSOR_PY"],
  ["functional.py", "functional.generated.ts", "FUNCTIONAL_PY"],
  ["nn.py", "nn.generated.ts", "NN_PY"],
  ["optim.py", "optim.generated.ts", "OPTIM_PY"],
  ["torch_compat.py", "torch_compat.generated.ts", "TORCH_COMPAT_PY"],
  ["utils_data.py", "utils_data.generated.ts", "UTILS_DATA_PY"],
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
process.stdout.write(`codegen: wrote ${wrote} files\n`);
