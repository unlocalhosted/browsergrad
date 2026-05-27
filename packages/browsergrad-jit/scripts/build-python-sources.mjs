#!/usr/bin/env node
/**
 * Codegen: src/python/*.py → src/python/*.generated.ts
 *
 * Each `.py` file becomes a TypeScript module exporting its content as a
 * string constant. The string is base64-encoded inside the generated file
 * to bypass every quoting/escaping concern (same idiom as browsergrad-grad).
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
  ["_ir.py", "_ir.generated.ts", "IR_PY"],
  ["_errors.py", "_errors.generated.ts", "ERRORS_PY"],
  ["_buffer_table.py", "_buffer_table.generated.ts", "BUFFER_TABLE_PY"],
  ["_realize.py", "_realize.generated.ts", "REALIZE_PY"],
  ["_fusion.py", "_fusion.generated.ts", "FUSION_PY"],
  ["_fusion_config.py", "_fusion_config.generated.ts", "FUSION_CONFIG_PY"],
  ["_vjp.py", "_vjp.generated.ts", "VJP_PY"],
  ["_trace_cache.py", "_trace_cache.generated.ts", "TRACE_CACHE_PY"],
  ["_safetensors.py", "_safetensors.generated.ts", "SAFETENSORS_PY"],
  ["_checkpoint.py", "_checkpoint.generated.ts", "CHECKPOINT_PY"],
  ["_utils_checkpoint.py", "_utils_checkpoint.generated.ts", "UTILS_CHECKPOINT_PY"],
  ["_amp.py", "_amp.generated.ts", "AMP_PY"],
  ["_bridge.py", "_bridge.generated.ts", "BRIDGE_PY"],
  ["_gpu_buffer_table.py", "_gpu_buffer_table.generated.ts", "GPU_BUFFER_TABLE_PY"],
  ["_realize_webgpu.py", "_realize_webgpu.generated.ts", "REALIZE_WEBGPU_PY"],
  ["_func.py", "_func.generated.ts", "FUNC_PY"],
  ["_vmap.py", "_vmap.generated.ts", "VMAP_PY"],
  ["_custom_kernel.py", "_custom_kernel.generated.ts", "CUSTOM_KERNEL_PY"],
  ["_onnx.py", "_onnx.generated.ts", "ONNX_PY"],
  ["_lab.py", "_lab.generated.ts", "LAB_PY"],
  ["_tensor_proxy.py", "_tensor_proxy.generated.ts", "TENSOR_PROXY_PY"],
  ["_nn.py", "_nn.generated.ts", "NN_PY"],
  ["_functional.py", "_functional.generated.ts", "FUNCTIONAL_PY"],
  ["_optim.py", "_optim.generated.ts", "OPTIM_PY"],
  ["_torch_compat.py", "_torch_compat.generated.ts", "TORCH_COMPAT_PY"],
];

function toBase64(content) {
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

// Fail-fast: every input file must exist before we emit anything.
// Without this, ENOENT mid-loop would leave the repo in a half-emitted state.
const missing = [];
for (const [py] of MODULES) {
  if (!existsSync(resolve(PYTHON_DIR, py))) missing.push(py);
}
if (missing.length) {
  process.stderr.write(
    `codegen: refusing to emit — missing source files:\n` +
      missing.map((m) => `  - ${m}\n`).join("") +
      `Check MODULES in scripts/build-python-sources.mjs.\n`,
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

process.stdout.write(`codegen: wrote ${wrote} files\n`);
