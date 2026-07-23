import { createHash } from "node:crypto";

export const CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_PROFILE =
  "browsergrad.compiler.cpp-cute.clang-cuda-runtime-wrapper.cuda-12.6-libcxx-x86_64@1";
export const CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_VIRTUAL_PATH =
  "__clang_cuda_runtime_wrapper.h";
export const CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_UPSTREAM_SHA256 =
  "877d48f0f311943eacdef11807c1935108c7d9b083da6974b764a97c478648bc";
export const CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_UPSTREAM_BYTE_LENGTH =
  "18624";

const ERROR_CODE =
  "BG-COMPILER-CPP-CUTE-BROWSER-CLANG-CUDA-RUNTIME-WRAPPER";
const TEXT_DECODER = new TextDecoder("utf-8", { fatal: true });
const TEXT_ENCODER = new TextEncoder();
const MAX_TEMPLATE_BYTES = 64 * 1024;
const CUDA_CLANG_GUARD = "#if defined(__CUDA__) && defined(__clang__)\n";
const CUDA_CLANG_GUARD_CONFIGURED = [
  CUDA_CLANG_GUARD.trimEnd(),
  "// BrowserGrad's closed x86_64/libc++ CUDA 12.6 profile uses NVIDIA's",
  "// header-provided escape hatch only while the Clang-owned wrapper is active.",
  '#pragma push_macro("_ALLOW_UNSUPPORTED_LIBCPP")',
  "#undef _ALLOW_UNSUPPORTED_LIBCPP",
  "#define _ALLOW_UNSUPPORTED_LIBCPP 1",
  "",
].join("\n");
const HOST_POISON = [
  '#pragma push_macro("__host__")',
  "#define __host__ UNEXPECTED_HOST_ATTRIBUTE",
].join("\n");
const HOST_POISON_CONFIGURED = [
  '#pragma push_macro("__host__")',
  "#undef __host__",
  "#define __host__ UNEXPECTED_HOST_ATTRIBUTE",
].join("\n");
const HOST_ERASURE = [
  "#define __host__",
  "#undef __CUDABE__",
].join("\n");
const HOST_ERASURE_CONFIGURED = [
  "#undef __host__",
  "#define __host__",
  "#undef __CUDABE__",
].join("\n");
const REDUNDANT_FAST_MATH_POP = [
  '#pragma pop_macro("uint3")',
  '#pragma pop_macro("__USE_FAST_MATH__")',
  '#pragma pop_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")',
].join("\n");
const CONFIGURED_FAST_MATH_POP = [
  '#pragma pop_macro("uint3")',
  '#pragma pop_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")',
].join("\n");
const CUDA_CLANG_CLOSE = "#endif // __CUDA__\n";
const CUDA_CLANG_CLOSE_CONFIGURED = [
  '#pragma pop_macro("_ALLOW_UNSUPPORTED_LIBCPP")',
  "",
  CUDA_CLANG_CLOSE.trimEnd(),
  "",
].join("\n");

export class CppCuteBrowserClangCudaRuntimeWrapperError extends Error {
  /**
   * @param {string} message
   * @param {ErrorOptions=} options
   */
  constructor(message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserClangCudaRuntimeWrapperError";
    this.code = ERROR_CODE;
  }
}

/**
 * Configures Clang 22's own forced CUDA wrapper for the exact browser host
 * profile. The transform is intentionally narrow:
 *
 * - NVIDIA's `_ALLOW_UNSUPPORTED_LIBCPP` escape hatch is scoped to the wrapper;
 * - Clang's two intentional `__host__` replacements first undefine the prior
 *   CUDA definition, preserving the global macro-redefinition error policy;
 * - one unmatched upstream `__USE_FAST_MATH__` pop is removed.
 *
 * Exact upstream identity is bound by the header-source plan and inventory.
 * This function additionally closes over the expected wrapper structure so an
 * upstream control-flow change fails instead of receiving a textual patch.
 *
 * @param {Uint8Array} templateBytes
 */
export function materializeCppCuteBrowserClangCudaRuntimeWrapper(templateBytes) {
  if (!(templateBytes instanceof Uint8Array) ||
      templateBytes.byteLength === 0 ||
      templateBytes.byteLength > MAX_TEMPLATE_BYTES) {
    invalid("expected one bounded Uint8Array Clang CUDA runtime wrapper");
  }
  /** @type {string} */
  let template;
  try {
    template = TEXT_DECODER.decode(templateBytes);
  } catch (cause) {
    invalid("Clang CUDA runtime wrapper is not strict UTF-8", { cause });
  }
  if (!template.endsWith("\n") ||
      !template.includes("#ifndef __CLANG_CUDA_RUNTIME_WRAPPER_H__\n") ||
      !template.includes("#define __CLANG_CUDA_RUNTIME_WRAPPER_H__\n") ||
      !template.endsWith(
        "#endif // __CUDA__\n#endif // __CLANG_CUDA_RUNTIME_WRAPPER_H__\n",
      ) ||
      template.includes("_ALLOW_UNSUPPORTED_LIBCPP")) {
    invalid("Clang CUDA runtime wrapper guard or compatibility surface differs");
  }
  /** @type {Array<readonly [string, string]>} */
  const replacements = [
    [CUDA_CLANG_GUARD, CUDA_CLANG_GUARD_CONFIGURED],
    [HOST_POISON, HOST_POISON_CONFIGURED],
    [HOST_ERASURE, HOST_ERASURE_CONFIGURED],
    [REDUNDANT_FAST_MATH_POP, CONFIGURED_FAST_MATH_POP],
    [CUDA_CLANG_CLOSE, CUDA_CLANG_CLOSE_CONFIGURED],
  ];
  const configured = replacements.reduce(
    (text, [needle, replacement]) => replaceExactlyOnce(text, needle, replacement),
    template,
  );
  if (count(configured, '#pragma push_macro("__host__")') !== 1 ||
      count(configured, '#pragma pop_macro("__host__")') !== 1 ||
      count(configured, '#pragma push_macro("__USE_FAST_MATH__")') !== 1 ||
      count(configured, '#pragma pop_macro("__USE_FAST_MATH__")') !== 1 ||
      count(configured, '#pragma push_macro("_ALLOW_UNSUPPORTED_LIBCPP")') !== 1 ||
      count(configured, '#pragma pop_macro("_ALLOW_UNSUPPORTED_LIBCPP")') !== 1) {
    invalid("configured Clang CUDA runtime wrapper macro scopes are unbalanced");
  }
  const bytes = TEXT_ENCODER.encode(configured);
  return Object.freeze({
    profile: CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_PROFILE,
    templateSha256: sha256(templateBytes),
    configuredSha256: sha256(bytes),
    bytes,
  });
}

/**
 * @param {string} value
 * @param {string} needle
 * @param {string} replacement
 */
function replaceExactlyOnce(value, needle, replacement) {
  const first = value.indexOf(needle);
  if (first < 0 || first !== value.lastIndexOf(needle)) {
    invalid(`expected exactly one reviewed wrapper fragment ${JSON.stringify(needle)}`);
  }
  return `${value.slice(0, first)}${replacement}${value.slice(first + needle.length)}`;
}

/** @param {string} value @param {string} needle */
function count(value, needle) {
  let occurrences = 0;
  let offset = 0;
  while (true) {
    const found = value.indexOf(needle, offset);
    if (found < 0) return occurrences;
    occurrences += 1;
    offset = found + needle.length;
  }
}

/** @param {Uint8Array} bytes */
function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

/**
 * @param {string} message
 * @param {ErrorOptions=} options
 * @returns {never}
 */
function invalid(message, options) {
  throw new CppCuteBrowserClangCudaRuntimeWrapperError(message, options);
}
