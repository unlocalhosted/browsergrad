import { createHash } from "node:crypto";

export const CPP_CUTE_BROWSER_LIBCXX_CONFIG_SITE_PROFILE =
  "browsergrad.compiler.cpp-cute.libcxx-config-site.linux-x86_64-glibc@1";

const ERROR_CODE =
  "BG-COMPILER-CPP-CUTE-BROWSER-LIBCXX-CONFIG-SITE";
const TEXT_DECODER = new TextDecoder("utf-8", { fatal: true });
const TEXT_ENCODER = new TextEncoder();
const MAX_TEMPLATE_BYTES = 16 * 1024;
const REPLACEMENTS = new Map([
  ["#cmakedefine _LIBCPP_ABI_VERSION @_LIBCPP_ABI_VERSION@", "#define _LIBCPP_ABI_VERSION 1"],
  ["#cmakedefine _LIBCPP_ABI_NAMESPACE @_LIBCPP_ABI_NAMESPACE@", "#define _LIBCPP_ABI_NAMESPACE __1"],
  ["#cmakedefine01 _LIBCPP_ABI_FORCE_ITANIUM", "#define _LIBCPP_ABI_FORCE_ITANIUM 0"],
  ["#cmakedefine01 _LIBCPP_ABI_FORCE_MICROSOFT", "#define _LIBCPP_ABI_FORCE_MICROSOFT 0"],
  ["#cmakedefine01 _LIBCPP_HAS_THREADS", "#define _LIBCPP_HAS_THREADS 1"],
  ["#cmakedefine01 _LIBCPP_HAS_MONOTONIC_CLOCK", "#define _LIBCPP_HAS_MONOTONIC_CLOCK 1"],
  ["#cmakedefine01 _LIBCPP_HAS_TERMINAL", "#define _LIBCPP_HAS_TERMINAL 1"],
  ["#cmakedefine01 _LIBCPP_HAS_MUSL_LIBC", "#define _LIBCPP_HAS_MUSL_LIBC 0"],
  ["#cmakedefine01 _LIBCPP_HAS_THREAD_API_PTHREAD", "#define _LIBCPP_HAS_THREAD_API_PTHREAD 0"],
  ["#cmakedefine01 _LIBCPP_HAS_THREAD_API_EXTERNAL", "#define _LIBCPP_HAS_THREAD_API_EXTERNAL 0"],
  ["#cmakedefine01 _LIBCPP_HAS_THREAD_API_WIN32", "#define _LIBCPP_HAS_THREAD_API_WIN32 0"],
  ["#cmakedefine _LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS", "/* #undef _LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS */"],
  ["#cmakedefine01 _LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS", "#define _LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS 0"],
  ["#cmakedefine _LIBCPP_NO_VCRUNTIME", "/* #undef _LIBCPP_NO_VCRUNTIME */"],
  [
    "#cmakedefine _LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION @_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION@",
    "/* #undef _LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION */",
  ],
  ["#cmakedefine01 _LIBCPP_HAS_FILESYSTEM", "#define _LIBCPP_HAS_FILESYSTEM 1"],
  ["#cmakedefine01 _LIBCPP_HAS_RANDOM_DEVICE", "#define _LIBCPP_HAS_RANDOM_DEVICE 1"],
  ["#cmakedefine01 _LIBCPP_HAS_LOCALIZATION", "#define _LIBCPP_HAS_LOCALIZATION 1"],
  ["#cmakedefine01 _LIBCPP_HAS_UNICODE", "#define _LIBCPP_HAS_UNICODE 1"],
  ["#cmakedefine01 _LIBCPP_HAS_WIDE_CHARACTERS", "#define _LIBCPP_HAS_WIDE_CHARACTERS 1"],
  ["#cmakedefine01 _LIBCPP_HAS_TIME_ZONE_DATABASE", "#define _LIBCPP_HAS_TIME_ZONE_DATABASE 1"],
  ["#cmakedefine01 _LIBCPP_INSTRUMENTED_WITH_ASAN", "#define _LIBCPP_INSTRUMENTED_WITH_ASAN 0"],
  ["#cmakedefine _LIBCPP_PSTL_BACKEND_SERIAL", "/* #undef _LIBCPP_PSTL_BACKEND_SERIAL */"],
  ["#cmakedefine _LIBCPP_PSTL_BACKEND_STD_THREAD", "#define _LIBCPP_PSTL_BACKEND_STD_THREAD 1"],
  ["#cmakedefine _LIBCPP_PSTL_BACKEND_LIBDISPATCH", "/* #undef _LIBCPP_PSTL_BACKEND_LIBDISPATCH */"],
  ["#cmakedefine _LIBCPP_HARDENING_MODE_DEFAULT @_LIBCPP_HARDENING_MODE_DEFAULT@", "#define _LIBCPP_HARDENING_MODE_DEFAULT 2"],
  [
    "#cmakedefine _LIBCPP_ASSERTION_SEMANTIC_DEFAULT @_LIBCPP_ASSERTION_SEMANTIC_DEFAULT@",
    "#define _LIBCPP_ASSERTION_SEMANTIC_DEFAULT 2",
  ],
  ["#cmakedefine01 _LIBCPP_LIBC_PICOLIBC", "#define _LIBCPP_LIBC_PICOLIBC 0"],
  ["#cmakedefine01 _LIBCPP_LIBC_NEWLIB", "#define _LIBCPP_LIBC_NEWLIB 0"],
  ["@_LIBCPP_ABI_DEFINES@", ""],
  ["@_LIBCPP_EXTRA_SITE_DEFINES@", ""],
]);

export class CppCuteBrowserLibcxxConfigSiteError extends Error {
  /**
   * @param {string} message
   * @param {ErrorOptions=} options
   */
  constructor(message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserLibcxxConfigSiteError";
    this.code = ERROR_CODE;
  }
}

/**
 * Applies the exact libc++ 22 Linux/x86_64/glibc configuration used by the
 * CUDA host pass. The upstream template remains the licensed source of the
 * generated header; this transform is deliberately closed over every CMake
 * placeholder so new upstream configuration inputs fail instead of drifting.
 *
 * @param {Uint8Array} templateBytes
 */
export function materializeCppCuteBrowserLibcxxConfigSite(templateBytes) {
  if (!(templateBytes instanceof Uint8Array) ||
      templateBytes.byteLength === 0 ||
      templateBytes.byteLength > MAX_TEMPLATE_BYTES) {
    invalid("expected one bounded Uint8Array libc++ configuration template");
  }
  /** @type {string} */
  let template;
  try {
    template = TEXT_DECODER.decode(templateBytes);
  } catch (cause) {
    invalid("libc++ configuration template is not strict UTF-8", { cause });
  }
  if (!template.endsWith("\n") ||
      !template.includes("#ifndef _LIBCPP___CONFIG_SITE\n") ||
      !template.includes("#endif // _LIBCPP___CONFIG_SITE\n")) {
    invalid("libc++ configuration template guard or final newline differs");
  }
  const seen = new Set();
  const configured = template.split("\n").map((line) => {
    const replacement = REPLACEMENTS.get(line);
    if (replacement === undefined) return line;
    if (seen.has(line)) invalid(`duplicate configuration directive ${JSON.stringify(line)}`);
    seen.add(line);
    return replacement;
  }).join("\n");
  if (seen.size !== REPLACEMENTS.size ||
      configured.includes("#cmakedefine") ||
      /@_[A-Z0-9_]+@/u.test(configured)) {
    invalid("libc++ configuration template has missing or unknown CMake placeholders");
  }
  const bytes = TEXT_ENCODER.encode(configured);
  return Object.freeze({
    profile: CPP_CUTE_BROWSER_LIBCXX_CONFIG_SITE_PROFILE,
    templateSha256: sha256(templateBytes),
    configuredSha256: sha256(bytes),
    bytes,
  });
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
  throw new CppCuteBrowserLibcxxConfigSiteError(message, options);
}
