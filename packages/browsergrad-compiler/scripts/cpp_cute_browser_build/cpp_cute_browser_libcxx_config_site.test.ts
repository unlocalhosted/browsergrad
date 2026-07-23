import { describe, expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_LIBCXX_CONFIG_SITE_PROFILE,
  CppCuteBrowserLibcxxConfigSiteError,
  materializeCppCuteBrowserLibcxxConfigSite,
} from "./cpp_cute_browser_libcxx_config_site.mjs";

describe("configured libc++ header materialization", () => {
  it("closes every upstream CMake placeholder over the browser CUDA host profile", () => {
    const configured = materializeCppCuteBrowserLibcxxConfigSite(
      new TextEncoder().encode(configSiteTemplate()),
    );
    const text = new TextDecoder().decode(configured.bytes);

    expect(configured.profile).toBe(CPP_CUTE_BROWSER_LIBCXX_CONFIG_SITE_PROFILE);
    expect(text).toContain("#define _LIBCPP_ABI_NAMESPACE __1");
    expect(text).toContain("#define _LIBCPP_HAS_THREADS 1");
    expect(text).toContain("#define _LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS 0");
    expect(text).toContain("#define _LIBCPP_PSTL_BACKEND_STD_THREAD 1");
    expect(text).toContain("#define _LIBCPP_HARDENING_MODE_DEFAULT 2");
    expect(text).not.toMatch(/#cmakedefine|@_[A-Z0-9_]+@/u);
  });

  it("fails closed on a new or missing upstream configuration input", () => {
    const template = configSiteTemplate();
    expect(() => materializeCppCuteBrowserLibcxxConfigSite(
      new TextEncoder().encode(template.replace(
        "#cmakedefine01 _LIBCPP_HAS_THREADS",
        "#cmakedefine01 _LIBCPP_HAS_FUTURE_CAPABILITY",
      )),
    )).toThrow(CppCuteBrowserLibcxxConfigSiteError);
  });
});

function configSiteTemplate(): string {
  return [
    "// LLVM libc++ configuration fixture",
    "#ifndef _LIBCPP___CONFIG_SITE",
    "#define _LIBCPP___CONFIG_SITE",
    "#cmakedefine _LIBCPP_ABI_VERSION @_LIBCPP_ABI_VERSION@",
    "#cmakedefine _LIBCPP_ABI_NAMESPACE @_LIBCPP_ABI_NAMESPACE@",
    "#cmakedefine01 _LIBCPP_ABI_FORCE_ITANIUM",
    "#cmakedefine01 _LIBCPP_ABI_FORCE_MICROSOFT",
    "#cmakedefine01 _LIBCPP_HAS_THREADS",
    "#cmakedefine01 _LIBCPP_HAS_MONOTONIC_CLOCK",
    "#cmakedefine01 _LIBCPP_HAS_TERMINAL",
    "#cmakedefine01 _LIBCPP_HAS_MUSL_LIBC",
    "#cmakedefine01 _LIBCPP_HAS_THREAD_API_PTHREAD",
    "#cmakedefine01 _LIBCPP_HAS_THREAD_API_EXTERNAL",
    "#cmakedefine01 _LIBCPP_HAS_THREAD_API_WIN32",
    "#define _LIBCPP_HAS_THREAD_API_C11 0 // FIXME: Is this guarding dead code?",
    "#cmakedefine _LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS",
    "#cmakedefine01 _LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS",
    "#cmakedefine _LIBCPP_NO_VCRUNTIME",
    "#cmakedefine _LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION @_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION@",
    "#cmakedefine01 _LIBCPP_HAS_FILESYSTEM",
    "#cmakedefine01 _LIBCPP_HAS_RANDOM_DEVICE",
    "#cmakedefine01 _LIBCPP_HAS_LOCALIZATION",
    "#cmakedefine01 _LIBCPP_HAS_UNICODE",
    "#cmakedefine01 _LIBCPP_HAS_WIDE_CHARACTERS",
    "#cmakedefine01 _LIBCPP_HAS_TIME_ZONE_DATABASE",
    "#cmakedefine01 _LIBCPP_INSTRUMENTED_WITH_ASAN",
    "#cmakedefine _LIBCPP_PSTL_BACKEND_SERIAL",
    "#cmakedefine _LIBCPP_PSTL_BACKEND_STD_THREAD",
    "#cmakedefine _LIBCPP_PSTL_BACKEND_LIBDISPATCH",
    "#cmakedefine _LIBCPP_HARDENING_MODE_DEFAULT @_LIBCPP_HARDENING_MODE_DEFAULT@",
    "#cmakedefine _LIBCPP_ASSERTION_SEMANTIC_DEFAULT @_LIBCPP_ASSERTION_SEMANTIC_DEFAULT@",
    "#cmakedefine01 _LIBCPP_LIBC_PICOLIBC",
    "#cmakedefine01 _LIBCPP_LIBC_NEWLIB",
    "@_LIBCPP_ABI_DEFINES@",
    "@_LIBCPP_EXTRA_SITE_DEFINES@",
    "#endif // _LIBCPP___CONFIG_SITE",
    "",
  ].join("\n");
}
