import { describe, expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_PROFILE,
  CppCuteBrowserClangCudaRuntimeWrapperError,
  materializeCppCuteBrowserClangCudaRuntimeWrapper,
} from "./cpp_cute_browser_clang_cuda_runtime_wrapper.mjs";

describe("configured Clang CUDA runtime wrapper", () => {
  it("scopes libc++ compatibility and preserves strict source macro policy", () => {
    const configured = materializeCppCuteBrowserClangCudaRuntimeWrapper(
      new TextEncoder().encode(wrapperFixture()),
    );
    const text = new TextDecoder().decode(configured.bytes);

    expect(configured.profile).toBe(
      CPP_CUTE_BROWSER_CLANG_CUDA_RUNTIME_WRAPPER_PROFILE,
    );
    expect(text).toContain('#pragma push_macro("_ALLOW_UNSUPPORTED_LIBCPP")');
    expect(text).toContain('#pragma pop_macro("_ALLOW_UNSUPPORTED_LIBCPP")');
    expect(text).toContain([
      '#pragma push_macro("__host__")',
      "#undef __host__",
      "#define __host__ UNEXPECTED_HOST_ATTRIBUTE",
    ].join("\n"));
    expect(text).toContain([
      "#undef __host__",
      "#define __host__",
      "#undef __CUDABE__",
    ].join("\n"));
    expect(text.match(/push_macro\("__USE_FAST_MATH__"\)/gu)).toHaveLength(1);
    expect(text.match(/pop_macro\("__USE_FAST_MATH__"\)/gu)).toHaveLength(1);
  });

  it("fails closed when a reviewed wrapper fragment drifts", () => {
    expect(() => materializeCppCuteBrowserClangCudaRuntimeWrapper(
      new TextEncoder().encode(wrapperFixture().replace(
        "#define __host__ UNEXPECTED_HOST_ATTRIBUTE",
        "#define __host__ CHANGED_ATTRIBUTE",
      )),
    )).toThrow(CppCuteBrowserClangCudaRuntimeWrapperError);
    expect(() => materializeCppCuteBrowserClangCudaRuntimeWrapper(
      new Uint8Array([0xff]),
    )).toThrow(CppCuteBrowserClangCudaRuntimeWrapperError);
  });
});

function wrapperFixture(): string {
  return [
    "#ifndef __CLANG_CUDA_RUNTIME_WRAPPER_H__",
    "#define __CLANG_CUDA_RUNTIME_WRAPPER_H__",
    "",
    "#if defined(__CUDA__) && defined(__clang__)",
    '#pragma push_macro("__host__")',
    "#define __host__ UNEXPECTED_HOST_ATTRIBUTE",
    '#pragma push_macro("__USE_FAST_MATH__")',
    '#pragma pop_macro("__USE_FAST_MATH__")',
    "#define __host__",
    "#undef __CUDABE__",
    '#pragma pop_macro("__host__")',
    '#pragma push_macro("uint3")',
    '#pragma pop_macro("uint3")',
    '#pragma pop_macro("__USE_FAST_MATH__")',
    '#pragma pop_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")',
    "#endif // __CUDA__",
    "#endif // __CLANG_CUDA_RUNTIME_WRAPPER_H__",
    "",
  ].join("\n");
}
