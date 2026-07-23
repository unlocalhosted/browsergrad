#include "BrowserGradCppCuteArtifactV3.h"
#include "BrowserGradCppCuteMetrics.h"
#include "BrowserGradCppCuteRuntime.h"

#include <cstdint>

#if defined(__EMSCRIPTEN__)
#define BG_CPP_CUTE_EXPORT \
  __attribute__((used)) __attribute__((visibility("default")))
#else
#define BG_CPP_CUTE_EXPORT
#endif

extern "C" {

BG_CPP_CUTE_EXPORT std::uint32_t bg_cpp_cute_abi_version(void) {
  return browsergrad::cpp_cute::runtime_abi_version();
}

BG_CPP_CUTE_EXPORT std::uint32_t bg_cpp_cute_alloc(
    std::uint32_t byte_length) {
  return browsergrad::cpp_cute::runtime_allocate(byte_length);
}

BG_CPP_CUTE_EXPORT std::uint32_t
bg_cpp_cute_allocator_metrics_pointer(void) {
  return browsergrad::cpp_cute::allocator_metrics_pointer();
}

BG_CPP_CUTE_EXPORT std::uint32_t
bg_cpp_cute_frontend_work_metrics_pointer(void) {
  return browsergrad::cpp_cute::frontend_work_metrics_pointer();
}

BG_CPP_CUTE_EXPORT std::uint32_t bg_cpp_cute_last_diagnostic_code(void) {
  return browsergrad::cpp_cute::runtime_last_diagnostic_code();
}

BG_CPP_CUTE_EXPORT std::int32_t bg_cpp_cute_compile(
    std::uint32_t input_pointer, std::uint32_t input_length) {
  return browsergrad::cpp_cute::runtime_compile(
      input_pointer, input_length,
      browsergrad::cpp_cute::build_artifact_v3);
}

BG_CPP_CUTE_EXPORT void bg_cpp_cute_free(std::uint32_t pointer,
                                         std::uint32_t byte_length) {
  browsergrad::cpp_cute::runtime_free(pointer, byte_length);
}

BG_CPP_CUTE_EXPORT void bg_cpp_cute_reset(void) {
  browsergrad::cpp_cute::runtime_reset();
}

BG_CPP_CUTE_EXPORT std::uint32_t bg_cpp_cute_result_length(void) {
  return browsergrad::cpp_cute::runtime_result_length();
}

BG_CPP_CUTE_EXPORT std::uint32_t bg_cpp_cute_result_pointer(void) {
  return browsergrad::cpp_cute::runtime_result_pointer();
}

BG_CPP_CUTE_EXPORT std::int32_t bg_cpp_cute_status(void) {
  return browsergrad::cpp_cute::runtime_status();
}

}  // extern "C"

#undef BG_CPP_CUTE_EXPORT
