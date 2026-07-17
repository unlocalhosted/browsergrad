#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {
bool g_metrics_healthy = true;
}

#define BG_CPP_CUTE_RUNTIME_TESTING 1
#include "extractor/BrowserGradCppCuteRuntime.cpp"

#include "extractor/BrowserGradCppCuteCompileSession.h"

namespace browsergrad::cpp_cute {
bool allocator_metrics_healthy() { return g_metrics_healthy; }
}

namespace {

using namespace browsergrad::cpp_cute;

static_assert(!std::is_copy_constructible_v<DecodedCompileSession>);
static_assert(!std::is_copy_assignable_v<DecodedCompileSession>);
static_assert(!std::is_move_constructible_v<DecodedCompileSession>);
static_assert(!std::is_move_assignable_v<DecodedCompileSession>);

#define BG_CHECK(condition)                                                   \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::fprintf(stderr, "compile-session check failed at line %d: %s\n", \
                   __LINE__, #condition);                                     \
      return 1;                                                               \
    }                                                                         \
  } while (false)

void* g_allocation = nullptr;
std::size_t g_allocation_length = 0U;
constexpr std::uint32_t kWirePointer = 0x1'0000U;

void* test_allocate(std::size_t byte_length) {
  if (g_allocation != nullptr) return nullptr;
  g_allocation = std::malloc(byte_length);
  g_allocation_length = g_allocation == nullptr ? 0U : byte_length;
  return g_allocation;
}

void test_release(void* pointer) {
  if (pointer == nullptr) return;
  if (pointer != g_allocation) std::abort();
  std::free(g_allocation);
  g_allocation = nullptr;
  g_allocation_length = 0U;
}

std::uint32_t test_wire_pointer(const void* pointer) {
  return pointer == g_allocation ? kWirePointer : 0U;
}

CompileSessionDecodeStatus g_decode_status =
    CompileSessionDecodeStatus::kInternalError;
CompileSessionDecodeFailure g_failure{};
std::string g_profile_hash;
std::string g_contract_hash;
std::string g_request_hash;
std::string g_request_id;
std::uint32_t g_output_limit = 0U;
bool g_ready_shape = false;

WireCompileStatus wire_status_for(CompileSessionDecodeStatus status) {
  switch (status) {
    case CompileSessionDecodeStatus::kInvalidFrame:
      return WireCompileStatus::kInvalidFrame;
    case CompileSessionDecodeStatus::kAbiMismatch:
      return WireCompileStatus::kAbiMismatch;
    case CompileSessionDecodeStatus::kResourceLimit:
      return WireCompileStatus::kResourceLimit;
    case CompileSessionDecodeStatus::kInternalError:
      return WireCompileStatus::kInternalError;
    case CompileSessionDecodeStatus::kReady:
      return WireCompileStatus::kInternalError;
  }
  return WireCompileStatus::kInternalError;
}

ArtifactV3CompileResult decode_callback(
    const ValidatedInputFrameRegions& regions, ArtifactV3ResultSink&) {
  CompileSessionDecodeResult decoded = decode_compile_session(regions);
  g_decode_status = decoded.status;
  g_failure = decoded.failure;
  if (decoded.status != CompileSessionDecodeStatus::kReady &&
      (decoded.failure.region == CompileSessionRegion::kRequest ||
       decoded.failure.region == CompileSessionRegion::kProfile)) {
    const bool request = decoded.failure.region == CompileSessionRegion::kRequest;
    const std::uint8_t* bytes = request ? regions.request_bytes() : regions.profile_bytes();
    const std::uint32_t length = request ? regions.request_byte_length()
                                         : regions.profile_byte_length();
    const std::uint32_t begin = decoded.failure.byte_offset > 80U
                                    ? decoded.failure.byte_offset - 80U : 0U;
    const std::uint32_t end = std::min<std::uint32_t>(
        length, decoded.failure.byte_offset + 120U);
    std::fprintf(stderr, "%s context: %.*s\n", request ? "request" : "profile",
                 end - begin, reinterpret_cast<const char*>(bytes + begin));
  }
  if (decoded.status == CompileSessionDecodeStatus::kReady && decoded.session) {
    g_profile_hash = decoded.session->profile_hash();
    g_contract_hash = decoded.session->compilation_contract_hash();
    g_request_hash = decoded.session->request_hash();
    g_request_id = decoded.session->request_id();
    g_output_limit = decoded.session->maximum_output_byte_length();
    constexpr CompilerOptionKind expected_kinds[] = {
        CompilerOptionKind::kDefine,
        CompilerOptionKind::kForcedInclude,
        CompilerOptionKind::kFrontendOption,
        CompilerOptionKind::kUndefine,
        CompilerOptionKind::kWarningPolicy,
    };
    bool exact_order = decoded.session->compiler_option_count() == 5U;
    for (std::size_t index = 0; index < decoded.session->compiler_option_count();
         ++index) {
      const CompilerOptionView option = decoded.session->compiler_option(index);
      exact_order = exact_order && option.ordinal == index &&
                    option.kind == expected_kinds[index];
    }
    g_ready_shape = exact_order &&
                    decoded.session->compiler_option(0).name_or_id == "CUTE_SM80_ENABLED" &&
                    decoded.session->compiler_option(1).include_root_id == "clang-resource" &&
                    decoded.session->compiler_option(2).name_or_id == "syntax-only" &&
                    decoded.session->compiler_option(3).name_or_id == "NDEBUG" &&
                    decoded.session->compiler_option(4).name_or_id == "clang.unused-variable" &&
                    decoded.session->semantic_pass_count() == 2U &&
                    decoded.session->include_root_count() == 6U &&
                    decoded.session->source_file_count() == 2U &&
                    decoded.session->entry_request().kind == "layout";
  }
  // Admission is not source verification, Clang execution, or artifact-v3
  // authority. Even a ready session remains blocked at this bridge.
  return {wire_status_for(decoded.status),
          ReviewOnlyBlocker::kCudaDualPassUnavailable};
}

CompileSessionDecodeStatus status_from_name(std::string_view name) {
  if (name == "ready") return CompileSessionDecodeStatus::kReady;
  if (name == "invalid") return CompileSessionDecodeStatus::kInvalidFrame;
  if (name == "abi") return CompileSessionDecodeStatus::kAbiMismatch;
  if (name == "resource") return CompileSessionDecodeStatus::kResourceLimit;
  return CompileSessionDecodeStatus::kInternalError;
}

int run(const char* frame_path, std::string_view expected_status,
        const char* expected_profile_hash, const char* expected_contract_hash,
        const char* expected_request_hash) {
  std::ifstream input(frame_path, std::ios::binary);
  BG_CHECK(input.good());
  const std::vector<std::uint8_t> frame(
      (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  BG_CHECK(!frame.empty());
  BG_CHECK(frame.size() <= std::numeric_limits<std::uint32_t>::max());

  g_runtime_test_allocation_hooks = {
      test_allocate,
      test_release,
      test_wire_pointer,
  };
  runtime_reset();
  const std::uint32_t pointer = runtime_allocate(
      static_cast<std::uint32_t>(frame.size()));
  BG_CHECK(pointer == kWirePointer);
  BG_CHECK(g_allocation_length == frame.size());
  std::memcpy(g_allocation, frame.data(), frame.size());
  const std::int32_t runtime_result = runtime_compile(
      pointer, static_cast<std::uint32_t>(frame.size()), decode_callback);
  if (g_decode_status != status_from_name(expected_status)) {
    std::fprintf(stderr, "decode status=%u region=%u reason=%u offset=%u\n",
                 static_cast<unsigned>(g_decode_status),
                 static_cast<unsigned>(g_failure.region),
                 static_cast<unsigned>(g_failure.reason),
                 g_failure.byte_offset);
  }
  BG_CHECK(g_decode_status == status_from_name(expected_status));
  if (g_decode_status == CompileSessionDecodeStatus::kReady) {
    BG_CHECK(runtime_result == static_cast<std::int32_t>(WireCompileStatus::kInternalError));
    BG_CHECK(runtime_result_pointer() == 0U);
    BG_CHECK(runtime_result_length() == 0U);
    BG_CHECK(g_ready_shape);
    BG_CHECK(g_profile_hash == expected_profile_hash);
    BG_CHECK(g_contract_hash == expected_contract_hash);
    BG_CHECK(g_request_hash == expected_request_hash);
    BG_CHECK(g_request_id == std::string("bg.cpp.frontend-request.sha256.") +
                                 expected_request_hash);
    BG_CHECK(g_output_limit == 8U * 1024U * 1024U);
  } else {
    BG_CHECK(runtime_result == static_cast<std::int32_t>(
        wire_status_for(g_decode_status)));
    BG_CHECK(!g_ready_shape);
  }
  runtime_reset();
  BG_CHECK(g_allocation == nullptr);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(stderr,
                 "usage: native-test FRAME STATUS PROFILE_HASH CONTRACT_HASH REQUEST_HASH\n");
    return 2;
  }
  return run(argv[1], argv[2], argv[3], argv[4], argv[5]);
}
