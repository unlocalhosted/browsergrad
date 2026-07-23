#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <type_traits>
#include <vector>

namespace {

bool g_allocator_metrics_healthy_for_test = true;
bool g_frontend_work_metrics_ready_for_test = true;
std::uint32_t g_frontend_work_reset_count_for_test = 0U;

}  // namespace

#define BG_CPP_CUTE_RUNTIME_TESTING 1
#include "extractor/BrowserGradCppCuteRuntime.cpp"

namespace browsergrad::cpp_cute {

bool allocator_metrics_healthy() {
  return g_allocator_metrics_healthy_for_test;
}

AllocatorMetricsFailureReason allocator_metrics_failure_reason() noexcept {
  return g_allocator_metrics_healthy_for_test
             ? AllocatorMetricsFailureReason::kNone
             : AllocatorMetricsFailureReason::kUntrackedFree;
}

bool frontend_work_metrics_ready() noexcept {
  return g_frontend_work_metrics_ready_for_test;
}

void reset_frontend_work_metrics() noexcept {
  ++g_frontend_work_reset_count_for_test;
}

}  // namespace browsergrad::cpp_cute

namespace {

using namespace browsergrad::cpp_cute;

#define BG_CHECK(condition)                                                   \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::fprintf(stderr, "runtime check failed at line %d: %s\n",         \
                   __LINE__, #condition);                                     \
      return 1;                                                               \
    }                                                                         \
  } while (false)

struct AllocationRecord {
  void* pointer = nullptr;
  std::size_t byte_length = 0U;
  std::uint32_t wire_pointer = 0U;
  bool live = false;
};

std::array<AllocationRecord, 256U> g_allocations{};
std::size_t g_allocation_count = 0U;
std::uint32_t g_next_wire_pointer = 0x1'0000U;
std::uint32_t g_forced_next_wire_pointer = 0U;
bool g_fail_next_allocation = false;
bool g_release_error = false;
std::uint32_t g_callback_count = 0U;
std::uint32_t g_reentrant_input_pointer = 0U;
std::uint32_t g_reentrant_input_length = 0U;
std::vector<std::uint8_t> g_artifact_bytes;

void* test_allocate(std::size_t byte_length) {
  if (g_fail_next_allocation) {
    g_fail_next_allocation = false;
    return nullptr;
  }
  if (g_allocation_count == g_allocations.size()) return nullptr;
  void* pointer = std::malloc(byte_length);
  if (pointer == nullptr) return nullptr;
  const bool forced_wire_pointer = g_forced_next_wire_pointer != 0U;
  const std::uint32_t wire_pointer = forced_wire_pointer
                                         ? g_forced_next_wire_pointer
                                         : g_next_wire_pointer;
  g_forced_next_wire_pointer = 0U;
  const std::uint64_t next = static_cast<std::uint64_t>(wire_pointer) +
                             byte_length + 0x100U;
  if (next > 0xffff'ffffULL && !forced_wire_pointer) {
    std::free(pointer);
    return nullptr;
  }
  if (next <= 0xffff'ffffULL) {
    g_next_wire_pointer = static_cast<std::uint32_t>(next);
  }
  g_allocations[g_allocation_count++] = {
      pointer,
      byte_length,
      wire_pointer,
      true,
  };
  return pointer;
}

void test_release(void* pointer) {
  if (pointer == nullptr) return;
  // Production metrics deliberately refuse tracked frees after poison; the
  // Worker then discards the complete module instance and its linear memory.
  if (!g_allocator_metrics_healthy_for_test) return;
  for (std::size_t index = 0U; index < g_allocation_count; ++index) {
    AllocationRecord& record = g_allocations[index];
    if (record.pointer == pointer && record.live) {
      record.live = false;
      std::free(pointer);
      return;
    }
  }
  g_release_error = true;
}

void discard_test_module_allocations() {
  for (std::size_t index = 0U; index < g_allocation_count; ++index) {
    AllocationRecord& record = g_allocations[index];
    if (!record.live) continue;
    record.live = false;
    std::free(record.pointer);
  }
}

std::uint32_t test_wire_pointer(const void* pointer) {
  for (std::size_t index = 0U; index < g_allocation_count; ++index) {
    const AllocationRecord& record = g_allocations[index];
    if (record.pointer == pointer && record.live) return record.wire_pointer;
  }
  return 0U;
}

void* live_pointer(std::uint32_t wire_pointer) {
  for (std::size_t index = 0U; index < g_allocation_count; ++index) {
    const AllocationRecord& record = g_allocations[index];
    if (record.wire_pointer == wire_pointer && record.live) {
      return record.pointer;
    }
  }
  return nullptr;
}

std::size_t live_allocation_count() {
  std::size_t count = 0U;
  for (std::size_t index = 0U; index < g_allocation_count; ++index) {
    if (g_allocations[index].live) ++count;
  }
  return count;
}

void write_u16_le(std::uint8_t* bytes, std::uint16_t value) {
  bytes[0] = static_cast<std::uint8_t>(value);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

void write_u32_le(std::uint8_t* bytes, std::uint32_t value) {
  bytes[0] = static_cast<std::uint8_t>(value);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
  bytes[2] = static_cast<std::uint8_t>(value >> 16U);
  bytes[3] = static_cast<std::uint8_t>(value >> 24U);
}

constexpr std::uint32_t kValidFrameByteLength = 80U;

std::uint32_t prepare_valid_input() {
  runtime_reset();
  if (runtime_status() != static_cast<std::int32_t>(WireCompileStatus::kIdle)) {
    return 0U;
  }
  const std::uint32_t wire_pointer = runtime_allocate(kValidFrameByteLength);
  auto* bytes = static_cast<std::uint8_t*>(live_pointer(wire_pointer));
  if (bytes == nullptr) return 0U;
  std::memset(bytes, 0, kValidFrameByteLength);
  std::memcpy(bytes, "BGCCABI1", 8U);
  write_u16_le(bytes + 8U, 1U);
  write_u16_le(bytes + 10U, 0U);
  write_u32_le(bytes + 12U, 64U);
  write_u32_le(bytes + 16U, kValidFrameByteLength);
  write_u32_le(bytes + 20U, 0U);
  write_u32_le(bytes + 24U, 64U);
  write_u32_le(bytes + 28U, 2U);
  write_u32_le(bytes + 32U, 72U);
  write_u32_le(bytes + 36U, 2U);
  std::memcpy(bytes + 64U, "{}", 2U);
  std::memcpy(bytes + 72U, "{}", 2U);
  return wire_pointer;
}

bool load_artifact_bytes(const char* path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) return false;
  const std::streamsize byte_length = input.tellg();
  if (byte_length <= 0 ||
      byte_length > static_cast<std::streamsize>(
                        ArtifactV3ResultSink::kAbiMaximumByteLength)) {
    return false;
  }
  g_artifact_bytes.resize(static_cast<std::size_t>(byte_length));
  input.seekg(0, std::ios::beg);
  return static_cast<bool>(input.read(
      reinterpret_cast<char*>(g_artifact_bytes.data()), byte_length));
}

ArtifactV3CompileResult successful_artifact(
    const ValidatedInputFrameRegions& regions, ArtifactV3ResultSink& sink) {
  ++g_callback_count;
  if (regions.profile_bytes() == nullptr ||
      regions.profile_byte_length() != 2U ||
      std::memcmp(regions.profile_bytes(), "{}", 2U) != 0 ||
      regions.request_bytes() == nullptr ||
      regions.request_byte_length() != 2U ||
      std::memcmp(regions.request_bytes(), "{}", 2U) != 0) {
    return {WireCompileStatus::kInvalidFrame, std::nullopt};
  }
  if (g_artifact_bytes.empty() ||
      !sink.bind_invocation_maximum_byte_length(
          static_cast<std::uint32_t>(g_artifact_bytes.size()))) {
    return {WireCompileStatus::kInternalError, std::nullopt};
  }
  auto* output = sink.allocate(
      static_cast<std::uint32_t>(g_artifact_bytes.size()));
  if (output == nullptr) {
    return {WireCompileStatus::kResourceLimit, std::nullopt};
  }
  std::memcpy(output, g_artifact_bytes.data(), g_artifact_bytes.size());
  if (!sink.commit()) {
    return {WireCompileStatus::kInternalError, std::nullopt};
  }
  return {WireCompileStatus::kArtifactReady, std::nullopt};
}

ArtifactV3CompileResult ready_without_output(
    const ValidatedInputFrameRegions&, ArtifactV3ResultSink& sink) {
  static_cast<void>(sink.bind_invocation_maximum_byte_length(8U));
  return {WireCompileStatus::kArtifactReady, std::nullopt};
}

ArtifactV3CompileResult output_without_invocation_limit(
    const ValidatedInputFrameRegions&, ArtifactV3ResultSink& sink) {
  static_cast<void>(sink.allocate(8U));
  return {WireCompileStatus::kArtifactReady, std::nullopt};
}

ArtifactV3CompileResult ready_with_review_blocker(
    const ValidatedInputFrameRegions& input,
    ArtifactV3ResultSink& sink) {
  ArtifactV3CompileResult result =
      successful_artifact(input, sink);
  result.blocker = ReviewOnlyBlocker::kCanonicalArtifactV3Unavailable;
  return result;
}

ArtifactV3CompileResult oversized_output(
    const ValidatedInputFrameRegions&, ArtifactV3ResultSink& sink) {
  static_cast<void>(sink.bind_invocation_maximum_byte_length(8U));
  static_cast<void>(sink.allocate(9U));
  return {WireCompileStatus::kArtifactReady, std::nullopt};
}

ArtifactV3CompileResult double_allocate(
    const ValidatedInputFrameRegions&, ArtifactV3ResultSink& sink) {
  static_cast<void>(sink.bind_invocation_maximum_byte_length(8U));
  static_cast<void>(sink.allocate(8U));
  static_cast<void>(sink.allocate(8U));
  return {WireCompileStatus::kArtifactReady, std::nullopt};
}

ArtifactV3CompileResult commit_without_allocation(
    const ValidatedInputFrameRegions&, ArtifactV3ResultSink& sink) {
  static_cast<void>(sink.bind_invocation_maximum_byte_length(8U));
  static_cast<void>(sink.commit());
  return {WireCompileStatus::kArtifactReady, std::nullopt};
}

ArtifactV3CompileResult double_commit(
    const ValidatedInputFrameRegions&, ArtifactV3ResultSink& sink) {
  static_cast<void>(sink.bind_invocation_maximum_byte_length(8U));
  auto* output = sink.allocate(8U);
  if (output != nullptr) std::memset(output, 0, 8U);
  static_cast<void>(sink.commit());
  static_cast<void>(sink.commit());
  return {WireCompileStatus::kArtifactReady, std::nullopt};
}

ArtifactV3CompileResult vfs_failure_with_output(
    const ValidatedInputFrameRegions&, ArtifactV3ResultSink& sink) {
  static_cast<void>(sink.bind_invocation_maximum_byte_length(8U));
  auto* output = sink.allocate(8U);
  if (output != nullptr) {
    std::memset(output, 0, 8U);
    static_cast<void>(sink.commit());
  }
  return {WireCompileStatus::kVfsError, std::nullopt};
}

ArtifactV3CompileResult invalid_terminal_status(
    const ValidatedInputFrameRegions&, ArtifactV3ResultSink&) {
  return {WireCompileStatus::kIdle, std::nullopt};
}

ArtifactV3CompileResult reentrant_compile(
    const ValidatedInputFrameRegions&, ArtifactV3ResultSink&) {
  static_cast<void>(runtime_compile(g_reentrant_input_pointer,
                                    g_reentrant_input_length,
                                    ready_without_output));
  return {WireCompileStatus::kArtifactReady, std::nullopt};
}

ArtifactV3CompileResult reentrant_reset(
    const ValidatedInputFrameRegions& input,
    ArtifactV3ResultSink&) {
  runtime_reset();
  if (runtime_status() !=
          static_cast<std::int32_t>(WireCompileStatus::kInvalidState) ||
      input.profile_bytes() == nullptr || input.profile_byte_length() != 2U ||
      std::memcmp(input.profile_bytes(), "{}", 2U) != 0 ||
      input.request_bytes() == nullptr || input.request_byte_length() != 2U ||
      std::memcmp(input.request_bytes(), "{}", 2U) != 0) {
    return {WireCompileStatus::kInternalError, std::nullopt};
  }
  return {WireCompileStatus::kArtifactReady, std::nullopt};
}

ArtifactV3CompileResult poison_after_commit(
    const ValidatedInputFrameRegions&, ArtifactV3ResultSink& sink) {
  static_cast<void>(sink.bind_invocation_maximum_byte_length(8U));
  auto* output = sink.allocate(8U);
  if (output != nullptr) {
    std::memset(output, 0, 8U);
    static_cast<void>(sink.commit());
  }
  g_allocator_metrics_healthy_for_test = false;
  return {WireCompileStatus::kArtifactReady, std::nullopt};
}

int run_runtime_tests() {
  static_assert(!std::is_default_constructible_v<ValidatedInputFrameRegions>);
  static_assert(!std::is_copy_constructible_v<ValidatedInputFrameRegions>);
  static_assert(!std::is_copy_assignable_v<ValidatedInputFrameRegions>);
  static_assert(!std::is_move_constructible_v<ValidatedInputFrameRegions>);
  static_assert(!std::is_move_assignable_v<ValidatedInputFrameRegions>);
  static_assert(!std::is_copy_constructible_v<ArtifactV3ResultSink>);
  static_assert(!std::is_copy_assignable_v<ArtifactV3ResultSink>);
  static_assert(static_cast<std::uint32_t>(NativeDiagnosticCode::kNone) == 0U);
  static_assert(
      static_cast<std::uint32_t>(
          NativeDiagnosticCode::kAllocatorUnknownFailure) == 32U);
  static_assert(
      static_cast<std::uint32_t>(
          NativeDiagnosticCode::kProducerVfsInvalidIncludeKind) == 46U);
  g_runtime_test_allocation_hooks = {
      test_allocate,
      test_release,
      test_wire_pointer,
  };

  runtime_reset();
  BG_CHECK(runtime_last_diagnostic_code() == 0U);
  report_native_diagnostic(NativeDiagnosticCode::kArtifactWriterInternal);
  BG_CHECK(runtime_last_diagnostic_code() == 13U);
  report_native_diagnostic(NativeDiagnosticCode::kProducerException);
  BG_CHECK(runtime_last_diagnostic_code() == 13U);
  runtime_reset();
  BG_CHECK(runtime_last_diagnostic_code() == 0U);

  const std::uint32_t resets_before_first_input =
      g_frontend_work_reset_count_for_test;

  std::uint32_t input = prepare_valid_input();
  BG_CHECK(input != 0U);
  BG_CHECK(g_frontend_work_reset_count_for_test ==
           resets_before_first_input + 1U);
  g_callback_count = 0U;
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           successful_artifact) == 0);
  BG_CHECK(runtime_last_diagnostic_code() == 0U);
  BG_CHECK(g_callback_count == 1U);
  BG_CHECK(runtime_status() == 0);
  BG_CHECK(runtime_result_length() == g_artifact_bytes.size());
  const std::uint32_t result = runtime_result_pointer();
  BG_CHECK(result != 0U);
  BG_CHECK(result != input);
  BG_CHECK(std::memcmp(live_pointer(result), g_artifact_bytes.data(),
                       g_artifact_bytes.size()) == 0);
  BG_CHECK(live_allocation_count() == 2U);
  runtime_free(input, kValidFrameByteLength);
  BG_CHECK(runtime_status() == 0);
  BG_CHECK(runtime_result_pointer() == result);
  BG_CHECK(runtime_result_length() == g_artifact_bytes.size());
  BG_CHECK(live_allocation_count() == 1U);
  runtime_reset();
  BG_CHECK(runtime_status() ==
           static_cast<std::int32_t>(WireCompileStatus::kIdle));
  BG_CHECK(runtime_last_diagnostic_code() == 0U);
  BG_CHECK(runtime_result_pointer() == 0U);
  BG_CHECK(runtime_result_length() == 0U);
  BG_CHECK(live_allocation_count() == 0U);

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  g_frontend_work_metrics_ready_for_test = false;
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           successful_artifact) ==
           static_cast<std::int32_t>(WireCompileStatus::kInternalError));
  BG_CHECK(runtime_result_pointer() == 0U);
  BG_CHECK(live_allocation_count() == 1U);
  g_frontend_work_metrics_ready_for_test = true;
  runtime_reset();

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           successful_artifact) == 0);
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           successful_artifact) ==
           static_cast<std::int32_t>(WireCompileStatus::kInvalidState));
  BG_CHECK(runtime_result_pointer() == 0U);
  BG_CHECK(live_allocation_count() == 2U);
  runtime_reset();
  BG_CHECK(live_allocation_count() == 0U);

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           successful_artifact) == 0);
  const std::uint32_t live_result = runtime_result_pointer();
  runtime_free(live_result, runtime_result_length());
  BG_CHECK(runtime_status() ==
           static_cast<std::int32_t>(WireCompileStatus::kInvalidArgument));
  BG_CHECK(runtime_result_pointer() == 0U);
  BG_CHECK(live_allocation_count() == 2U);
  runtime_reset();
  BG_CHECK(live_allocation_count() == 0U);

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           ready_without_output) ==
           static_cast<std::int32_t>(WireCompileStatus::kInternalError));
  BG_CHECK(runtime_result_pointer() == 0U);
  runtime_reset();

  for (ArtifactV3Compile producer : {
           output_without_invocation_limit,
           ready_with_review_blocker,
       }) {
    input = prepare_valid_input();
    BG_CHECK(input != 0U);
    BG_CHECK(runtime_compile(input, kValidFrameByteLength, producer) ==
             static_cast<std::int32_t>(WireCompileStatus::kInternalError));
    BG_CHECK(runtime_result_pointer() == 0U);
    BG_CHECK(live_allocation_count() == 1U);
    runtime_reset();
  }

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  g_forced_next_wire_pointer = input + 8U;
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           successful_artifact) ==
           static_cast<std::int32_t>(WireCompileStatus::kInternalError));
  BG_CHECK(runtime_result_pointer() == 0U);
  BG_CHECK(live_allocation_count() == 1U);
  runtime_reset();

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  g_forced_next_wire_pointer = 0xffff'fff0U;
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           successful_artifact) ==
           static_cast<std::int32_t>(WireCompileStatus::kInternalError));
  BG_CHECK(runtime_result_pointer() == 0U);
  BG_CHECK(live_allocation_count() == 1U);
  runtime_reset();

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  BG_CHECK(runtime_compile(input, kValidFrameByteLength, oversized_output) ==
           static_cast<std::int32_t>(WireCompileStatus::kResourceLimit));
  BG_CHECK(live_allocation_count() == 1U);
  runtime_reset();

  for (ArtifactV3Compile producer : {
           double_allocate,
           commit_without_allocation,
           double_commit,
       }) {
    input = prepare_valid_input();
    BG_CHECK(input != 0U);
    BG_CHECK(runtime_compile(input, kValidFrameByteLength, producer) ==
             static_cast<std::int32_t>(WireCompileStatus::kInternalError));
    BG_CHECK(runtime_result_pointer() == 0U);
    BG_CHECK(live_allocation_count() == 1U);
    runtime_reset();
  }

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           vfs_failure_with_output) ==
           static_cast<std::int32_t>(WireCompileStatus::kVfsError));
  BG_CHECK(runtime_result_pointer() == 0U);
  BG_CHECK(live_allocation_count() == 1U);
  runtime_reset();

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           invalid_terminal_status) ==
           static_cast<std::int32_t>(WireCompileStatus::kInternalError));
  runtime_reset();

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  g_fail_next_allocation = true;
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           successful_artifact) ==
           static_cast<std::int32_t>(WireCompileStatus::kResourceLimit));
  BG_CHECK(live_allocation_count() == 1U);
  runtime_reset();

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  g_reentrant_input_pointer = input;
  g_reentrant_input_length = kValidFrameByteLength;
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           reentrant_compile) ==
           static_cast<std::int32_t>(WireCompileStatus::kInternalError));
  BG_CHECK(runtime_result_pointer() == 0U);
  runtime_reset();

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           reentrant_reset) ==
           static_cast<std::int32_t>(WireCompileStatus::kInternalError));
  BG_CHECK(runtime_result_pointer() == 0U);
  BG_CHECK(live_allocation_count() == 1U);
  runtime_reset();
  BG_CHECK(live_allocation_count() == 0U);

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  g_callback_count = 0U;
  auto* frame = static_cast<std::uint8_t*>(live_pointer(input));
  BG_CHECK(frame != nullptr);
  frame[0] = 0U;
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           successful_artifact) ==
           static_cast<std::int32_t>(WireCompileStatus::kInvalidFrame));
  BG_CHECK(g_callback_count == 0U);
  runtime_reset();

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  BG_CHECK(runtime_compile(input + 8U, kValidFrameByteLength,
                           successful_artifact) ==
           static_cast<std::int32_t>(WireCompileStatus::kInvalidArgument));
  runtime_reset();

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  BG_CHECK(runtime_compile(input, kValidFrameByteLength, nullptr) ==
           static_cast<std::int32_t>(WireCompileStatus::kInternalError));
  runtime_reset();

  input = prepare_valid_input();
  BG_CHECK(input != 0U);
  BG_CHECK(runtime_compile(input, kValidFrameByteLength,
                           poison_after_commit) ==
           static_cast<std::int32_t>(WireCompileStatus::kInternalError));
  BG_CHECK(runtime_result_pointer() == 0U);
  BG_CHECK(live_allocation_count() == 2U);
  runtime_reset();
  BG_CHECK(runtime_status() ==
           static_cast<std::int32_t>(WireCompileStatus::kInternalError));
  BG_CHECK(runtime_last_diagnostic_code() == 28U);
  BG_CHECK(live_allocation_count() == 2U);
  discard_test_module_allocations();
  BG_CHECK(live_allocation_count() == 0U);
  BG_CHECK(!g_release_error);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || !load_artifact_bytes(argv[1])) {
    std::fprintf(stderr, "expected one bounded canonical artifact fixture\n");
    return 1;
  }
  return run_runtime_tests();
}
