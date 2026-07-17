#include "BrowserGradCppCuteRuntime.h"

#include "BrowserGradCppCuteMetrics.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace browsergrad::cpp_cute {
namespace {

constexpr std::uint32_t kRuntimeAbiVersion = 0x0001'0001U;
constexpr std::uint32_t kInputFrameMaximumByteLength = 4U * 1024U * 1024U;
constexpr std::uint32_t kInputFrameHeaderByteLength = 64U;
constexpr std::uint32_t kInputFrameAlignment = 8U;
constexpr std::array<std::uint8_t, 8> kInputFrameMagic = {
    'B', 'G', 'C', 'C', 'A', 'B', 'I', '1'};

enum class RuntimePhase {
  kIdle,
  kInputAllocated,
  kFailed,
};

struct RuntimeState {
  RuntimePhase phase = RuntimePhase::kIdle;
  WireCompileStatus status = WireCompileStatus::kIdle;
  std::uint8_t* input = nullptr;
  std::uint32_t input_byte_length = 0;
  ReviewOnlyBlocker blocker = ReviewOnlyBlocker::kCudaDualPassUnavailable;
};

RuntimeState g_runtime;

std::uint16_t read_u16_le(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t read_u32_le(const std::uint8_t* bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

bool all_zero(const std::uint8_t* begin, const std::uint8_t* end) {
  for (const auto* cursor = begin; cursor != end; ++cursor) {
    if (*cursor != 0U) return false;
  }
  return true;
}

bool validate_frame_envelope(const std::uint8_t* bytes,
                             std::uint32_t byte_length) {
  if (bytes == nullptr || byte_length < kInputFrameHeaderByteLength ||
      byte_length > kInputFrameMaximumByteLength) {
    return false;
  }
  for (std::size_t index = 0; index < kInputFrameMagic.size(); ++index) {
    if (bytes[index] != kInputFrameMagic[index]) return false;
  }
  if (read_u16_le(bytes + 8U) != 1U || read_u16_le(bytes + 10U) != 0U ||
      read_u32_le(bytes + 12U) != kInputFrameHeaderByteLength ||
      read_u32_le(bytes + 16U) != byte_length ||
      read_u32_le(bytes + 20U) != 0U ||
      read_u32_le(bytes + 24U) != kInputFrameHeaderByteLength ||
      !all_zero(bytes + 40U, bytes + kInputFrameHeaderByteLength)) {
    return false;
  }

  const std::uint64_t profile_offset = read_u32_le(bytes + 24U);
  const std::uint64_t profile_byte_length = read_u32_le(bytes + 28U);
  const std::uint64_t request_offset = read_u32_le(bytes + 32U);
  const std::uint64_t request_byte_length = read_u32_le(bytes + 36U);
  if (profile_byte_length == 0U || request_byte_length == 0U ||
      request_offset % kInputFrameAlignment != 0U) {
    return false;
  }
  const std::uint64_t profile_end = profile_offset + profile_byte_length;
  const std::uint64_t expected_request_offset =
      align_up(profile_end, kInputFrameAlignment);
  const std::uint64_t request_end = request_offset + request_byte_length;
  const std::uint64_t expected_total =
      align_up(request_end, kInputFrameAlignment);
  if (profile_end > byte_length || request_end > byte_length ||
      request_offset != expected_request_offset || expected_total != byte_length) {
    return false;
  }
  return all_zero(bytes + profile_end, bytes + request_offset) &&
         all_zero(bytes + request_end, bytes + byte_length);
}

void release_input() {
  std::free(g_runtime.input);
  g_runtime.input = nullptr;
  g_runtime.input_byte_length = 0;
}

std::int32_t wire_status(WireCompileStatus status) {
  return static_cast<std::int32_t>(status);
}

}  // namespace

std::uint32_t runtime_abi_version() { return kRuntimeAbiVersion; }

std::uint32_t runtime_allocate(std::uint32_t byte_length) {
  if (!allocator_metrics_healthy()) {
    g_runtime.status = WireCompileStatus::kInternalError;
    g_runtime.phase = RuntimePhase::kFailed;
    return 0U;
  }
  if (g_runtime.phase != RuntimePhase::kIdle || byte_length == 0U ||
      byte_length > kInputFrameMaximumByteLength) {
    g_runtime.status = WireCompileStatus::kInvalidArgument;
    g_runtime.phase = RuntimePhase::kFailed;
    return 0U;
  }
  auto* allocation = static_cast<std::uint8_t*>(std::malloc(byte_length));
  if (allocation == nullptr) {
    g_runtime.status = allocator_metrics_healthy()
                           ? WireCompileStatus::kResourceLimit
                           : WireCompileStatus::kInternalError;
    g_runtime.phase = RuntimePhase::kFailed;
    return 0U;
  }
  const auto pointer = reinterpret_cast<std::uintptr_t>(allocation);
  if (pointer > std::numeric_limits<std::uint32_t>::max()) {
    std::free(allocation);
    g_runtime.status = WireCompileStatus::kInternalError;
    g_runtime.phase = RuntimePhase::kFailed;
    return 0U;
  }
  g_runtime.input = allocation;
  g_runtime.input_byte_length = byte_length;
  g_runtime.status = WireCompileStatus::kInputAllocated;
  g_runtime.phase = RuntimePhase::kInputAllocated;
  return static_cast<std::uint32_t>(pointer);
}

std::int32_t runtime_compile(std::uint32_t input_pointer,
                             std::uint32_t input_length,
                             ArtifactV3Compile compile_artifact) {
  if (!allocator_metrics_healthy()) {
    g_runtime.status = WireCompileStatus::kInternalError;
    g_runtime.phase = RuntimePhase::kFailed;
    return wire_status(g_runtime.status);
  }
  if (g_runtime.phase != RuntimePhase::kInputAllocated ||
      g_runtime.input == nullptr) {
    g_runtime.status = WireCompileStatus::kInvalidState;
    g_runtime.phase = RuntimePhase::kFailed;
    return wire_status(g_runtime.status);
  }
  const auto expected_pointer = reinterpret_cast<std::uintptr_t>(g_runtime.input);
  if (input_pointer != expected_pointer ||
      input_length != g_runtime.input_byte_length) {
    g_runtime.status = WireCompileStatus::kInvalidArgument;
    g_runtime.phase = RuntimePhase::kFailed;
    return wire_status(g_runtime.status);
  }
  if (!validate_frame_envelope(g_runtime.input, g_runtime.input_byte_length)) {
    g_runtime.status = WireCompileStatus::kInvalidFrame;
    g_runtime.phase = RuntimePhase::kFailed;
    return wire_status(g_runtime.status);
  }

  // The only wired producer is the fail-closed artifact-v3 placeholder. Keep
  // status zero unreachable until lifecycle/result ownership exists here.
  const ArtifactV3CompileResult result =
      compile_artifact == nullptr
          ? ArtifactV3CompileResult{
                WireCompileStatus::kInternalError,
                ReviewOnlyBlocker::kCanonicalArtifactV3Unavailable}
          : compile_artifact(g_runtime.input, g_runtime.input_byte_length);
  g_runtime.blocker = result.blocker;
  g_runtime.status = !allocator_metrics_healthy() ||
                             result.status == WireCompileStatus::kArtifactReady
                         ? WireCompileStatus::kInternalError
                         : result.status;
  g_runtime.phase = RuntimePhase::kFailed;
  return wire_status(g_runtime.status);
}

void runtime_free(std::uint32_t pointer, std::uint32_t byte_length) {
  if (g_runtime.input == nullptr ||
      pointer != reinterpret_cast<std::uintptr_t>(g_runtime.input) ||
      byte_length != g_runtime.input_byte_length) {
    g_runtime.status = WireCompileStatus::kInvalidArgument;
    g_runtime.phase = RuntimePhase::kFailed;
    return;
  }
  const bool was_allocated = g_runtime.phase == RuntimePhase::kInputAllocated;
  release_input();
  if (!allocator_metrics_healthy()) {
    g_runtime.phase = RuntimePhase::kFailed;
    g_runtime.status = WireCompileStatus::kInternalError;
    return;
  }
  if (was_allocated) {
    g_runtime.phase = RuntimePhase::kIdle;
    g_runtime.status = WireCompileStatus::kIdle;
  }
}

void runtime_reset() {
  release_input();
  g_runtime = RuntimeState{};
  if (!allocator_metrics_healthy()) {
    g_runtime.phase = RuntimePhase::kFailed;
    g_runtime.status = WireCompileStatus::kInternalError;
  }
}

std::uint32_t runtime_result_length() { return 0U; }

std::uint32_t runtime_result_pointer() { return 0U; }

std::int32_t runtime_status() { return wire_status(g_runtime.status); }

}  // namespace browsergrad::cpp_cute
