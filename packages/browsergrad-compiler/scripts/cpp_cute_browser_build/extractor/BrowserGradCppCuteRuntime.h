#pragma once

#include <cstdint>

namespace browsergrad::cpp_cute {

enum class WireCompileStatus : std::int32_t {
  kArtifactReady = 0,
  kIdle = 1,
  kInputAllocated = 2,
  kInvalidState = 100,
  kInvalidArgument = 101,
  kInvalidFrame = 102,
  kAbiMismatch = 103,
  kVfsError = 104,
  kResourceLimit = 105,
  kInternalError = 106,
};

/**
 * Review-only blockers are deliberately distinct from the C ABI status. They
 * keep incomplete producer work from ever masquerading as an artifact.
 */
enum class ReviewOnlyBlocker {
  kCudaDualPassUnavailable,
  kCanonicalArtifactV3Unavailable,
};

struct ArtifactV3CompileResult {
  WireCompileStatus status = WireCompileStatus::kInternalError;
  ReviewOnlyBlocker blocker = ReviewOnlyBlocker::kCudaDualPassUnavailable;
};

using ArtifactV3Compile = ArtifactV3CompileResult (*)(
    const std::uint8_t* input, std::uint32_t input_byte_length);

std::uint32_t runtime_abi_version();
std::uint32_t runtime_allocate(std::uint32_t byte_length);
std::int32_t runtime_compile(std::uint32_t input_pointer,
                             std::uint32_t input_length,
                             ArtifactV3Compile compile_artifact);
void runtime_free(std::uint32_t pointer, std::uint32_t byte_length);
void runtime_reset();
std::uint32_t runtime_result_length();
std::uint32_t runtime_result_pointer();
std::int32_t runtime_status();

}  // namespace browsergrad::cpp_cute
