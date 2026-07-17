#pragma once

#include <cstdint>
#include <optional>

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
  std::optional<ReviewOnlyBlocker> blocker =
      ReviewOnlyBlocker::kCudaDualPassUnavailable;
};

class ArtifactV3ResultSink;

using ArtifactV3Compile = ArtifactV3CompileResult (*)(
    const std::uint8_t* input, std::uint32_t input_byte_length,
    ArtifactV3ResultSink& result_sink);

/**
 * Single-allocation sink for one complete canonical artifact-v3 payload.
 *
 * The runtime owns the allocation and either adopts it atomically after a
 * successful compile or releases it before returning a failure. Producers
 * cannot supply, alias, or resize result storage; the returned write pointer
 * is call-scoped and must not be retained.
 */
class ArtifactV3ResultSink final {
 public:
  static constexpr std::uint32_t kAbiMaximumByteLength =
      8U * 1024U * 1024U;

  ArtifactV3ResultSink(const ArtifactV3ResultSink&) = delete;
  ArtifactV3ResultSink& operator=(const ArtifactV3ResultSink&) = delete;

  /** Bind the strictly decoded invocation ceiling exactly once. */
  bool bind_invocation_maximum_byte_length(std::uint32_t byte_length);
  std::uint8_t* allocate(std::uint32_t byte_length);
  bool commit();

 private:
  friend std::int32_t runtime_compile(std::uint32_t input_pointer,
                                      std::uint32_t input_length,
                                      ArtifactV3Compile compile_artifact);

  ArtifactV3ResultSink() = default;
  ~ArtifactV3ResultSink();

  void discard();

  std::uint8_t* bytes_ = nullptr;
  std::uint32_t byte_length_ = 0;
  std::uint32_t invocation_maximum_byte_length_ = 0;
  std::uint32_t wire_pointer_ = 0;
  WireCompileStatus failure_status_ = WireCompileStatus::kInternalError;
  bool allocation_attempted_ = false;
  bool invocation_limit_bound_ = false;
  bool committed_ = false;
  bool adopted_ = false;
  bool failed_ = false;
};

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
