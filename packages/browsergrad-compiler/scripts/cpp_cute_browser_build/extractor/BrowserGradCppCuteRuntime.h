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
 * Fixed, source-independent diagnostic codes emitted only through the pinned
 * stdout/stderr host capability. They explain a rejected native invocation but
 * never participate in ABI status, artifact identity, or authority.
 */
enum class NativeDiagnosticCode : std::uint32_t {
  kNone = 0U,
  kProducerPolicyInstallFailure = 1U,
  kProducerVfsFailure = 2U,
  kProducerDiagnosticCaptureLimit = 3U,
  kProducerFrontendWorkLimit = 4U,
  kProducerInvocationFailure = 5U,
  kProducerException = 6U,
  kArtifactDecodeInternal = 7U,
  kArtifactPlanInternal = 8U,
  kArtifactSinkBindInternal = 9U,
  kArtifactFrontendBeginInternal = 10U,
  kArtifactFrontendCompleteInternal = 11U,
  kArtifactProducerInternal = 12U,
  kArtifactWriterInternal = 13U,
  kRuntimePostCompilePhaseInvariant = 14U,
  kRuntimeArtifactReadyInvariant = 15U,
  kRuntimeTerminalStatusInvalid = 16U,
  kAllocatorInvalidAllocationTableShape = 17U,
  kAllocatorAllocationTableProbeExhausted = 18U,
  kAllocatorAllocationTableRehashFailure = 19U,
  kAllocatorAllocationTableInsertFailure = 20U,
  kAllocatorAllocationTableEraseFailure = 21U,
  kAllocatorCreationCounterOverflow = 22U,
  kAllocatorReleaseInvariantFailure = 23U,
  kAllocatorReallocationInvariantFailure = 24U,
  kAllocatorFailedAllocationCounterOverflow = 25U,
  kAllocatorReentrantHook = 26U,
  kAllocatorDuplicateBuiltinPointer = 27U,
  kAllocatorUntrackedFree = 28U,
  kAllocatorUntrackedReallocation = 29U,
  kAllocatorReplacementPointerCollision = 30U,
  kAllocatorInvalidMetricsPointer = 31U,
  kAllocatorUnknownFailure = 32U,
  kProducerVfsInvalidLimits = 33U,
  kProducerVfsInvalidSuccessfulRead = 34U,
  kProducerVfsInconsistentSuccessfulRead = 35U,
  kProducerVfsOpenedFileLimit = 36U,
  kProducerVfsInvalidIncludeEdge = 37U,
  kProducerVfsIncludeEdgeLimit = 38U,
  kProducerVfsInvalidIncludeSourceRange = 39U,
  kProducerVfsInvalidIncludeResolvedPath = 40U,
  kProducerVfsInvalidIncludeIncludingPath = 41U,
  kProducerVfsInvalidIncludeSpelling = 42U,
  kProducerVfsInvalidIncludeOffsets = 43U,
  kProducerVfsInvalidIncludeSourceOrdinal = 44U,
  kProducerVfsInvalidForcedIncludeShape = 45U,
  kProducerVfsInvalidIncludeKind = 46U,
};

void report_native_diagnostic(NativeDiagnosticCode code) noexcept;

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

class ValidatedInputFrameRegions;
class ArtifactV3ResultSink;

using ArtifactV3Compile = ArtifactV3CompileResult (*)(
    const ValidatedInputFrameRegions& input,
    ArtifactV3ResultSink& result_sink);

/**
 * Immutable region views produced only after runtime-v1 frame validation.
 *
 * The compile callback receives no raw frame/header/padding bytes. Typed
 * session admission must independently validate both regions as canonical
 * JSON before any VFS or Clang work.
 */
class ValidatedInputFrameRegions final {
 public:
  ValidatedInputFrameRegions(const ValidatedInputFrameRegions&) = delete;
  ValidatedInputFrameRegions& operator=(const ValidatedInputFrameRegions&) =
      delete;
  ValidatedInputFrameRegions(ValidatedInputFrameRegions&&) = delete;
  ValidatedInputFrameRegions& operator=(ValidatedInputFrameRegions&&) = delete;

  const std::uint8_t* profile_bytes() const noexcept {
    return profile_bytes_;
  }
  std::uint32_t profile_byte_length() const noexcept {
    return profile_byte_length_;
  }
  const std::uint8_t* request_bytes() const noexcept {
    return request_bytes_;
  }
  std::uint32_t request_byte_length() const noexcept {
    return request_byte_length_;
  }

 private:
  friend std::int32_t runtime_compile(std::uint32_t input_pointer,
                                      std::uint32_t input_length,
                                      ArtifactV3Compile compile_artifact);

  ValidatedInputFrameRegions(const std::uint8_t* profile_bytes,
                             std::uint32_t profile_byte_length,
                             const std::uint8_t* request_bytes,
                             std::uint32_t request_byte_length) noexcept
      : profile_bytes_(profile_bytes),
        profile_byte_length_(profile_byte_length),
        request_bytes_(request_bytes),
        request_byte_length_(request_byte_length) {}

  const std::uint8_t* profile_bytes_ = nullptr;
  std::uint32_t profile_byte_length_ = 0;
  const std::uint8_t* request_bytes_ = nullptr;
  std::uint32_t request_byte_length_ = 0;
};

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
std::uint32_t runtime_last_diagnostic_code() noexcept;
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
