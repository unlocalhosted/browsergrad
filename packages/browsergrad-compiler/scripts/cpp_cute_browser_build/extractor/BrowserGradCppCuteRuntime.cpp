#include "BrowserGradCppCuteRuntime.h"

#include "BrowserGradCppCuteMetrics.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <unistd.h>

namespace browsergrad::cpp_cute {
namespace {

constexpr std::uint32_t kRuntimeAbiVersion = 0x0001'0002U;
constexpr std::uint32_t kInputFrameMaximumByteLength = 4U * 1024U * 1024U;
constexpr std::uint32_t kInputFrameHeaderByteLength = 64U;
constexpr std::uint32_t kInputFrameAlignment = 8U;
constexpr std::array<std::uint8_t, 8> kInputFrameMagic = {
    'B', 'G', 'C', 'C', 'A', 'B', 'I', '1'};

#if defined(__EMSCRIPTEN__) && defined(BG_CPP_CUTE_RUNTIME_TESTING)
#error "runtime test hooks must never be enabled in the Wasm producer"
#endif

#if defined(BG_CPP_CUTE_RUNTIME_TESTING)
struct RuntimeTestAllocationHooks {
  void* (*allocate)(std::size_t) = nullptr;
  void (*release)(void*) = nullptr;
  std::uint32_t (*wire_pointer)(const void*) = nullptr;
};

RuntimeTestAllocationHooks g_runtime_test_allocation_hooks;
#endif

enum class RuntimePhase {
  kIdle,
  kInputAllocated,
  kCompiling,
  kArtifactReady,
  kFailed,
};

struct RuntimeState {
  RuntimePhase phase = RuntimePhase::kIdle;
  WireCompileStatus status = WireCompileStatus::kIdle;
  std::uint8_t* input = nullptr;
  std::uint32_t input_byte_length = 0;
  std::uint32_t input_wire_pointer = 0;
  std::uint8_t* result = nullptr;
  std::uint32_t result_byte_length = 0;
  std::uint32_t result_wire_pointer = 0;
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

void* allocate_bytes(std::size_t byte_length) {
#if defined(BG_CPP_CUTE_RUNTIME_TESTING)
  if (g_runtime_test_allocation_hooks.allocate != nullptr) {
    return g_runtime_test_allocation_hooks.allocate(byte_length);
  }
#endif
  return std::malloc(byte_length);
}

void release_bytes(void* pointer) {
  if (pointer == nullptr) return;
#if defined(BG_CPP_CUTE_RUNTIME_TESTING)
  if (g_runtime_test_allocation_hooks.release != nullptr) {
    g_runtime_test_allocation_hooks.release(pointer);
    return;
  }
#endif
  std::free(pointer);
}

bool encode_wire_pointer(const void* pointer, std::uint32_t byte_length,
                         std::uint32_t* wire_pointer) {
  if (pointer == nullptr || byte_length == 0U || wire_pointer == nullptr) {
    return false;
  }
#if defined(BG_CPP_CUTE_RUNTIME_TESTING)
  if (g_runtime_test_allocation_hooks.wire_pointer != nullptr) {
    const std::uint32_t encoded =
        g_runtime_test_allocation_hooks.wire_pointer(pointer);
    const std::uint64_t end =
        static_cast<std::uint64_t>(encoded) + byte_length;
    if (encoded == 0U || end > (std::uint64_t{1U} << 32U)) return false;
    *wire_pointer = encoded;
    return true;
  }
#endif
  const auto encoded = reinterpret_cast<std::uintptr_t>(pointer);
  const std::uint64_t end = static_cast<std::uint64_t>(encoded) + byte_length;
  if (encoded == 0U || encoded > std::numeric_limits<std::uint32_t>::max() ||
      end > (std::uint64_t{1U} << 32U)) {
    return false;
  }
  *wire_pointer = static_cast<std::uint32_t>(encoded);
  return true;
}

bool ranges_overlap(std::uint32_t left_pointer, std::uint32_t left_length,
                    std::uint32_t right_pointer,
                    std::uint32_t right_length) {
  const std::uint64_t left_begin = left_pointer;
  const std::uint64_t left_end = left_begin + left_length;
  const std::uint64_t right_begin = right_pointer;
  const std::uint64_t right_end = right_begin + right_length;
  return left_begin < right_end && right_begin < left_end;
}

struct InputFrameRegionOffsets {
  std::uint32_t profile_offset = 0U;
  std::uint32_t profile_byte_length = 0U;
  std::uint32_t request_offset = 0U;
  std::uint32_t request_byte_length = 0U;
};

bool validate_frame_envelope(const std::uint8_t* bytes,
                             std::uint32_t byte_length,
                             InputFrameRegionOffsets* regions) {
  if (bytes == nullptr || byte_length < kInputFrameHeaderByteLength ||
      byte_length > kInputFrameMaximumByteLength || regions == nullptr) {
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
  if (!all_zero(bytes + profile_end, bytes + request_offset) ||
      !all_zero(bytes + request_end, bytes + byte_length)) {
    return false;
  }
  *regions = InputFrameRegionOffsets{
      static_cast<std::uint32_t>(profile_offset),
      static_cast<std::uint32_t>(profile_byte_length),
      static_cast<std::uint32_t>(request_offset),
      static_cast<std::uint32_t>(request_byte_length),
  };
  return true;
}

void release_input() {
  release_bytes(g_runtime.input);
  g_runtime.input = nullptr;
  g_runtime.input_byte_length = 0;
  g_runtime.input_wire_pointer = 0;
}

void release_result() {
  release_bytes(g_runtime.result);
  g_runtime.result = nullptr;
  g_runtime.result_byte_length = 0;
  g_runtime.result_wire_pointer = 0;
}

std::int32_t wire_status(WireCompileStatus status) {
  return static_cast<std::int32_t>(status);
}

void write_native_diagnostic(const std::string_view message) noexcept {
  std::string_view remaining = message;
  while (!remaining.empty()) {
    const ssize_t written =
        ::write(STDERR_FILENO, remaining.data(), remaining.size());
    if (written <= 0) return;
    remaining.remove_prefix(static_cast<std::size_t>(written));
  }
}

void report_allocator_metrics_failure() noexcept {
  switch (allocator_metrics_failure_reason()) {
    case AllocatorMetricsFailureReason::kNone:
      report_native_diagnostic(NativeDiagnosticCode::kAllocatorUnknownFailure);
      return;
    case AllocatorMetricsFailureReason::kInvalidAllocationTableShape:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorInvalidAllocationTableShape);
      return;
    case AllocatorMetricsFailureReason::kAllocationTableProbeExhausted:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorAllocationTableProbeExhausted);
      return;
    case AllocatorMetricsFailureReason::kAllocationTableRehashFailure:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorAllocationTableRehashFailure);
      return;
    case AllocatorMetricsFailureReason::kAllocationTableInsertFailure:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorAllocationTableInsertFailure);
      return;
    case AllocatorMetricsFailureReason::kAllocationTableEraseFailure:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorAllocationTableEraseFailure);
      return;
    case AllocatorMetricsFailureReason::kCreationCounterOverflow:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorCreationCounterOverflow);
      return;
    case AllocatorMetricsFailureReason::kReleaseInvariantFailure:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorReleaseInvariantFailure);
      return;
    case AllocatorMetricsFailureReason::kReallocationInvariantFailure:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorReallocationInvariantFailure);
      return;
    case AllocatorMetricsFailureReason::kFailedAllocationCounterOverflow:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorFailedAllocationCounterOverflow);
      return;
    case AllocatorMetricsFailureReason::kReentrantAllocatorHook:
      report_native_diagnostic(NativeDiagnosticCode::kAllocatorReentrantHook);
      return;
    case AllocatorMetricsFailureReason::kDuplicateBuiltinPointer:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorDuplicateBuiltinPointer);
      return;
    case AllocatorMetricsFailureReason::kUntrackedFree:
      report_native_diagnostic(NativeDiagnosticCode::kAllocatorUntrackedFree);
      return;
    case AllocatorMetricsFailureReason::kUntrackedReallocation:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorUntrackedReallocation);
      return;
    case AllocatorMetricsFailureReason::kReplacementPointerCollision:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorReplacementPointerCollision);
      return;
    case AllocatorMetricsFailureReason::kInvalidMetricsPointer:
      report_native_diagnostic(
          NativeDiagnosticCode::kAllocatorInvalidMetricsPointer);
      return;
  }
  report_native_diagnostic(NativeDiagnosticCode::kAllocatorUnknownFailure);
}

}  // namespace

void report_native_diagnostic(const NativeDiagnosticCode code) noexcept {
  switch (code) {
    case NativeDiagnosticCode::kProducerPolicyInstallFailure:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:producer-policy-install-failure\n");
      return;
    case NativeDiagnosticCode::kProducerVfsFailure:
      write_native_diagnostic("BG-CPP-CUTE-DIAGNOSTIC:producer-vfs-failure\n");
      return;
    case NativeDiagnosticCode::kProducerDiagnosticCaptureLimit:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:producer-diagnostic-capture-limit\n");
      return;
    case NativeDiagnosticCode::kProducerFrontendWorkLimit:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:producer-frontend-work-limit\n");
      return;
    case NativeDiagnosticCode::kProducerInvocationFailure:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:producer-invocation-failure\n");
      return;
    case NativeDiagnosticCode::kProducerException:
      write_native_diagnostic("BG-CPP-CUTE-DIAGNOSTIC:producer-exception\n");
      return;
    case NativeDiagnosticCode::kArtifactDecodeInternal:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:artifact-decode-internal\n");
      return;
    case NativeDiagnosticCode::kArtifactPlanInternal:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:artifact-plan-internal\n");
      return;
    case NativeDiagnosticCode::kArtifactSinkBindInternal:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:artifact-sink-bind-internal\n");
      return;
    case NativeDiagnosticCode::kArtifactFrontendBeginInternal:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:artifact-frontend-begin-internal\n");
      return;
    case NativeDiagnosticCode::kArtifactFrontendCompleteInternal:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:artifact-frontend-complete-internal\n");
      return;
    case NativeDiagnosticCode::kArtifactProducerInternal:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:artifact-producer-internal\n");
      return;
    case NativeDiagnosticCode::kArtifactWriterInternal:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:artifact-writer-internal\n");
      return;
    case NativeDiagnosticCode::kRuntimePostCompilePhaseInvariant:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:runtime-post-compile-phase-invariant\n");
      return;
    case NativeDiagnosticCode::kRuntimeArtifactReadyInvariant:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:runtime-artifact-ready-invariant\n");
      return;
    case NativeDiagnosticCode::kRuntimeTerminalStatusInvalid:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:runtime-terminal-status-invalid\n");
      return;
    case NativeDiagnosticCode::kAllocatorInvalidAllocationTableShape:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-invalid-table-shape\n");
      return;
    case NativeDiagnosticCode::kAllocatorAllocationTableProbeExhausted:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-table-probe-exhausted\n");
      return;
    case NativeDiagnosticCode::kAllocatorAllocationTableRehashFailure:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-table-rehash-failure\n");
      return;
    case NativeDiagnosticCode::kAllocatorAllocationTableInsertFailure:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-table-insert-failure\n");
      return;
    case NativeDiagnosticCode::kAllocatorAllocationTableEraseFailure:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-table-erase-failure\n");
      return;
    case NativeDiagnosticCode::kAllocatorCreationCounterOverflow:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-creation-counter-overflow\n");
      return;
    case NativeDiagnosticCode::kAllocatorReleaseInvariantFailure:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-release-invariant-failure\n");
      return;
    case NativeDiagnosticCode::kAllocatorReallocationInvariantFailure:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-reallocation-invariant-failure\n");
      return;
    case NativeDiagnosticCode::kAllocatorFailedAllocationCounterOverflow:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-failed-counter-overflow\n");
      return;
    case NativeDiagnosticCode::kAllocatorReentrantHook:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-reentrant-hook\n");
      return;
    case NativeDiagnosticCode::kAllocatorDuplicateBuiltinPointer:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-duplicate-builtin-pointer\n");
      return;
    case NativeDiagnosticCode::kAllocatorUntrackedFree:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-untracked-free\n");
      return;
    case NativeDiagnosticCode::kAllocatorUntrackedReallocation:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-untracked-reallocation\n");
      return;
    case NativeDiagnosticCode::kAllocatorReplacementPointerCollision:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-replacement-pointer-collision\n");
      return;
    case NativeDiagnosticCode::kAllocatorInvalidMetricsPointer:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-invalid-metrics-pointer\n");
      return;
    case NativeDiagnosticCode::kAllocatorUnknownFailure:
      write_native_diagnostic(
          "BG-CPP-CUTE-DIAGNOSTIC:allocator-unknown-failure\n");
      return;
  }
}

bool ArtifactV3ResultSink::bind_invocation_maximum_byte_length(
    std::uint32_t byte_length) {
  if (failed_ || invocation_limit_bound_ || allocation_attempted_ || committed_ ||
      byte_length == 0U || byte_length > kAbiMaximumByteLength) {
    failed_ = true;
    failure_status_ = WireCompileStatus::kInternalError;
    return false;
  }
  invocation_maximum_byte_length_ = byte_length;
  invocation_limit_bound_ = true;
  return true;
}

std::uint8_t* ArtifactV3ResultSink::allocate(std::uint32_t byte_length) {
  if (failed_ || allocation_attempted_ || committed_) {
    failed_ = true;
    failure_status_ = WireCompileStatus::kInternalError;
    return nullptr;
  }
  allocation_attempted_ = true;
  if (!invocation_limit_bound_ || byte_length == 0U) {
    failed_ = true;
    failure_status_ = WireCompileStatus::kInternalError;
    return nullptr;
  }
  if (byte_length > invocation_maximum_byte_length_) {
    failed_ = true;
    failure_status_ = WireCompileStatus::kResourceLimit;
    return nullptr;
  }
  if (!allocator_metrics_healthy()) {
    failed_ = true;
    failure_status_ = WireCompileStatus::kInternalError;
    return nullptr;
  }

  bytes_ = static_cast<std::uint8_t*>(allocate_bytes(byte_length));
  if (bytes_ == nullptr) {
    failed_ = true;
    failure_status_ = allocator_metrics_healthy()
                          ? WireCompileStatus::kResourceLimit
                          : WireCompileStatus::kInternalError;
    return nullptr;
  }
  byte_length_ = byte_length;
  if (!encode_wire_pointer(bytes_, byte_length_, &wire_pointer_) ||
      !allocator_metrics_healthy()) {
    discard();
    failed_ = true;
    failure_status_ = WireCompileStatus::kInternalError;
    return nullptr;
  }
  return bytes_;
}

bool ArtifactV3ResultSink::commit() {
  if (failed_ || !allocation_attempted_ || bytes_ == nullptr || committed_) {
    failed_ = true;
    failure_status_ = WireCompileStatus::kInternalError;
    return false;
  }
  if (!allocator_metrics_healthy()) {
    failed_ = true;
    failure_status_ = WireCompileStatus::kInternalError;
    return false;
  }
  committed_ = true;
  return true;
}

void ArtifactV3ResultSink::discard() {
  if (!adopted_) release_bytes(bytes_);
  bytes_ = nullptr;
  byte_length_ = 0;
  wire_pointer_ = 0;
  committed_ = false;
}

ArtifactV3ResultSink::~ArtifactV3ResultSink() { discard(); }

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
  auto* allocation = static_cast<std::uint8_t*>(allocate_bytes(byte_length));
  if (allocation == nullptr) {
    g_runtime.status = allocator_metrics_healthy()
                           ? WireCompileStatus::kResourceLimit
                           : WireCompileStatus::kInternalError;
    g_runtime.phase = RuntimePhase::kFailed;
    return 0U;
  }
  std::uint32_t wire_pointer = 0U;
  if (!encode_wire_pointer(allocation, byte_length, &wire_pointer) ||
      !allocator_metrics_healthy()) {
    release_bytes(allocation);
    g_runtime.status = WireCompileStatus::kInternalError;
    g_runtime.phase = RuntimePhase::kFailed;
    return 0U;
  }
  g_runtime.input = allocation;
  g_runtime.input_byte_length = byte_length;
  g_runtime.input_wire_pointer = wire_pointer;
  g_runtime.status = WireCompileStatus::kInputAllocated;
  g_runtime.phase = RuntimePhase::kInputAllocated;
  return wire_pointer;
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
  if (input_pointer != g_runtime.input_wire_pointer ||
      input_length != g_runtime.input_byte_length) {
    g_runtime.status = WireCompileStatus::kInvalidArgument;
    g_runtime.phase = RuntimePhase::kFailed;
    return wire_status(g_runtime.status);
  }
  InputFrameRegionOffsets input_offsets;
  if (!validate_frame_envelope(g_runtime.input, g_runtime.input_byte_length,
                               &input_offsets)) {
    g_runtime.status = WireCompileStatus::kInvalidFrame;
    g_runtime.phase = RuntimePhase::kFailed;
    return wire_status(g_runtime.status);
  }
  const ValidatedInputFrameRegions input_regions(
      g_runtime.input + input_offsets.profile_offset,
      input_offsets.profile_byte_length,
      g_runtime.input + input_offsets.request_offset,
      input_offsets.request_byte_length);

  g_runtime.phase = RuntimePhase::kCompiling;
  ArtifactV3ResultSink result_sink;
  const ArtifactV3CompileResult result =
      compile_artifact == nullptr
          ? ArtifactV3CompileResult{
                WireCompileStatus::kInternalError,
                ReviewOnlyBlocker::kCanonicalArtifactV3Unavailable}
          : compile_artifact(input_regions, result_sink);
  if (g_runtime.phase != RuntimePhase::kCompiling ||
      !allocator_metrics_healthy()) {
    if (g_runtime.phase != RuntimePhase::kCompiling) {
      report_native_diagnostic(
          NativeDiagnosticCode::kRuntimePostCompilePhaseInvariant);
    }
    if (!allocator_metrics_healthy()) {
      report_allocator_metrics_failure();
    }
    result_sink.discard();
    g_runtime.status = WireCompileStatus::kInternalError;
    g_runtime.phase = RuntimePhase::kFailed;
    return wire_status(g_runtime.status);
  }

  if (result.status == WireCompileStatus::kArtifactReady) {
    if (result.blocker.has_value() || result_sink.failed_ ||
        !result_sink.invocation_limit_bound_ || !result_sink.committed_ ||
        result_sink.bytes_ == nullptr || result_sink.byte_length_ == 0U ||
        !frontend_work_metrics_ready() ||
        ranges_overlap(g_runtime.input_wire_pointer,
                       g_runtime.input_byte_length,
                       result_sink.wire_pointer_,
                       result_sink.byte_length_)) {
      report_native_diagnostic(
          NativeDiagnosticCode::kRuntimeArtifactReadyInvariant);
      if (!allocator_metrics_healthy()) {
        report_allocator_metrics_failure();
      }
      const WireCompileStatus failure =
          result.blocker.has_value()
              ? WireCompileStatus::kInternalError
              : result_sink.failed_ ? result_sink.failure_status_
                                    : WireCompileStatus::kInternalError;
      result_sink.discard();
      g_runtime.status = !allocator_metrics_healthy()
                             ? WireCompileStatus::kInternalError
                             : failure;
      g_runtime.phase = RuntimePhase::kFailed;
      return wire_status(g_runtime.status);
    }
    g_runtime.result = result_sink.bytes_;
    g_runtime.result_byte_length = result_sink.byte_length_;
    g_runtime.result_wire_pointer = result_sink.wire_pointer_;
    result_sink.adopted_ = true;
    g_runtime.status = WireCompileStatus::kArtifactReady;
    g_runtime.phase = RuntimePhase::kArtifactReady;
    return wire_status(g_runtime.status);
  }

  const WireCompileStatus terminal_status = result_sink.failed_
                                                ? result_sink.failure_status_
                                                : result.status;
  result_sink.discard();
  switch (terminal_status) {
    case WireCompileStatus::kInvalidFrame:
    case WireCompileStatus::kAbiMismatch:
    case WireCompileStatus::kVfsError:
    case WireCompileStatus::kResourceLimit:
    case WireCompileStatus::kInternalError:
      g_runtime.status = terminal_status;
      break;
    default:
      report_native_diagnostic(
          NativeDiagnosticCode::kRuntimeTerminalStatusInvalid);
      g_runtime.status = WireCompileStatus::kInternalError;
      break;
  }
  if (!allocator_metrics_healthy()) {
    report_allocator_metrics_failure();
    g_runtime.status = WireCompileStatus::kInternalError;
  }
  g_runtime.phase = RuntimePhase::kFailed;
  return wire_status(g_runtime.status);
}

void runtime_free(std::uint32_t pointer, std::uint32_t byte_length) {
  if (g_runtime.phase == RuntimePhase::kCompiling) {
    g_runtime.status = WireCompileStatus::kInvalidState;
    g_runtime.phase = RuntimePhase::kFailed;
    return;
  }
  if (g_runtime.input == nullptr ||
      pointer != g_runtime.input_wire_pointer ||
      byte_length != g_runtime.input_byte_length) {
    g_runtime.status = WireCompileStatus::kInvalidArgument;
    g_runtime.phase = RuntimePhase::kFailed;
    return;
  }
  const bool was_allocated = g_runtime.phase == RuntimePhase::kInputAllocated;
  const bool artifact_ready = g_runtime.phase == RuntimePhase::kArtifactReady;
  release_input();
  if (!allocator_metrics_healthy()) {
    g_runtime.phase = RuntimePhase::kFailed;
    g_runtime.status = WireCompileStatus::kInternalError;
    return;
  }
  if (was_allocated) {
    g_runtime.phase = RuntimePhase::kIdle;
    g_runtime.status = WireCompileStatus::kIdle;
  } else if (artifact_ready) {
    g_runtime.phase = RuntimePhase::kArtifactReady;
    g_runtime.status = WireCompileStatus::kArtifactReady;
  }
}

void runtime_reset() {
  if (g_runtime.phase == RuntimePhase::kCompiling) {
    g_runtime.status = WireCompileStatus::kInvalidState;
    g_runtime.phase = RuntimePhase::kFailed;
    return;
  }
  release_input();
  release_result();
  g_runtime = RuntimeState{};
  reset_frontend_work_metrics();
  if (!allocator_metrics_healthy()) {
    g_runtime.phase = RuntimePhase::kFailed;
    g_runtime.status = WireCompileStatus::kInternalError;
  }
}

std::uint32_t runtime_result_length() {
  return g_runtime.phase == RuntimePhase::kArtifactReady
             ? g_runtime.result_byte_length
             : 0U;
}

std::uint32_t runtime_result_pointer() {
  return g_runtime.phase == RuntimePhase::kArtifactReady
             ? g_runtime.result_wire_pointer
             : 0U;
}

std::int32_t runtime_status() { return wire_status(g_runtime.status); }

}  // namespace browsergrad::cpp_cute
