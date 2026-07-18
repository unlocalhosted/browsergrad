#pragma once

#include "BrowserGradCppCuteCompileSession.h"
#include "BrowserGradCppCuteProducer.h"
#include "BrowserGradCppCuteRuntime.h"

#include <cstdint>

namespace browsergrad::cpp_cute {

enum class ArtifactV3WriteStatus : std::uint8_t {
  kReady,
  kInvalidObservation,
  kResourceLimit,
  kInternalError,
};

/**
 * Builds, hashes, validates, and atomically commits one canonical Artifact V3.
 *
 * The function accepts only the immutable decoded admission authority and the
 * owned two-pass producer observation. It performs no filesystem or compiler
 * work and does not allocate result storage until all artifact bytes have been
 * constructed and revalidated as canonical JSON.
 */
ArtifactV3WriteStatus write_cpp_cute_artifact_v3(
    const ProducerReviewResult& producer,
    const DecodedCompileSession& session,
    ArtifactV3ResultSink& result_sink) noexcept;

}  // namespace browsergrad::cpp_cute
