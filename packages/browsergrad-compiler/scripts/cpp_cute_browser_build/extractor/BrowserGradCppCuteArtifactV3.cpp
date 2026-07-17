#include "BrowserGradCppCuteArtifactV3.h"

namespace browsergrad::cpp_cute {

ArtifactV3CompileResult build_artifact_v3_placeholder(
    const ValidatedInputFrameRegions& input,
    ArtifactV3ResultSink& result_sink) {
  static_cast<void>(input);
  static_cast<void>(result_sink);

  // Gate 3 establishes ownership only. A review trace is not an artifact.
  // Explicit CUDA device/host passes and the canonical artifact-v3 writer are
  // still absent, so the existing internal-error behavior remains mandatory.
  return {
      WireCompileStatus::kInternalError,
      ReviewOnlyBlocker::kCudaDualPassUnavailable,
  };
}

}  // namespace browsergrad::cpp_cute
