#pragma once

#include "BrowserGradCppCuteRuntime.h"

#include <cstdint>

namespace browsergrad::cpp_cute {

ArtifactV3CompileResult build_artifact_v3_placeholder(
    const ValidatedInputFrameRegions& input,
    ArtifactV3ResultSink& result_sink);

}  // namespace browsergrad::cpp_cute
