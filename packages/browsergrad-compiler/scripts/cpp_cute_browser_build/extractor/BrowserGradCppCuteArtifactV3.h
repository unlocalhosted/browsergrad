#pragma once

#include "BrowserGradCppCuteRuntime.h"

#include <cstdint>

namespace browsergrad::cpp_cute {

ArtifactV3CompileResult build_artifact_v3_placeholder(
    const std::uint8_t* input, std::uint32_t input_byte_length,
    ArtifactV3ResultSink& result_sink);

}  // namespace browsergrad::cpp_cute
