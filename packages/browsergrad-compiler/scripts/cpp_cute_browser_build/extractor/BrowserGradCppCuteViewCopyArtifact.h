#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "BrowserGradCppCuteArtifactJson.h"
#include "BrowserGradCppCuteProducer.h"

namespace browsergrad::cpp_cute {

struct ViewCopyDeclarationSpans final {
  std::string_view declaration_span_id;
  std::string_view identity_span_id;
};

struct ViewCopyArtifactBuildInput final {
  std::string_view main_virtual_path;
  std::string_view main_content_sha256;
  std::uint64_t main_byte_length = 0U;
  std::size_t maximum_artifact_byte_length = 0U;
  ViewCopyDeclarationSpans function_spans;
  std::array<ViewCopyDeclarationSpans, 2U> parameter_spans;
  std::array<ViewCopyDeclarationSpans, 2U> tensor_spans;
  std::string_view copy_span_id;
};

struct ViewCopyArtifactGraph final {
  artifact_json::Json::Array types;
  artifact_json::Json::Array declarations;
  artifact_json::Json::Array facts;
  artifact_json::Json::Array function_bodies;
  artifact_json::Json::Array source_entities;
  artifact_json::Json source_abi;
  artifact_json::Json entry;
  std::string entry_id;
  std::vector<std::string> fact_ids;
  std::string selected_source_entity_id;
  std::string shared_surface_sha256;
};

/**
 * Projects one exact AST-derived CuTe Tensor copy into the closed Artifact V3
 * graph. The caller owns input-closure and span serialization; this function
 * owns all semantic graph identities and rejects incomplete or substituted
 * producer observations before any artifact bytes can be committed.
 */
ViewCopyArtifactGraph build_view_copy_artifact_graph(
    const ProducerViewCopyObservation& observation,
    const ViewCopyArtifactBuildInput& input);

bool equivalent_view_copy_observation(
    const ProducerViewCopyObservation& left,
    const ProducerViewCopyObservation& right) noexcept;

}  // namespace browsergrad::cpp_cute
