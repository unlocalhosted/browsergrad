#include "BrowserGradCppCuteViewCopyArtifact.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace browsergrad::cpp_cute {
namespace {

using artifact_json::array;
using artifact_json::hash_json;
using artifact_json::InvalidObservation;
using artifact_json::Json;
using artifact_json::object;
using artifact_json::stable_id;

constexpr std::size_t kMaximumArtifactStringByteLength = 16U * 1024U;
constexpr std::size_t kMaximumHierarchyNodeCount = 4096U;
constexpr std::uint32_t kMaximumHierarchyDepth = 64U;

bool bounded_text(const std::string_view value, const std::size_t maximum,
                  const bool allow_empty = false) noexcept {
  return (allow_empty || !value.empty()) && value.size() <= maximum &&
         value.find('\0') == std::string_view::npos;
}

bool lowercase_sha256(const std::string_view value) noexcept {
  if (value.size() != 64U) return false;
  return std::all_of(value.begin(), value.end(), [](const char byte) {
    return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
  });
}

bool cute_header_path(const std::string_view path) noexcept {
  return path.starts_with("/toolchain/cutlass/include/cute/") &&
         bounded_text(path, 4096U);
}

bool source_scoped_clang_usr(const std::string_view usr,
                             const std::string_view virtual_path,
                             const std::uint64_t declaration_begin) noexcept {
  const std::size_t separator = virtual_path.find_last_of('/');
  const std::string_view basename =
      separator == std::string_view::npos
          ? virtual_path
          : virtual_path.substr(separator + 1U);
  if (basename.empty() || basename.find('@') != std::string_view::npos ||
      basename.find('\\') != std::string_view::npos) {
    return false;
  }
  std::array<char, 32U> offset{};
  const auto converted = std::to_chars(
      offset.data(), offset.data() + offset.size(), declaration_begin);
  if (converted.ec != std::errc{}) return false;
  const std::string_view decimal(offset.data(), converted.ptr);
  if (!usr.starts_with("c:") ||
      usr.size() <= 2U + basename.size() + 1U + decimal.size() + 1U) {
    return false;
  }
  return usr.substr(2U).starts_with(basename) &&
         usr[2U + basename.size()] == '@' &&
         usr.substr(3U + basename.size()).starts_with(decimal) &&
         usr[3U + basename.size() + decimal.size()] == '@';
}

std::string decimal_u64(const std::uint64_t value) {
  std::array<char, 32U> output{};
  const auto converted =
      std::to_chars(output.data(), output.data() + output.size(), value);
  if (converted.ec != std::errc{}) throw InvalidObservation();
  return std::string(output.data(), converted.ptr);
}

Json qualifiers(const bool is_const = false) {
  return object(
      {{"const", is_const}, {"restrict", false}, {"volatile", false}});
}

Json cuda_attributes(const bool host = false, const bool device = false,
                     const bool global = false,
                     const bool force_inline = false) {
  return object({{"device", device},
                 {"forceInline", force_inline},
                 {"global", global},
                 {"host", host}});
}

Json source_origin(const std::string_view span_id) {
  return object({{"kind", "source"}, {"spanId", span_id}});
}

Json template_substitution_origin(const std::string_view span_id) {
  return object({{"anchorSpanId", span_id},
                 {"kind", "implicit"},
                 {"reason", "template-substitution"}});
}

struct HierarchySummary final {
  std::vector<std::int64_t> shape_leaves;
  std::vector<std::int64_t> stride_leaves;
  std::size_t nodes = 0U;
};

void validate_hierarchy_pair(const ProducerIntegerHierarchy& shape,
                             const ProducerIntegerHierarchy& stride,
                             HierarchySummary& summary,
                             const std::uint32_t depth) {
  if (depth > kMaximumHierarchyDepth ||
      summary.nodes > kMaximumHierarchyNodeCount - 2U ||
      shape.tuple != stride.tuple) {
    throw InvalidObservation();
  }
  summary.nodes += 2U;
  if (!shape.tuple) {
    if (!shape.elements.empty() || !stride.elements.empty() ||
        shape.value <= 0 || stride.value < 0) {
      throw InvalidObservation();
    }
    summary.shape_leaves.push_back(shape.value);
    summary.stride_leaves.push_back(stride.value);
    return;
  }
  if (shape.elements.empty() ||
      shape.elements.size() != stride.elements.size()) {
    throw InvalidObservation();
  }
  for (std::size_t index = 0U; index < shape.elements.size(); ++index) {
    validate_hierarchy_pair(shape.elements[index], stride.elements[index],
                            summary, depth + 1U);
  }
}

bool same_hierarchy(const ProducerIntegerHierarchy& left,
                    const ProducerIntegerHierarchy& right) noexcept {
  if (left.tuple != right.tuple || left.value != right.value ||
      left.elements.size() != right.elements.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.elements.size(); ++index) {
    if (!same_hierarchy(left.elements[index], right.elements[index])) {
      return false;
    }
  }
  return true;
}

void validate_tensor_layout(const ProducerViewCopyTensorObservation& tensor) {
  HierarchySummary summary;
  validate_hierarchy_pair(tensor.shape, tensor.stride, summary, 1U);
  const std::uint32_t top_rank =
      tensor.shape.tuple
          ? static_cast<std::uint32_t>(tensor.shape.elements.size())
          : 1U;
  if (tensor.rank != top_rank ||
      tensor.leaf_rank != summary.shape_leaves.size()) {
    throw InvalidObservation();
  }
  std::int64_t size = 1;
  std::int64_t cosize = 1;
  for (std::size_t index = 0U; index < summary.shape_leaves.size(); ++index) {
    const std::int64_t shape = summary.shape_leaves[index];
    const std::int64_t stride = summary.stride_leaves[index];
    if (size > std::numeric_limits<std::int64_t>::max() / shape) {
      throw InvalidObservation();
    }
    size *= shape;
    const std::int64_t extent = shape - 1;
    if (stride != 0 &&
        extent > std::numeric_limits<std::int64_t>::max() / stride) {
      throw InvalidObservation();
    }
    const std::int64_t contribution = extent * stride;
    if (cosize > std::numeric_limits<std::int64_t>::max() - contribution) {
      throw InvalidObservation();
    }
    cosize += contribution;
  }
  if (tensor.size != size || tensor.cosize != cosize) {
    throw InvalidObservation();
  }
}

Json hierarchy_json(const ProducerIntegerHierarchy& hierarchy) {
  if (!hierarchy.tuple) {
    return object(
        {{"kind", "scalar"},
         {"value", object({{"kind", "integer"},
                           {"value", decimal_u64(static_cast<std::uint64_t>(
                                         hierarchy.value))}})}});
  }
  Json::Array elements;
  elements.reserve(hierarchy.elements.size());
  for (const ProducerIntegerHierarchy& element : hierarchy.elements) {
    elements.push_back(hierarchy_json(element));
  }
  return object({{"elements", Json(std::move(elements))}, {"kind", "tuple"}});
}

void validate_declaration_spans(const ViewCopyDeclarationSpans& spans) {
  if (!bounded_text(spans.declaration_span_id, 256U) ||
      !bounded_text(spans.identity_span_id, 256U)) {
    throw InvalidObservation();
  }
}

void validate_observation(const ProducerViewCopyObservation& observation,
                          const ViewCopyArtifactBuildInput& input) {
  if (!observation.selected || observation.ambiguous ||
      !observation.resolved_function || !observation.resolved_copy ||
      observation.cuda_host || !observation.cuda_device ||
      observation.cuda_global ||
      !bounded_text(observation.canonical_usr,
                    kMaximumArtifactStringByteLength) ||
      !std::string_view(observation.canonical_usr).starts_with("c:@") ||
      !bounded_text(observation.canonical_name,
                    kMaximumArtifactStringByteLength) ||
      !bounded_text(observation.canonical_type,
                    kMaximumArtifactStringByteLength) ||
      observation.copy_callee_name != "cute::copy" ||
      !bounded_text(observation.copy_callee_usr,
                    kMaximumArtifactStringByteLength) ||
      !std::string_view(observation.copy_callee_usr).starts_with("c:@") ||
      !cute_header_path(observation.copy_callee_path) ||
      observation.declaration_begin_byte >= observation.declaration_end_byte ||
      observation.identity_begin_byte >= observation.identity_end_byte ||
      observation.identity_begin_byte < observation.declaration_begin_byte ||
      observation.identity_end_byte > observation.declaration_end_byte ||
      observation.copy_begin_byte >= observation.copy_end_byte ||
      observation.copy_begin_byte < observation.declaration_begin_byte ||
      observation.copy_end_byte > observation.declaration_end_byte ||
      observation.parameters.size() != 2U || observation.tensors.size() != 2U ||
      observation.parameters[0U].scalar_kind ==
          ProducerViewCopyScalarKind::kUnsupported ||
      observation.parameters[0U].scalar_kind !=
          observation.parameters[1U].scalar_kind ||
      observation.source_tensor_ordinal >= observation.tensors.size() ||
      observation.destination_tensor_ordinal >= observation.tensors.size() ||
      observation.source_tensor_ordinal ==
          observation.destination_tensor_ordinal ||
      !bounded_text(input.main_virtual_path, 4096U) ||
      !lowercase_sha256(input.main_content_sha256) ||
      observation.declaration_end_byte > input.main_byte_length ||
      input.maximum_artifact_byte_length == 0U ||
      !bounded_text(input.copy_span_id, 256U)) {
    throw InvalidObservation();
  }
  validate_declaration_spans(input.function_spans);

  std::set<std::string, std::less<>> canonical_usrs;
  canonical_usrs.emplace(observation.canonical_usr);
  for (std::size_t index = 0U; index < observation.parameters.size(); ++index) {
    const ProducerViewCopyParameterObservation& parameter =
        observation.parameters[index];
    validate_declaration_spans(input.parameter_spans[index]);
    if (!parameter.resolved_pointer ||
        parameter.pointee_const != (index == 0U) ||
        parameter.ordinal != index ||
        !bounded_text(parameter.canonical_usr,
                      kMaximumArtifactStringByteLength) ||
        !source_scoped_clang_usr(parameter.canonical_usr,
                                 input.main_virtual_path,
                                 parameter.declaration_begin_byte) ||
        !bounded_text(parameter.canonical_name,
                      kMaximumArtifactStringByteLength) ||
        !bounded_text(parameter.canonical_type,
                      kMaximumArtifactStringByteLength) ||
        parameter.declaration_begin_byte >= parameter.declaration_end_byte ||
        parameter.identity_begin_byte >= parameter.identity_end_byte ||
        parameter.declaration_begin_byte < observation.declaration_begin_byte ||
        parameter.declaration_end_byte > observation.declaration_end_byte ||
        parameter.identity_begin_byte < parameter.declaration_begin_byte ||
        parameter.identity_end_byte > parameter.declaration_end_byte ||
        !canonical_usrs.emplace(parameter.canonical_usr).second) {
      throw InvalidObservation();
    }
  }

  for (std::size_t index = 0U; index < observation.tensors.size(); ++index) {
    const ProducerViewCopyTensorObservation& tensor =
        observation.tensors[index];
    validate_declaration_spans(input.tensor_spans[index]);
    if (!tensor.resolved_tensor_type || !tensor.resolved_static_affine_layout ||
        !tensor.initializer_parameter_bound ||
        tensor.engine_parameter_ordinal > 1U ||
        tensor.engine_pointee_const !=
            (tensor.engine_parameter_ordinal == 0U) ||
        !bounded_text(tensor.canonical_usr, kMaximumArtifactStringByteLength) ||
        !source_scoped_clang_usr(tensor.canonical_usr,
                                 input.main_virtual_path,
                                 tensor.declaration_begin_byte) ||
        !bounded_text(tensor.canonical_name,
                      kMaximumArtifactStringByteLength) ||
        tensor.canonical_name.size() > kMaximumArtifactStringByteLength -
                                           std::string_view(".layout").size() ||
        !bounded_text(tensor.canonical_type,
                      kMaximumArtifactStringByteLength) ||
        !bounded_text(tensor.layout_canonical_type,
                      kMaximumArtifactStringByteLength) ||
        !bounded_text(tensor.initializer_callee_usr,
                      kMaximumArtifactStringByteLength) ||
        !std::string_view(tensor.initializer_callee_usr).starts_with("c:@") ||
        tensor.initializer_callee_name != "cute::make_tensor" ||
        !cute_header_path(tensor.tensor_template_path) ||
        !cute_header_path(tensor.layout_template_path) ||
        !cute_header_path(tensor.initializer_callee_path) ||
        tensor.declaration_begin_byte >= tensor.declaration_end_byte ||
        tensor.identity_begin_byte >= tensor.identity_end_byte ||
        tensor.declaration_begin_byte < observation.declaration_begin_byte ||
        tensor.declaration_end_byte > observation.declaration_end_byte ||
        tensor.identity_begin_byte < tensor.declaration_begin_byte ||
        tensor.identity_end_byte > tensor.declaration_end_byte ||
        !canonical_usrs.emplace(tensor.canonical_usr).second) {
      throw InvalidObservation();
    }
    validate_tensor_layout(tensor);
  }
  const ProducerViewCopyTensorObservation& source =
      observation.tensors[observation.source_tensor_ordinal];
  const ProducerViewCopyTensorObservation& destination =
      observation.tensors[observation.destination_tensor_ordinal];
  if (source.engine_parameter_ordinal != 0U ||
      destination.engine_parameter_ordinal != 1U ||
      source.rank != destination.rank ||
      source.leaf_rank != destination.leaf_rank ||
      source.size != destination.size ||
      !same_hierarchy(source.shape, destination.shape)) {
    throw InvalidObservation();
  }
}

Json source_entity_id_projection(const std::string_view canonical_identity,
                                 const std::string_view entity_kind,
                                 const ViewCopyArtifactBuildInput& input,
                                 const std::uint64_t begin,
                                 const std::uint64_t end) {
  const Json canonical_range =
      object({{"contentSha256", input.main_content_sha256},
              {"endByte", decimal_u64(end)},
              {"startByte", decimal_u64(begin)},
              {"virtualPath", input.main_virtual_path}});
  return object(
      {{"canonicalIdentity", canonical_identity},
       {"domain", "browsergrad.compiler.cpp-cute.source-entity-id.v1"},
       {"entityKind", entity_kind},
       {"origin",
        object({{"kind", "source"},
                {"span", object({{"expansion", canonical_range},
                                 {"spelling", canonical_range}})}})}});
}

template <typename Id>
void sort_records(Json::Array& records, Id id) {
  std::sort(records.begin(), records.end(),
            [id](const Json& left, const Json& right) {
              return std::get<std::string>(
                         std::get<Json::Object>(left.value).at(id).value) <
                     std::get<std::string>(
                         std::get<Json::Object>(right.value).at(id).value);
            });
}

struct TypeIds final {
  std::string void_type;
  std::string scalar_type;
  std::string const_scalar_type;
  std::array<std::string, 2U> pointer_types;
  std::string function_type;
  std::array<std::string, 2U> layout_types;
  std::array<std::string, 2U> tensor_types;
};

struct ScalarDescriptor final {
  std::string_view canonical_name;
  std::string_view builtin;
};

ScalarDescriptor scalar_descriptor(
    const ProducerViewCopyScalarKind kind) {
  switch (kind) {
    case ProducerViewCopyScalarKind::kFloat32:
      return {"float", "float"};
    case ProducerViewCopyScalarKind::kSignedInt32:
      return {"int", "int"};
    case ProducerViewCopyScalarKind::kUnsignedInt32:
      return {"unsigned int", "unsigned-int"};
    case ProducerViewCopyScalarKind::kUnsupported:
      throw InvalidObservation();
  }
  throw InvalidObservation();
}

Json builtin_type(const std::string_view type_id,
                  const std::string_view canonical_name,
                  const std::string_view builtin, const Json& origin,
                  const bool is_const = false) {
  return object({{"builtin", builtin},
                 {"canonicalName", canonical_name},
                 {"kind", "builtin"},
                 {"origin", origin},
                 {"qualifiers", qualifiers(is_const)},
                 {"typeId", type_id}});
}

Json declaration_record(
    const std::string_view declaration_id, const std::string_view kind,
    const std::string_view canonical_usr, const std::string_view canonical_name,
    const Json& origin, const Json& identity_span_id,
    const Json& lexical_parent_id, const Json& semantic_parent_id,
    const std::string_view type_id, const std::string_view definition_kind,
    const std::string_view linkage, const std::string_view storage_duration,
    const std::string_view memory_space, const Json& attributes) {
  return object({{"canonicalName", canonical_name},
                 {"canonicalUsr", canonical_usr},
                 {"cudaAttributes", attributes},
                 {"declarationId", declaration_id},
                 {"definitionKind", definition_kind},
                 {"identitySpanId", identity_span_id},
                 {"initializerExpressionId", nullptr},
                 {"kind", kind},
                 {"lexicalParentId", lexical_parent_id},
                 {"linkage", linkage},
                 {"mangledName", nullptr},
                 {"memorySpace", memory_space},
                 {"origin", origin},
                 {"semanticParentId", semantic_parent_id},
                 {"storageDuration", storage_duration},
                 {"targetTypeId", nullptr},
                 {"typeId", type_id}});
}

}  // namespace

bool equivalent_view_copy_observation(
    const ProducerViewCopyObservation& left,
    const ProducerViewCopyObservation& right) noexcept {
  if (left.selected != right.selected || left.ambiguous != right.ambiguous ||
      left.resolved_function != right.resolved_function ||
      left.resolved_copy != right.resolved_copy ||
      left.cuda_host != right.cuda_host ||
      left.cuda_device != right.cuda_device ||
      left.cuda_global != right.cuda_global ||
      left.force_inline != right.force_inline ||
      left.canonical_usr != right.canonical_usr ||
      left.canonical_name != right.canonical_name ||
      left.canonical_type != right.canonical_type ||
      left.copy_callee_usr != right.copy_callee_usr ||
      left.copy_callee_name != right.copy_callee_name ||
      left.copy_callee_path != right.copy_callee_path ||
      left.declaration_begin_byte != right.declaration_begin_byte ||
      left.declaration_end_byte != right.declaration_end_byte ||
      left.identity_begin_byte != right.identity_begin_byte ||
      left.identity_end_byte != right.identity_end_byte ||
      left.copy_begin_byte != right.copy_begin_byte ||
      left.copy_end_byte != right.copy_end_byte ||
      left.source_tensor_ordinal != right.source_tensor_ordinal ||
      left.destination_tensor_ordinal != right.destination_tensor_ordinal ||
      left.parameters.size() != right.parameters.size() ||
      left.tensors.size() != right.tensors.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.parameters.size(); ++index) {
    const ProducerViewCopyParameterObservation& a = left.parameters[index];
    const ProducerViewCopyParameterObservation& b = right.parameters[index];
    if (a.resolved_pointer != b.resolved_pointer ||
        a.scalar_kind != b.scalar_kind ||
        a.pointee_const != b.pointee_const || a.ordinal != b.ordinal ||
        a.canonical_usr != b.canonical_usr ||
        a.canonical_name != b.canonical_name ||
        a.canonical_type != b.canonical_type ||
        a.declaration_begin_byte != b.declaration_begin_byte ||
        a.declaration_end_byte != b.declaration_end_byte ||
        a.identity_begin_byte != b.identity_begin_byte ||
        a.identity_end_byte != b.identity_end_byte) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < left.tensors.size(); ++index) {
    const ProducerViewCopyTensorObservation& a = left.tensors[index];
    const ProducerViewCopyTensorObservation& b = right.tensors[index];
    if (a.resolved_tensor_type != b.resolved_tensor_type ||
        a.resolved_static_affine_layout != b.resolved_static_affine_layout ||
        a.initializer_parameter_bound != b.initializer_parameter_bound ||
        a.engine_pointee_const != b.engine_pointee_const ||
        a.engine_parameter_ordinal != b.engine_parameter_ordinal ||
        a.canonical_usr != b.canonical_usr ||
        a.canonical_name != b.canonical_name ||
        a.canonical_type != b.canonical_type ||
        a.tensor_template_path != b.tensor_template_path ||
        a.initializer_callee_usr != b.initializer_callee_usr ||
        a.initializer_callee_name != b.initializer_callee_name ||
        a.initializer_callee_path != b.initializer_callee_path ||
        a.layout_canonical_type != b.layout_canonical_type ||
        a.layout_template_path != b.layout_template_path ||
        a.declaration_begin_byte != b.declaration_begin_byte ||
        a.declaration_end_byte != b.declaration_end_byte ||
        a.identity_begin_byte != b.identity_begin_byte ||
        a.identity_end_byte != b.identity_end_byte || a.rank != b.rank ||
        a.leaf_rank != b.leaf_rank || a.size != b.size ||
        a.cosize != b.cosize || !same_hierarchy(a.shape, b.shape) ||
        !same_hierarchy(a.stride, b.stride)) {
      return false;
    }
  }
  return true;
}

ViewCopyArtifactGraph build_view_copy_artifact_graph(
    const ProducerViewCopyObservation& observation,
    const ViewCopyArtifactBuildInput& input) {
  validate_observation(observation, input);
  const std::size_t maximum = input.maximum_artifact_byte_length;
  const ProducerViewCopyTensorObservation& source =
      observation.tensors[observation.source_tensor_ordinal];
  const ProducerViewCopyTensorObservation& destination =
      observation.tensors[observation.destination_tensor_ordinal];
  const ProducerViewCopyParameterObservation& source_parameter =
      observation.parameters[0U];
  const ProducerViewCopyParameterObservation& destination_parameter =
      observation.parameters[1U];
  const ScalarDescriptor scalar =
      scalar_descriptor(source_parameter.scalar_kind);
  const std::string const_scalar_name =
      std::string("const ") + std::string(scalar.canonical_name);

  const Json function_origin =
      source_origin(input.function_spans.declaration_span_id);
  const Json source_parameter_origin =
      source_origin(input.parameter_spans[0U].declaration_span_id);
  const Json destination_parameter_origin =
      source_origin(input.parameter_spans[1U].declaration_span_id);
  const Json source_tensor_origin =
      source_origin(input.tensor_spans[observation.source_tensor_ordinal]
                        .declaration_span_id);
  const Json destination_tensor_origin =
      source_origin(input.tensor_spans[observation.destination_tensor_ordinal]
                        .declaration_span_id);
  const Json copy_origin = source_origin(input.copy_span_id);

  const Json source_layout_origin = template_substitution_origin(
      input.tensor_spans[observation.source_tensor_ordinal]
          .declaration_span_id);
  const Json destination_layout_origin = template_substitution_origin(
      input.tensor_spans[observation.destination_tensor_ordinal]
          .declaration_span_id);
  const Json layout_template_origin = source_layout_origin;
  const Json tensor_template_origin = source_layout_origin;

  const std::string layout_template_declaration_id =
      stable_id("declaration",
                object({{"canonicalName", "cute::Layout"},
                        {"canonicalUsr", "c:@N@cute@ST@Layout"},
                        {"kind", "template"},
                        {"origin", layout_template_origin}}),
                maximum);
  const std::string tensor_template_declaration_id =
      stable_id("declaration",
                object({{"canonicalName", "cute::Tensor"},
                        {"canonicalUsr", "c:@N@cute@ST@Tensor"},
                        {"kind", "template"},
                        {"origin", tensor_template_origin}}),
                maximum);

  TypeIds type_ids;
  type_ids.void_type = stable_id("type",
                                 object({{"builtin", "void"},
                                         {"canonicalName", "void"},
                                         {"origin", function_origin},
                                         {"qualifiers", qualifiers()}}),
                                 maximum);
  type_ids.scalar_type =
      stable_id("type",
                object({{"builtin", scalar.builtin},
                        {"canonicalName", scalar.canonical_name},
                        {"origin", destination_parameter_origin},
                        {"qualifiers", qualifiers()}}),
                maximum);
  type_ids.const_scalar_type =
      stable_id("type",
                object({{"builtin", scalar.builtin},
                        {"canonicalName", const_scalar_name},
                        {"origin", source_parameter_origin},
                        {"qualifiers", qualifiers(true)}}),
                maximum);
  type_ids.pointer_types[0U] =
      stable_id("type",
                object({{"addressSpace", "global"},
                        {"canonicalName", source_parameter.canonical_type},
                        {"kind", "pointer"},
                        {"origin", source_parameter_origin},
                        {"pointeeTypeId", type_ids.const_scalar_type},
                        {"qualifiers", qualifiers()}}),
                maximum);
  type_ids.pointer_types[1U] =
      stable_id("type",
                object({{"addressSpace", "global"},
                        {"canonicalName", destination_parameter.canonical_type},
                        {"kind", "pointer"},
                        {"origin", destination_parameter_origin},
                        {"pointeeTypeId", type_ids.scalar_type},
                        {"qualifiers", qualifiers()}}),
                maximum);
  type_ids.function_type = stable_id(
      "type",
      object({{"callingConvention", "cuda-device"},
              {"canonicalName", observation.canonical_type},
              {"kind", "function"},
              {"origin", function_origin},
              {"parameterTypeIds",
               array({type_ids.pointer_types[0U], type_ids.pointer_types[1U]})},
              {"qualifiers", qualifiers()},
              {"returnTypeId", type_ids.void_type},
              {"variadic", false}}),
      maximum);

  const std::array<const ProducerViewCopyTensorObservation*, 2U> tensors = {
      &source, &destination};
  const std::array<std::size_t, 2U> tensor_span_ordinals = {
      observation.source_tensor_ordinal,
      observation.destination_tensor_ordinal};
  const std::array<Json, 2U> tensor_origins = {source_tensor_origin,
                                               destination_tensor_origin};
  const std::array<Json, 2U> layout_origins = {source_layout_origin,
                                               destination_layout_origin};
  for (std::size_t index = 0U; index < tensors.size(); ++index) {
    type_ids.layout_types[index] = stable_id(
        "type",
        object({{"canonicalName", tensors[index]->layout_canonical_type},
                {"kind", "template-specialization"},
                {"origin", tensor_origins[index]},
                {"templateDeclarationId", layout_template_declaration_id}}),
        maximum);
    type_ids.tensor_types[index] = stable_id(
        "type",
        object({{"arguments",
                 array({object({{"kind", "type"},
                                {"typeId", type_ids.pointer_types[index]}}),
                        object({{"kind", "type"},
                                {"typeId", type_ids.layout_types[index]}})})},
                {"canonicalName", tensors[index]->canonical_type},
                {"kind", "template-specialization"},
                {"origin", tensor_origins[index]},
                {"templateDeclarationId", tensor_template_declaration_id}}),
        maximum);
  }

  Json::Array types = {
      builtin_type(type_ids.void_type, "void", "void", function_origin),
      builtin_type(type_ids.scalar_type, scalar.canonical_name, scalar.builtin,
                   destination_parameter_origin),
      builtin_type(type_ids.const_scalar_type, const_scalar_name, scalar.builtin,
                   source_parameter_origin, true),
      object({{"addressSpace", "global"},
              {"canonicalName", source_parameter.canonical_type},
              {"kind", "pointer"},
              {"origin", source_parameter_origin},
              {"pointeeTypeId", type_ids.const_scalar_type},
              {"qualifiers", qualifiers()},
              {"typeId", type_ids.pointer_types[0U]}}),
      object({{"addressSpace", "global"},
              {"canonicalName", destination_parameter.canonical_type},
              {"kind", "pointer"},
              {"origin", destination_parameter_origin},
              {"pointeeTypeId", type_ids.scalar_type},
              {"qualifiers", qualifiers()},
              {"typeId", type_ids.pointer_types[1U]}}),
      object({{"callingConvention", "cuda-device"},
              {"canonicalName", observation.canonical_type},
              {"kind", "function"},
              {"origin", function_origin},
              {"parameterTypeIds",
               array({type_ids.pointer_types[0U], type_ids.pointer_types[1U]})},
              {"qualifiers", qualifiers()},
              {"returnTypeId", type_ids.void_type},
              {"typeId", type_ids.function_type},
              {"variadic", false}}),
  };
  for (std::size_t index = 0U; index < tensors.size(); ++index) {
    types.push_back(
        object({{"arguments", Json::Array{}},
                {"canonicalName", tensors[index]->layout_canonical_type},
                {"kind", "template-specialization"},
                {"origin", tensor_origins[index]},
                {"qualifiers", qualifiers()},
                {"templateDeclarationId", layout_template_declaration_id},
                {"typeId", type_ids.layout_types[index]}}));
    types.push_back(
        object({{"arguments",
                 array({object({{"kind", "type"},
                                {"typeId", type_ids.pointer_types[index]}}),
                        object({{"kind", "type"},
                                {"typeId", type_ids.layout_types[index]}})})},
                {"canonicalName", tensors[index]->canonical_type},
                {"kind", "template-specialization"},
                {"origin", tensor_origins[index]},
                {"qualifiers", qualifiers()},
                {"templateDeclarationId", tensor_template_declaration_id},
                {"typeId", type_ids.tensor_types[index]}}));
  }
  sort_records(types, "typeId");

  const std::string function_declaration_id =
      stable_id("declaration",
                object({{"canonicalName", observation.canonical_name},
                        {"canonicalUsr", observation.canonical_usr},
                        {"kind", "function"},
                        {"origin", function_origin},
                        {"typeId", type_ids.function_type}}),
                maximum);
  std::array<std::string, 2U> parameter_declaration_ids;
  for (std::size_t index = 0U; index < 2U; ++index) {
    const ProducerViewCopyParameterObservation& parameter =
        observation.parameters[index];
    const Json parameter_origin =
        index == 0U ? source_parameter_origin : destination_parameter_origin;
    parameter_declaration_ids[index] =
        stable_id("declaration",
                  object({{"canonicalName", parameter.canonical_name},
                          {"canonicalUsr", parameter.canonical_usr},
                          {"kind", "parameter"},
                          {"origin", parameter_origin},
                          {"semanticParentId", function_declaration_id},
                          {"typeId", type_ids.pointer_types[index]}}),
                  maximum);
  }

  std::array<std::string, 2U> tensor_declaration_ids;
  std::array<std::string, 2U> layout_declaration_ids;
  std::array<std::string, 2U> layout_canonical_usrs;
  std::array<std::string, 2U> layout_canonical_names;
  for (std::size_t index = 0U; index < tensors.size(); ++index) {
    tensor_declaration_ids[index] =
        stable_id("declaration",
                  object({{"canonicalName", tensors[index]->canonical_name},
                          {"canonicalUsr", tensors[index]->canonical_usr},
                          {"kind", "variable"},
                          {"origin", tensor_origins[index]},
                          {"semanticParentId", function_declaration_id},
                          {"typeId", type_ids.tensor_types[index]}}),
                  maximum);
    layout_canonical_usrs[index] =
        std::string("c:@BG@implicit-layout@") +
        hash_json(
            object({{"tensorCanonicalUsr", tensors[index]->canonical_usr}}),
            maximum);
    layout_canonical_names[index] = tensors[index]->canonical_name + ".layout";
    layout_declaration_ids[index] =
        stable_id("declaration",
                  object({{"canonicalName", layout_canonical_names[index]},
                          {"canonicalUsr", layout_canonical_usrs[index]},
                          {"kind", "variable"},
                          {"origin", layout_origins[index]},
                          {"semanticParentId", function_declaration_id},
                          {"typeId", type_ids.layout_types[index]}}),
                  maximum);
  }

  Json::Array declarations;
  declarations.reserve(9U);
  declarations.push_back(declaration_record(
      layout_template_declaration_id, "template", "c:@N@cute@ST@Layout",
      "cute::Layout", layout_template_origin, nullptr, nullptr, nullptr,
      type_ids.layout_types[0U], "external", "external", "none", "generic",
      cuda_attributes()));
  declarations.push_back(declaration_record(
      tensor_template_declaration_id, "template", "c:@N@cute@ST@Tensor",
      "cute::Tensor", tensor_template_origin, nullptr, nullptr, nullptr,
      type_ids.tensor_types[0U], "external", "external", "none", "generic",
      cuda_attributes()));
  declarations.push_back(declaration_record(
      function_declaration_id, "function", observation.canonical_usr,
      observation.canonical_name, function_origin,
      Json(input.function_spans.identity_span_id), nullptr, nullptr,
      type_ids.function_type, "definition", "external", "none", "generic",
      cuda_attributes(observation.cuda_host, observation.cuda_device,
                      observation.cuda_global, observation.force_inline)));
  for (std::size_t index = 0U; index < 2U; ++index) {
    const ProducerViewCopyParameterObservation& parameter =
        observation.parameters[index];
    const Json parameter_origin =
        index == 0U ? source_parameter_origin : destination_parameter_origin;
    declarations.push_back(declaration_record(
        parameter_declaration_ids[index], "parameter", parameter.canonical_usr,
        parameter.canonical_name, parameter_origin,
        Json(input.parameter_spans[index].identity_span_id),
        Json(function_declaration_id), Json(function_declaration_id),
        type_ids.pointer_types[index], "definition", "none", "automatic",
        "global", cuda_attributes()));
  }
  for (std::size_t index = 0U; index < tensors.size(); ++index) {
    declarations.push_back(declaration_record(
        tensor_declaration_ids[index], "variable",
        tensors[index]->canonical_usr, tensors[index]->canonical_name,
        tensor_origins[index],
        Json(input.tensor_spans[tensor_span_ordinals[index]].identity_span_id),
        Json(function_declaration_id), Json(function_declaration_id),
        type_ids.tensor_types[index], "definition", "none", "automatic",
        "generic", cuda_attributes()));
    declarations.push_back(declaration_record(
        layout_declaration_ids[index], "variable", layout_canonical_usrs[index],
        layout_canonical_names[index], layout_origins[index], nullptr,
        Json(function_declaration_id), Json(function_declaration_id),
        type_ids.layout_types[index], "definition", "none", "automatic",
        "generic", cuda_attributes()));
  }
  sort_records(declarations, "declarationId");

  std::array<std::string, 2U> layout_fact_ids;
  std::array<std::string, 2U> tensor_fact_ids;
  for (std::size_t index = 0U; index < tensors.size(); ++index) {
    layout_fact_ids[index] = stable_id(
        "fact",
        object({{"kind", "affine-layout"},
                {"origin", layout_origins[index]},
                {"resultDeclarationId", layout_declaration_ids[index]},
                {"shape", hierarchy_json(tensors[index]->shape)},
                {"stride", hierarchy_json(tensors[index]->stride)}}),
        maximum);
    tensor_fact_ids[index] = stable_id(
        "fact",
        object({{"elementTypeId", type_ids.scalar_type},
                {"engine", object({{"kind", "global-pointer"},
                                   {"nullable", false},
                                   {"pointerDeclarationId",
                                    parameter_declaration_ids[index]}})},
                {"kind", "tensor"},
                {"layoutFactId", layout_fact_ids[index]},
                {"origin", tensor_origins[index]},
                {"resultDeclarationId", tensor_declaration_ids[index]}}),
        maximum);
  }

  const std::string source_expression_id =
      stable_id("expression",
                object({{"declarationId", tensor_declaration_ids[0U]},
                        {"kind", "declaration-reference"},
                        {"origin", copy_origin},
                        {"typeId", type_ids.tensor_types[0U]},
                        {"valueCategory", "lvalue"}}),
                maximum);
  const std::string destination_expression_id =
      stable_id("expression",
                object({{"declarationId", tensor_declaration_ids[1U]},
                        {"kind", "declaration-reference"},
                        {"origin", copy_origin},
                        {"typeId", type_ids.tensor_types[1U]},
                        {"valueCategory", "lvalue"}}),
                maximum);
  const std::string intrinsic_fact_id = stable_id(
      "fact",
      object({{"availability", object({{"kind", "portable-candidate"}})},
              {"effects", object({{"convergent", false},
                                  {"readsMemory", true},
                                  {"synchronizes", false},
                                  {"writesMemory", true}})},
              {"familyId", "cute:copy@1"},
              {"kind", "target-intrinsic"},
              {"operandExpressionIds",
               array({source_expression_id, destination_expression_id})},
              {"operation", object({{"asynchronous", false},
                                    {"destinationSpace", "global"},
                                    {"kind", "copy"},
                                    {"sourceSpace", "global"},
                                    {"transferBits", 32U}})},
              {"origin", copy_origin},
              {"resultTypeId", type_ids.void_type}}),
      maximum);
  const std::string operation_expression_id =
      stable_id("expression",
                object({{"intrinsicFactId", intrinsic_fact_id},
                        {"kind", "target-intrinsic"},
                        {"origin", copy_origin},
                        {"typeId", type_ids.void_type},
                        {"valueCategory", "prvalue"}}),
                maximum);

  Json::Array facts;
  facts.reserve(5U);
  for (std::size_t index = 0U; index < tensors.size(); ++index) {
    facts.push_back(object(
        {{"cosize", object({{"kind", "integer"},
                            {"value", decimal_u64(static_cast<std::uint64_t>(
                                          tensors[index]->cosize))}})},
         {"factId", layout_fact_ids[index]},
         {"kind", "affine-layout"},
         {"leafRank", tensors[index]->leaf_rank},
         {"origin", layout_origins[index]},
         {"rank", tensors[index]->rank},
         {"resultDeclarationId", layout_declaration_ids[index]},
         {"shape", hierarchy_json(tensors[index]->shape)},
         {"size", object({{"kind", "integer"},
                          {"value", decimal_u64(static_cast<std::uint64_t>(
                                        tensors[index]->size))}})},
         {"stride", hierarchy_json(tensors[index]->stride)}}));
    facts.push_back(object(
        {{"elementTypeId", type_ids.scalar_type},
         {"engine",
          object({{"kind", "global-pointer"},
                  {"nullable", false},
                  {"pointerDeclarationId", parameter_declaration_ids[index]}})},
         {"factId", tensor_fact_ids[index]},
         {"kind", "tensor"},
         {"layoutFactId", layout_fact_ids[index]},
         {"memorySpace", "global"},
         {"origin", tensor_origins[index]},
         {"resultDeclarationId", tensor_declaration_ids[index]}}));
  }
  facts.push_back(
      object({{"availability", object({{"kind", "portable-candidate"}})},
              {"effects", object({{"convergent", false},
                                  {"readsMemory", true},
                                  {"synchronizes", false},
                                  {"writesMemory", true}})},
              {"factId", intrinsic_fact_id},
              {"familyId", "cute:copy@1"},
              {"kind", "target-intrinsic"},
              {"operandExpressionIds",
               array({source_expression_id, destination_expression_id})},
              {"operation", object({{"asynchronous", false},
                                    {"destinationSpace", "global"},
                                    {"kind", "copy"},
                                    {"sourceSpace", "global"},
                                    {"transferBits", 32U}})},
              {"origin", copy_origin},
              {"resultTypeId", type_ids.void_type}}));
  sort_records(facts, "factId");

  const std::string source_statement_id =
      stable_id("statement",
                object({{"declarationId", tensor_declaration_ids[0U]},
                        {"kind", "declaration"},
                        {"origin", source_tensor_origin}}),
                maximum);
  const std::string destination_statement_id =
      stable_id("statement",
                object({{"declarationId", tensor_declaration_ids[1U]},
                        {"kind", "declaration"},
                        {"origin", destination_tensor_origin}}),
                maximum);
  const std::string operation_statement_id =
      stable_id("statement",
                object({{"expressionId", operation_expression_id},
                        {"kind", "expression"},
                        {"origin", copy_origin}}),
                maximum);
  const std::string root_statement_id =
      stable_id("statement",
                object({{"kind", "block"},
                        {"origin", function_origin},
                        {"statementIds",
                         array({source_statement_id, destination_statement_id,
                                operation_statement_id})}}),
                maximum);
  const std::string body_id =
      stable_id("body",
                object({{"declarationId", function_declaration_id},
                        {"rootStatementId", root_statement_id}}),
                maximum);

  Json::Array statements = {
      object({{"kind", "block"},
              {"origin", function_origin},
              {"statementId", root_statement_id},
              {"statementIds",
               array({source_statement_id, destination_statement_id,
                      operation_statement_id})}}),
      object({{"declarationId", tensor_declaration_ids[0U]},
              {"kind", "declaration"},
              {"origin", source_tensor_origin},
              {"statementId", source_statement_id}}),
      object({{"declarationId", tensor_declaration_ids[1U]},
              {"kind", "declaration"},
              {"origin", destination_tensor_origin},
              {"statementId", destination_statement_id}}),
      object({{"expressionId", operation_expression_id},
              {"kind", "expression"},
              {"origin", copy_origin},
              {"statementId", operation_statement_id}}),
  };
  sort_records(statements, "statementId");
  Json::Array expressions = {
      object({{"declarationId", tensor_declaration_ids[0U]},
              {"expressionId", source_expression_id},
              {"kind", "declaration-reference"},
              {"origin", copy_origin},
              {"typeId", type_ids.tensor_types[0U]},
              {"valueCategory", "lvalue"}}),
      object({{"declarationId", tensor_declaration_ids[1U]},
              {"expressionId", destination_expression_id},
              {"kind", "declaration-reference"},
              {"origin", copy_origin},
              {"typeId", type_ids.tensor_types[1U]},
              {"valueCategory", "lvalue"}}),
      object({{"expressionId", operation_expression_id},
              {"intrinsicFactId", intrinsic_fact_id},
              {"kind", "target-intrinsic"},
              {"origin", copy_origin},
              {"typeId", type_ids.void_type},
              {"valueCategory", "prvalue"}}),
  };
  sort_records(expressions, "expressionId");
  Json::Array function_bodies = {
      object({{"bodyId", body_id},
              {"declarationId", function_declaration_id},
              {"expressions", Json(std::move(expressions))},
              {"rootStatementId", root_statement_id},
              {"statements", Json(std::move(statements))}}),
  };

  const std::string function_source_entity_id =
      std::string("bg.cpp.source-entity.sha256.") +
      hash_json(
          source_entity_id_projection(observation.canonical_usr, "function",
                                      input, observation.declaration_begin_byte,
                                      observation.declaration_end_byte),
          maximum);
  const std::string scalar_source_entity_id =
      std::string("bg.cpp.source-entity.sha256.") +
      hash_json(source_entity_id_projection(
                    scalar.canonical_name, "type", input,
                    destination_parameter.declaration_begin_byte,
                    destination_parameter.declaration_end_byte),
                maximum);
  Json::Array source_entities = {
      object({{"canonicalIdentity", observation.canonical_usr},
              {"domains", array({"device", "host"})},
              {"entityKind", "function"},
              {"origin", function_origin},
              {"sourceEntityId", function_source_entity_id}}),
      object({{"canonicalIdentity", scalar.canonical_name},
              {"domains", array({"device", "host"})},
              {"entityKind", "type"},
              {"origin", destination_parameter_origin},
              {"sourceEntityId", scalar_source_entity_id}}),
  };
  sort_records(source_entities, "sourceEntityId");

  Json::Array abi_types = {
      object({{"alignmentBits", "32"},
              {"bases", Json::Array{}},
              {"deviceTypeId", type_ids.scalar_type},
              {"domain", "device"},
              {"fields", Json::Array{}},
              {"shared", true},
              {"sizeBits", "32"},
              {"sourceTypeEntityId", scalar_source_entity_id}}),
      object({{"alignmentBits", "32"},
              {"bases", Json::Array{}},
              {"deviceTypeId", nullptr},
              {"domain", "host"},
              {"fields", Json::Array{}},
              {"shared", true},
              {"sizeBits", "32"},
              {"sourceTypeEntityId", scalar_source_entity_id}}),
  };
  const Json source_abi = object(
      {{"functions", Json::Array{}}, {"types", Json(std::move(abi_types))}});

  const std::string entry_id = stable_id(
      "entry",
      object({{"destinationTensorFactId", tensor_fact_ids[1U]},
              {"kind", "view-copy"},
              {"operationExpressionId", operation_expression_id},
              {"selectedRootDeclarationIds", array({function_declaration_id})},
              {"sourceTensorFactId", tensor_fact_ids[0U]}}),
      maximum);
  const Json entry =
      object({{"destinationTensorFactId", tensor_fact_ids[1U]},
              {"entryId", entry_id},
              {"kind", "view-copy"},
              {"operationExpressionId", operation_expression_id},
              {"selectedRootDeclarationIds", array({function_declaration_id})},
              {"sourceTensorFactId", tensor_fact_ids[0U]}});

  Json::Array shared_types = {
      object({{"bases", Json::Array{}},
              {"fields", Json::Array{}},
              {"sourceTypeEntityId", scalar_source_entity_id}}),
  };
  const std::string shared_surface_sha256 = hash_json(
      object(
          {{"domain", "browsergrad.compiler.cpp-cute.shared-source-surface.v2"},
           {"functions", Json::Array{}},
           {"selectedSourceRootEntityIds", array({function_source_entity_id})},
           {"types", Json(std::move(shared_types))}}),
      maximum);

  std::vector<std::string> fact_ids = {layout_fact_ids[0U], layout_fact_ids[1U],
                                       tensor_fact_ids[0U], tensor_fact_ids[1U],
                                       intrinsic_fact_id};
  std::sort(fact_ids.begin(), fact_ids.end());

  return {
      std::move(types),
      std::move(declarations),
      std::move(facts),
      std::move(function_bodies),
      std::move(source_entities),
      source_abi,
      entry,
      entry_id,
      std::move(fact_ids),
      function_source_entity_id,
      shared_surface_sha256,
  };
}

}  // namespace browsergrad::cpp_cute
