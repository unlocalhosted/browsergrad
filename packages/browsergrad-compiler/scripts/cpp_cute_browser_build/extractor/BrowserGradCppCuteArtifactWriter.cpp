#include "BrowserGradCppCuteArtifactWriter.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "BrowserGradCppCuteArtifactJson.h"
#include "BrowserGradCppCuteCanonicalJson.h"
#include "BrowserGradCppCuteViewCopyArtifact.h"

namespace browsergrad::cpp_cute {
namespace {

using artifact_json::array;
using artifact_json::ArtifactResourceLimit;
using artifact_json::canonical_json;
using artifact_json::hash_json;
using artifact_json::InvalidObservation;
using artifact_json::Json;
using artifact_json::object;
using artifact_json::stable_id;

constexpr std::size_t kMaximumArtifactStringByteLength = 16U * 1024U;
constexpr std::size_t kMaximumObservedFileCount = 4096U;
constexpr std::size_t kMaximumObservedIncludeEdgeCount = 16384U;
constexpr std::size_t kMaximumHierarchyNodeCount = 4096U;
constexpr std::uint32_t kMaximumHierarchyDepth = 64U;
constexpr std::uint32_t kMaximumJsonDepth = 128U;
constexpr std::uint32_t kMaximumJsonNodes = 1000000U;
constexpr std::uint32_t kMaximumJsonArrayLength = 65536U;
constexpr std::uint32_t kMaximumJsonObjectPropertyCount = 512U;

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

std::string decimal_u64(const std::uint64_t value) {
  std::array<char, 32U> output{};
  const auto converted =
      std::to_chars(output.data(), output.data() + output.size(), value);
  if (converted.ec != std::errc{}) throw InvalidObservation();
  return std::string(output.data(), converted.ptr);
}

bool parse_u64(const std::string_view value, std::uint64_t& output) noexcept {
  if (value.empty() || (value.size() > 1U && value.front() == '0'))
    return false;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), output);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}

bool path_is_below(const std::string_view path,
                   const std::string_view root) noexcept {
  if (root == "/") return path != "/" && !path.empty() && path.front() == '/';
  return path.size() > root.size() && path.starts_with(root) &&
         path[root.size()] == '/';
}

Json owner_json(const IncludeRootView& root) {
  if (root.owner_kind == "source") return object({{"kind", "source"}});
  if (root.owner_kind == "compiler-resource-directory") {
    return object({{"kind", "compiler-resource-directory"}});
  }
  if (root.owner_kind == "dependency" &&
      bounded_text(root.dependency_id, 256U)) {
    return object(
        {{"dependencyId", root.dependency_id}, {"kind", "dependency"}});
  }
  throw InvalidObservation();
}

struct RootRecord final {
  IncludeRootView view;
  Json json;
};

std::vector<RootRecord> build_roots(const DecodedCompileSession& session) {
  if (session.include_root_count() == 0U ||
      session.include_root_count() > 64U) {
    throw InvalidObservation();
  }
  std::vector<RootRecord> roots;
  roots.reserve(session.include_root_count());
  std::set<std::string, std::less<>> ids;
  for (std::size_t index = 0U; index < session.include_root_count(); ++index) {
    IncludeRootView root = session.include_root(index);
    if (root.ordinal != index ||
        (root.mode != "quote" && root.mode != "system") ||
        !bounded_text(root.include_root_id, 256U) ||
        !bounded_text(root.virtual_path, 4096U) ||
        !lowercase_sha256(root.manifest_sha256) ||
        !ids.emplace(root.include_root_id).second) {
      throw InvalidObservation();
    }
    Json owner = owner_json(root);
    roots.push_back({root, object({{"includeRootId", root.include_root_id},
                                   {"manifestSha256", root.manifest_sha256},
                                   {"mode", root.mode},
                                   {"ordinal", root.ordinal},
                                   {"owner", std::move(owner)},
                                   {"virtualPath", root.virtual_path}})});
  }
  return roots;
}

const RootRecord& root_by_id(const std::vector<RootRecord>& roots,
                             const std::string_view id) {
  const auto found = std::find_if(
      roots.begin(), roots.end(),
      [id](const RootRecord& root) { return root.view.include_root_id == id; });
  if (found == roots.end()) throw InvalidObservation();
  return *found;
}

const RootRecord& root_for_path(const std::vector<RootRecord>& roots,
                                const std::string_view path) {
  const RootRecord* selected = nullptr;
  for (const RootRecord& root : roots) {
    if (!path_is_below(path, root.view.virtual_path)) continue;
    if (selected == nullptr ||
        root.view.virtual_path.size() > selected->view.virtual_path.size()) {
      selected = &root;
    } else if (root.view.virtual_path.size() ==
                   selected->view.virtual_path.size() &&
               root.view.include_root_id != selected->view.include_root_id) {
      throw InvalidObservation();
    }
  }
  if (selected == nullptr) throw InvalidObservation();
  return *selected;
}

struct OpenedIdentity final {
  std::string content_sha256;
  std::uint64_t byte_length = 0U;
};

using OpenedIdentityMap = std::map<std::string, OpenedIdentity, std::less<>>;

OpenedIdentityMap merge_opened_files(const ProducerReviewResult& producer) {
  OpenedIdentityMap opened;
  for (const ProducerPassObservation& pass : producer.passes) {
    if (pass.opened_files.size() > kMaximumObservedFileCount ||
        pass.opened_file_paths.size() != pass.opened_files.size()) {
      throw ArtifactResourceLimit();
    }
    std::set<std::string, std::less<>> pass_paths;
    for (const ProducerOpenedFileObservation& file : pass.opened_files) {
      if (!bounded_text(file.virtual_path, 4096U) ||
          !lowercase_sha256(file.content_sha256) ||
          !pass_paths.emplace(file.virtual_path).second) {
        throw InvalidObservation();
      }
      const auto [iterator, inserted] =
          opened.emplace(file.virtual_path,
                         OpenedIdentity{file.content_sha256, file.byte_length});
      if (!inserted &&
          (iterator->second.content_sha256 != file.content_sha256 ||
           iterator->second.byte_length != file.byte_length)) {
        throw InvalidObservation();
      }
    }
    std::vector<std::string> declared = pass.opened_file_paths;
    std::sort(declared.begin(), declared.end());
    if (std::adjacent_find(declared.begin(), declared.end()) !=
            declared.end() ||
        !std::equal(declared.begin(), declared.end(), pass_paths.begin(),
                    pass_paths.end())) {
      throw InvalidObservation();
    }
  }
  if (opened.empty() || opened.size() > kMaximumObservedFileCount) {
    throw ArtifactResourceLimit();
  }
  return opened;
}

struct FileRecord final {
  std::string file_id;
  std::string virtual_path;
  std::string include_root_id;
  bool has_include_root = false;
  std::uint64_t byte_length = 0U;
  Json json;
};

using SourceViewMap = std::map<std::string, SourceFileView, std::less<>>;

SourceViewMap source_views(const DecodedCompileSession& session) {
  if (session.source_file_count() == 0U ||
      session.source_file_count() >
          session.request_semantic_limit(CompileSemanticLimit::kSourceFiles)) {
    throw InvalidObservation();
  }
  SourceViewMap result;
  for (std::size_t index = 0U; index < session.source_file_count(); ++index) {
    const SourceFileView source = session.source_file(index);
    if (!bounded_text(source.file_id, 128U) ||
        !bounded_text(source.virtual_path, 4096U) ||
        !bounded_text(source.role, 64U) ||
        !lowercase_sha256(source.content_sha256) ||
        !result.emplace(std::string(source.virtual_path), source).second) {
      throw InvalidObservation();
    }
  }
  return result;
}

std::vector<FileRecord> build_files(const DecodedCompileSession& session,
                                    const std::vector<RootRecord>& roots,
                                    const OpenedIdentityMap& opened) {
  const SourceViewMap sources = source_views(session);
  std::vector<FileRecord> files;
  files.reserve(opened.size());
  for (const auto& [path, identity] : opened) {
    const auto source = sources.find(path);
    if (source != sources.end()) {
      const SourceFileView view = source->second;
      std::uint64_t expected_length = 0U;
      if (!parse_u64(view.byte_length, expected_length) ||
          view.content_sha256 != identity.content_sha256 ||
          expected_length != identity.byte_length) {
        throw InvalidObservation();
      }
      if ((view.role == "main-source") !=
          (path == session.main_virtual_path())) {
        throw InvalidObservation();
      }
      std::string include_root_id;
      if (view.has_include_root) {
        const RootRecord& root = root_by_id(roots, view.include_root_id);
        if (root.view.owner_kind != "source" ||
            !path_is_below(path, root.view.virtual_path)) {
          throw InvalidObservation();
        }
        include_root_id = view.include_root_id;
      } else if (view.role != "main-source") {
        throw InvalidObservation();
      }
      Json value = object(
          {{"byteLength", view.byte_length},
           {"contentSha256", view.content_sha256},
           {"fileId", view.file_id},
           {"includeRootId",
            view.has_include_root ? Json(view.include_root_id) : Json(nullptr)},
           {"owner", object({{"kind", "source"}})},
           {"role", view.role},
           {"virtualPath", view.virtual_path}});
      files.push_back({std::string(view.file_id), path,
                       std::move(include_root_id), view.has_include_root,
                       expected_length, std::move(value)});
      continue;
    }

    const RootRecord& root = root_for_path(roots, path);
    if (root.view.owner_kind == "source") throw InvalidObservation();
    const std::string role =
        root.view.owner_kind == "compiler-resource-directory"
            ? "compiler-header"
            : "dependency-header";
    Json owner = owner_json(root.view);
    Json identity_value =
        object({{"byteLength", decimal_u64(identity.byte_length)},
                {"contentSha256", identity.content_sha256},
                {"includeRootId", root.view.include_root_id},
                {"owner", owner},
                {"role", role},
                {"virtualPath", path}});
    const std::string file_id = stable_id(
        "file", identity_value, ArtifactV3ResultSink::kAbiMaximumByteLength);
    Json value = object({{"byteLength", decimal_u64(identity.byte_length)},
                         {"contentSha256", identity.content_sha256},
                         {"fileId", file_id},
                         {"includeRootId", root.view.include_root_id},
                         {"owner", std::move(owner)},
                         {"role", role},
                         {"virtualPath", path}});
    files.push_back({file_id, path, std::string(root.view.include_root_id),
                     true, identity.byte_length, std::move(value)});
  }
  std::sort(files.begin(), files.end(),
            [](const FileRecord& left, const FileRecord& right) {
              return left.file_id < right.file_id;
            });
  if (std::adjacent_find(files.begin(), files.end(),
                         [](const FileRecord& left, const FileRecord& right) {
                           return left.file_id == right.file_id ||
                                  left.virtual_path == right.virtual_path;
                         }) != files.end()) {
    throw InvalidObservation();
  }
  return files;
}

const FileRecord& file_by_path(const std::vector<FileRecord>& files,
                               const std::string_view path) {
  const auto found = std::find_if(
      files.begin(), files.end(),
      [path](const FileRecord& file) { return file.virtual_path == path; });
  if (found == files.end()) throw InvalidObservation();
  return *found;
}

Json file_range(const std::string_view file_id, const std::uint64_t begin,
                const std::uint64_t end) {
  if (begin > end) throw InvalidObservation();
  return object({{"endByte", decimal_u64(end)},
                 {"fileId", file_id},
                 {"startByte", decimal_u64(begin)}});
}

struct SpanRecord final {
  std::string span_id;
  Json json;
};

SpanRecord span_record(const std::string_view file_id,
                       const std::uint64_t begin, const std::uint64_t end) {
  Json range = file_range(file_id, begin, end);
  const std::string span_id =
      stable_id("span", object({{"expansion", range}, {"spelling", range}}),
                ArtifactV3ResultSink::kAbiMaximumByteLength);
  return {span_id, object({{"expansion", range},
                           {"macroExpansionId", nullptr},
                           {"spanId", span_id},
                           {"spelling", std::move(range)}})};
}

struct EdgeRecord final {
  std::string edge_id;
  std::string logical_key;
  Json json;
  SpanRecord directive_span;
  bool has_directive_span = false;
};

struct PassEdgeIds final {
  std::array<std::vector<std::string>, 2U> ids;
};

EdgeRecord edge_record(const ProducerIncludeEdgeObservation& edge,
                       const std::vector<FileRecord>& files) {
  if (edge.kind == ProducerIncludeKind::kCompilerForced) {
    const FileRecord& file = file_by_path(files, edge.resolved_file_path);
    if (!file.has_include_root || !edge.including_file_path.empty() ||
        !edge.spelling.empty() || edge.directive_start_byte_offset != 0U ||
        edge.directive_end_byte_offset != 0U) {
      throw InvalidObservation();
    }
    Json identity =
        object({{"compilerOptionOrdinal", edge.compiler_option_ordinal},
                {"fileId", file.file_id},
                {"includeRootId", file.include_root_id},
                {"kind", "compiler-forced"}});
    const std::string edge_id = stable_id(
        "include-edge", identity, ArtifactV3ResultSink::kAbiMaximumByteLength);
    return {edge_id,
            std::string("forced:") + decimal_u64(edge.compiler_option_ordinal),
            object({{"compilerOptionOrdinal", edge.compiler_option_ordinal},
                    {"fileId", file.file_id},
                    {"includeEdgeId", edge_id},
                    {"includeRootId", file.include_root_id},
                    {"kind", "compiler-forced"}}),
            {},
            false};
  }
  const FileRecord& including = file_by_path(files, edge.including_file_path);
  const FileRecord& resolved = file_by_path(files, edge.resolved_file_path);
  if (!resolved.has_include_root || !bounded_text(edge.spelling, 4096U) ||
      edge.directive_start_byte_offset >= edge.directive_end_byte_offset ||
      edge.directive_end_byte_offset > including.byte_length) {
    throw InvalidObservation();
  }
  SpanRecord span =
      span_record(including.file_id, edge.directive_start_byte_offset,
                  edge.directive_end_byte_offset);
  const char* mode =
      edge.kind == ProducerIncludeKind::kSourceQuote ? "quote" : "angle";
  Json identity = object(
      {{"directiveSpanId", span.span_id},
       {"includingFileId", including.file_id},
       {"kind", "source-directive"},
       {"mode", mode},
       {"resolution", object({{"fileId", resolved.file_id},
                              {"includeRootId", resolved.include_root_id},
                              {"kind", "resolved"}})},
       {"spelling", edge.spelling}});
  const std::string edge_id = stable_id(
      "include-edge", identity, ArtifactV3ResultSink::kAbiMaximumByteLength);
  return {edge_id,
          std::string("source:") + edge.including_file_path + ":" +
              decimal_u64(edge.directive_start_byte_offset) + ":" +
              decimal_u64(edge.directive_end_byte_offset),
          object({{"directiveSpanId", span.span_id},
                  {"includeEdgeId", edge_id},
                  {"includingFileId", including.file_id},
                  {"kind", "source-directive"},
                  {"mode", mode},
                  {"resolution",
                   object({{"fileId", resolved.file_id},
                           {"includeRootId", resolved.include_root_id},
                           {"kind", "resolved"}})},
                  {"spelling", edge.spelling}}),
          std::move(span), true};
}

std::pair<std::vector<EdgeRecord>, PassEdgeIds> build_edges(
    const ProducerReviewResult& producer, const DecodedCompileSession& session,
    const std::vector<FileRecord>& files) {
  std::map<std::string, EdgeRecord, std::less<>> by_id;
  std::map<std::string, std::string, std::less<>> logical_ids;
  PassEdgeIds pass_ids;
  for (std::size_t pass_index = 0U; pass_index < producer.completed_pass_count;
       ++pass_index) {
    const ProducerPassObservation& pass = producer.passes[pass_index];
    if (pass.include_edges.size() > kMaximumObservedIncludeEdgeCount) {
      throw ArtifactResourceLimit();
    }
    std::set<std::string, std::less<>> unique;
    for (const ProducerIncludeEdgeObservation& observation :
         pass.include_edges) {
      EdgeRecord record = edge_record(observation, files);
      const auto logical =
          logical_ids.emplace(record.logical_key, record.edge_id);
      if (!logical.second && logical.first->second != record.edge_id) {
        throw InvalidObservation();
      }
      unique.emplace(record.edge_id);
      by_id.emplace(record.edge_id, std::move(record));
    }
    pass_ids.ids[pass_index].assign(unique.begin(), unique.end());
  }

  std::set<std::uint32_t> forced_ordinals;
  for (std::size_t index = 0U; index < session.compiler_option_count();
       ++index) {
    const CompilerOptionView option = session.compiler_option(index);
    if (option.kind != CompilerOptionKind::kForcedInclude) continue;
    forced_ordinals.emplace(option.ordinal);
    const FileRecord& file = file_by_path(files, option.virtual_path);
    if (!file.has_include_root ||
        file.include_root_id != option.include_root_id) {
      throw InvalidObservation();
    }
  }
  for (std::size_t pass_index = 0U; pass_index < producer.completed_pass_count;
       ++pass_index) {
    std::set<std::uint32_t> observed;
    for (const ProducerIncludeEdgeObservation& edge :
         producer.passes[pass_index].include_edges) {
      if (edge.kind == ProducerIncludeKind::kCompilerForced) {
        observed.emplace(edge.compiler_option_ordinal);
      }
    }
    if (observed != forced_ordinals) throw InvalidObservation();
  }

  std::vector<EdgeRecord> records;
  records.reserve(by_id.size());
  for (auto& [id, record] : by_id) {
    static_cast<void>(id);
    records.push_back(std::move(record));
  }
  if (records.size() > kMaximumObservedIncludeEdgeCount) {
    throw ArtifactResourceLimit();
  }
  return {std::move(records), std::move(pass_ids)};
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

bool same_layout(const ProducerLayoutObservation& left,
                 const ProducerLayoutObservation& right) noexcept {
  return left.selected == right.selected &&
         left.resolved_layout_type == right.resolved_layout_type &&
         left.resolved_static_affine_layout ==
             right.resolved_static_affine_layout &&
         left.canonical_usr == right.canonical_usr &&
         left.canonical_name == right.canonical_name &&
         left.canonical_type == right.canonical_type &&
         left.identity_begin_byte == right.identity_begin_byte &&
         left.identity_end_byte == right.identity_end_byte &&
         left.rank == right.rank && left.leaf_rank == right.leaf_rank &&
         left.size == right.size && left.cosize == right.cosize &&
         same_hierarchy(left.shape, right.shape) &&
         same_hierarchy(left.stride, right.stride);
}

bool empty_hierarchy(const ProducerIntegerHierarchy& hierarchy) noexcept {
  return !hierarchy.tuple && hierarchy.value == 0 && hierarchy.elements.empty();
}

bool empty_layout(const ProducerLayoutObservation& layout) noexcept {
  return !layout.selected && !layout.resolved_layout_type &&
         !layout.resolved_static_affine_layout &&
         layout.canonical_usr.empty() && layout.canonical_name.empty() &&
         layout.canonical_type.empty() && layout.initializer_callee.empty() &&
         layout.identity_begin_byte == 0U && layout.identity_end_byte == 0U &&
         layout.rank == 0U && layout.leaf_rank == 0U && layout.size == 0 &&
         layout.cosize == 0 && empty_hierarchy(layout.shape) &&
         empty_hierarchy(layout.stride);
}

bool empty_view_copy(const ProducerViewCopyObservation& observation) noexcept {
  return !observation.selected && !observation.ambiguous &&
         !observation.resolved_function && !observation.resolved_copy &&
         !observation.cuda_host && !observation.cuda_device &&
         !observation.cuda_global && !observation.force_inline &&
         observation.canonical_usr.empty() &&
         observation.canonical_name.empty() &&
         observation.canonical_type.empty() &&
         observation.copy_callee_usr.empty() &&
         observation.copy_callee_name.empty() &&
         observation.copy_callee_path.empty() &&
         observation.declaration_begin_byte == 0U &&
         observation.declaration_end_byte == 0U &&
         observation.identity_begin_byte == 0U &&
         observation.identity_end_byte == 0U &&
         observation.copy_begin_byte == 0U && observation.copy_end_byte == 0U &&
         observation.source_tensor_ordinal == 0U &&
         observation.destination_tensor_ordinal == 0U &&
         observation.parameters.empty() && observation.tensors.empty();
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

void validate_layout(const ProducerLayoutObservation& layout,
                     const EntryRequestView& entry) {
  std::uint64_t request_begin = 0U;
  std::uint64_t request_end = 0U;
  if (!layout.selected || !layout.resolved_layout_type ||
      !layout.resolved_static_affine_layout ||
      !bounded_text(layout.canonical_usr, kMaximumArtifactStringByteLength) ||
      !std::string_view(layout.canonical_usr).starts_with("c:@") ||
      !bounded_text(layout.canonical_name, kMaximumArtifactStringByteLength) ||
      !bounded_text(layout.canonical_type, kMaximumArtifactStringByteLength) ||
      !parse_u64(entry.begin_byte, request_begin) ||
      !parse_u64(entry.end_byte, request_end) ||
      layout.identity_begin_byte != request_begin ||
      layout.identity_end_byte != request_end || request_begin >= request_end) {
    throw InvalidObservation();
  }
  HierarchySummary summary;
  validate_hierarchy_pair(layout.shape, layout.stride, summary, 1U);
  const std::uint32_t top_rank =
      layout.shape.tuple
          ? static_cast<std::uint32_t>(layout.shape.elements.size())
          : 1U;
  if (layout.rank != top_rank ||
      layout.leaf_rank != summary.shape_leaves.size()) {
    throw InvalidObservation();
  }
  std::int64_t computed_size = 1;
  std::int64_t computed_cosize = 1;
  for (std::size_t index = 0U; index < summary.shape_leaves.size(); ++index) {
    const std::int64_t shape = summary.shape_leaves[index];
    const std::int64_t stride = summary.stride_leaves[index];
    if (computed_size > std::numeric_limits<std::int64_t>::max() / shape) {
      throw InvalidObservation();
    }
    computed_size *= shape;
    const std::int64_t extent = shape - 1;
    if (stride != 0 &&
        extent > std::numeric_limits<std::int64_t>::max() / stride) {
      throw InvalidObservation();
    }
    const std::int64_t contribution = extent * stride;
    if (computed_cosize >
        std::numeric_limits<std::int64_t>::max() - contribution) {
      throw InvalidObservation();
    }
    computed_cosize += contribution;
  }
  if (layout.size != computed_size || layout.cosize != computed_cosize) {
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

Json qualifiers() {
  return object({{"const", false}, {"restrict", false}, {"volatile", false}});
}

Json cuda_attributes() {
  return object({{"device", false},
                 {"forceInline", false},
                 {"global", false},
                 {"host", false}});
}

Json source_origin(const std::string_view span_id) {
  return object({{"kind", "source"}, {"spanId", span_id}});
}

Json template_substitution_origin(const std::string_view span_id) {
  return object({{"anchorSpanId", span_id},
                 {"kind", "implicit"},
                 {"reason", "template-substitution"}});
}

Json::Array json_array_of_roots(const std::vector<RootRecord>& roots) {
  Json::Array values;
  values.reserve(roots.size());
  for (const RootRecord& root : roots) values.push_back(root.json);
  return values;
}

Json::Array json_array_of_files(const std::vector<FileRecord>& files) {
  Json::Array values;
  values.reserve(files.size());
  for (const FileRecord& file : files) values.push_back(file.json);
  return values;
}

Json::Array json_array_of_edges(const std::vector<EdgeRecord>& edges) {
  Json::Array values;
  values.reserve(edges.size());
  for (const EdgeRecord& edge : edges) values.push_back(edge.json);
  return values;
}

Json file_set_projection(const std::string_view kind,
                         const std::vector<FileRecord>& files) {
  Json::Array selected;
  for (const FileRecord& file : files) {
    const Json::Object& record = std::get<Json::Object>(file.json.value);
    const std::string& role = std::get<std::string>(record.at("role").value);
    if ((kind == "source") != (role == "main-source")) continue;
    selected.push_back(object({{"byteLength", record.at("byteLength")},
                               {"contentSha256", record.at("contentSha256")},
                               {"includeRootId", record.at("includeRootId")},
                               {"owner", record.at("owner")},
                               {"role", record.at("role")},
                               {"virtualPath", record.at("virtualPath")}}));
  }
  return object({{"domain", std::string("browsergrad.compiler.cpp-cute.") +
                                std::string(kind) + "-set.v2"},
                 {"files", Json(std::move(selected))}});
}

std::vector<std::string> pass_opened_ids(const ProducerPassObservation& pass,
                                         const std::vector<FileRecord>& files) {
  std::vector<std::string> ids;
  ids.reserve(pass.opened_files.size());
  for (const ProducerOpenedFileObservation& opened : pass.opened_files) {
    ids.push_back(file_by_path(files, opened.virtual_path).file_id);
  }
  std::sort(ids.begin(), ids.end());
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
    throw InvalidObservation();
  }
  return ids;
}

Json strings_json(const std::vector<std::string>& values) {
  Json::Array result;
  result.reserve(values.size());
  for (const std::string& value : values) result.emplace_back(value);
  return Json(std::move(result));
}

Json semantic_pass_input_projection(const SemanticPassView& pass,
                                    const std::vector<RootRecord>& roots,
                                    const std::vector<FileRecord>& files,
                                    const std::vector<EdgeRecord>& edges,
                                    const std::vector<std::string>& opened_ids,
                                    const std::vector<std::string>& edge_ids) {
  const std::set<std::string, std::less<>> opened(opened_ids.begin(),
                                                  opened_ids.end());
  const std::set<std::string, std::less<>> included_edges(edge_ids.begin(),
                                                          edge_ids.end());
  Json::Array selected_files;
  for (const FileRecord& file : files) {
    if (opened.contains(file.file_id)) selected_files.push_back(file.json);
  }
  Json::Array selected_edges;
  for (const EdgeRecord& edge : edges) {
    if (included_edges.contains(edge.edge_id))
      selected_edges.push_back(edge.json);
  }
  return object(
      {{"domain",
        "browsergrad.compiler.cpp-cute.semantic-pass-input-closure.v1"},
       {"files", Json(std::move(selected_files))},
       {"includeEdges", Json(std::move(selected_edges))},
       {"includeRoots", Json(json_array_of_roots(roots))},
       {"passId", pass.pass_id}});
}

Json shared_surface_projection(const std::string_view source_entity_id) {
  return object(
      {{"domain", "browsergrad.compiler.cpp-cute.shared-source-surface.v2"},
       {"functions", Json::Array{}},
       {"selectedSourceRootEntityIds", array({Json(source_entity_id)})},
       {"types", Json::Array{}}});
}

Json source_entity_id_projection(const ProducerLayoutObservation& layout,
                                 const FileRecord& main_file,
                                 const std::uint64_t begin,
                                 const std::uint64_t end) {
  Json canonical_range = object(
      {{"contentSha256",
        std::get<std::string>(std::get<Json::Object>(main_file.json.value)
                                  .at("contentSha256")
                                  .value)},
       {"endByte", decimal_u64(end)},
       {"startByte", decimal_u64(begin)},
       {"virtualPath", main_file.virtual_path}});
  return object(
      {{"canonicalIdentity", layout.canonical_usr},
       {"domain", "browsergrad.compiler.cpp-cute.source-entity-id.v1"},
       {"entityKind", "variable"},
       {"origin",
        object({{"kind", "source"},
                {"span", object({{"expansion", canonical_range},
                                 {"spelling", canonical_range}})}})}});
}

struct AcceptedInputClosure final {
  std::vector<RootRecord> roots;
  std::vector<FileRecord> files;
  std::vector<EdgeRecord> edges;
  PassEdgeIds pass_edge_ids;
  std::string closure_sha256;
  Json inputs;
};

AcceptedInputClosure build_accepted_input_closure(
    const ProducerReviewResult& producer, const DecodedCompileSession& session,
    const std::size_t maximum_byte_length) {
  std::vector<RootRecord> roots = build_roots(session);
  const OpenedIdentityMap opened = merge_opened_files(producer);
  std::vector<FileRecord> files = build_files(session, roots, opened);
  const FileRecord& main_file =
      file_by_path(files, session.main_virtual_path());
  const Json::Object& main_json = std::get<Json::Object>(main_file.json.value);
  if (std::get<std::string>(main_json.at("role").value) != "main-source") {
    throw InvalidObservation();
  }
  for (const ProducerPassObservation& pass : producer.passes) {
    if (std::none_of(pass.opened_files.begin(), pass.opened_files.end(),
                     [&session](const ProducerOpenedFileObservation& file) {
                       return file.virtual_path == session.main_virtual_path();
                     })) {
      throw InvalidObservation();
    }
  }
  auto [edges, pass_edge_ids] = build_edges(producer, session, files);
  const std::string source_set_sha256 =
      hash_json(file_set_projection("source", files), maximum_byte_length);
  const std::string header_set_sha256 =
      hash_json(file_set_projection("header", files), maximum_byte_length);
  const Json projection =
      object({{"domain", "browsergrad.compiler.cpp-cute.input-closure.v2"},
              {"files", Json(json_array_of_files(files))},
              {"headerSetSha256", header_set_sha256},
              {"includeEdges", Json(json_array_of_edges(edges))},
              {"includeRoots", Json(json_array_of_roots(roots))},
              {"mainFileId", main_file.file_id},
              {"sourceSetSha256", source_set_sha256}});
  const std::string closure_sha256 = hash_json(projection, maximum_byte_length);
  Json inputs = object({{"closureSha256", closure_sha256},
                        {"files", Json(json_array_of_files(files))},
                        {"headerSetSha256", header_set_sha256},
                        {"includeEdges", Json(json_array_of_edges(edges))},
                        {"includeRoots", Json(json_array_of_roots(roots))},
                        {"mainFileId", main_file.file_id},
                        {"sourceSetSha256", source_set_sha256}});
  return {std::move(roots),         std::move(files), std::move(edges),
          std::move(pass_edge_ids), closure_sha256,   std::move(inputs)};
}

Json build_view_copy_accepted_payload(const ProducerReviewResult& producer,
                                      const DecodedCompileSession& session,
                                      const std::size_t maximum_byte_length) {
  const EntryRequestView entry = session.entry_request();
  if (entry.kind != "view-copy" || entry.declaration_kind != "function" ||
      entry.virtual_path != session.main_virtual_path() ||
      !empty_layout(producer.passes[0].layout) ||
      !empty_layout(producer.passes[1].layout) ||
      !equivalent_view_copy_observation(producer.passes[0].view_copy,
                                        producer.passes[1].view_copy)) {
    throw InvalidObservation();
  }
  const ProducerViewCopyObservation& view = producer.passes[0].view_copy;
  std::uint64_t request_begin = 0U;
  std::uint64_t request_end = 0U;
  if (!parse_u64(entry.begin_byte, request_begin) ||
      !parse_u64(entry.end_byte, request_end) ||
      request_begin != view.identity_begin_byte ||
      request_end != view.identity_end_byte) {
    throw InvalidObservation();
  }

  const AcceptedInputClosure closure =
      build_accepted_input_closure(producer, session, maximum_byte_length);
  const std::vector<RootRecord>& roots = closure.roots;
  const std::vector<FileRecord>& files = closure.files;
  const std::vector<EdgeRecord>& edges = closure.edges;
  const PassEdgeIds& pass_edge_ids = closure.pass_edge_ids;
  const FileRecord& main_file =
      file_by_path(files, session.main_virtual_path());
  const Json::Object& main_json = std::get<Json::Object>(main_file.json.value);
  static_cast<void>(file_by_path(files, view.copy_callee_path));
  for (const ProducerViewCopyTensorObservation& tensor : view.tensors) {
    static_cast<void>(file_by_path(files, tensor.tensor_template_path));
    static_cast<void>(file_by_path(files, tensor.layout_template_path));
    static_cast<void>(file_by_path(files, tensor.initializer_callee_path));
  }
  std::map<std::string, SpanRecord, std::less<>> spans;
  const auto add_span = [&spans, &main_file](const std::uint64_t begin,
                                             const std::uint64_t end) {
    if (begin >= end || end > main_file.byte_length) {
      throw InvalidObservation();
    }
    SpanRecord record = span_record(main_file.file_id, begin, end);
    const std::string id = record.span_id;
    spans.emplace(id, std::move(record));
    return id;
  };
  const std::array<std::string, 2U> function_span_storage = {
      add_span(view.declaration_begin_byte, view.declaration_end_byte),
      add_span(view.identity_begin_byte, view.identity_end_byte)};
  const ViewCopyDeclarationSpans function_spans = {function_span_storage[0U],
                                                   function_span_storage[1U]};
  std::array<ViewCopyDeclarationSpans, 2U> parameter_spans;
  std::array<ViewCopyDeclarationSpans, 2U> tensor_spans;
  std::array<std::array<std::string, 2U>, 2U> parameter_span_storage;
  std::array<std::array<std::string, 2U>, 2U> tensor_span_storage;
  for (std::size_t index = 0U; index < 2U; ++index) {
    const ProducerViewCopyParameterObservation& parameter =
        view.parameters.at(index);
    parameter_span_storage[index] = {
        add_span(parameter.declaration_begin_byte,
                 parameter.declaration_end_byte),
        add_span(parameter.identity_begin_byte, parameter.identity_end_byte)};
    parameter_spans[index] = {parameter_span_storage[index][0U],
                              parameter_span_storage[index][1U]};
    const ProducerViewCopyTensorObservation& tensor = view.tensors.at(index);
    tensor_span_storage[index] = {
        add_span(tensor.declaration_begin_byte, tensor.declaration_end_byte),
        add_span(tensor.identity_begin_byte, tensor.identity_end_byte)};
    tensor_spans[index] = {tensor_span_storage[index][0U],
                           tensor_span_storage[index][1U]};
  }
  const std::string copy_span_id =
      add_span(view.copy_begin_byte, view.copy_end_byte);
  for (const EdgeRecord& edge : edges) {
    if (edge.has_directive_span) {
      spans.emplace(edge.directive_span.span_id, edge.directive_span);
    }
  }

  const ViewCopyArtifactGraph graph = build_view_copy_artifact_graph(
      view, {main_file.virtual_path,
             std::get<std::string>(main_json.at("contentSha256").value),
             main_file.byte_length, maximum_byte_length, function_spans,
             parameter_spans, tensor_spans, copy_span_id});
  Json::Array span_values;
  span_values.reserve(spans.size());
  for (const auto& [id, span] : spans) {
    static_cast<void>(id);
    span_values.push_back(span.json);
  }

  Json::Array semantic_passes;
  semantic_passes.reserve(2U);
  for (std::size_t index = 0U; index < 2U; ++index) {
    const SemanticPassView pass = session.semantic_pass(index);
    const std::vector<std::string> opened_ids =
        pass_opened_ids(producer.passes[index], files);
    const std::vector<std::string>& edge_ids = pass_edge_ids.ids[index];
    const std::string observed_hash =
        hash_json(semantic_pass_input_projection(pass, roots, files, edges,
                                                 opened_ids, edge_ids),
                  maximum_byte_length);
    semantic_passes.push_back(
        object({{"auxiliaryTargetTriple", pass.auxiliary_target_triple},
                {"deviceArchitecture", pass.device_architecture},
                {"diagnosticIds", Json::Array{}},
                {"domain", pass.domain},
                {"factIds", index == 0U ? strings_json(graph.fact_ids)
                                        : Json(Json::Array{})},
                {"includeEdgeIds", strings_json(edge_ids)},
                {"invocationMode", pass.invocation_mode},
                {"observedInputClosureSha256", observed_hash},
                {"openedFileIds", strings_json(opened_ids)},
                {"ordinal", pass.ordinal},
                {"passId", pass.pass_id},
                {"role", pass.role},
                {"selectedSourceRootEntityIds",
                 array({graph.selected_source_entity_id})},
                {"sharedSurfaceSha256", graph.shared_surface_sha256},
                {"status", "succeeded"},
                {"targetTriple", pass.target_triple}}));
  }

  return object(
      {{"compilationContractHash", session.compilation_contract_hash()},
       {"constants", Json::Array{}},
       {"declarations", Json(graph.declarations)},
       {"diagnostics", Json::Array{}},
       {"entries", array({graph.entry})},
       {"extraction", object({{"appliedTransforms", Json::Array{}},
                              {"compilationContractHash",
                               session.compilation_contract_hash()},
                              {"inputClosureSha256", closure.closure_sha256}})},
       {"facts", Json(graph.facts)},
       {"functionBodies", Json(graph.function_bodies)},
       {"initializerExpressions", Json::Array{}},
       {"inputs", closure.inputs},
       {"macroExpansions", Json::Array{}},
       {"outcome", object({{"kind", "accepted"},
                           {"selectedEntryIds", array({graph.entry_id})}})},
       {"overloadResolutions", Json::Array{}},
       {"semanticGraphOwnerPassId", "cuda-device-sema"},
       {"semanticPasses", Json(std::move(semantic_passes))},
       {"sourceAbi", graph.source_abi},
       {"sourceEntities", Json(graph.source_entities)},
       {"spans", Json(std::move(span_values))},
       {"templateInstantiations", Json::Array{}},
       {"types", Json(graph.types)}});
}

Json build_accepted_payload(const ProducerReviewResult& producer,
                            const DecodedCompileSession& session,
                            const std::size_t maximum_byte_length) {
  const EntryRequestView entry = session.entry_request();
  if (entry.kind == "view-copy") {
    return build_view_copy_accepted_payload(producer, session,
                                            maximum_byte_length);
  }
  if (entry.kind != "layout" || entry.declaration_kind != "variable" ||
      entry.virtual_path != session.main_virtual_path() ||
      !empty_view_copy(producer.passes[0].view_copy) ||
      !empty_view_copy(producer.passes[1].view_copy) ||
      !same_layout(producer.passes[0].layout, producer.passes[1].layout)) {
    throw InvalidObservation();
  }
  const ProducerLayoutObservation& layout = producer.passes[0].layout;
  validate_layout(layout, entry);

  const AcceptedInputClosure closure =
      build_accepted_input_closure(producer, session, maximum_byte_length);
  const std::vector<RootRecord>& roots = closure.roots;
  const std::vector<FileRecord>& files = closure.files;
  const std::vector<EdgeRecord>& edges = closure.edges;
  const PassEdgeIds& pass_edge_ids = closure.pass_edge_ids;
  const FileRecord& main_file =
      file_by_path(files, session.main_virtual_path());
  std::uint64_t identity_begin = 0U;
  std::uint64_t identity_end = 0U;
  if (!parse_u64(entry.begin_byte, identity_begin) ||
      !parse_u64(entry.end_byte, identity_end)) {
    throw InvalidObservation();
  }
  SpanRecord identity_span =
      span_record(main_file.file_id, identity_begin, identity_end);
  if (identity_end > main_file.byte_length) throw InvalidObservation();

  std::map<std::string, SpanRecord, std::less<>> spans;
  spans.emplace(identity_span.span_id, identity_span);
  for (const EdgeRecord& edge : edges) {
    if (edge.has_directive_span) {
      spans.emplace(edge.directive_span.span_id, edge.directive_span);
    }
  }
  Json::Array span_values;
  span_values.reserve(spans.size());
  for (const auto& [id, span] : spans) {
    static_cast<void>(id);
    span_values.push_back(span.json);
  }

  const std::string source_entity_id =
      std::string("bg.cpp.source-entity.sha256.") +
      hash_json(source_entity_id_projection(layout, main_file, identity_begin,
                                            identity_end),
                maximum_byte_length);
  const Json variable_origin = source_origin(identity_span.span_id);
  const Json template_origin =
      template_substitution_origin(identity_span.span_id);
  const std::string template_declaration_id =
      stable_id("declaration",
                object({{"canonicalName", "cute::Layout"},
                        {"canonicalUsr", "c:@N@cute@ST@Layout"},
                        {"kind", "template"},
                        {"origin", template_origin}}),
                maximum_byte_length);
  const std::string layout_type_id =
      stable_id("type",
                object({{"canonicalName", layout.canonical_type},
                        {"kind", "template-specialization"},
                        {"origin", template_origin},
                        {"templateDeclarationId", template_declaration_id}}),
                maximum_byte_length);
  const std::string variable_declaration_id =
      stable_id("declaration",
                object({{"canonicalName", layout.canonical_name},
                        {"canonicalUsr", layout.canonical_usr},
                        {"kind", "variable"},
                        {"origin", variable_origin},
                        {"typeId", layout_type_id}}),
                maximum_byte_length);
  const std::string fact_id =
      stable_id("fact",
                object({{"kind", "affine-layout"},
                        {"origin", variable_origin},
                        {"resultDeclarationId", variable_declaration_id},
                        {"shape", hierarchy_json(layout.shape)},
                        {"stride", hierarchy_json(layout.stride)}}),
                maximum_byte_length);
  const std::string entry_id = stable_id(
      "entry",
      object(
          {{"kind", "layout"},
           {"layoutFactId", fact_id},
           {"selectedRootDeclarationIds", array({variable_declaration_id})}}),
      maximum_byte_length);

  const Json type = object({{"arguments", Json::Array{}},
                            {"canonicalName", layout.canonical_type},
                            {"kind", "template-specialization"},
                            {"origin", template_origin},
                            {"qualifiers", qualifiers()},
                            {"templateDeclarationId", template_declaration_id},
                            {"typeId", layout_type_id}});
  const Json template_declaration =
      object({{"canonicalName", "cute::Layout"},
              {"canonicalUsr", "c:@N@cute@ST@Layout"},
              {"cudaAttributes", cuda_attributes()},
              {"declarationId", template_declaration_id},
              {"definitionKind", "external"},
              {"identitySpanId", nullptr},
              {"initializerExpressionId", nullptr},
              {"kind", "template"},
              {"lexicalParentId", nullptr},
              {"linkage", "external"},
              {"mangledName", nullptr},
              {"memorySpace", "generic"},
              {"origin", template_origin},
              {"semanticParentId", nullptr},
              {"storageDuration", "none"},
              {"targetTypeId", nullptr},
              {"typeId", layout_type_id}});
  const Json variable_declaration =
      object({{"canonicalName", layout.canonical_name},
              {"canonicalUsr", layout.canonical_usr},
              {"cudaAttributes", cuda_attributes()},
              {"declarationId", variable_declaration_id},
              {"definitionKind", "definition"},
              {"identitySpanId", identity_span.span_id},
              {"initializerExpressionId", nullptr},
              {"kind", "variable"},
              {"lexicalParentId", nullptr},
              {"linkage", "external"},
              {"mangledName", nullptr},
              {"memorySpace", "generic"},
              {"origin", variable_origin},
              {"semanticParentId", nullptr},
              {"storageDuration", "static"},
              {"targetTypeId", nullptr},
              {"typeId", layout_type_id}});
  Json::Array declarations = {template_declaration, variable_declaration};
  std::sort(
      declarations.begin(), declarations.end(),
      [](const Json& left, const Json& right) {
        const auto& left_object = std::get<Json::Object>(left.value);
        const auto& right_object = std::get<Json::Object>(right.value);
        return std::get<std::string>(left_object.at("declarationId").value) <
               std::get<std::string>(right_object.at("declarationId").value);
      });
  const Json fact = object(
      {{"cosize", object({{"kind", "integer"},
                          {"value", decimal_u64(static_cast<std::uint64_t>(
                                        layout.cosize))}})},
       {"factId", fact_id},
       {"kind", "affine-layout"},
       {"leafRank", layout.leaf_rank},
       {"origin", variable_origin},
       {"rank", layout.rank},
       {"resultDeclarationId", variable_declaration_id},
       {"shape", hierarchy_json(layout.shape)},
       {"size", object({{"kind", "integer"},
                        {"value", decimal_u64(static_cast<std::uint64_t>(
                                      layout.size))}})},
       {"stride", hierarchy_json(layout.stride)}});
  const Json frontend_entry = object(
      {{"entryId", entry_id},
       {"kind", "layout"},
       {"layoutFactId", fact_id},
       {"selectedRootDeclarationIds", array({variable_declaration_id})}});
  const Json source_entity =
      object({{"canonicalIdentity", layout.canonical_usr},
              {"domains", array({"device", "host"})},
              {"entityKind", "variable"},
              {"origin", variable_origin},
              {"sourceEntityId", source_entity_id}});

  const std::string shared_surface_sha256 = hash_json(
      shared_surface_projection(source_entity_id), maximum_byte_length);
  Json::Array semantic_passes;
  semantic_passes.reserve(2U);
  for (std::size_t index = 0U; index < 2U; ++index) {
    const SemanticPassView pass = session.semantic_pass(index);
    const std::vector<std::string> opened_ids =
        pass_opened_ids(producer.passes[index], files);
    const std::vector<std::string>& edge_ids = pass_edge_ids.ids[index];
    const std::string observed_hash =
        hash_json(semantic_pass_input_projection(pass, roots, files, edges,
                                                 opened_ids, edge_ids),
                  maximum_byte_length);
    semantic_passes.push_back(object(
        {{"auxiliaryTargetTriple", pass.auxiliary_target_triple},
         {"deviceArchitecture", pass.device_architecture},
         {"diagnosticIds", Json::Array{}},
         {"domain", pass.domain},
         {"factIds", index == 0U ? array({fact_id}) : Json(Json::Array{})},
         {"includeEdgeIds", strings_json(edge_ids)},
         {"invocationMode", pass.invocation_mode},
         {"observedInputClosureSha256", observed_hash},
         {"openedFileIds", strings_json(opened_ids)},
         {"ordinal", pass.ordinal},
         {"passId", pass.pass_id},
         {"role", pass.role},
         {"selectedSourceRootEntityIds", array({source_entity_id})},
         {"sharedSurfaceSha256", shared_surface_sha256},
         {"status", "succeeded"},
         {"targetTriple", pass.target_triple}}));
  }

  return object(
      {{"compilationContractHash", session.compilation_contract_hash()},
       {"constants", Json::Array{}},
       {"declarations", Json(std::move(declarations))},
       {"diagnostics", Json::Array{}},
       {"entries", array({frontend_entry})},
       {"extraction", object({{"appliedTransforms", Json::Array{}},
                              {"compilationContractHash",
                               session.compilation_contract_hash()},
                              {"inputClosureSha256", closure.closure_sha256}})},
       {"facts", array({fact})},
       {"functionBodies", Json::Array{}},
       {"initializerExpressions", Json::Array{}},
       {"inputs", closure.inputs},
       {"macroExpansions", Json::Array{}},
       {"outcome", object({{"kind", "accepted"},
                           {"selectedEntryIds", array({entry_id})}})},
       {"overloadResolutions", Json::Array{}},
       {"semanticGraphOwnerPassId", "cuda-device-sema"},
       {"semanticPasses", Json(std::move(semantic_passes))},
       {"sourceAbi",
        object({{"functions", Json::Array{}}, {"types", Json::Array{}}})},
       {"sourceEntities", array({source_entity})},
       {"spans", Json(std::move(span_values))},
       {"templateInstantiations", Json::Array{}},
       {"types", array({type})}});
}

std::string_view diagnostic_phase_name(const DiagnosticPhase phase) {
  switch (phase) {
    case DiagnosticPhase::kPreprocessing:
      return "preprocessing";
    case DiagnosticPhase::kParsing:
      return "parsing";
    case DiagnosticPhase::kNameLookup:
      return "name-lookup";
    case DiagnosticPhase::kOverloadResolution:
      return "overload-resolution";
    case DiagnosticPhase::kTemplateInstantiation:
      return "template-instantiation";
    case DiagnosticPhase::kCudaSema:
      return "cuda-sema";
    case DiagnosticPhase::kArtifactExtraction:
      return "artifact-extraction";
  }
  throw InvalidObservation();
}

std::string_view diagnostic_severity_name(const DiagnosticSeverity severity) {
  switch (severity) {
    case DiagnosticSeverity::kRemark:
      return "remark";
    case DiagnosticSeverity::kNote:
      return "note";
    case DiagnosticSeverity::kWarning:
      return "warning";
    case DiagnosticSeverity::kError:
      return "error";
    case DiagnosticSeverity::kFatal:
      return "fatal";
    case DiagnosticSeverity::kNone:
      break;
  }
  throw InvalidObservation();
}

Json diagnostic_subject_json(const NormalizedDiagnosticSubject& subject) {
  switch (subject.kind) {
    case DiagnosticSubjectKind::kCompiler:
      if (!subject.entity_id.empty()) throw InvalidObservation();
      return object({{"kind", "compiler"}});
    case DiagnosticSubjectKind::kFile:
      return object({{"fileId", subject.entity_id}, {"kind", "file"}});
    case DiagnosticSubjectKind::kDeclaration:
      return object(
          {{"declarationId", subject.entity_id}, {"kind", "declaration"}});
    case DiagnosticSubjectKind::kType:
      return object({{"kind", "type"}, {"typeId", subject.entity_id}});
    case DiagnosticSubjectKind::kExpression:
      return object(
          {{"expressionId", subject.entity_id}, {"kind", "expression"}});
    case DiagnosticSubjectKind::kFact:
      return object({{"factId", subject.entity_id}, {"kind", "fact"}});
  }
  throw InvalidObservation();
}

Json normalized_diagnostic_json(const NormalizedDiagnostic& diagnostic) {
  Json location;
  if (diagnostic.location.has_source) {
    Json::Array related;
    related.reserve(diagnostic.location.related.size());
    for (const NormalizedDiagnosticRelatedLocation& entry :
         diagnostic.location.related) {
      related.push_back(object(
          {{"message", entry.rendered_message}, {"spanId", entry.span_id}}));
    }
    location = object({{"kind", "source"},
                       {"primarySpanId", diagnostic.location.primary_span_id},
                       {"related", Json(std::move(related))}});
  } else {
    location = object({{"kind", "none"}});
  }
  return object(
      {{"code", diagnostic.code},
       {"diagnosticId", diagnostic.diagnostic_id},
       {"location", std::move(location)},
       {"parentDiagnosticId", diagnostic.parent_diagnostic_id.empty()
                                  ? Json(nullptr)
                                  : Json(diagnostic.parent_diagnostic_id)},
       {"phase", diagnostic_phase_name(diagnostic.phase)},
       {"renderedMessage", diagnostic.rendered_message},
       {"severity", diagnostic_severity_name(diagnostic.severity)},
       {"subject", diagnostic_subject_json(diagnostic.subject)}});
}

struct DiagnosticRecord final {
  std::string diagnostic_id;
  bool root_blocking = false;
  Json json;
};

struct DiagnosticBuild final {
  std::vector<DiagnosticRecord> records;
  std::array<std::vector<std::string>, 2U> pass_ids;
  std::vector<std::string> blocking_ids;
};

using DiagnosticSpanIds = std::array<std::vector<std::string>, 2U>;

DiagnosticSpanIds add_diagnostic_spans(
    const ProducerReviewResult& producer, const std::vector<FileRecord>& files,
    std::map<std::string, SpanRecord, std::less<>>& spans) {
  DiagnosticSpanIds result;
  for (std::size_t pass_index = 0U; pass_index < producer.completed_pass_count;
       ++pass_index) {
    const ProducerPassObservation& pass = producer.passes[pass_index];
    result[pass_index].resize(pass.diagnostics.size());
    for (std::size_t index = 0U; index < pass.diagnostics.size(); ++index) {
      const ProducerDiagnosticObservation& diagnostic = pass.diagnostics[index];
      if (!diagnostic.has_source_location) {
        if (!diagnostic.virtual_path.empty() || diagnostic.byte_offset != 0U) {
          throw InvalidObservation();
        }
        continue;
      }
      const FileRecord& file = file_by_path(files, diagnostic.virtual_path);
      if (diagnostic.byte_offset > file.byte_length) {
        throw InvalidObservation();
      }
      SpanRecord span = span_record(file.file_id, diagnostic.byte_offset,
                                    diagnostic.byte_offset);
      const std::string span_id = span.span_id;
      const auto [found, inserted] = spans.emplace(span_id, std::move(span));
      static_cast<void>(inserted);
      result[pass_index][index] = found->first;
    }
  }
  return result;
}

bool opened_in_pass(const ProducerPassObservation& pass,
                    const std::string& path) noexcept {
  return !path.empty() &&
         std::find(pass.opened_file_paths.begin(), pass.opened_file_paths.end(),
                   path) != pass.opened_file_paths.end();
}

bool view_copy_origins_opened(
    const ProducerPassObservation& pass) noexcept {
  if (!opened_in_pass(pass, pass.view_copy.copy_callee_path)) return false;
  return std::all_of(
      pass.view_copy.tensors.begin(), pass.view_copy.tensors.end(),
      [&pass](const ProducerViewCopyTensorObservation& tensor) {
        return opened_in_pass(pass, tensor.tensor_template_path) &&
               opened_in_pass(pass, tensor.layout_template_path) &&
               opened_in_pass(pass, tensor.initializer_callee_path);
      });
}

std::string_view view_copy_extraction_failure_message(
    const ProducerPassObservation& pass) noexcept {
  const ProducerViewCopyObservation& view_copy = pass.view_copy;
  if (!view_copy.selected) {
    return "selected view-copy function was not resolved";
  }
  if (view_copy.ambiguous) {
    return "selected view-copy function resolved ambiguously";
  }
  if (view_copy.canonical_usr.empty()) {
    return "selected view-copy function has no canonical identity";
  }
  if (view_copy.cuda_host || !view_copy.cuda_device ||
      view_copy.cuda_global) {
    return "selected view-copy function has unsupported CUDA attributes";
  }
  if (view_copy.parameters.size() != 2U) {
    return "selected view-copy function does not have two resolved parameters";
  }
  for (std::size_t index = 0U; index < view_copy.parameters.size(); ++index) {
    const ProducerViewCopyParameterObservation& parameter =
        view_copy.parameters[index];
    if (!parameter.resolved_pointer ||
        !parameter.resolved_float_pointee ||
        parameter.pointee_const != (index == 0U) ||
        parameter.canonical_usr.empty()) {
      return "selected view-copy function has unsupported parameter semantics";
    }
  }
  if (view_copy.tensors.size() != 2U) {
    return "selected view-copy function does not have two resolved tensors";
  }
  for (const ProducerViewCopyTensorObservation& tensor :
       view_copy.tensors) {
    if (!tensor.resolved_tensor_type) {
      return "selected view-copy function has an unsupported tensor type";
    }
    if (!tensor.resolved_static_affine_layout) {
      return "selected view-copy function has an unresolved static affine layout";
    }
    if (!tensor.initializer_parameter_bound) {
      return "selected view-copy tensor is not bound to its function parameter";
    }
    if (tensor.canonical_usr.empty() ||
        tensor.initializer_callee_usr.empty()) {
      return "selected view-copy tensor has no canonical identity";
    }
  }
  if (view_copy.copy_callee_name != "cute::copy" ||
      view_copy.copy_callee_usr.empty() || !view_copy.resolved_copy) {
    return "selected view-copy function has no supported resolved cute::copy";
  }
  if (!view_copy_origins_opened(pass)) {
    return "selected view-copy function references an unopened CuTe origin";
  }
  if (!view_copy.resolved_function) {
    return "selected view-copy function failed final semantic validation";
  }
  return "selected frontend extraction did not yield a supported resolved "
         "semantic fact";
}

bool pass_observation_failed(const ProducerPassObservation& pass) noexcept {
  const bool resolved_layout = pass.layout.selected &&
                               pass.layout.resolved_layout_type &&
                               pass.layout.resolved_static_affine_layout &&
                               !pass.layout.canonical_usr.empty();
  const bool resolved_view_copy =
      pass.view_copy.selected && !pass.view_copy.ambiguous &&
      pass.view_copy.resolved_function && pass.view_copy.resolved_copy &&
      !pass.view_copy.canonical_usr.empty() &&
      view_copy_origins_opened(pass);
  return !pass.invocation_succeeded || pass.policy_failed ||
         pass.policy_violation_count != 0U || pass.clang_error_count != 0U ||
         !((resolved_layout && empty_view_copy(pass.view_copy)) ||
           (resolved_view_copy && empty_layout(pass.layout)));
}

bool rejected_pass_failed(const ProducerReviewResult& producer,
                          const std::size_t pass_index) noexcept {
  if (pass_index >= producer.completed_pass_count) return false;
  if (pass_observation_failed(producer.passes[pass_index])) return true;
  return pass_index == 1U && producer.completed_pass_count == 2U &&
         !producer.shared_surface_converged;
}

DiagnosticBuild normalize_diagnostics(
    const ProducerReviewResult& producer, const DecodedCompileSession& session,
    const std::vector<FileRecord>& files,
    const std::map<std::string, SpanRecord, std::less<>>& spans,
    const DiagnosticSpanIds& diagnostic_span_ids) {
  std::vector<std::string_view> opened_span_ids;
  opened_span_ids.reserve(spans.size());
  for (const auto& [id, span] : spans) {
    static_cast<void>(span);
    opened_span_ids.emplace_back(id);
  }
  std::vector<std::string_view> opened_virtual_paths;
  opened_virtual_paths.reserve(files.size());
  for (const FileRecord& file : files) {
    opened_virtual_paths.emplace_back(file.virtual_path);
  }

  DiagnosticBuild output;
  std::map<std::string, DiagnosticRecord, std::less<>> records;
  for (std::size_t pass_index = 0U; pass_index < producer.completed_pass_count;
       ++pass_index) {
    const ProducerPassObservation& pass = producer.passes[pass_index];
    const SemanticPassView semantic_pass = session.semantic_pass(pass_index);
    DiagnosticNormalizerConfig config = {
        session.compilation_contract_hash(),
        semantic_pass.pass_id,
        static_cast<std::uint32_t>(
            session.request_semantic_limit(CompileSemanticLimit::kDiagnostics)),
        session.maximum_output_byte_length(),
        opened_span_ids,
        opened_virtual_paths,
    };
    CppCuteDiagnosticNormalizer normalizer(config);
    if (!normalizer.configured()) throw InvalidObservation();

    for (std::size_t index = 0U; index < pass.diagnostics.size(); ++index) {
      if (normalizer.poisoned()) break;
      const ProducerDiagnosticObservation& diagnostic = pass.diagnostics[index];
      RawDiagnosticInput input;
      input.stage = diagnostic.stage;
      input.severity = diagnostic.severity;
      input.custom = diagnostic.custom;
      input.custom_code = diagnostic.custom_code;
      input.raw_diagnostic_id = diagnostic.raw_diagnostic_id;
      input.rendered_message = diagnostic.rendered_message;
      if (diagnostic.has_source_location) {
        const FileRecord& file = file_by_path(files, diagnostic.virtual_path);
        input.location.has_source = true;
        input.location.primary_span_id = diagnostic_span_ids[pass_index][index];
        input.subject.kind = DiagnosticSubjectKind::kFile;
        input.subject.entity_id = file.file_id;
      } else {
        input.subject.kind = DiagnosticSubjectKind::kCompiler;
      }
      static_cast<void>(normalizer.normalize(input));
    }

    bool has_root_blocking = false;
    for (std::size_t index = 0U; index < normalizer.diagnostic_count();
         ++index) {
      const NormalizedDiagnostic* diagnostic = normalizer.diagnostic(index);
      if (diagnostic == nullptr) throw InvalidObservation();
      if (diagnostic->parent_diagnostic_id.empty() &&
          (diagnostic->severity == DiagnosticSeverity::kError ||
           diagnostic->severity == DiagnosticSeverity::kFatal)) {
        has_root_blocking = true;
      }
    }
    if (rejected_pass_failed(producer, pass_index) && !has_root_blocking &&
        !normalizer.poisoned()) {
      RawDiagnosticInput synthetic;
      synthetic.stage = RawDiagnosticStage::kArtifactExtractor;
      synthetic.severity = RawDiagnosticSeverity::kError;
      synthetic.custom = true;
      if (pass_index == 1U && producer.completed_pass_count == 2U &&
          !producer.shared_surface_converged) {
        synthetic.custom_code =
            CustomDiagnosticCode::kHostDeviceSurfaceDivergence;
        synthetic.rendered_message =
            "host and device selected source surfaces did not converge";
      } else {
        synthetic.custom_code = CustomDiagnosticCode::kSemanticExtractionFailed;
        synthetic.rendered_message =
            session.entry_request().kind == "view-copy"
                ? view_copy_extraction_failure_message(pass)
                : std::string_view(
                      "selected frontend extraction did not yield a supported "
                      "resolved semantic fact");
      }
      synthetic.subject.kind = DiagnosticSubjectKind::kCompiler;
      static_cast<void>(normalizer.normalize(synthetic));
    }

    std::vector<std::string>& pass_ids = output.pass_ids[pass_index];
    pass_ids.reserve(normalizer.diagnostic_count());
    for (std::size_t index = 0U; index < normalizer.diagnostic_count();
         ++index) {
      const NormalizedDiagnostic* diagnostic = normalizer.diagnostic(index);
      if (diagnostic == nullptr) throw InvalidObservation();
      const bool root_blocking =
          diagnostic->parent_diagnostic_id.empty() &&
          (diagnostic->severity == DiagnosticSeverity::kError ||
           diagnostic->severity == DiagnosticSeverity::kFatal);
      pass_ids.push_back(diagnostic->diagnostic_id);
      if (!records
               .emplace(
                   diagnostic->diagnostic_id,
                   DiagnosticRecord{diagnostic->diagnostic_id, root_blocking,
                                    normalized_diagnostic_json(*diagnostic)})
               .second) {
        throw InvalidObservation();
      }
    }
    std::sort(pass_ids.begin(), pass_ids.end());
    if (rejected_pass_failed(producer, pass_index) &&
        std::none_of(pass_ids.begin(), pass_ids.end(),
                     [&records](const std::string& id) {
                       return records.at(id).root_blocking;
                     })) {
      if (normalizer.terminal_status() ==
          DiagnosticNormalizationStatus::kResourceLimit) {
        throw ArtifactResourceLimit();
      }
      throw InvalidObservation();
    }
  }
  if (records.size() >
      session.request_semantic_limit(CompileSemanticLimit::kDiagnostics)) {
    throw ArtifactResourceLimit();
  }
  output.records.reserve(records.size());
  for (auto& [id, record] : records) {
    static_cast<void>(id);
    if (record.root_blocking)
      output.blocking_ids.push_back(record.diagnostic_id);
    output.records.push_back(std::move(record));
  }
  if (output.blocking_ids.empty()) throw InvalidObservation();
  return output;
}

Json empty_shared_surface_projection() {
  return object(
      {{"domain", "browsergrad.compiler.cpp-cute.shared-source-surface.v2"},
       {"functions", Json::Array{}},
       {"selectedSourceRootEntityIds", Json::Array{}},
       {"types", Json::Array{}}});
}

Json build_rejected_payload(const ProducerReviewResult& producer,
                            const DecodedCompileSession& session,
                            const std::size_t maximum_byte_length) {
  const std::vector<RootRecord> roots = build_roots(session);
  const OpenedIdentityMap opened = merge_opened_files(producer);
  const std::vector<FileRecord> files = build_files(session, roots, opened);
  const FileRecord& main_file =
      file_by_path(files, session.main_virtual_path());
  const Json::Object& main_json = std::get<Json::Object>(main_file.json.value);
  if (std::get<std::string>(main_json.at("role").value) != "main-source") {
    throw InvalidObservation();
  }
  for (std::size_t pass_index = 0U; pass_index < producer.completed_pass_count;
       ++pass_index) {
    const ProducerPassObservation& pass = producer.passes[pass_index];
    if (std::none_of(pass.opened_files.begin(), pass.opened_files.end(),
                     [&session](const ProducerOpenedFileObservation& file) {
                       return file.virtual_path == session.main_virtual_path();
                     })) {
      throw InvalidObservation();
    }
  }

  auto [edges, pass_edge_ids] = build_edges(producer, session, files);
  std::map<std::string, SpanRecord, std::less<>> spans;
  for (const EdgeRecord& edge : edges) {
    if (edge.has_directive_span) {
      spans.emplace(edge.directive_span.span_id, edge.directive_span);
    }
  }
  const DiagnosticSpanIds diagnostic_span_ids =
      add_diagnostic_spans(producer, files, spans);
  const DiagnosticBuild diagnostic_build = normalize_diagnostics(
      producer, session, files, spans, diagnostic_span_ids);

  Json::Array span_values;
  span_values.reserve(spans.size());
  for (const auto& [id, span] : spans) {
    static_cast<void>(id);
    span_values.push_back(span.json);
  }
  Json::Array diagnostic_values;
  diagnostic_values.reserve(diagnostic_build.records.size());
  for (const DiagnosticRecord& diagnostic : diagnostic_build.records) {
    diagnostic_values.push_back(diagnostic.json);
  }

  const std::string source_set_sha256 =
      hash_json(file_set_projection("source", files), maximum_byte_length);
  const std::string header_set_sha256 =
      hash_json(file_set_projection("header", files), maximum_byte_length);
  const Json closure_projection =
      object({{"domain", "browsergrad.compiler.cpp-cute.input-closure.v2"},
              {"files", Json(json_array_of_files(files))},
              {"headerSetSha256", header_set_sha256},
              {"includeEdges", Json(json_array_of_edges(edges))},
              {"includeRoots", Json(json_array_of_roots(roots))},
              {"mainFileId", main_file.file_id},
              {"sourceSetSha256", source_set_sha256}});
  const std::string closure_sha256 =
      hash_json(closure_projection, maximum_byte_length);
  const Json inputs =
      object({{"closureSha256", closure_sha256},
              {"files", Json(json_array_of_files(files))},
              {"headerSetSha256", header_set_sha256},
              {"includeEdges", Json(json_array_of_edges(edges))},
              {"includeRoots", Json(json_array_of_roots(roots))},
              {"mainFileId", main_file.file_id},
              {"sourceSetSha256", source_set_sha256}});

  const std::string empty_surface_sha256 =
      hash_json(empty_shared_surface_projection(), maximum_byte_length);
  Json::Array semantic_passes;
  semantic_passes.reserve(2U);
  for (std::size_t index = 0U; index < 2U; ++index) {
    const SemanticPassView pass = session.semantic_pass(index);
    const bool executed = index < producer.completed_pass_count;
    const bool failed = rejected_pass_failed(producer, index);
    std::vector<std::string> opened_ids;
    std::string observed_hash;
    if (executed) {
      opened_ids = pass_opened_ids(producer.passes[index], files);
      observed_hash = hash_json(
          semantic_pass_input_projection(pass, roots, files, edges, opened_ids,
                                         pass_edge_ids.ids[index]),
          maximum_byte_length);
    }
    semantic_passes.push_back(object(
        {{"auxiliaryTargetTriple", pass.auxiliary_target_triple},
         {"deviceArchitecture", pass.device_architecture},
         {"diagnosticIds", strings_json(diagnostic_build.pass_ids[index])},
         {"domain", pass.domain},
         {"factIds", Json::Array{}},
         {"includeEdgeIds", executed ? strings_json(pass_edge_ids.ids[index])
                                     : Json(Json::Array{})},
         {"invocationMode", pass.invocation_mode},
         {"observedInputClosureSha256",
          executed ? Json(observed_hash) : Json(nullptr)},
         {"openedFileIds",
          executed ? strings_json(opened_ids) : Json(Json::Array{})},
         {"ordinal", pass.ordinal},
         {"passId", pass.pass_id},
         {"role", pass.role},
         {"selectedSourceRootEntityIds", Json::Array{}},
         {"sharedSurfaceSha256",
          executed && !failed ? Json(empty_surface_sha256) : Json(nullptr)},
         {"status", !executed ? "not-run" : (failed ? "failed" : "succeeded")},
         {"targetTriple", pass.target_triple}}));
  }

  return object(
      {{"compilationContractHash", session.compilation_contract_hash()},
       {"constants", Json::Array{}},
       {"declarations", Json::Array{}},
       {"diagnostics", Json(std::move(diagnostic_values))},
       {"entries", Json::Array{}},
       {"extraction", object({{"appliedTransforms", Json::Array{}},
                              {"compilationContractHash",
                               session.compilation_contract_hash()},
                              {"inputClosureSha256", closure_sha256}})},
       {"facts", Json::Array{}},
       {"functionBodies", Json::Array{}},
       {"initializerExpressions", Json::Array{}},
       {"inputs", inputs},
       {"macroExpansions", Json::Array{}},
       {"outcome", object({{"blockingDiagnosticIds",
                            strings_json(diagnostic_build.blocking_ids)},
                           {"kind", "rejected"}})},
       {"overloadResolutions", Json::Array{}},
       {"semanticGraphOwnerPassId", "cuda-device-sema"},
       {"semanticPasses", Json(std::move(semantic_passes))},
       {"sourceAbi",
        object({{"functions", Json::Array{}}, {"types", Json::Array{}}})},
       {"sourceEntities", Json::Array{}},
       {"spans", Json(std::move(span_values))},
       {"templateInstantiations", Json::Array{}},
       {"types", Json::Array{}}});
}

void validate_producer(const ProducerReviewResult& producer) {
  const bool accepted =
      producer.status == ProducerReviewStatus::kReviewComplete;
  const bool rejected =
      producer.status ==
      ProducerReviewStatus::kReviewCompleteWithBlockingDiagnostics;
  if ((!accepted && !rejected) || producer.completed_pass_count == 0U ||
      producer.completed_pass_count > producer.passes.size()) {
    throw InvalidObservation();
  }
  if (accepted && (producer.completed_pass_count != 2U ||
                   producer.blocking_diagnostic_pass_count != 0U ||
                   !producer.shared_surface_converged)) {
    throw InvalidObservation();
  }
  if (rejected && (producer.blocking_diagnostic_pass_count == 0U ||
                   producer.shared_surface_converged)) {
    throw InvalidObservation();
  }
  bool observed_failed_pass = false;
  for (std::size_t index = 0U; index < producer.passes.size(); ++index) {
    const ProducerPassObservation& pass = producer.passes[index];
    if (index >= producer.completed_pass_count) {
      if (pass.invocation_succeeded || pass.policy_installed ||
          pass.policy_failed || pass.vfs_failed ||
          pass.policy_violation_count != 0U || pass.clang_error_count != 0U ||
          pass.diagnostic_capture_failed || !pass.diagnostics.empty() ||
          !pass.opened_file_paths.empty() || !pass.opened_files.empty() ||
          !pass.include_edges.empty() || pass.layout.selected ||
          pass.layout.resolved_layout_type ||
          pass.layout.resolved_static_affine_layout ||
          !pass.layout.canonical_usr.empty() ||
          !pass.layout.canonical_name.empty() ||
          !pass.layout.canonical_type.empty() || pass.view_copy.selected ||
          pass.view_copy.ambiguous || pass.view_copy.resolved_function ||
          pass.view_copy.resolved_copy || pass.view_copy.cuda_host ||
          pass.view_copy.cuda_device || pass.view_copy.cuda_global ||
          pass.view_copy.force_inline ||
          !pass.view_copy.canonical_usr.empty() ||
          !pass.view_copy.canonical_name.empty() ||
          !pass.view_copy.canonical_type.empty() ||
          !pass.view_copy.copy_callee_usr.empty() ||
          !pass.view_copy.copy_callee_name.empty() ||
          !pass.view_copy.copy_callee_path.empty() ||
          pass.view_copy.declaration_begin_byte != 0U ||
          pass.view_copy.declaration_end_byte != 0U ||
          pass.view_copy.identity_begin_byte != 0U ||
          pass.view_copy.identity_end_byte != 0U ||
          pass.view_copy.copy_begin_byte != 0U ||
          pass.view_copy.copy_end_byte != 0U ||
          pass.view_copy.source_tensor_ordinal != 0U ||
          pass.view_copy.destination_tensor_ordinal != 0U ||
          !pass.view_copy.parameters.empty() ||
          !pass.view_copy.tensors.empty()) {
        throw InvalidObservation();
      }
      continue;
    }
    if (!pass.policy_installed || pass.vfs_failed ||
        pass.diagnostic_capture_failed ||
        (!pass.invocation_succeeded && pass.clang_error_count == 0U &&
         !pass.policy_failed)) {
      throw InvalidObservation();
    }
    const bool failed = rejected_pass_failed(producer, index);
    observed_failed_pass = observed_failed_pass || failed;
    if (accepted && (failed || pass.policy_violation_count != 0U ||
                     pass.clang_error_count != 0U)) {
      throw InvalidObservation();
    }
  }
  if (rejected && !observed_failed_pass) throw InvalidObservation();
}

}  // namespace

ArtifactV3WriteStatus write_cpp_cute_artifact_v3(
    const ProducerReviewResult& producer, const DecodedCompileSession& session,
    ArtifactV3ResultSink& result_sink) noexcept {
  try {
    validate_producer(producer);
    const std::size_t maximum_byte_length =
        session.maximum_output_byte_length();
    if (maximum_byte_length == 0U ||
        maximum_byte_length > ArtifactV3ResultSink::kAbiMaximumByteLength) {
      return ArtifactV3WriteStatus::kInvalidObservation;
    }
    Json payload;
    try {
      payload =
          producer.status == ProducerReviewStatus::kReviewComplete
              ? build_accepted_payload(producer, session, maximum_byte_length)
              : build_rejected_payload(producer, session, maximum_byte_length);
    } catch (const ArtifactResourceLimit&) {
      report_native_diagnostic(
          NativeDiagnosticCode::kArtifactWriterPayloadResourceLimit);
      return ArtifactV3WriteStatus::kResourceLimit;
    } catch (const std::bad_alloc&) {
      report_native_diagnostic(
          NativeDiagnosticCode::kArtifactWriterAllocationFailure);
      return ArtifactV3WriteStatus::kResourceLimit;
    }

    Json envelope;
    try {
      const std::string artifact_hash = hash_json(
          object(
              {{"domain",
                "browsergrad.compiler.cpp-cute.frontend-artifact.v3"},
               {"payload", payload},
               {"requiredExtensions", Json::Array{}},
               {"schema", "browsergrad.compiler.cpp-cute.frontend-artifact"},
               {"version", object({{"major", 3U}, {"minor", 0U}})}}),
          maximum_byte_length);
      envelope = object(
          {{"artifactId",
            std::string("bg.artifact.cpp-cute-frontend.sha256.") +
                artifact_hash},
           {"payload", std::move(payload)},
           {"producer", object({{"id", "browsergrad-tools/cpp-cute-frontend"},
                                {"version", "0.1.0"}})},
           {"requiredExtensions", Json::Array{}},
           {"schema", "browsergrad.compiler.cpp-cute.frontend-artifact"},
           {"version", object({{"major", 3U}, {"minor", 0U}})}});
    } catch (const ArtifactResourceLimit&) {
      report_native_diagnostic(
          NativeDiagnosticCode::kArtifactWriterSerializationResourceLimit);
      return ArtifactV3WriteStatus::kResourceLimit;
    } catch (const std::bad_alloc&) {
      report_native_diagnostic(
          NativeDiagnosticCode::kArtifactWriterAllocationFailure);
      return ArtifactV3WriteStatus::kResourceLimit;
    }

    std::string bytes;
    try {
      bytes = canonical_json(envelope, maximum_byte_length);
    } catch (const ArtifactResourceLimit&) {
      report_native_diagnostic(
          NativeDiagnosticCode::kArtifactWriterSerializationResourceLimit);
      return ArtifactV3WriteStatus::kResourceLimit;
    } catch (const std::bad_alloc&) {
      report_native_diagnostic(
          NativeDiagnosticCode::kArtifactWriterAllocationFailure);
      return ArtifactV3WriteStatus::kResourceLimit;
    }
    if (bytes.empty() ||
        bytes.size() > std::numeric_limits<std::uint32_t>::max()) {
      report_native_diagnostic(
          NativeDiagnosticCode::kArtifactWriterSerializationResourceLimit);
      return ArtifactV3WriteStatus::kResourceLimit;
    }
    const CanonicalJsonLimits limits = {
        static_cast<std::uint32_t>(maximum_byte_length),
        kMaximumJsonDepth,
        kMaximumJsonNodes,
        static_cast<std::uint32_t>(maximum_byte_length),
        kMaximumJsonArrayLength,
        kMaximumJsonObjectPropertyCount,
    };
    const CanonicalJsonValidation validation = validate_canonical_json(
        reinterpret_cast<const std::uint8_t*>(bytes.data()),
        static_cast<std::uint32_t>(bytes.size()), limits);
    if (validation.status == CanonicalJsonStatus::kResourceLimit) {
      report_native_diagnostic(
          NativeDiagnosticCode::kArtifactWriterSerializationResourceLimit);
      return ArtifactV3WriteStatus::kResourceLimit;
    }
    if (validation.status != CanonicalJsonStatus::kValid) {
      return ArtifactV3WriteStatus::kInvalidObservation;
    }
    std::uint8_t* destination =
        result_sink.allocate(static_cast<std::uint32_t>(bytes.size()));
    if (destination == nullptr) {
      report_native_diagnostic(
          NativeDiagnosticCode::kArtifactWriterResultAllocationLimit);
      return ArtifactV3WriteStatus::kResourceLimit;
    }
    std::memcpy(destination, bytes.data(), bytes.size());
    if (!result_sink.commit()) return ArtifactV3WriteStatus::kInternalError;
    return ArtifactV3WriteStatus::kReady;
  } catch (const ArtifactResourceLimit&) {
    report_native_diagnostic(
        NativeDiagnosticCode::kArtifactWriterPayloadResourceLimit);
    return ArtifactV3WriteStatus::kResourceLimit;
  } catch (const InvalidObservation&) {
    return ArtifactV3WriteStatus::kInvalidObservation;
  } catch (const std::bad_alloc&) {
    report_native_diagnostic(
        NativeDiagnosticCode::kArtifactWriterAllocationFailure);
    return ArtifactV3WriteStatus::kResourceLimit;
  } catch (...) {
    return ArtifactV3WriteStatus::kInternalError;
  }
}

}  // namespace browsergrad::cpp_cute
