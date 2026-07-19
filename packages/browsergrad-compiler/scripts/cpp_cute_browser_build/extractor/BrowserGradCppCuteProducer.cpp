#include "BrowserGradCppCuteProducer.h"

#include "BrowserGradCppCuteClangAction.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace browsergrad::cpp_cute {
namespace {

bool parse_u32(const std::string_view value, std::uint32_t& output) noexcept {
  if (value.empty() || (value.size() > 1U && value.front() == '0')) return false;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(),
                                      output);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}

bool source_anchor(const DecodedCompileSession& session,
                   SourceAnchor& output) {
  const EntryRequestView entry = session.entry_request();
  std::uint32_t begin = 0U;
  std::uint32_t end = 0U;
  if (entry.kind != "layout" || !parse_u32(entry.begin_byte, begin) ||
      !parse_u32(entry.end_byte, end) || begin >= end) {
    return false;
  }
  output = {std::string(entry.virtual_path), begin, end};
  return true;
}

ProducerIncludeKind producer_include_kind(
    const ImportedVfsIncludeKind kind) noexcept {
  switch (kind) {
    case ImportedVfsIncludeKind::kSourceQuote:
      return ProducerIncludeKind::kSourceQuote;
    case ImportedVfsIncludeKind::kSourceAngle:
      return ProducerIncludeKind::kSourceAngle;
    case ImportedVfsIncludeKind::kCompilerForced:
      return ProducerIncludeKind::kCompilerForced;
  }
  return ProducerIncludeKind::kSourceQuote;
}

ProducerIntegerHierarchy producer_hierarchy(LayoutIntegerHierarchy&& source) {
  ProducerIntegerHierarchy destination;
  destination.tuple = source.tuple;
  destination.value = source.value;
  destination.elements.reserve(source.elements.size());
  for (LayoutIntegerHierarchy& element : source.elements) {
    destination.elements.push_back(producer_hierarchy(std::move(element)));
  }
  return destination;
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

void commit_pass_review(ClangPassReview&& source,
                        ProducerPassObservation& destination) {
  destination.invocation_succeeded = source.invocation_succeeded;
  destination.policy_installed = source.policy_install_status ==
      CppCutePreprocessorPolicyInstallStatus::kInstalled;
  destination.policy_failed = source.policy_failed;
  destination.vfs_failed = source.vfs_failed;
  destination.policy_violation_count = source.policy_violation_count;
  destination.clang_error_count = source.clang_error_count;
  destination.diagnostic_capture_failed = source.diagnostic_capture_failed;
  destination.diagnostics.reserve(source.diagnostics.size());
  for (ClangDiagnosticObservation& diagnostic : source.diagnostics) {
    destination.diagnostics.push_back({
        diagnostic.stage,
        diagnostic.severity,
        diagnostic.custom,
        diagnostic.custom_code,
        diagnostic.raw_diagnostic_id,
        std::move(diagnostic.rendered_message),
        diagnostic.has_source_location,
        std::move(diagnostic.virtual_path),
        diagnostic.byte_offset,
    });
  }
  destination.opened_file_paths =
      std::move(source.vfs_observation.opened_file_paths);
  destination.opened_files.reserve(source.vfs_observation.opened_files.size());
  for (ImportedVfsOpenedFileObservation& file :
       source.vfs_observation.opened_files) {
    destination.opened_files.push_back({
        std::move(file.virtual_path),
        std::move(file.content_sha256),
        file.byte_length,
    });
  }
  destination.include_edges.reserve(source.vfs_observation.include_edges.size());
  for (ImportedVfsIncludeEdgeObservation& edge :
       source.vfs_observation.include_edges) {
    destination.include_edges.push_back({
        producer_include_kind(edge.kind),
        std::move(edge.including_file_path),
        std::move(edge.resolved_file_path),
        std::move(edge.spelling),
        edge.directive_start_byte_offset,
        edge.directive_end_byte_offset,
        edge.compiler_option_ordinal,
    });
  }
  destination.layout = {
      source.layout_trace.selected,
      source.layout_trace.resolved_layout_type,
      source.layout_trace.resolved_static_affine_layout,
      std::move(source.layout_trace.canonical_usr),
      std::move(source.layout_trace.canonical_name),
      std::move(source.layout_trace.canonical_type),
      std::move(source.layout_trace.initializer_callee),
      source.layout_trace.identity_begin_byte,
      source.layout_trace.identity_end_byte,
      source.layout_trace.rank,
      source.layout_trace.leaf_rank,
      source.layout_trace.size,
      source.layout_trace.cosize,
      producer_hierarchy(std::move(source.layout_trace.shape)),
      producer_hierarchy(std::move(source.layout_trace.stride)),
  };
}

}  // namespace

ProducerReviewResult run_cpp_cute_producer_review(
    const PreparedCppCuteCompilePlan& plan,
    const DecodedCompileSession& session) noexcept {
  ProducerReviewResult result;
  try {
    if (plan.compilation_contract_hash() !=
            session.compilation_contract_hash() ||
        session.semantic_pass_count() != 2U ||
        plan.device_arguments().empty() || plan.host_arguments().empty()) {
      result.status = ProducerReviewStatus::kInvalidPlan;
      return result;
    }
    SourceAnchor anchor;
    if (!source_anchor(session, anchor)) {
      result.status = ProducerReviewStatus::kInvalidPlan;
      return result;
    }
    std::vector<ClangForcedIncludeObservation> forced_includes;
    forced_includes.reserve(session.compiler_option_count());
    for (std::size_t index = 0U; index < session.compiler_option_count();
         ++index) {
      const CompilerOptionView option = session.compiler_option(index);
      if (option.kind == CompilerOptionKind::kForcedInclude) {
        forced_includes.push_back({option.virtual_path, option.ordinal});
      }
    }
    const std::uint64_t opened_limit =
        static_cast<std::uint64_t>(session.maximum_source_file_count()) +
        session.maximum_header_file_count();
    ImportedVfsObservationLimits observation_limits;
    observation_limits.max_opened_file_count = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(opened_limit,
                               kImportedVfsMaximumObservedFileCount));
    observation_limits.max_include_edge_count =
        kImportedVfsMaximumObservedIncludeEdgeCount;
    if (observation_limits.max_opened_file_count == 0U) {
      result.status = ProducerReviewStatus::kInvalidPlan;
      return result;
    }

    const std::array<std::span<const std::string>, 2U> arguments = {
        plan.device_arguments(), plan.host_arguments()};
    for (std::size_t pass_index = 0U; pass_index < arguments.size();
         ++pass_index) {
      const std::span<const std::string> pass_arguments = arguments[pass_index];
      std::vector<std::string> owned_arguments(pass_arguments.begin(),
                                               pass_arguments.end());
      ClangPassReview review;
      static_cast<void>(run_cpp_cute_clang_pass_for_review(
          owned_arguments, anchor, forced_includes, observation_limits,
          session.maximum_diagnostic_count(),
          session.maximum_output_byte_length(),
          review));
      if (review.policy_install_status !=
          CppCutePreprocessorPolicyInstallStatus::kInstalled) {
        result.status = ProducerReviewStatus::kInternalError;
        return result;
      }
      if (review.vfs_failed) {
        result.status = ProducerReviewStatus::kVfsError;
        return result;
      }
      if (review.diagnostic_capture_failed) {
        result.status = ProducerReviewStatus::kResourceLimit;
        return result;
      }
      if (review.frontend_work_limit_exceeded) {
        result.status = ProducerReviewStatus::kResourceLimit;
        return result;
      }
      if (!review.invocation_succeeded && review.clang_error_count == 0U &&
          !review.policy_failed) {
        result.status = ProducerReviewStatus::kInternalError;
        return result;
      }
      const bool extraction_failed = review.invocation_succeeded &&
          !review.policy_failed && review.clang_error_count == 0U &&
          (!review.layout_trace.selected ||
           !review.layout_trace.resolved_layout_type ||
           !review.layout_trace.resolved_static_affine_layout ||
           review.layout_trace.canonical_usr.empty());
      commit_pass_review(std::move(review), result.passes[pass_index]);
      ++result.completed_pass_count;
      const ProducerPassObservation& committed = result.passes[pass_index];
      if (committed.policy_failed || committed.clang_error_count != 0U ||
          extraction_failed) {
        ++result.blocking_diagnostic_pass_count;
        break;
      }
    }
    if (result.completed_pass_count == 2U &&
        result.blocking_diagnostic_pass_count == 0U) {
      const ProducerLayoutObservation& device = result.passes[0].layout;
      const ProducerLayoutObservation& host = result.passes[1].layout;
      result.shared_surface_converged =
          device.canonical_usr == host.canonical_usr &&
          device.canonical_name == host.canonical_name &&
          device.canonical_type == host.canonical_type &&
          device.identity_begin_byte == host.identity_begin_byte &&
          device.identity_end_byte == host.identity_end_byte &&
          device.rank == host.rank && device.leaf_rank == host.leaf_rank &&
          device.size == host.size && device.cosize == host.cosize &&
          same_hierarchy(device.shape, host.shape) &&
          same_hierarchy(device.stride, host.stride);
      if (!result.shared_surface_converged) {
        ++result.blocking_diagnostic_pass_count;
      }
    }
    result.status = result.blocking_diagnostic_pass_count == 0U
                        ? ProducerReviewStatus::kReviewComplete
                        : ProducerReviewStatus::
                              kReviewCompleteWithBlockingDiagnostics;
    return result;
  } catch (const std::bad_alloc&) {
    result.status = ProducerReviewStatus::kResourceLimit;
    return result;
  } catch (...) {
    result.status = ProducerReviewStatus::kInternalError;
    return result;
  }
}

}  // namespace browsergrad::cpp_cute
