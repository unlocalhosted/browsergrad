#include "BrowserGradCppCuteProducer.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "BrowserGradCppCuteClangAction.h"
#include "BrowserGradCppCuteRuntime.h"

namespace browsergrad::cpp_cute {
namespace {

bool parse_u32(const std::string_view value, std::uint32_t& output) noexcept {
  if (value.empty() || (value.size() > 1U && value.front() == '0'))
    return false;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), output);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}

bool source_anchor(const DecodedCompileSession& session, SourceAnchor& output) {
  const EntryRequestView entry = session.entry_request();
  std::uint32_t begin = 0U;
  std::uint32_t end = 0U;
  const bool layout =
      entry.kind == "layout" && entry.declaration_kind == "variable";
  const bool view_copy =
      entry.kind == "view-copy" && entry.declaration_kind == "function";
  if ((!layout && !view_copy) || !parse_u32(entry.begin_byte, begin) ||
      !parse_u32(entry.end_byte, end) || begin >= end) {
    return false;
  }
  output = {
      std::string(entry.virtual_path),
      begin,
      end,
      layout ? SourceAnchorKind::kLayoutVariable
             : SourceAnchorKind::kViewCopyFunction,
  };
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

NativeDiagnosticCode producer_vfs_diagnostic(
    const ImportedVfsObserverFailure failure) noexcept {
  switch (failure) {
    case ImportedVfsObserverFailure::kNone:
      return NativeDiagnosticCode::kProducerVfsFailure;
    case ImportedVfsObserverFailure::kInvalidLimits:
      return NativeDiagnosticCode::kProducerVfsInvalidLimits;
    case ImportedVfsObserverFailure::kInvalidSuccessfulRead:
      return NativeDiagnosticCode::kProducerVfsInvalidSuccessfulRead;
    case ImportedVfsObserverFailure::kInconsistentSuccessfulRead:
      return NativeDiagnosticCode::kProducerVfsInconsistentSuccessfulRead;
    case ImportedVfsObserverFailure::kOpenedFileLimit:
      return NativeDiagnosticCode::kProducerVfsOpenedFileLimit;
    case ImportedVfsObserverFailure::kInvalidIncludeEdge:
      return NativeDiagnosticCode::kProducerVfsInvalidIncludeEdge;
    case ImportedVfsObserverFailure::kIncludeEdgeLimit:
      return NativeDiagnosticCode::kProducerVfsIncludeEdgeLimit;
    case ImportedVfsObserverFailure::kInvalidIncludeSourceRange:
      return NativeDiagnosticCode::kProducerVfsInvalidIncludeSourceRange;
    case ImportedVfsObserverFailure::kInvalidIncludeResolvedPath:
      return NativeDiagnosticCode::kProducerVfsInvalidIncludeResolvedPath;
    case ImportedVfsObserverFailure::kInvalidIncludeIncludingPath:
      return NativeDiagnosticCode::kProducerVfsInvalidIncludeIncludingPath;
    case ImportedVfsObserverFailure::kInvalidIncludeSpelling:
      return NativeDiagnosticCode::kProducerVfsInvalidIncludeSpelling;
    case ImportedVfsObserverFailure::kInvalidIncludeOffsets:
      return NativeDiagnosticCode::kProducerVfsInvalidIncludeOffsets;
    case ImportedVfsObserverFailure::kInvalidIncludeSourceOrdinal:
      return NativeDiagnosticCode::kProducerVfsInvalidIncludeSourceOrdinal;
    case ImportedVfsObserverFailure::kInvalidForcedIncludeShape:
      return NativeDiagnosticCode::kProducerVfsInvalidForcedIncludeShape;
    case ImportedVfsObserverFailure::kInvalidIncludeKind:
      return NativeDiagnosticCode::kProducerVfsInvalidIncludeKind;
  }
  return NativeDiagnosticCode::kProducerVfsFailure;
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

bool same_view_copy_parameter(
    const ProducerViewCopyParameterObservation& left,
    const ProducerViewCopyParameterObservation& right) noexcept {
  return left.resolved_pointer == right.resolved_pointer &&
         left.resolved_float_pointee == right.resolved_float_pointee &&
         left.pointee_const == right.pointee_const &&
         left.ordinal == right.ordinal &&
         left.canonical_usr == right.canonical_usr &&
         left.canonical_name == right.canonical_name &&
         left.canonical_type == right.canonical_type &&
         left.declaration_begin_byte == right.declaration_begin_byte &&
         left.declaration_end_byte == right.declaration_end_byte &&
         left.identity_begin_byte == right.identity_begin_byte &&
         left.identity_end_byte == right.identity_end_byte;
}

bool same_view_copy_tensor(
    const ProducerViewCopyTensorObservation& left,
    const ProducerViewCopyTensorObservation& right) noexcept {
  return left.resolved_tensor_type == right.resolved_tensor_type &&
         left.resolved_static_affine_layout ==
             right.resolved_static_affine_layout &&
         left.initializer_parameter_bound ==
             right.initializer_parameter_bound &&
         left.engine_pointee_const == right.engine_pointee_const &&
         left.engine_parameter_ordinal == right.engine_parameter_ordinal &&
         left.canonical_usr == right.canonical_usr &&
         left.canonical_name == right.canonical_name &&
         left.canonical_type == right.canonical_type &&
         left.tensor_template_path == right.tensor_template_path &&
         left.initializer_callee_usr == right.initializer_callee_usr &&
         left.initializer_callee_name == right.initializer_callee_name &&
         left.initializer_callee_path == right.initializer_callee_path &&
         left.layout_canonical_type == right.layout_canonical_type &&
         left.layout_template_path == right.layout_template_path &&
         left.declaration_begin_byte == right.declaration_begin_byte &&
         left.declaration_end_byte == right.declaration_end_byte &&
         left.identity_begin_byte == right.identity_begin_byte &&
         left.identity_end_byte == right.identity_end_byte &&
         left.rank == right.rank && left.leaf_rank == right.leaf_rank &&
         left.size == right.size && left.cosize == right.cosize &&
         same_hierarchy(left.shape, right.shape) &&
         same_hierarchy(left.stride, right.stride);
}

bool same_view_copy(const ProducerViewCopyObservation& left,
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
    if (!same_view_copy_parameter(left.parameters[index],
                                  right.parameters[index])) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < left.tensors.size(); ++index) {
    if (!same_view_copy_tensor(left.tensors[index], right.tensors[index])) {
      return false;
    }
  }
  return true;
}

bool view_copy_origins_opened(const ViewCopyTrace& trace,
                              const ImportedVfsPassObservation& vfs) {
  const auto opened = [&vfs](const std::string& path) {
    return !path.empty() &&
           std::find(vfs.opened_file_paths.begin(), vfs.opened_file_paths.end(),
                     path) != vfs.opened_file_paths.end();
  };
  if (!opened(trace.copy_callee_path)) return false;
  for (const ViewCopyTensorTrace& tensor : trace.tensors) {
    if (!opened(tensor.tensor_template_path) ||
        !opened(tensor.layout_template_path) ||
        !opened(tensor.initializer_callee_path)) {
      return false;
    }
  }
  return true;
}

ProducerViewCopyParameterObservation producer_parameter(
    ViewCopyParameterTrace&& source) {
  return {
      source.resolved_pointer,
      source.resolved_float_pointee,
      source.pointee_const,
      source.ordinal,
      std::move(source.canonical_usr),
      std::move(source.canonical_name),
      std::move(source.canonical_type),
      source.declaration_begin_byte,
      source.declaration_end_byte,
      source.identity_begin_byte,
      source.identity_end_byte,
  };
}

ProducerViewCopyTensorObservation producer_tensor(
    ViewCopyTensorTrace&& source) {
  return {
      source.resolved_tensor_type,
      source.resolved_static_affine_layout,
      source.initializer_parameter_bound,
      source.engine_pointee_const,
      source.engine_parameter_ordinal,
      std::move(source.canonical_usr),
      std::move(source.canonical_name),
      std::move(source.canonical_type),
      std::move(source.tensor_template_path),
      std::move(source.initializer_callee_usr),
      std::move(source.initializer_callee_name),
      std::move(source.initializer_callee_path),
      std::move(source.layout_canonical_type),
      std::move(source.layout_template_path),
      source.declaration_begin_byte,
      source.declaration_end_byte,
      source.identity_begin_byte,
      source.identity_end_byte,
      source.rank,
      source.leaf_rank,
      source.size,
      source.cosize,
      producer_hierarchy(std::move(source.shape)),
      producer_hierarchy(std::move(source.stride)),
  };
}

ProducerViewCopyObservation producer_view_copy(ViewCopyTrace&& source) {
  ProducerViewCopyObservation destination;
  destination.selected = source.selected;
  destination.ambiguous = source.ambiguous;
  destination.resolved_function = source.resolved_function;
  destination.resolved_copy = source.resolved_copy;
  destination.cuda_host = source.cuda_host;
  destination.cuda_device = source.cuda_device;
  destination.cuda_global = source.cuda_global;
  destination.force_inline = source.force_inline;
  destination.canonical_usr = std::move(source.canonical_usr);
  destination.canonical_name = std::move(source.canonical_name);
  destination.canonical_type = std::move(source.canonical_type);
  destination.copy_callee_usr = std::move(source.copy_callee_usr);
  destination.copy_callee_name = std::move(source.copy_callee_name);
  destination.copy_callee_path = std::move(source.copy_callee_path);
  destination.declaration_begin_byte = source.declaration_begin_byte;
  destination.declaration_end_byte = source.declaration_end_byte;
  destination.identity_begin_byte = source.identity_begin_byte;
  destination.identity_end_byte = source.identity_end_byte;
  destination.copy_begin_byte = source.copy_begin_byte;
  destination.copy_end_byte = source.copy_end_byte;
  destination.source_tensor_ordinal = source.source_tensor_ordinal;
  destination.destination_tensor_ordinal = source.destination_tensor_ordinal;
  destination.parameters.reserve(source.parameters.size());
  for (ViewCopyParameterTrace& parameter : source.parameters) {
    destination.parameters.push_back(producer_parameter(std::move(parameter)));
  }
  destination.tensors.reserve(source.tensors.size());
  for (ViewCopyTensorTrace& tensor : source.tensors) {
    destination.tensors.push_back(producer_tensor(std::move(tensor)));
  }
  return destination;
}

void commit_pass_review(ClangPassReview&& source,
                        ProducerPassObservation& destination) {
  destination.invocation_succeeded = source.invocation_succeeded;
  destination.policy_installed =
      source.policy_install_status ==
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
  destination.include_edges.reserve(
      source.vfs_observation.include_edges.size());
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
  destination.view_copy = producer_view_copy(std::move(source.view_copy_trace));
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
        session.request_semantic_limit(CompileSemanticLimit::kSourceFiles) +
        session.request_semantic_limit(CompileSemanticLimit::kHeaderFiles);
    ImportedVfsObservationLimits observation_limits;
    observation_limits.max_opened_file_count =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(
            opened_limit, kImportedVfsMaximumObservedFileCount));
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
          static_cast<std::uint32_t>(session.request_semantic_limit(
              CompileSemanticLimit::kDiagnostics)),
          session.maximum_output_byte_length(), review));
      if (review.policy_install_status !=
          CppCutePreprocessorPolicyInstallStatus::kInstalled) {
        report_native_diagnostic(
            NativeDiagnosticCode::kProducerPolicyInstallFailure);
        result.status = ProducerReviewStatus::kInternalError;
        return result;
      }
      if (review.vfs_failed) {
        report_native_diagnostic(producer_vfs_diagnostic(review.vfs_failure));
        result.status = ProducerReviewStatus::kVfsError;
        return result;
      }
      if (review.diagnostic_capture_failed) {
        report_native_diagnostic(
            NativeDiagnosticCode::kProducerDiagnosticCaptureLimit);
        result.status = ProducerReviewStatus::kResourceLimit;
        return result;
      }
      if (review.frontend_work_limit_exceeded) {
        report_native_diagnostic(
            NativeDiagnosticCode::kProducerFrontendWorkLimit);
        result.status = ProducerReviewStatus::kResourceLimit;
        return result;
      }
      if (!review.invocation_succeeded && review.clang_error_count == 0U &&
          !review.policy_failed) {
        report_native_diagnostic(
            NativeDiagnosticCode::kProducerInvocationFailure);
        result.status = ProducerReviewStatus::kInternalError;
        return result;
      }
      const bool layout_extraction_failed =
          anchor.kind == SourceAnchorKind::kLayoutVariable &&
          (!review.layout_trace.selected ||
           !review.layout_trace.resolved_layout_type ||
           !review.layout_trace.resolved_static_affine_layout ||
           review.layout_trace.canonical_usr.empty());
      const bool view_copy_extraction_failed =
          anchor.kind == SourceAnchorKind::kViewCopyFunction &&
          (!review.view_copy_trace.selected ||
           review.view_copy_trace.ambiguous ||
           !review.view_copy_trace.resolved_function ||
           !review.view_copy_trace.resolved_copy ||
           review.view_copy_trace.canonical_usr.empty() ||
           !view_copy_origins_opened(review.view_copy_trace,
                                     review.vfs_observation));
      const bool extraction_failed =
          review.invocation_succeeded && !review.policy_failed &&
          review.clang_error_count == 0U &&
          (layout_extraction_failed || view_copy_extraction_failed);
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
      if (anchor.kind == SourceAnchorKind::kLayoutVariable) {
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
      } else {
        result.shared_surface_converged = same_view_copy(
            result.passes[0].view_copy, result.passes[1].view_copy);
      }
      if (!result.shared_surface_converged) {
        ++result.blocking_diagnostic_pass_count;
      }
    }
    result.status =
        result.blocking_diagnostic_pass_count == 0U
            ? ProducerReviewStatus::kReviewComplete
            : ProducerReviewStatus::kReviewCompleteWithBlockingDiagnostics;
    return result;
  } catch (const std::bad_alloc&) {
    result.status = ProducerReviewStatus::kResourceLimit;
    return result;
  } catch (...) {
    report_native_diagnostic(NativeDiagnosticCode::kProducerException);
    result.status = ProducerReviewStatus::kInternalError;
    return result;
  }
}

}  // namespace browsergrad::cpp_cute
