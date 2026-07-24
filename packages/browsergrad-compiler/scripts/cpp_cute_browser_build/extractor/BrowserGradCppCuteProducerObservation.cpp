#include "BrowserGradCppCuteProducerObservation.h"

#include <algorithm>
#include <string>
#include <string_view>

namespace browsergrad::cpp_cute {

bool empty_producer_layout_observation(
    const ProducerLayoutObservation& observation) noexcept {
  const auto empty_hierarchy = [](const ProducerIntegerHierarchy& hierarchy) {
    return !hierarchy.tuple && hierarchy.value == 0 &&
           hierarchy.elements.empty();
  };
  return !observation.selected && !observation.resolved_layout_type &&
         !observation.resolved_static_affine_layout &&
         observation.canonical_usr.empty() &&
         observation.canonical_name.empty() &&
         observation.canonical_type.empty() &&
         observation.initializer_callee.empty() &&
         observation.identity_begin_byte == 0U &&
         observation.identity_end_byte == 0U && observation.rank == 0U &&
         observation.leaf_rank == 0U && observation.size == 0 &&
         observation.cosize == 0 && empty_hierarchy(observation.shape) &&
         empty_hierarchy(observation.stride);
}

bool empty_producer_view_copy_observation(
    const ProducerViewCopyObservation& observation) noexcept {
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
         observation.copy_begin_byte == 0U &&
         observation.copy_end_byte == 0U &&
         observation.source_tensor_ordinal == 0U &&
         observation.destination_tensor_ordinal == 0U &&
         observation.parameters.empty() && observation.tensors.empty();
}

bool producer_view_copy_origins_opened(
    const ProducerPassObservation& pass) noexcept {
  const auto opened = [&pass](const std::string& path) {
    return !path.empty() &&
           std::find(pass.opened_file_paths.begin(),
                     pass.opened_file_paths.end(),
                     path) != pass.opened_file_paths.end();
  };
  if (!opened(pass.view_copy.copy_callee_path)) return false;
  return std::all_of(
      pass.view_copy.tensors.begin(), pass.view_copy.tensors.end(),
      [&opened](const ProducerViewCopyTensorObservation& tensor) {
        return opened(tensor.tensor_template_path) &&
               opened(tensor.layout_template_path) &&
               opened(tensor.initializer_callee_path);
      });
}

std::string_view producer_view_copy_extraction_failure_message(
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
  for (const ProducerViewCopyTensorObservation& tensor : view_copy.tensors) {
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
  if (!producer_view_copy_origins_opened(pass)) {
    return "selected view-copy function references an unopened CuTe origin";
  }
  if (!view_copy.resolved_function) {
    return "selected view-copy function failed final semantic validation";
  }
  return "selected frontend extraction did not yield a supported resolved "
         "semantic fact";
}

bool producer_pass_observation_failed(
    const ProducerPassObservation& pass) noexcept {
  const bool resolved_layout = pass.layout.selected &&
                               pass.layout.resolved_layout_type &&
                               pass.layout.resolved_static_affine_layout &&
                               !pass.layout.canonical_usr.empty();
  const bool resolved_view_copy =
      pass.view_copy.selected && !pass.view_copy.ambiguous &&
      pass.view_copy.resolved_function && pass.view_copy.resolved_copy &&
      !pass.view_copy.canonical_usr.empty() &&
      producer_view_copy_origins_opened(pass);
  return !pass.invocation_succeeded || pass.policy_failed ||
         pass.policy_violation_count != 0U || pass.clang_error_count != 0U ||
         !((resolved_layout &&
            empty_producer_view_copy_observation(pass.view_copy)) ||
           (resolved_view_copy &&
            empty_producer_layout_observation(pass.layout)));
}

bool producer_rejected_pass_failed(
    const ProducerReviewResult& producer,
    const std::size_t pass_index) noexcept {
  if (pass_index >= producer.completed_pass_count) return false;
  if (producer_pass_observation_failed(producer.passes[pass_index])) {
    return true;
  }
  return pass_index == 1U && producer.completed_pass_count == 2U &&
         !producer.shared_surface_converged;
}

}  // namespace browsergrad::cpp_cute
