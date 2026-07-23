#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "extractor/BrowserGradCppCuteClangAction.h"
#include "extractor/BrowserGradCppCuteMetrics.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace {

llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> g_file_system;

#define BG_CHECK(condition)                                            \
  do {                                                                 \
    if (!(condition)) {                                                \
      std::fprintf(stderr, "clang-pass check failed at line %d: %s\n", \
                   __LINE__, #condition);                              \
      return 1;                                                        \
    }                                                                  \
  } while (false)

void install_source(const std::string_view source,
                    const std::string_view header_path = {},
                    const std::string_view header_source = {}) {
  g_file_system = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  g_file_system->addFile(
      "/workspace/main.cu", 0,
      llvm::MemoryBuffer::getMemBufferCopy(source, "/workspace/main.cu"));
  if (!header_path.empty()) {
    g_file_system->addFile(
        header_path, 0,
        llvm::MemoryBuffer::getMemBufferCopy(header_source, header_path));
  }
}

std::vector<std::string> arguments(const std::string_view pass) {
  return {
      "clang++",
      "--no-default-config",
      "-x",
      "cuda",
      "-std=c++17",
      pass == "device" ? "--cuda-device-only" : "--cuda-host-only",
      "--target=x86_64-unknown-linux-gnu",
      "--cuda-gpu-arch=sm_80",
      "-resource-dir",
      "/toolchain/clang/lib/clang/22",
      "--cuda-path=/toolchain/cuda",
      "--cuda-path-ignore-env",
      "-nostdinc",
      "-nostdinc++",
      "-nogpuinc",
      "-nogpulib",
      "-iquote",
      "/workspace",
      "-isystem",
      "/toolchain/clang/lib/clang/22/include",
      "-isystem",
      "/toolchain/cuda/include",
      "-isystem",
      "/toolchain/cutlass/include",
      "-isystem",
      "/toolchain/cxx/include/c++/v1",
      "-isystem",
      "/toolchain/sysroot/usr/include",
      "-Werror=builtin-macro-redefined",
      "-Werror=date-time",
      "-Werror=macro-redefined",
      "-DCUTE_SM80_ENABLED=1",
      "-fsyntax-only",
      "-ferror-limit=4096",
      "-fconstexpr-steps=10000000",
      "-ftemplate-depth=1024",
      "/workspace/main.cu",
  };
}

}  // namespace

namespace browsergrad::cpp_cute {

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> imported_closed_vfs() {
  return g_file_system;
}

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> imported_closed_vfs(
    std::shared_ptr<ImportedVfsObserver>) {
  return g_file_system;
}

}  // namespace browsergrad::cpp_cute

int main() {
  using namespace browsergrad::cpp_cute;
  constexpr std::string_view source =
      "namespace cute {\n"
      "template <auto Value> struct C {};\n"
      "template <class... Values> struct tuple {};\n"
      "template <class... Values> using Shape = tuple<Values...>;\n"
      "template <class... Values> using Stride = tuple<Values...>;\n"
      "template <class ShapeType, class StrideType> struct Layout {};\n"
      "}\n"
      "auto layout = cute::Layout<\n"
      "  cute::Shape<cute::C<4>, cute::C<2>>,\n"
      "  cute::Stride<cute::C<1>, cute::C<4>>>{};\n";
  install_source(source);
  const std::size_t begin = source.find("layout");
  BG_CHECK(begin != std::string_view::npos);
  SourceAnchor anchor{
      "/workspace/main.cu",
      static_cast<std::uint32_t>(begin),
      static_cast<std::uint32_t>(begin + std::string_view("layout").size()),
  };
  constexpr FrontendWorkLimitsV1 frontend_limits{
      64U, 1'000'000U, 1'000'000U, 1'000'000U, 1'000'000U, 1'000'000U, 1'000U,
  };
  BG_CHECK(begin_frontend_work_invocation(frontend_limits));
  for (const std::string_view pass :
       {std::string_view("device"), std::string_view("host")}) {
    ClangPassReview review;
    const std::vector<std::string> argv = arguments(pass);
    const bool succeeded = run_cpp_cute_clang_pass_for_review(
        argv, anchor, {}, ImportedVfsObservationLimits{}, 1024U, 1024U * 1024U,
        review);
    if (!succeeded) {
      std::fprintf(
          stderr, "%.*s pass failed: invocation=%d policy=%d clang-errors=%u\n",
          static_cast<int>(pass.size()), pass.data(),
          review.invocation_succeeded ? 1 : 0, review.policy_failed ? 1 : 0,
          review.clang_error_count);
      for (const ClangDiagnosticObservation& diagnostic : review.diagnostics) {
        std::fprintf(stderr, "diagnostic: %s\n",
                     diagnostic.rendered_message.c_str());
      }
    }
    BG_CHECK(succeeded);
    BG_CHECK(review.invocation_succeeded);
    BG_CHECK(review.policy_install_status ==
             CppCutePreprocessorPolicyInstallStatus::kInstalled);
    BG_CHECK(!review.policy_failed);
    BG_CHECK(!review.vfs_failed);
    BG_CHECK(review.clang_error_count == 0U);
    BG_CHECK(review.layout_trace.selected);
    BG_CHECK(review.layout_trace.resolved_layout_type);
    BG_CHECK(review.layout_trace.resolved_static_affine_layout);
    BG_CHECK(review.layout_trace.canonical_name == "layout");
    BG_CHECK(review.layout_trace.rank == 2U);
    BG_CHECK(review.layout_trace.leaf_rank == 2U);
    BG_CHECK(review.layout_trace.size == 8);
    BG_CHECK(review.layout_trace.cosize == 8);
    BG_CHECK(review.layout_trace.shape.tuple);
    BG_CHECK(review.layout_trace.shape.elements.size() == 2U);
    BG_CHECK(review.layout_trace.shape.elements[0].value == 4);
    BG_CHECK(review.layout_trace.shape.elements[1].value == 2);
  }
  BG_CHECK(complete_frontend_work_invocation(2U));
  BG_CHECK(frontend_work_metrics_ready());
  const FrontendWorkMetricsRecordV1& accepted_work =
      frontend_work_metrics_record_for_testing();
  BG_CHECK(accepted_work.preprocessed_tokens != 0U);
  BG_CHECK(accepted_work.ast_nodes != 0U);
  BG_CHECK(accepted_work.template_instantiations != 0U);
  BG_CHECK(accepted_work.template_depth != 0U);
  BG_CHECK(accepted_work.completed_semantic_passes == 2U);

  constexpr std::string_view view_copy_header_source =
      "namespace cute {\n"
      "template <auto Value> struct C {};\n"
      "template <class... Values> struct tuple {};\n"
      "template <class... Values> using Shape = tuple<Values...>;\n"
      "template <class... Values> using Stride = tuple<Values...>;\n"
      "template <class ShapeType, class StrideType> struct Layout {};\n"
      "template <class Engine, class LayoutType> struct Tensor {};\n"
      "template <class Engine, class LayoutType>\n"
      "__attribute__((host, device)) Tensor<Engine, LayoutType>\n"
      "make_tensor(Engine, LayoutType) { return {}; }\n"
      "template <class Source, class Destination>\n"
      "__attribute__((device)) void copy(const Source&, Destination&) {}\n"
      "}\n";
  constexpr std::string_view view_copy_source =
      "#include <cute/tensor.hpp>\n"
      "using SourceLayout = cute::Layout<\n"
      "  cute::Shape<cute::C<2>, cute::C<3>>,\n"
      "  cute::Stride<cute::C<1>, cute::C<2>>>;\n"
      "using DestinationLayout = cute::Layout<\n"
      "  cute::Shape<cute::C<2>, cute::C<3>>,\n"
      "  cute::Stride<cute::C<3>, cute::C<1>>>;\n"
      "__attribute__((device))\n"
      "void copy_views(const float* source, float* destination) {\n"
      "  auto source_tensor = cute::make_tensor(source, SourceLayout{});\n"
      "  auto destination_tensor =\n"
      "      cute::make_tensor(destination, DestinationLayout{});\n"
      "  cute::copy(source_tensor, destination_tensor);\n"
      "}\n";
  reset_frontend_work_metrics();
  BG_CHECK(begin_frontend_work_invocation(frontend_limits));
  install_source(view_copy_source, "/toolchain/cutlass/include/cute/tensor.hpp",
                 view_copy_header_source);
  const std::size_t view_copy_begin = view_copy_source.find("copy_views");
  BG_CHECK(view_copy_begin != std::string_view::npos);
  const SourceAnchor view_copy_anchor{
      "/workspace/main.cu",
      static_cast<std::uint32_t>(view_copy_begin),
      static_cast<std::uint32_t>(view_copy_begin +
                                 std::string_view("copy_views").size()),
      SourceAnchorKind::kViewCopyFunction,
  };
  for (const std::string_view pass :
       {std::string_view("device"), std::string_view("host")}) {
    ClangPassReview review;
    const std::vector<std::string> argv = arguments(pass);
    const bool succeeded = run_cpp_cute_clang_pass_for_review(
        argv, view_copy_anchor, {}, ImportedVfsObservationLimits{}, 1024U,
        1024U * 1024U, review);
    if (!succeeded) {
      std::fprintf(stderr,
                   "%.*s view-copy pass failed: invocation=%d policy=%d "
                   "clang-errors=%u\n",
                   static_cast<int>(pass.size()), pass.data(),
                   review.invocation_succeeded ? 1 : 0,
                   review.policy_failed ? 1 : 0, review.clang_error_count);
      for (const ClangDiagnosticObservation& diagnostic : review.diagnostics) {
        std::fprintf(stderr, "diagnostic: %s\n",
                     diagnostic.rendered_message.c_str());
      }
    }
    BG_CHECK(succeeded);
    BG_CHECK(review.invocation_succeeded);
    BG_CHECK(!review.policy_failed);
    BG_CHECK(!review.vfs_failed);
    BG_CHECK(review.clang_error_count == 0U);
    const ViewCopyTrace& trace = review.view_copy_trace;
    BG_CHECK(trace.selected);
    BG_CHECK(!trace.ambiguous);
    BG_CHECK(trace.resolved_function);
    BG_CHECK(trace.resolved_copy);
    BG_CHECK(!trace.cuda_host);
    BG_CHECK(trace.cuda_device);
    BG_CHECK(!trace.cuda_global);
    BG_CHECK(trace.canonical_name == "copy_views");
    BG_CHECK(!trace.canonical_usr.empty());
    BG_CHECK(trace.parameters.size() == 2U);
    BG_CHECK(trace.parameters[0].ordinal == 0U);
    BG_CHECK(trace.parameters[0].resolved_pointer);
    BG_CHECK(trace.parameters[0].resolved_float_pointee);
    BG_CHECK(trace.parameters[0].pointee_const);
    BG_CHECK(trace.parameters[1].ordinal == 1U);
    BG_CHECK(trace.parameters[1].resolved_pointer);
    BG_CHECK(trace.parameters[1].resolved_float_pointee);
    BG_CHECK(!trace.parameters[1].pointee_const);
    BG_CHECK(trace.tensors.size() == 2U);
    BG_CHECK(trace.source_tensor_ordinal == 0U);
    BG_CHECK(trace.destination_tensor_ordinal == 1U);
    BG_CHECK(trace.copy_callee_name == "cute::copy");
    BG_CHECK(!trace.copy_callee_usr.empty());
    BG_CHECK(trace.copy_callee_path ==
             "/toolchain/cutlass/include/cute/tensor.hpp");
    for (const ViewCopyTensorTrace& tensor : trace.tensors) {
      BG_CHECK(tensor.resolved_tensor_type);
      BG_CHECK(tensor.resolved_static_affine_layout);
      BG_CHECK(tensor.initializer_parameter_bound);
      BG_CHECK(tensor.initializer_callee_name == "cute::make_tensor");
      BG_CHECK(!tensor.initializer_callee_usr.empty());
      BG_CHECK(tensor.tensor_template_path ==
               "/toolchain/cutlass/include/cute/tensor.hpp");
      BG_CHECK(tensor.layout_template_path ==
               "/toolchain/cutlass/include/cute/tensor.hpp");
      BG_CHECK(tensor.initializer_callee_path ==
               "/toolchain/cutlass/include/cute/tensor.hpp");
      BG_CHECK(tensor.rank == 2U);
      BG_CHECK(tensor.leaf_rank == 2U);
      BG_CHECK(tensor.size == 6);
      BG_CHECK(tensor.cosize == 6);
      BG_CHECK(tensor.shape.tuple);
      BG_CHECK(tensor.shape.elements.size() == 2U);
      BG_CHECK(tensor.stride.tuple);
      BG_CHECK(tensor.stride.elements.size() == 2U);
    }
    BG_CHECK(trace.tensors[0].engine_parameter_ordinal == 0U);
    BG_CHECK(trace.tensors[0].engine_pointee_const);
    BG_CHECK(trace.tensors[1].engine_parameter_ordinal == 1U);
    BG_CHECK(!trace.tensors[1].engine_pointee_const);
    BG_CHECK(trace.declaration_begin_byte < trace.identity_begin_byte);
    BG_CHECK(trace.identity_end_byte < trace.declaration_end_byte);
    BG_CHECK(trace.copy_begin_byte < trace.copy_end_byte);
  }
  BG_CHECK(complete_frontend_work_invocation(2U));
  BG_CHECK(frontend_work_metrics_ready());

  std::string decoy_copy_source(view_copy_source);
  const std::string_view semantic_call =
      "cute::copy(source_tensor, destination_tensor);";
  const std::size_t semantic_call_begin =
      decoy_copy_source.rfind(semantic_call);
  BG_CHECK(semantic_call_begin != std::string::npos);
  decoy_copy_source.replace(semantic_call_begin, semantic_call.size(),
                            "(void)source_tensor; (void)destination_tensor;");
  reset_frontend_work_metrics();
  BG_CHECK(begin_frontend_work_invocation(frontend_limits));
  install_source(decoy_copy_source,
                 "/toolchain/cutlass/include/cute/tensor.hpp",
                 view_copy_header_source);
  ClangPassReview decoy;
  BG_CHECK(run_cpp_cute_clang_pass_for_review(
      arguments("host"), view_copy_anchor, {}, ImportedVfsObservationLimits{},
      1024U, 1024U * 1024U, decoy));
  BG_CHECK(decoy.view_copy_trace.selected);
  BG_CHECK(!decoy.view_copy_trace.resolved_copy);
  BG_CHECK(!decoy.view_copy_trace.resolved_function);
  BG_CHECK(complete_frontend_work_invocation(1U));
  BG_CHECK(frontend_work_metrics_ready());

  std::string spoofed_header(view_copy_header_source);
  constexpr std::string_view upstream_copy =
      "template <class Source, class Destination>\n"
      "__attribute__((device)) void copy(const Source&, Destination&) {}\n";
  const std::size_t upstream_copy_begin = spoofed_header.find(upstream_copy);
  BG_CHECK(upstream_copy_begin != std::string::npos);
  spoofed_header.erase(upstream_copy_begin, upstream_copy.size());
  std::string spoofed_copy_source(view_copy_source);
  constexpr std::string_view selected_function_marker =
      "__attribute__((device))\n"
      "void copy_views";
  const std::size_t selected_function_begin =
      spoofed_copy_source.find(selected_function_marker);
  BG_CHECK(selected_function_begin != std::string::npos);
  spoofed_copy_source.insert(selected_function_begin,
                             "namespace cute {\n"
                             "__attribute__((device)) void copy(\n"
                             "    Tensor<const float*, ::SourceLayout>&,\n"
                             "    Tensor<float*, ::DestinationLayout>&) {}\n"
                             "}\n");
  reset_frontend_work_metrics();
  BG_CHECK(begin_frontend_work_invocation(frontend_limits));
  install_source(spoofed_copy_source,
                 "/toolchain/cutlass/include/cute/tensor.hpp", spoofed_header);
  const std::size_t spoofed_anchor_begin =
      spoofed_copy_source.find("copy_views");
  BG_CHECK(spoofed_anchor_begin != std::string::npos);
  const SourceAnchor spoofed_anchor{
      "/workspace/main.cu",
      static_cast<std::uint32_t>(spoofed_anchor_begin),
      static_cast<std::uint32_t>(spoofed_anchor_begin +
                                 std::string_view("copy_views").size()),
      SourceAnchorKind::kViewCopyFunction,
  };
  ClangPassReview spoofed;
  BG_CHECK(run_cpp_cute_clang_pass_for_review(
      arguments("host"), spoofed_anchor, {}, ImportedVfsObservationLimits{},
      1024U, 1024U * 1024U, spoofed));
  BG_CHECK(spoofed.view_copy_trace.selected);
  BG_CHECK(spoofed.view_copy_trace.resolved_copy);
  BG_CHECK(spoofed.view_copy_trace.copy_callee_path == "/workspace/main.cu");
  BG_CHECK(!spoofed.view_copy_trace.resolved_function);
  BG_CHECK(complete_frontend_work_invocation(1U));
  BG_CHECK(frontend_work_metrics_ready());

  constexpr std::string_view macro_include_source =
      "#define BG_HEADER \"nested/../dependency.h\"\n"
      "#include BG_HEADER\n"
      "namespace cute {\n"
      "template <auto Value> struct C {};\n"
      "template <class... Values> struct tuple {};\n"
      "template <class... Values> using Shape = tuple<Values...>;\n"
      "template <class... Values> using Stride = tuple<Values...>;\n"
      "template <class ShapeType, class StrideType> struct Layout {};\n"
      "}\n"
      "auto layout = cute::Layout<\n"
      "  cute::Shape<cute::C<4>, cute::C<2>>,\n"
      "  cute::Stride<cute::C<1>, cute::C<4>>>{};\n";
  reset_frontend_work_metrics();
  BG_CHECK(begin_frontend_work_invocation(frontend_limits));
  install_source(macro_include_source, "/workspace/dependency.h",
                 "static_assert(true, \"macro include reached\");\n");
  const std::size_t macro_layout_begin = macro_include_source.find("layout");
  BG_CHECK(macro_layout_begin != std::string_view::npos);
  const SourceAnchor macro_anchor{
      "/workspace/main.cu",
      static_cast<std::uint32_t>(macro_layout_begin),
      static_cast<std::uint32_t>(macro_layout_begin +
                                 std::string_view("layout").size()),
  };
  ClangPassReview macro_include;
  const bool macro_include_succeeded = run_cpp_cute_clang_pass_for_review(
      arguments("host"), macro_anchor, {}, ImportedVfsObservationLimits{},
      1024U, 1024U * 1024U, macro_include);
  if (!macro_include_succeeded) {
    std::fprintf(stderr,
                 "macro include failed: invocation=%d policy=%d vfs=%d "
                 "vfs-failure=%u clang-errors=%u\n",
                 macro_include.invocation_succeeded ? 1 : 0,
                 macro_include.policy_failed ? 1 : 0,
                 macro_include.vfs_failed ? 1 : 0,
                 static_cast<unsigned>(macro_include.vfs_failure),
                 macro_include.clang_error_count);
    for (const ClangDiagnosticObservation& diagnostic :
         macro_include.diagnostics) {
      std::fprintf(stderr, "diagnostic: %s\n",
                   diagnostic.rendered_message.c_str());
    }
  }
  BG_CHECK(macro_include_succeeded);
  BG_CHECK(!macro_include.vfs_failed);
  BG_CHECK(macro_include.vfs_failure == ImportedVfsObserverFailure::kNone);
  BG_CHECK(complete_frontend_work_invocation(1U));
  BG_CHECK(frontend_work_metrics_ready());

  reset_frontend_work_metrics();
  BG_CHECK(begin_frontend_work_invocation(frontend_limits));
  install_source(source, "/workspace/dependency.h",
                 "static_assert(true, \"forced include reached\");\n");
  std::vector<std::string> forced_arguments = arguments("host");
  forced_arguments.insert(forced_arguments.end() - 1,
                          {"-include", "/workspace/dependency.h"});
  constexpr std::array<ClangForcedIncludeObservation, 1U> forced_includes{{
      {"/workspace/dependency.h", 1U},
  }};
  ClangPassReview forced_include;
  BG_CHECK(run_cpp_cute_clang_pass_for_review(
      forced_arguments, anchor, forced_includes, ImportedVfsObservationLimits{},
      1024U, 1024U * 1024U, forced_include));
  BG_CHECK(!forced_include.vfs_failed);
  BG_CHECK(forced_include.vfs_failure == ImportedVfsObserverFailure::kNone);
  BG_CHECK(complete_frontend_work_invocation(1U));
  BG_CHECK(frontend_work_metrics_ready());

  constexpr std::string_view rejected_source =
      "const char* temporal = __DATE__;\n";
  reset_frontend_work_metrics();
  BG_CHECK(begin_frontend_work_invocation(frontend_limits));
  install_source(rejected_source);
  ClangPassReview rejected;
  const std::vector<std::string> host = arguments("host");
  SourceAnchor rejected_anchor{"/workspace/main.cu", 12U, 20U};
  BG_CHECK(!run_cpp_cute_clang_pass_for_review(host, rejected_anchor, {},
                                               ImportedVfsObservationLimits{},
                                               1024U, 1024U * 1024U, rejected));
  BG_CHECK(rejected.policy_failed);
  BG_CHECK(rejected.policy_violation_count == 1U);
  BG_CHECK(rejected.clang_error_count != 0U);
  BG_CHECK(!rejected.diagnostic_capture_failed);
  BG_CHECK(!rejected.diagnostics.empty());
  const auto custom_diagnostic =
      std::find_if(rejected.diagnostics.begin(), rejected.diagnostics.end(),
                   [](const ClangDiagnosticObservation& diagnostic) {
                     return diagnostic.custom;
                   });
  BG_CHECK(custom_diagnostic != rejected.diagnostics.end());
  BG_CHECK(custom_diagnostic->custom_code ==
           CustomDiagnosticCode::kTemporalMacroForbidden);
  BG_CHECK(custom_diagnostic->stage == RawDiagnosticStage::kPreprocessor);
  BG_CHECK(custom_diagnostic->severity == RawDiagnosticSeverity::kError);
  BG_CHECK(custom_diagnostic->virtual_path == "/workspace/main.cu");
  BG_CHECK(complete_frontend_work_invocation(1U));
  BG_CHECK(frontend_work_metrics_ready());
  BG_CHECK(
      frontend_work_metrics_record_for_testing().completed_semantic_passes ==
      1U);

  reset_frontend_work_metrics();
  BG_CHECK(begin_frontend_work_invocation(frontend_limits));
  install_source(source);
  ClangPassReview invalid_driver_argument;
  std::vector<std::string> invalid_driver_arguments = host;
  invalid_driver_arguments.insert(
      invalid_driver_arguments.end() - 1,
      "-fdefinitely-unsupported-browsergrad-option");
  BG_CHECK(!run_cpp_cute_clang_pass_for_review(
      invalid_driver_arguments, anchor, {}, ImportedVfsObservationLimits{},
      1024U, 1024U * 1024U, invalid_driver_argument));
  BG_CHECK(!invalid_driver_argument.invocation_succeeded);
  BG_CHECK(invalid_driver_argument.policy_install_status ==
           CppCutePreprocessorPolicyInstallStatus::kInstalled);
  BG_CHECK(!invalid_driver_argument.policy_failed);
  BG_CHECK(invalid_driver_argument.clang_error_count != 0U);
  BG_CHECK(!invalid_driver_argument.diagnostic_capture_failed);
  BG_CHECK(std::any_of(
      invalid_driver_argument.diagnostics.begin(),
      invalid_driver_argument.diagnostics.end(),
      [](const ClangDiagnosticObservation& diagnostic) {
        return diagnostic.severity == RawDiagnosticSeverity::kError ||
               diagnostic.severity == RawDiagnosticSeverity::kFatal;
      }));
  BG_CHECK(complete_frontend_work_invocation(1U));
  BG_CHECK(frontend_work_metrics_ready());

  reset_frontend_work_metrics();
  BG_CHECK(begin_frontend_work_invocation(frontend_limits));
  install_source(source);
  ClangPassReview clean_after_rejection;
  BG_CHECK(run_cpp_cute_clang_pass_for_review(
      host, anchor, {}, ImportedVfsObservationLimits{}, 1024U, 1024U * 1024U,
      clean_after_rejection));
  BG_CHECK(!clean_after_rejection.policy_failed);
  BG_CHECK(clean_after_rejection.policy_violation_count == 0U);
  BG_CHECK(complete_frontend_work_invocation(1U));
  BG_CHECK(frontend_work_metrics_ready());
  BG_CHECK(frontend_work_metrics_record_for_testing().ast_nodes != 0U);
  return 0;
}
