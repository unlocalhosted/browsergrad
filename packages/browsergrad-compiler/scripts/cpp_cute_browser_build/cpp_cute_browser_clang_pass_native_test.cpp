#include "extractor/BrowserGradCppCuteClangAction.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> g_file_system;

#define BG_CHECK(condition)                                                   \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::fprintf(stderr, "clang-pass check failed at line %d: %s\n",      \
                   __LINE__, #condition);                                     \
      return 1;                                                               \
    }                                                                         \
  } while (false)

void install_source(const std::string_view source) {
  g_file_system = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  g_file_system->addFile(
      "/workspace/main.cu", 0,
      llvm::MemoryBuffer::getMemBufferCopy(source, "/workspace/main.cu"));
}

std::vector<std::string> arguments(const std::string_view pass) {
  return {
      "clang++",
      "-x",
      "cuda",
      pass == "device" ? "--cuda-device-only" : "--cuda-host-only",
      "--target=x86_64-unknown-linux-gnu",
      "--cuda-gpu-arch=sm_80",
      "-nocudainc",
      "-nogpulib",
      "-fsyntax-only",
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
  for (const std::string_view pass : {std::string_view("device"),
                                      std::string_view("host")}) {
    ClangPassReview review;
    const std::vector<std::string> argv = arguments(pass);
    BG_CHECK(run_cpp_cute_clang_pass_for_review(
        argv, anchor, {}, ImportedVfsObservationLimits{}, 1024U,
        1024U * 1024U, review));
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

  constexpr std::string_view rejected_source =
      "const char* temporal = __DATE__;\n";
  install_source(rejected_source);
  ClangPassReview rejected;
  const std::vector<std::string> host = arguments("host");
  SourceAnchor rejected_anchor{"/workspace/main.cu", 12U, 20U};
  BG_CHECK(!run_cpp_cute_clang_pass_for_review(
      host, rejected_anchor, {}, ImportedVfsObservationLimits{}, 1024U,
      1024U * 1024U, rejected));
  BG_CHECK(rejected.policy_failed);
  BG_CHECK(rejected.policy_violation_count == 1U);
  BG_CHECK(rejected.clang_error_count != 0U);
  BG_CHECK(!rejected.diagnostic_capture_failed);
  BG_CHECK(!rejected.diagnostics.empty());
  BG_CHECK(rejected.diagnostics[0].custom);
  BG_CHECK(rejected.diagnostics[0].custom_code ==
           CustomDiagnosticCode::kTemporalMacroForbidden);
  BG_CHECK(rejected.diagnostics[0].stage ==
           RawDiagnosticStage::kPreprocessor);
  BG_CHECK(rejected.diagnostics[0].severity ==
           RawDiagnosticSeverity::kError);
  BG_CHECK(rejected.diagnostics[0].virtual_path == "/workspace/main.cu");

  install_source(source);
  ClangPassReview clean_after_rejection;
  BG_CHECK(run_cpp_cute_clang_pass_for_review(
      host, anchor, {}, ImportedVfsObservationLimits{},
      1024U, 1024U * 1024U,
      clean_after_rejection));
  BG_CHECK(!clean_after_rejection.policy_failed);
  BG_CHECK(clean_after_rejection.policy_violation_count == 0U);
  return 0;
}
