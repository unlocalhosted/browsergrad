#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {
bool g_metrics_healthy = true;
bool g_frontend_collecting = false;
bool g_frontend_failed = false;
std::uint64_t g_frontend_completed_passes = 0U;
}

#define BG_CPP_CUTE_RUNTIME_TESTING 1
#include "extractor/BrowserGradCppCuteRuntime.cpp"

#include "extractor/BrowserGradCppCuteCompileSession.h"
#include "extractor/BrowserGradCppCuteCompilePlan.h"
#include "extractor/BrowserGradCppCuteArtifactV3.h"
#include "extractor/BrowserGradCppCuteProducer.h"

namespace browsergrad::cpp_cute {
std::string g_producer_mode = "success";

bool allocator_metrics_healthy() { return g_metrics_healthy; }

bool begin_frontend_work_invocation(FrontendWorkLimitsV1 limits) noexcept {
  if (g_frontend_collecting || g_frontend_failed ||
      limits.max_include_depth == 0U || limits.max_macro_expansions == 0U ||
      limits.max_preprocessed_tokens == 0U || limits.max_ast_nodes == 0U ||
      limits.max_constexpr_steps == 0U ||
      limits.max_template_instantiations == 0U ||
      limits.max_template_depth == 0U) {
    g_frontend_failed = true;
    return false;
  }
  g_frontend_collecting = true;
  g_frontend_completed_passes = 0U;
  return true;
}

bool begin_frontend_work_semantic_pass() noexcept {
  return g_frontend_collecting && !g_frontend_failed;
}

bool complete_frontend_work_semantic_pass() noexcept {
  if (!g_frontend_collecting || g_frontend_failed) return false;
  ++g_frontend_completed_passes;
  return true;
}

bool complete_frontend_work_invocation(
    std::uint64_t expected_semantic_passes) noexcept {
  if (!g_frontend_collecting || g_frontend_failed ||
      expected_semantic_passes == 0U ||
      g_frontend_completed_passes != expected_semantic_passes) {
    g_frontend_failed = true;
    return false;
  }
  g_frontend_collecting = false;
  return true;
}

void fail_frontend_work_invocation() noexcept {
  g_frontend_collecting = false;
  g_frontend_failed = true;
}

void reset_frontend_work_metrics() noexcept {
  g_frontend_collecting = false;
  g_frontend_failed = false;
  g_frontend_completed_passes = 0U;
}

bool frontend_work_metrics_ready() noexcept {
  return !g_frontend_collecting && !g_frontend_failed &&
         g_frontend_completed_passes >= 1U &&
         g_frontend_completed_passes <= 2U;
}

ProducerReviewResult run_cpp_cute_producer_review(
    const PreparedCppCuteCompilePlan&,
    const DecodedCompileSession& session) noexcept {
  ProducerReviewResult result;
  if (!begin_frontend_work_semantic_pass() ||
      !complete_frontend_work_semantic_pass()) {
    result.status = ProducerReviewStatus::kInternalError;
    return result;
  }
  result.status = ProducerReviewStatus::kReviewComplete;
  result.completed_pass_count = 2U;
  result.shared_surface_converged = true;
  const EntryRequestView entry = session.entry_request();
  std::uint32_t identity_begin = 0U;
  std::uint32_t identity_end = 0U;
  const auto parsed_begin = std::from_chars(
      entry.begin_byte.data(), entry.begin_byte.data() + entry.begin_byte.size(),
      identity_begin);
  const auto parsed_end = std::from_chars(
      entry.end_byte.data(), entry.end_byte.data() + entry.end_byte.size(),
      identity_end);
  if (parsed_begin.ec != std::errc{} ||
      parsed_begin.ptr != entry.begin_byte.data() + entry.begin_byte.size() ||
      parsed_end.ec != std::errc{} ||
      parsed_end.ptr != entry.end_byte.data() + entry.end_byte.size()) {
    result.status = ProducerReviewStatus::kInternalError;
    return result;
  }
  for (ProducerPassObservation& pass : result.passes) {
    pass.invocation_succeeded = true;
    pass.policy_installed = true;
    for (std::size_t index = 0U; index < session.source_file_count(); ++index) {
      const SourceFileView source = session.source_file(index);
      std::uint64_t byte_length = 0U;
      const auto parsed = std::from_chars(
          source.byte_length.data(),
          source.byte_length.data() + source.byte_length.size(), byte_length);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != source.byte_length.data() + source.byte_length.size()) {
        result.status = ProducerReviewStatus::kInternalError;
        return result;
      }
      pass.opened_file_paths.emplace_back(source.virtual_path);
      pass.opened_files.push_back({std::string(source.virtual_path),
                                   std::string(source.content_sha256),
                                   byte_length});
    }
    pass.include_edges.push_back({
        ProducerIncludeKind::kSourceQuote,
        "/workspace/src/main.cu",
        "/workspace/src/project.hpp",
        "project.hpp",
        0U,
        22U,
        0U,
    });
    for (std::size_t index = 0U; index < session.compiler_option_count();
         ++index) {
      const CompilerOptionView option = session.compiler_option(index);
      if (option.kind != CompilerOptionKind::kForcedInclude) continue;
      pass.opened_file_paths.emplace_back(option.virtual_path);
      pass.opened_files.push_back({std::string(option.virtual_path),
                                   std::string(64U, 'a'), 1U});
      pass.include_edges.push_back({
          ProducerIncludeKind::kCompilerForced,
          {},
          std::string(option.virtual_path),
          {},
          0U,
          0U,
          option.ordinal,
      });
    }
    pass.layout = {
        true,
        true,
        true,
        "c:@layout",
        "layout",
        "cute::Layout<cute::C<2>, cute::C<1>>",
        "make_layout",
        identity_begin,
        identity_end,
        1U,
        1U,
        2,
        2,
        ProducerIntegerHierarchy{false, 2, {}},
        ProducerIntegerHierarchy{false, 1, {}},
    };
  }
  if (g_producer_mode == "layout-drift") {
    result.passes[1].layout.canonical_usr = "c:@other_layout";
  } else if (g_producer_mode == "content-drift") {
    result.passes[1].opened_files[0].content_sha256 = std::string(64U, 'f');
  } else if (g_producer_mode == "invalid-utf8") {
    result.passes[0].layout.canonical_name = std::string("layout\x80", 7U);
    result.passes[1].layout.canonical_name = result.passes[0].layout.canonical_name;
  } else if (g_producer_mode == "semantic-failure") {
    result.status =
        ProducerReviewStatus::kReviewCompleteWithBlockingDiagnostics;
    result.completed_pass_count = 1U;
    result.blocking_diagnostic_pass_count = 1U;
    result.shared_surface_converged = false;
    result.passes[0].layout = ProducerLayoutObservation{};
    result.passes[1] = ProducerPassObservation{};
  } else if (g_producer_mode == "surface-divergence") {
    result.status =
        ProducerReviewStatus::kReviewCompleteWithBlockingDiagnostics;
    result.blocking_diagnostic_pass_count = 1U;
    result.shared_surface_converged = false;
    result.passes[1].layout.canonical_usr = "c:@other_layout";
  }
  if (result.completed_pass_count == 2U &&
      (!begin_frontend_work_semantic_pass() ||
       !complete_frontend_work_semantic_pass())) {
    result.status = ProducerReviewStatus::kInternalError;
    result.completed_pass_count = 0U;
  }
  return result;
}
}  // namespace browsergrad::cpp_cute

namespace {

using namespace browsergrad::cpp_cute;

static_assert(!std::is_copy_constructible_v<DecodedCompileSession>);
static_assert(!std::is_copy_assignable_v<DecodedCompileSession>);
static_assert(!std::is_move_constructible_v<DecodedCompileSession>);
static_assert(!std::is_move_assignable_v<DecodedCompileSession>);

#define BG_CHECK(condition)                                                   \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::fprintf(stderr, "compile-session check failed at line %d: %s\n", \
                   __LINE__, #condition);                                     \
      return 1;                                                               \
    }                                                                         \
  } while (false)

void* g_input_allocation = nullptr;
void* g_result_allocation = nullptr;
bool g_input_slot_used = false;
std::size_t g_input_allocation_length = 0U;
std::size_t g_result_allocation_length = 0U;
constexpr std::uint32_t kInputWirePointer = 0x1'0000U;
constexpr std::uint32_t kResultWirePointer = 0x2'0000U;

void* test_allocate(std::size_t byte_length) {
  void* allocation = std::malloc(byte_length);
  if (allocation == nullptr) return nullptr;
  if (!g_input_slot_used) {
    g_input_slot_used = true;
    g_input_allocation = allocation;
    g_input_allocation_length = byte_length;
    return allocation;
  }
  if (g_result_allocation == nullptr) {
    g_result_allocation = allocation;
    g_result_allocation_length = byte_length;
    return allocation;
  }
  std::free(allocation);
  return nullptr;
}

void test_release(void* pointer) {
  if (pointer == nullptr) return;
  if (pointer == g_input_allocation) {
    std::free(g_input_allocation);
    g_input_allocation = nullptr;
    g_input_allocation_length = 0U;
    return;
  }
  if (pointer == g_result_allocation) {
    std::free(g_result_allocation);
    g_result_allocation = nullptr;
    g_result_allocation_length = 0U;
    return;
  }
  std::abort();
}

std::uint32_t test_wire_pointer(const void* pointer) {
  if (pointer == g_input_allocation) return kInputWirePointer;
  if (pointer == g_result_allocation) return kResultWirePointer;
  return 0U;
}

CompileSessionDecodeStatus g_decode_status =
    CompileSessionDecodeStatus::kInternalError;
CompileSessionDecodeFailure g_failure{};
std::string g_profile_hash;
std::string g_contract_hash;
std::string g_request_hash;
std::string g_request_id;
std::uint32_t g_output_limit = 0U;
bool g_ready_shape = false;
bool g_compile_plan_ready = false;

WireCompileStatus wire_status_for(CompileSessionDecodeStatus status) {
  switch (status) {
    case CompileSessionDecodeStatus::kInvalidFrame:
      return WireCompileStatus::kInvalidFrame;
    case CompileSessionDecodeStatus::kAbiMismatch:
      return WireCompileStatus::kAbiMismatch;
    case CompileSessionDecodeStatus::kResourceLimit:
      return WireCompileStatus::kResourceLimit;
    case CompileSessionDecodeStatus::kInternalError:
      return WireCompileStatus::kInternalError;
    case CompileSessionDecodeStatus::kReady:
      return WireCompileStatus::kInternalError;
  }
  return WireCompileStatus::kInternalError;
}

ArtifactV3CompileResult decode_callback(
    const ValidatedInputFrameRegions& regions, ArtifactV3ResultSink& sink) {
  CompileSessionDecodeResult decoded = decode_compile_session(regions);
  g_decode_status = decoded.status;
  g_failure = decoded.failure;
  if (decoded.status != CompileSessionDecodeStatus::kReady &&
      (decoded.failure.region == CompileSessionRegion::kRequest ||
       decoded.failure.region == CompileSessionRegion::kProfile)) {
    const bool request = decoded.failure.region == CompileSessionRegion::kRequest;
    const std::uint8_t* bytes = request ? regions.request_bytes() : regions.profile_bytes();
    const std::uint32_t length = request ? regions.request_byte_length()
                                         : regions.profile_byte_length();
    const std::uint32_t begin = decoded.failure.byte_offset > 80U
                                    ? decoded.failure.byte_offset - 80U : 0U;
    const std::uint32_t end = std::min<std::uint32_t>(
        length, decoded.failure.byte_offset + 120U);
    std::fprintf(stderr, "%s context: %.*s\n", request ? "request" : "profile",
                 end - begin, reinterpret_cast<const char*>(bytes + begin));
  }
  if (decoded.status == CompileSessionDecodeStatus::kReady && decoded.session) {
    g_profile_hash = decoded.session->profile_hash();
    g_contract_hash = decoded.session->compilation_contract_hash();
    g_request_hash = decoded.session->request_hash();
    g_request_id = decoded.session->request_id();
    g_output_limit = decoded.session->maximum_output_byte_length();
    constexpr CompilerOptionKind expected_kinds[] = {
        CompilerOptionKind::kDefine,
        CompilerOptionKind::kForcedInclude,
        CompilerOptionKind::kFrontendOption,
        CompilerOptionKind::kFrontendOption,
        CompilerOptionKind::kUndefine,
        CompilerOptionKind::kWarningPolicy,
    };
    bool exact_order = decoded.session->compiler_option_count() == 6U;
    for (std::size_t index = 0; index < decoded.session->compiler_option_count();
         ++index) {
      const CompilerOptionView option = decoded.session->compiler_option(index);
      exact_order = exact_order && option.ordinal == index &&
                    option.kind == expected_kinds[index];
    }
    g_ready_shape = exact_order &&
                    decoded.session->compiler_resource_directory_virtual_path() ==
                        "/toolchain/clang/lib/clang/22" &&
                    decoded.session->compiler_option(0).name_or_id == "CUTE_SM80_ENABLED" &&
                    decoded.session->compiler_option(1).include_root_id == "clang-resource" &&
                    decoded.session->compiler_option(2).name_or_id == "syntax-only" &&
                    decoded.session->compiler_option(3).name_or_id == "error-limit" &&
                    decoded.session->compiler_option(3).value_or_disposition == "100000" &&
                    decoded.session->compiler_option(4).name_or_id == "NDEBUG" &&
                    decoded.session->compiler_option(5).name_or_id == "clang.unused-variable" &&
                    decoded.session->semantic_pass_count() == 2U &&
                    decoded.session->include_root_count() == 6U &&
                    decoded.session->source_file_count() == 2U &&
                    decoded.session->entry_request().kind == "layout";
    PrepareCppCuteCompilePlanResult plan =
        prepare_cpp_cute_compile_plan(*decoded.session);
    if (plan.status == CompilePlanStatus::kReady && plan.plan) {
      const std::span<const std::string> device = plan.plan->device_arguments();
      const std::span<const std::string> host = plan.plan->host_arguments();
      g_compile_plan_ready =
          plan.plan->compilation_contract_hash() == g_contract_hash &&
          plan.plan->maximum_output_byte_length() == 8U * 1024U * 1024U &&
          plan.plan->maximum_diagnostic_count() == 100000U &&
          !device.empty() && !host.empty() &&
          device.front() == "clang++" && host.front() == "clang++" &&
          std::find(device.begin(), device.end(), "/workspace/src/main.cu") !=
              device.end() &&
          std::find(host.begin(), host.end(), "/workspace/src/main.cu") !=
              host.end() &&
          std::find(device.begin(), device.end(), "--cuda-device-only") !=
              device.end() &&
          std::find(host.begin(), host.end(), "--cuda-host-only") != host.end() &&
          std::find(device.begin(), device.end(), "-ferror-limit=100000") !=
              device.end() &&
          std::find(host.begin(), host.end(), "-ferror-limit=100000") !=
              host.end() &&
          std::find(device.begin(), device.end(),
                    "-fno-experimental-new-constant-interpreter") ==
              device.end() &&
          std::find(host.begin(), host.end(),
                    "-fno-experimental-new-constant-interpreter") ==
              host.end() &&
          std::find(device.begin(), device.end(), "-fconstexpr-steps=10000000") !=
              device.end() &&
          std::find(host.begin(), host.end(), "-fconstexpr-steps=10000000") !=
              host.end() &&
          std::find(device.begin(), device.end(), "-ftemplate-depth=1024") !=
              device.end() &&
          std::find(host.begin(), host.end(), "-ftemplate-depth=1024") !=
              host.end();
    }
  }
  return build_artifact_v3(regions, sink);
}

CompileSessionDecodeStatus status_from_name(std::string_view name) {
  if (name == "ready") return CompileSessionDecodeStatus::kReady;
  if (name == "invalid") return CompileSessionDecodeStatus::kInvalidFrame;
  if (name == "abi") return CompileSessionDecodeStatus::kAbiMismatch;
  if (name == "resource") return CompileSessionDecodeStatus::kResourceLimit;
  return CompileSessionDecodeStatus::kInternalError;
}

int run(const char* frame_path, const char* artifact_path,
        const std::string_view producer_mode,
        std::string_view expected_status,
        const char* expected_profile_hash, const char* expected_contract_hash,
        const char* expected_request_hash) {
  std::ifstream input(frame_path, std::ios::binary);
  BG_CHECK(input.good());
  const std::vector<std::uint8_t> frame(
      (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  BG_CHECK(!frame.empty());
  BG_CHECK(frame.size() <= std::numeric_limits<std::uint32_t>::max());

  g_runtime_test_allocation_hooks = {
      test_allocate,
      test_release,
      test_wire_pointer,
  };
  browsergrad::cpp_cute::g_producer_mode = producer_mode;
  runtime_reset();
  const std::uint32_t pointer = runtime_allocate(
      static_cast<std::uint32_t>(frame.size()));
  BG_CHECK(pointer == kInputWirePointer);
  BG_CHECK(g_input_allocation_length == frame.size());
  std::memcpy(g_input_allocation, frame.data(), frame.size());
  const std::int32_t runtime_result = runtime_compile(
      pointer, static_cast<std::uint32_t>(frame.size()), decode_callback);
  if (g_decode_status != status_from_name(expected_status)) {
    std::fprintf(stderr, "decode status=%u region=%u reason=%u offset=%u\n",
                 static_cast<unsigned>(g_decode_status),
                 static_cast<unsigned>(g_failure.region),
                 static_cast<unsigned>(g_failure.reason),
                 g_failure.byte_offset);
  }
  BG_CHECK(g_decode_status == status_from_name(expected_status));
  const bool artifact_expected = producer_mode == "success" ||
      producer_mode == "semantic-failure" ||
      producer_mode == "surface-divergence";
  if (g_decode_status == CompileSessionDecodeStatus::kReady &&
      artifact_expected) {
    BG_CHECK(runtime_result == static_cast<std::int32_t>(WireCompileStatus::kArtifactReady));
    BG_CHECK(runtime_result_pointer() == kResultWirePointer);
    BG_CHECK(runtime_result_length() == g_result_allocation_length);
    BG_CHECK(g_result_allocation != nullptr);
    std::ofstream artifact(artifact_path, std::ios::binary);
    BG_CHECK(artifact.good());
    artifact.write(static_cast<const char*>(g_result_allocation),
                   static_cast<std::streamsize>(g_result_allocation_length));
    BG_CHECK(artifact.good());
    BG_CHECK(g_ready_shape);
    BG_CHECK(g_compile_plan_ready);
    BG_CHECK(g_profile_hash == expected_profile_hash);
    BG_CHECK(g_contract_hash == expected_contract_hash);
    BG_CHECK(g_request_hash == expected_request_hash);
    BG_CHECK(g_request_id == std::string("bg.cpp.frontend-request.sha256.") +
                                 expected_request_hash);
    BG_CHECK(g_output_limit == 8U * 1024U * 1024U);
  } else if (g_decode_status == CompileSessionDecodeStatus::kReady) {
    BG_CHECK(runtime_result ==
             static_cast<std::int32_t>(WireCompileStatus::kInvalidFrame));
    BG_CHECK(runtime_result_pointer() == 0U);
    BG_CHECK(runtime_result_length() == 0U);
    BG_CHECK(g_result_allocation == nullptr);
  } else {
    BG_CHECK(runtime_result == static_cast<std::int32_t>(
        wire_status_for(g_decode_status)));
    BG_CHECK(!g_ready_shape);
  }
  runtime_reset();
  BG_CHECK(g_input_allocation == nullptr);
  BG_CHECK(g_result_allocation == nullptr);
  g_input_slot_used = false;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr,
                 "usage: native-test FRAME ARTIFACT PRODUCER_MODE STATUS PROFILE_HASH CONTRACT_HASH REQUEST_HASH\n");
    return 2;
  }
  return run(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
}
