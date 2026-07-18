#include "extractor/BrowserGradCppCuteImportedVfs.h"

#include <cstdio>
#include <type_traits>

namespace {

using namespace browsergrad::cpp_cute;

constexpr std::string_view kContentSha256 =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

std::error_code record_read(ImportedVfsObserver& observer,
                            std::string_view path,
                            std::uint64_t byte_length = 1U) {
  return observer.record_successful_read(path, kContentSha256, byte_length);
}

#define BG_CHECK(condition)                                                   \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::fprintf(stderr, "imported VFS observer check failed at line %d: " \
                           "%s\n",                                           \
                   __LINE__, #condition);                                     \
      return 1;                                                               \
    }                                                                         \
  } while (false)

ImportedVfsIncludeEdgeObservation source_edge(
    ImportedVfsIncludeKind kind, std::string including_file_path,
    std::string resolved_file_path, std::string spelling,
    std::uint64_t directive_start_byte_offset,
    std::uint64_t directive_end_byte_offset) {
  return ImportedVfsIncludeEdgeObservation{
      kind,
      std::move(including_file_path),
      std::move(resolved_file_path),
      std::move(spelling),
      directive_start_byte_offset,
      directive_end_byte_offset,
      kImportedVfsNoCompilerOptionOrdinal,
  };
}

ImportedVfsIncludeEdgeObservation forced_edge(
    std::string resolved_file_path, std::uint32_t compiler_option_ordinal) {
  return ImportedVfsIncludeEdgeObservation{
      ImportedVfsIncludeKind::kCompilerForced,
      "",
      std::move(resolved_file_path),
      "",
      0U,
      0U,
      compiler_option_ordinal,
  };
}

bool same_edge(const ImportedVfsIncludeEdgeObservation& left,
               const ImportedVfsIncludeEdgeObservation& right) {
  return left.kind == right.kind &&
         left.including_file_path == right.including_file_path &&
         left.resolved_file_path == right.resolved_file_path &&
         left.spelling == right.spelling &&
         left.directive_start_byte_offset ==
             right.directive_start_byte_offset &&
         left.directive_end_byte_offset == right.directive_end_byte_offset &&
         left.compiler_option_ordinal == right.compiler_option_ordinal;
}

bool same_observation(const ImportedVfsPassObservation& left,
                      const ImportedVfsPassObservation& right) {
  if (left.opened_file_paths != right.opened_file_paths ||
      left.opened_files.size() != right.opened_files.size() ||
      left.include_edges.size() != right.include_edges.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.opened_files.size(); ++index) {
    const ImportedVfsOpenedFileObservation& left_file =
        left.opened_files[index];
    const ImportedVfsOpenedFileObservation& right_file =
        right.opened_files[index];
    if (left_file.virtual_path != right_file.virtual_path ||
        left_file.content_sha256 != right_file.content_sha256 ||
        left_file.byte_length != right_file.byte_length) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < left.include_edges.size(); ++index) {
    if (!same_edge(left.include_edges[index], right.include_edges[index])) {
      return false;
    }
  }
  return true;
}

int run_observer_tests() {
  static_assert(!std::is_copy_constructible_v<ImportedVfsObserver>);
  static_assert(!std::is_copy_assignable_v<ImportedVfsObserver>);
  static_assert(!std::is_move_constructible_v<ImportedVfsObserver>);
  static_assert(!std::is_move_assignable_v<ImportedVfsObserver>);

  const auto quote = source_edge(
      ImportedVfsIncludeKind::kSourceQuote, "/src/main.cu",
      "/src/detail/local.hpp", "detail/local.hpp", 9U, 36U);
  const auto angle = source_edge(
      ImportedVfsIncludeKind::kSourceAngle, "/src/main.cu",
      "/toolchain/cute/layout.hpp", "cute/layout.hpp", 40U, 67U);
  const auto forced = forced_edge(
      "/toolchain/clang/__clang_cuda_runtime_wrapper.h", 1U);

  ImportedVfsObserver first(
      ImportedVfsObservationLimits{4U, 4U});
  // Include callbacks may arrive before read closure. They remain staged and
  // cannot pollute the snapshot while either participating read is absent.
  BG_CHECK(!first.record_resolved_include_edge(angle));
  BG_CHECK(!first.record_resolved_include_edge(quote));
  BG_CHECK(!first.record_resolved_include_edge(forced));
  BG_CHECK(!first.record_resolved_include_edge(angle));
  ImportedVfsPassObservation observation;
  BG_CHECK(!first.snapshot(observation));
  BG_CHECK(observation.opened_file_paths.empty());
  BG_CHECK(observation.include_edges.empty());

  BG_CHECK(!record_read(first, "/toolchain/cute/layout.hpp"));
  BG_CHECK(!record_read(first, "/src/main.cu"));
  BG_CHECK(!record_read(first, "/src/detail/local.hpp"));
  BG_CHECK(!record_read(first,
      "/toolchain/clang/__clang_cuda_runtime_wrapper.h"));
  BG_CHECK(!record_read(first, "/src/main.cu"));
  BG_CHECK(!first.snapshot(observation));
  BG_CHECK(observation.opened_file_paths.size() == 4U);
  BG_CHECK(observation.opened_file_paths[0] == "/src/detail/local.hpp");
  BG_CHECK(observation.opened_file_paths[1] == "/src/main.cu");
  BG_CHECK(observation.opened_file_paths[2] ==
           "/toolchain/clang/__clang_cuda_runtime_wrapper.h");
  BG_CHECK(observation.opened_file_paths[3] ==
           "/toolchain/cute/layout.hpp");
  BG_CHECK(observation.opened_files.size() == 4U);
  BG_CHECK(observation.opened_files[1].virtual_path == "/src/main.cu");
  BG_CHECK(observation.opened_files[1].content_sha256 == kContentSha256);
  BG_CHECK(observation.opened_files[1].byte_length == 1U);
  BG_CHECK(observation.include_edges.size() == 3U);
  BG_CHECK(same_edge(observation.include_edges[0], quote));
  BG_CHECK(same_edge(observation.include_edges[1], angle));
  BG_CHECK(same_edge(observation.include_edges[2], forced));

  // Snapshot bytes are independent of callback and read order.
  ImportedVfsObserver second(
      ImportedVfsObservationLimits{4U, 4U});
  BG_CHECK(!record_read(second,
      "/toolchain/clang/__clang_cuda_runtime_wrapper.h"));
  BG_CHECK(!record_read(second, "/src/detail/local.hpp"));
  BG_CHECK(!record_read(second, "/src/main.cu"));
  BG_CHECK(!record_read(second, "/toolchain/cute/layout.hpp"));
  BG_CHECK(!second.record_resolved_include_edge(forced));
  BG_CHECK(!second.record_resolved_include_edge(quote));
  BG_CHECK(!second.record_resolved_include_edge(angle));
  ImportedVfsPassObservation reverse_observation;
  BG_CHECK(!second.snapshot(reverse_observation));
  BG_CHECK(same_observation(observation, reverse_observation));

  // Returned records do not alias observer-owned state.
  reverse_observation.opened_file_paths[0] = "/mutated";
  reverse_observation.opened_files[0].content_sha256 = std::string(64U, 'f');
  reverse_observation.include_edges.clear();
  ImportedVfsPassObservation fresh_observation;
  BG_CHECK(!second.snapshot(fresh_observation));
  BG_CHECK(same_observation(observation, fresh_observation));

  // Unique-record limits are strict; duplicates do not consume capacity.
  ImportedVfsObserver read_limited(
      ImportedVfsObservationLimits{1U, 1U});
  BG_CHECK(!record_read(read_limited, "/src/main.cu"));
  BG_CHECK(!record_read(read_limited, "/src/main.cu"));
  BG_CHECK(record_read(read_limited, "/src/other.cu") ==
           std::make_error_code(std::errc::value_too_large));
  BG_CHECK(read_limited.terminal_error() ==
           std::make_error_code(std::errc::value_too_large));
  BG_CHECK(read_limited.snapshot(fresh_observation) ==
           read_limited.terminal_error());

  ImportedVfsObserver edge_limited(
      ImportedVfsObservationLimits{2U, 1U});
  BG_CHECK(!edge_limited.record_resolved_include_edge(quote));
  BG_CHECK(!edge_limited.record_resolved_include_edge(quote));
  BG_CHECK(edge_limited.record_resolved_include_edge(angle) ==
           std::make_error_code(std::errc::value_too_large));

  // Invalid records poison only their own pass observer.
  ImportedVfsObserver invalid_path;
  BG_CHECK(record_read(invalid_path, "/src/../secret") ==
           std::make_error_code(std::errc::invalid_argument));
  BG_CHECK(invalid_path.terminal_error() ==
           std::make_error_code(std::errc::invalid_argument));
  BG_CHECK(!first.terminal_error());

  ImportedVfsObserver invalid_edge;
  auto malformed = quote;
  malformed.directive_end_byte_offset = malformed.directive_start_byte_offset;
  BG_CHECK(invalid_edge.record_resolved_include_edge(std::move(malformed)) ==
           std::make_error_code(std::errc::invalid_argument));

  ImportedVfsObserver oversized_limits(ImportedVfsObservationLimits{
      kImportedVfsMaximumObservedFileCount + 1U,
      kImportedVfsMaximumObservedIncludeEdgeCount,
  });
  BG_CHECK(oversized_limits.terminal_error() ==
           std::make_error_code(std::errc::invalid_argument));

  // Separate observers are pass-scoped and never share records or poison.
  ImportedVfsObserver device;
  ImportedVfsObserver host;
  BG_CHECK(!record_read(device, "/src/device-only.cuh"));
  BG_CHECK(!record_read(host, "/src/host-only.hpp"));
  ImportedVfsPassObservation device_observation;
  ImportedVfsPassObservation host_observation;
  BG_CHECK(!device.snapshot(device_observation));
  BG_CHECK(!host.snapshot(host_observation));
  BG_CHECK(device_observation.opened_file_paths ==
           std::vector<std::string>{"/src/device-only.cuh"});
  BG_CHECK(host_observation.opened_file_paths ==
           std::vector<std::string>{"/src/host-only.hpp"});

  // Strict UTF-8 accepts scalar values and rejects overlong/surrogate forms.
  ImportedVfsObserver unicode;
  BG_CHECK(!record_read(unicode, "/src/\xc3\xa9.hpp"));
  ImportedVfsObserver overlong;
  BG_CHECK(record_read(overlong, "/src/\xc0\xaf.hpp") ==
           std::make_error_code(std::errc::invalid_argument));
  ImportedVfsObserver surrogate;
  BG_CHECK(record_read(surrogate, "/src/\xed\xa0\x80.hpp") ==
           std::make_error_code(std::errc::invalid_argument));

  return 0;
}

}  // namespace

int main() { return run_observer_tests(); }
