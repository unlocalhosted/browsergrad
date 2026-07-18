#pragma once

#include "BrowserGradCppCuteVirtualPath.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace browsergrad::cpp_cute {

inline constexpr std::uint32_t kImportedVfsMaximumObservedFileCount = 110000U;
inline constexpr std::uint32_t kImportedVfsMaximumObservedIncludeEdgeCount =
    262144U;
inline constexpr std::uint32_t kImportedVfsNoCompilerOptionOrdinal =
    std::numeric_limits<std::uint32_t>::max();

struct ImportedVfsObservationLimits {
  std::uint32_t max_opened_file_count =
      kImportedVfsMaximumObservedFileCount;
  std::uint32_t max_include_edge_count =
      kImportedVfsMaximumObservedIncludeEdgeCount;
};

enum class ImportedVfsIncludeKind : std::uint8_t {
  kSourceQuote = 0,
  kSourceAngle = 1,
  kCompilerForced = 2,
};

struct ImportedVfsIncludeEdgeObservation {
  ImportedVfsIncludeKind kind = ImportedVfsIncludeKind::kSourceQuote;
  std::string including_file_path;
  std::string resolved_file_path;
  std::string spelling;
  std::uint64_t directive_start_byte_offset = 0U;
  std::uint64_t directive_end_byte_offset = 0U;
  std::uint32_t compiler_option_ordinal =
      kImportedVfsNoCompilerOptionOrdinal;
};

struct ImportedVfsOpenedFileObservation {
  std::string virtual_path;
  std::string content_sha256;
  std::uint64_t byte_length = 0U;
};

struct ImportedVfsPassObservation {
  // Despite the artifact field name, a path enters this set only after its
  // complete contents have been read successfully into stable Wasm memory.
  std::vector<std::string> opened_file_paths;
  std::vector<ImportedVfsOpenedFileObservation> opened_files;
  std::vector<ImportedVfsIncludeEdgeObservation> include_edges;
};

namespace imported_vfs_detail {

inline int compare_utf8_bytes(std::string_view left, std::string_view right) {
  const auto limit = std::min(left.size(), right.size());
  for (std::size_t index = 0U; index < limit; ++index) {
    const auto left_byte = static_cast<unsigned char>(left[index]);
    const auto right_byte = static_cast<unsigned char>(right[index]);
    if (left_byte < right_byte) return -1;
    if (left_byte > right_byte) return 1;
  }
  if (left.size() < right.size()) return -1;
  if (left.size() > right.size()) return 1;
  return 0;
}

inline bool valid_utf8(std::string_view value) {
  std::size_t index = 0U;
  while (index < value.size()) {
    const auto first = static_cast<unsigned char>(value[index]);
    if (first <= 0x7fU) {
      ++index;
      continue;
    }
    const auto continuation = [&value](std::size_t at) {
      return at < value.size() &&
             (static_cast<unsigned char>(value[at]) & 0xc0U) == 0x80U;
    };
    if (first >= 0xc2U && first <= 0xdfU) {
      if (!continuation(index + 1U)) return false;
      index += 2U;
      continue;
    }
    if (first >= 0xe0U && first <= 0xefU) {
      if (index + 2U >= value.size() || !continuation(index + 1U) ||
          !continuation(index + 2U)) {
        return false;
      }
      const auto second = static_cast<unsigned char>(value[index + 1U]);
      if ((first == 0xe0U && second < 0xa0U) ||
          (first == 0xedU && second >= 0xa0U)) {
        return false;
      }
      index += 3U;
      continue;
    }
    if (first >= 0xf0U && first <= 0xf4U) {
      if (index + 3U >= value.size() || !continuation(index + 1U) ||
          !continuation(index + 2U) || !continuation(index + 3U)) {
        return false;
      }
      const auto second = static_cast<unsigned char>(value[index + 1U]);
      if ((first == 0xf0U && second < 0x90U) ||
          (first == 0xf4U && second >= 0x90U)) {
        return false;
      }
      index += 4U;
      continue;
    }
    return false;
  }
  return true;
}

inline bool valid_include_spelling(std::string_view spelling) {
  constexpr std::size_t kMaximumSpellingByteLength = 4096U;
  if (spelling.empty() || spelling.size() > kMaximumSpellingByteLength ||
      !valid_utf8(spelling)) {
    return false;
  }
  return std::none_of(spelling.begin(), spelling.end(), [](char value) {
    const auto byte = static_cast<unsigned char>(value);
    return byte == 0U || byte < 0x20U || byte == 0x7fU;
  });
}

struct Utf8ByteLess {
  using is_transparent = void;

  bool operator()(std::string_view left, std::string_view right) const {
    return compare_utf8_bytes(left, right) < 0;
  }
};

struct IncludeEdgeLess {
  bool operator()(const ImportedVfsIncludeEdgeObservation& left,
                  const ImportedVfsIncludeEdgeObservation& right) const {
    if (left.kind != right.kind) {
      return static_cast<std::uint8_t>(left.kind) <
             static_cast<std::uint8_t>(right.kind);
    }
    const auto compare_string = [](std::string_view left_value,
                                   std::string_view right_value) {
      return compare_utf8_bytes(left_value, right_value);
    };
    if (const auto order = compare_string(left.including_file_path,
                                          right.including_file_path);
        order != 0) {
      return order < 0;
    }
    if (const auto order = compare_string(left.resolved_file_path,
                                          right.resolved_file_path);
        order != 0) {
      return order < 0;
    }
    if (const auto order = compare_string(left.spelling, right.spelling);
        order != 0) {
      return order < 0;
    }
    if (left.directive_start_byte_offset !=
        right.directive_start_byte_offset) {
      return left.directive_start_byte_offset <
             right.directive_start_byte_offset;
    }
    if (left.directive_end_byte_offset != right.directive_end_byte_offset) {
      return left.directive_end_byte_offset < right.directive_end_byte_offset;
    }
    return left.compiler_option_ordinal < right.compiler_option_ordinal;
  }
};

}  // namespace imported_vfs_detail

class ImportedVfsObserver final {
 public:
  explicit ImportedVfsObserver(
      ImportedVfsObservationLimits limits = ImportedVfsObservationLimits{})
      : limits_(limits) {
    if (limits_.max_opened_file_count >
            kImportedVfsMaximumObservedFileCount ||
        limits_.max_include_edge_count >
            kImportedVfsMaximumObservedIncludeEdgeCount) {
      terminal_error_ = std::make_error_code(std::errc::invalid_argument);
    }
  }

  ImportedVfsObserver(const ImportedVfsObserver&) = delete;
  ImportedVfsObserver& operator=(const ImportedVfsObserver&) = delete;
  ImportedVfsObserver(ImportedVfsObserver&&) = delete;
  ImportedVfsObserver& operator=(ImportedVfsObserver&&) = delete;

  // ImportedVfsFile calls this only after every read chunk has committed.
  // Status, directory lookup, open, and failed/partial reads must not call it.
  std::error_code record_successful_read(
      std::string_view canonical_path, std::string_view content_sha256,
      std::uint64_t byte_length) {
    if (terminal_error_) return terminal_error_;
    if (!cpp_cute_valid_canonical_virtual_path(canonical_path) ||
        content_sha256.size() != 64U ||
        !std::all_of(content_sha256.begin(), content_sha256.end(),
                     [](const char byte) {
                       return (byte >= '0' && byte <= '9') ||
                              (byte >= 'a' && byte <= 'f');
                     })) {
      return poison(std::make_error_code(std::errc::invalid_argument));
    }
    if (const auto found = opened_files_.find(canonical_path);
        found != opened_files_.end()) {
      if (found->second.content_sha256 != content_sha256 ||
          found->second.byte_length != byte_length) {
        return poison(std::make_error_code(std::errc::state_not_recoverable));
      }
      return {};
    }
    if (opened_files_.size() >= limits_.max_opened_file_count) {
      return poison(std::make_error_code(std::errc::value_too_large));
    }
    opened_files_.emplace(
        std::string(canonical_path),
        ImportedVfsOpenedFileObservation{
            std::string(canonical_path), std::string(content_sha256),
            byte_length});
    return {};
  }

  // Only resolved include observations belong here. Unresolved directives are
  // diagnostic observations and must not fabricate an opened-file edge.
  std::error_code record_resolved_include_edge(
      ImportedVfsIncludeEdgeObservation edge) {
    if (terminal_error_) return terminal_error_;
    if (!valid_include_edge(edge)) {
      return poison(std::make_error_code(std::errc::invalid_argument));
    }
    if (include_edges_.find(edge) != include_edges_.end()) return {};
    if (include_edges_.size() >= limits_.max_include_edge_count) {
      return poison(std::make_error_code(std::errc::value_too_large));
    }
    include_edges_.insert(std::move(edge));
    return {};
  }

  // Produces a unique byte-ordered snapshot. Resolved callbacks are staged,
  // then admitted only when both participating files reached read closure.
  // This prevents a failed include/read probe from polluting pass evidence.
  std::error_code snapshot(ImportedVfsPassObservation& observation) const {
    if (terminal_error_) return terminal_error_;
    ImportedVfsPassObservation candidate;
    candidate.opened_file_paths.reserve(opened_files_.size());
    candidate.opened_files.reserve(opened_files_.size());
    for (const auto& [path, file] : opened_files_) {
      candidate.opened_file_paths.push_back(path);
      candidate.opened_files.push_back(file);
    }
    candidate.include_edges.reserve(include_edges_.size());
    for (const auto& edge : include_edges_) {
      if (opened_files_.find(edge.resolved_file_path) ==
          opened_files_.end()) {
        continue;
      }
      if (edge.kind != ImportedVfsIncludeKind::kCompilerForced &&
          opened_files_.find(edge.including_file_path) ==
              opened_files_.end()) {
        continue;
      }
      candidate.include_edges.push_back(edge);
    }
    observation = std::move(candidate);
    return {};
  }

  std::error_code terminal_error() const { return terminal_error_; }

 private:
  static bool valid_include_edge(
      const ImportedVfsIncludeEdgeObservation& edge) {
    if (!cpp_cute_valid_canonical_virtual_path(edge.resolved_file_path)) {
      return false;
    }
    switch (edge.kind) {
      case ImportedVfsIncludeKind::kSourceQuote:
      case ImportedVfsIncludeKind::kSourceAngle:
        return cpp_cute_valid_canonical_virtual_path(
                   edge.including_file_path) &&
               imported_vfs_detail::valid_include_spelling(edge.spelling) &&
               edge.directive_start_byte_offset <
                   edge.directive_end_byte_offset &&
               edge.compiler_option_ordinal ==
                   kImportedVfsNoCompilerOptionOrdinal;
      case ImportedVfsIncludeKind::kCompilerForced:
        return edge.including_file_path.empty() && edge.spelling.empty() &&
               edge.directive_start_byte_offset == 0U &&
               edge.directive_end_byte_offset == 0U &&
               edge.compiler_option_ordinal !=
                   kImportedVfsNoCompilerOptionOrdinal;
    }
    return false;
  }

  std::error_code poison(std::error_code error) {
    if (!terminal_error_) terminal_error_ = error;
    return terminal_error_;
  }

  ImportedVfsObservationLimits limits_;
  std::map<std::string, ImportedVfsOpenedFileObservation,
           imported_vfs_detail::Utf8ByteLess> opened_files_;
  std::set<ImportedVfsIncludeEdgeObservation,
           imported_vfs_detail::IncludeEdgeLess>
      include_edges_;
  std::error_code terminal_error_;
};

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> imported_closed_vfs();
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> imported_closed_vfs(
    std::shared_ptr<ImportedVfsObserver> observer);

}  // namespace browsergrad::cpp_cute
