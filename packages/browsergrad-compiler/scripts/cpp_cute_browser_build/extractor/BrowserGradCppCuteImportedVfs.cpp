#include "BrowserGradCppCuteImportedVfs.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

#if defined(__wasm__)
#define BG_CPP_CUTE_VFS_IMPORT(name)                                      \
  __attribute__((import_module("browsergrad_vfs_v1")))                  \
  __attribute__((import_name(name)))
#else
#define BG_CPP_CUTE_VFS_IMPORT(name)
#endif

extern "C" {

BG_CPP_CUTE_VFS_IMPORT("bg_vfs_status")
std::int32_t bg_vfs_status(std::uint32_t path_pointer,
                           std::uint32_t path_length,
                           std::uint32_t metadata_pointer);
BG_CPP_CUTE_VFS_IMPORT("bg_vfs_open")
std::int32_t bg_vfs_open(std::uint32_t path_pointer,
                         std::uint32_t path_length,
                         std::uint32_t open_result_pointer);
BG_CPP_CUTE_VFS_IMPORT("bg_vfs_read")
std::int32_t bg_vfs_read(std::uint32_t handle, std::uint32_t offset_low,
                         std::uint32_t offset_high,
                         std::uint32_t destination_pointer,
                         std::uint32_t byte_length);
BG_CPP_CUTE_VFS_IMPORT("bg_vfs_close")
std::int32_t bg_vfs_close(std::uint32_t handle);
BG_CPP_CUTE_VFS_IMPORT("bg_vfs_directory_count")
std::int32_t bg_vfs_directory_count(std::uint32_t path_pointer,
                                    std::uint32_t path_length,
                                    std::uint32_t count_pointer);
BG_CPP_CUTE_VFS_IMPORT("bg_vfs_directory_entry")
std::int32_t bg_vfs_directory_entry(
    std::uint32_t path_pointer, std::uint32_t path_length,
    std::uint32_t index, std::uint32_t name_pointer,
    std::uint32_t name_capacity, std::uint32_t metadata_pointer);

}  // extern "C"

#undef BG_CPP_CUTE_VFS_IMPORT

namespace browsergrad::cpp_cute {
namespace {

constexpr std::uint32_t kVfsMaximumPathByteLength = 4096U;
constexpr std::uint32_t kVfsMaximumDirectoryEntryCount = 262144U;
constexpr std::uint32_t kVfsMaximumLiveHandleCount = 65536U;
constexpr std::uint64_t kVfsMaximumReadableFileByteLength = 402653184ULL;
constexpr std::uint32_t kVfsReadChunkByteLength = 64U * 1024U;

enum class ImportedVfsStatus : std::int32_t {
  kOk = 0,
  kNotFound = 1,
  kNotDirectory = 2,
  kIsDirectory = 3,
  kInvalidPath = 4,
  kBufferTooSmall = 5,
  kOutOfRange = 6,
  kInvalidHandle = 7,
  kResourceLimit = 8,
  kSessionClosed = 9,
  kInternalError = 10,
};

struct ImportedVfsMetadata {
  std::uint32_t kind = 0;
  std::uint32_t name_byte_length = 0;
  std::uint64_t file_byte_length = 0;
  std::uint64_t unique_id_device = 0;
  std::uint64_t unique_id_file = 0;
};

struct ImportedVfsOpenResult {
  std::uint32_t handle = 0;
  std::uint64_t file_byte_length = 0;
};

struct ImportedVfsHandleBudget {
  std::uint32_t live_handle_count = 0;
  std::error_code terminal_error;
};

std::uint32_t read_u32_le(const std::uint8_t* bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint64_t read_u64_le(const std::uint8_t* bytes) {
  return static_cast<std::uint64_t>(read_u32_le(bytes)) |
         (static_cast<std::uint64_t>(read_u32_le(bytes + 4U)) << 32U);
}

std::error_code imported_vfs_error(std::int32_t wire_status) {
  switch (static_cast<ImportedVfsStatus>(wire_status)) {
    case ImportedVfsStatus::kOk:
      return {};
    case ImportedVfsStatus::kNotFound:
      return std::make_error_code(std::errc::no_such_file_or_directory);
    case ImportedVfsStatus::kNotDirectory:
      return std::make_error_code(std::errc::not_a_directory);
    case ImportedVfsStatus::kIsDirectory:
      return std::make_error_code(std::errc::is_a_directory);
    case ImportedVfsStatus::kInvalidPath:
      return std::make_error_code(std::errc::invalid_argument);
    case ImportedVfsStatus::kBufferTooSmall:
      return std::make_error_code(std::errc::no_buffer_space);
    case ImportedVfsStatus::kOutOfRange:
      return std::make_error_code(std::errc::result_out_of_range);
    case ImportedVfsStatus::kInvalidHandle:
      return std::make_error_code(std::errc::bad_file_descriptor);
    case ImportedVfsStatus::kResourceLimit:
      return std::make_error_code(std::errc::not_enough_memory);
    case ImportedVfsStatus::kSessionClosed:
      return std::make_error_code(std::errc::operation_canceled);
    case ImportedVfsStatus::kInternalError:
      return std::make_error_code(std::errc::io_error);
  }
  return std::make_error_code(std::errc::protocol_error);
}

void poison_handle_budget(
    const std::shared_ptr<ImportedVfsHandleBudget>& handle_budget,
    std::error_code error) {
  if (handle_budget != nullptr && !handle_budget->terminal_error) {
    handle_budget->terminal_error =
        error ? error : std::make_error_code(std::errc::protocol_error);
  }
}

std::error_code close_imported_handle(
    std::uint32_t handle,
    const std::shared_ptr<ImportedVfsHandleBudget>& handle_budget) {
  const auto error = imported_vfs_error(bg_vfs_close(handle));
  if (error) poison_handle_budget(handle_budget, error);
  return error;
}

bool wasm_pointer(const void* pointer, std::uint32_t& result) {
  const auto value = reinterpret_cast<std::uintptr_t>(pointer);
  if (value > std::numeric_limits<std::uint32_t>::max()) return false;
  result = static_cast<std::uint32_t>(value);
  return true;
}

bool valid_utf8(llvm::StringRef value) {
  const auto* begin = reinterpret_cast<const llvm::UTF8*>(value.data());
  const auto* cursor = begin;
  return llvm::isLegalUTF8String(&cursor, begin + value.size()) != 0;
}

bool valid_canonical_path(llvm::StringRef path) {
  if (path.empty() || path.size() > kVfsMaximumPathByteLength ||
      path.front() != '/' || !valid_utf8(path) ||
      (path.size() > 1U && path.back() == '/')) {
    return false;
  }
  if (path == "/") return true;

  std::size_t segment_begin = 1U;
  for (std::size_t index = 1U; index <= path.size(); ++index) {
    if (index < path.size()) {
      const auto byte = static_cast<unsigned char>(path[index]);
      if (byte == '\\' || byte == 0U || byte < 0x20U || byte == 0x7fU) {
        return false;
      }
      if (byte != '/') continue;
    }
    const llvm::StringRef segment = path.slice(segment_begin, index);
    if (segment.empty() || segment == "." || segment == "..") return false;
    segment_begin = index + 1U;
  }
  return true;
}

bool valid_basename(llvm::StringRef name) {
  if (name.empty() || name.size() > kVfsMaximumPathByteLength ||
      name == "." || name == ".." || !valid_utf8(name)) {
    return false;
  }
  for (const unsigned char byte : name.bytes()) {
    if (byte == '/' || byte == '\\' || byte == 0U || byte < 0x20U ||
        byte == 0x7fU) {
      return false;
    }
  }
  return true;
}

bool utf8_byte_less(llvm::StringRef left, llvm::StringRef right) {
  const auto limit = std::min(left.size(), right.size());
  for (std::size_t index = 0; index < limit; ++index) {
    const auto left_byte = static_cast<unsigned char>(left[index]);
    const auto right_byte = static_cast<unsigned char>(right[index]);
    if (left_byte != right_byte) return left_byte < right_byte;
  }
  return left.size() < right.size();
}

std::error_code imported_path_pointer(llvm::StringRef path,
                                      std::uint32_t& pointer) {
  if (!valid_canonical_path(path) || !wasm_pointer(path.data(), pointer)) {
    return std::make_error_code(std::errc::invalid_argument);
  }
  return {};
}

bool decode_metadata(const std::array<std::uint8_t, 32>& bytes,
                     ImportedVfsMetadata& metadata) {
  metadata.kind = read_u32_le(bytes.data());
  metadata.name_byte_length = read_u32_le(bytes.data() + 4U);
  metadata.file_byte_length = read_u64_le(bytes.data() + 8U);
  metadata.unique_id_device = read_u64_le(bytes.data() + 16U);
  metadata.unique_id_file = read_u64_le(bytes.data() + 24U);
  return metadata.kind == 1U || metadata.kind == 2U;
}

llvm::vfs::Status llvm_status(llvm::StringRef path,
                              const ImportedVfsMetadata& metadata) {
  const auto type = metadata.kind == 1U
                        ? llvm::sys::fs::file_type::regular_file
                        : llvm::sys::fs::file_type::directory_file;
  return llvm::vfs::Status(
      path,
      llvm::sys::fs::UniqueID(metadata.unique_id_device,
                              metadata.unique_id_file),
      llvm::sys::TimePoint<>(), 0U, 0U, metadata.file_byte_length, type,
      llvm::sys::fs::all_read);
}

llvm::ErrorOr<ImportedVfsMetadata> imported_status(llvm::StringRef path) {
  std::uint32_t path_pointer = 0;
  if (const auto error = imported_path_pointer(path, path_pointer)) {
    return error;
  }
  alignas(8) std::array<std::uint8_t, 32> bytes{};
  std::uint32_t metadata_pointer = 0;
  if (!wasm_pointer(bytes.data(), metadata_pointer)) {
    return std::make_error_code(std::errc::value_too_large);
  }
  const auto status = bg_vfs_status(
      path_pointer, static_cast<std::uint32_t>(path.size()), metadata_pointer);
  if (status != static_cast<std::int32_t>(ImportedVfsStatus::kOk)) {
    return imported_vfs_error(status);
  }
  ImportedVfsMetadata metadata;
  if (!decode_metadata(bytes, metadata) || metadata.name_byte_length != 0U ||
      (metadata.kind == 2U && metadata.file_byte_length != 0U) ||
      metadata.file_byte_length > kVfsMaximumReadableFileByteLength) {
    return std::make_error_code(std::errc::protocol_error);
  }
  return metadata;
}

llvm::ErrorOr<ImportedVfsOpenResult> imported_open(
    llvm::StringRef path,
    const std::shared_ptr<ImportedVfsHandleBudget>& handle_budget) {
  std::uint32_t path_pointer = 0;
  if (const auto error = imported_path_pointer(path, path_pointer)) {
    return error;
  }
  alignas(8) std::array<std::uint8_t, 16> bytes{};
  std::uint32_t result_pointer = 0;
  if (!wasm_pointer(bytes.data(), result_pointer)) {
    return std::make_error_code(std::errc::value_too_large);
  }
  const auto status = bg_vfs_open(
      path_pointer, static_cast<std::uint32_t>(path.size()), result_pointer);
  if (status != static_cast<std::int32_t>(ImportedVfsStatus::kOk)) {
    return imported_vfs_error(status);
  }
  if (read_u32_le(bytes.data() + 4U) != 0U) {
    const auto close_error =
        close_imported_handle(read_u32_le(bytes.data()), handle_budget);
    return close_error ? close_error
                       : std::make_error_code(std::errc::protocol_error);
  }
  ImportedVfsOpenResult result{
      read_u32_le(bytes.data()), read_u64_le(bytes.data() + 8U)};
  if (result.file_byte_length > kVfsMaximumReadableFileByteLength) {
    const auto close_error = close_imported_handle(result.handle, handle_budget);
    return close_error ? close_error
                       : std::make_error_code(std::errc::file_too_large);
  }
  return result;
}

class ImportedVfsFile final : public llvm::vfs::File {
 public:
  ImportedVfsFile(std::string path, llvm::vfs::Status status,
                  ImportedVfsOpenResult open_result,
                  std::shared_ptr<ImportedVfsHandleBudget> handle_budget)
      : path_(std::move(path)),
        status_(std::move(status)),
        handle_(open_result.handle),
        file_byte_length_(open_result.file_byte_length),
        handle_budget_(std::move(handle_budget)) {}

  ~ImportedVfsFile() override { static_cast<void>(close()); }

  llvm::ErrorOr<llvm::vfs::Status> status() override { return status_; }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> getBuffer(
      const llvm::Twine& name, std::int64_t file_size,
      bool requires_null_terminator, bool is_volatile) override {
    static_cast<void>(name);
    static_cast<void>(requires_null_terminator);
    static_cast<void>(is_volatile);
    if (!live_) return std::make_error_code(std::errc::bad_file_descriptor);
    if (handle_budget_ == nullptr || handle_budget_->terminal_error) {
      return handle_budget_ == nullptr
                 ? std::make_error_code(std::errc::protocol_error)
                 : handle_budget_->terminal_error;
    }
    if (file_size < -1 ||
        (file_size >= 0 &&
         static_cast<std::uint64_t>(file_size) != file_byte_length_)) {
      return std::make_error_code(std::errc::invalid_argument);
    }
    if (file_byte_length_ > std::numeric_limits<std::size_t>::max()) {
      return std::make_error_code(std::errc::file_too_large);
    }

    auto buffer = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(
        static_cast<std::size_t>(file_byte_length_), path_);
    if (!buffer) return std::make_error_code(std::errc::not_enough_memory);
    auto* destination =
        reinterpret_cast<std::uint8_t*>(buffer->getBufferStart());
    const std::uint64_t read_count = file_byte_length_ == 0U
                                         ? 1U
                                         : (file_byte_length_ +
                                            kVfsReadChunkByteLength - 1U) /
                                               kVfsReadChunkByteLength;
    for (std::uint64_t chunk = 0; chunk < read_count; ++chunk) {
      const std::uint64_t offset = chunk * kVfsReadChunkByteLength;
      const auto byte_length = static_cast<std::uint32_t>(
          std::min<std::uint64_t>(kVfsReadChunkByteLength,
                                  file_byte_length_ - offset));
      std::uint32_t destination_pointer = 0;
      if (!wasm_pointer(destination + offset, destination_pointer)) {
        return std::make_error_code(std::errc::value_too_large);
      }
      const auto read_status = bg_vfs_read(
          handle_, static_cast<std::uint32_t>(offset),
          static_cast<std::uint32_t>(offset >> 32U), destination_pointer,
          byte_length);
      if (read_status != static_cast<std::int32_t>(ImportedVfsStatus::kOk)) {
        return imported_vfs_error(read_status);
      }
    }
    return std::unique_ptr<llvm::MemoryBuffer>(std::move(buffer));
  }

  std::error_code close() override {
    if (!live_) return {};
    live_ = false;
    const bool valid_budget =
        handle_budget_ != nullptr && handle_budget_->live_handle_count > 0U;
    const auto close_error = close_imported_handle(handle_, handle_budget_);
    if (!valid_budget) {
      const auto protocol_error =
          std::make_error_code(std::errc::protocol_error);
      poison_handle_budget(handle_budget_, protocol_error);
      return close_error ? close_error : protocol_error;
    }
    if (close_error) return close_error;
    --handle_budget_->live_handle_count;
    return {};
  }

 private:
  std::string path_;
  llvm::vfs::Status status_;
  std::uint32_t handle_ = 0;
  std::uint64_t file_byte_length_ = 0;
  std::shared_ptr<ImportedVfsHandleBudget> handle_budget_;
  bool live_ = true;
};

class ImportedVfsOpenGuard final {
 public:
  ImportedVfsOpenGuard(
      std::uint32_t handle,
      std::shared_ptr<ImportedVfsHandleBudget> handle_budget)
      : handle_(handle), handle_budget_(std::move(handle_budget)) {}

  ImportedVfsOpenGuard(const ImportedVfsOpenGuard&) = delete;
  ImportedVfsOpenGuard& operator=(const ImportedVfsOpenGuard&) = delete;

  ~ImportedVfsOpenGuard() {
    if (live_) {
      static_cast<void>(close_imported_handle(handle_, handle_budget_));
    }
  }

  void release() { live_ = false; }

 private:
  std::uint32_t handle_ = 0;
  std::shared_ptr<ImportedVfsHandleBudget> handle_budget_;
  bool live_ = true;
};

class ImportedVfsDirectoryIterator final
    : public llvm::vfs::detail::DirIterImpl {
 public:
  ImportedVfsDirectoryIterator(
      std::string directory, std::uint32_t entry_count,
      std::shared_ptr<ImportedVfsHandleBudget> handle_budget)
      : directory_(std::move(directory)),
        entry_count_(entry_count),
        handle_budget_(std::move(handle_budget)) {}

  std::error_code initialize() { return load_current(); }

  std::error_code increment() override {
    ++index_;
    return load_current();
  }

 private:
  std::error_code load_current() {
    CurrentEntry = llvm::vfs::directory_entry{};
    if (index_ >= entry_count_) return {};
    if (handle_budget_ == nullptr) {
      return std::make_error_code(std::errc::protocol_error);
    }
    if (handle_budget_->terminal_error) return handle_budget_->terminal_error;

    std::uint32_t path_pointer = 0;
    if (const auto error = imported_path_pointer(directory_, path_pointer)) {
      return error;
    }
    alignas(8) std::array<std::uint8_t, 32> metadata_bytes{};
    std::uint32_t metadata_pointer = 0;
    if (!wasm_pointer(metadata_bytes.data(), metadata_pointer)) {
      return std::make_error_code(std::errc::value_too_large);
    }
    const auto sizing_status = bg_vfs_directory_entry(
        path_pointer, static_cast<std::uint32_t>(directory_.size()), index_,
        0U, 0U, metadata_pointer);
    const auto required_name_byte_length =
        read_u32_le(metadata_bytes.data() + 4U);
    if (sizing_status !=
            static_cast<std::int32_t>(ImportedVfsStatus::kBufferTooSmall) ||
        required_name_byte_length == 0U ||
        required_name_byte_length > kVfsMaximumPathByteLength) {
      return std::make_error_code(std::errc::protocol_error);
    }

    std::string name(required_name_byte_length, '\0');
    std::uint32_t name_pointer = 0;
    if (!wasm_pointer(name.data(), name_pointer)) {
      return std::make_error_code(std::errc::value_too_large);
    }
    metadata_bytes.fill(0U);
    const auto entry_status = bg_vfs_directory_entry(
        path_pointer, static_cast<std::uint32_t>(directory_.size()), index_,
        name_pointer, required_name_byte_length, metadata_pointer);
    ImportedVfsMetadata metadata;
    if (entry_status != static_cast<std::int32_t>(ImportedVfsStatus::kOk) ||
        !decode_metadata(metadata_bytes, metadata) ||
        metadata.name_byte_length != required_name_byte_length ||
        !valid_basename(name) ||
        (!previous_name_.empty() && !utf8_byte_less(previous_name_, name))) {
      return std::make_error_code(std::errc::protocol_error);
    }
    std::string child =
        directory_ == "/" ? "/" + name : directory_ + "/" + name;
    if (!valid_canonical_path(child) ||
        (metadata.kind == 2U && metadata.file_byte_length != 0U) ||
        metadata.file_byte_length > kVfsMaximumReadableFileByteLength) {
      return std::make_error_code(std::errc::protocol_error);
    }
    CurrentEntry = llvm::vfs::directory_entry(
        std::move(child), metadata.kind == 1U
                              ? llvm::sys::fs::file_type::regular_file
                              : llvm::sys::fs::file_type::directory_file);
    previous_name_ = std::move(name);
    return {};
  }

  std::string directory_;
  std::string previous_name_;
  std::uint32_t entry_count_ = 0;
  std::uint32_t index_ = 0;
  std::shared_ptr<ImportedVfsHandleBudget> handle_budget_;
};

class ImportedVfsFileSystem final : public llvm::vfs::FileSystem {
 public:
  llvm::ErrorOr<llvm::vfs::Status> status(
      const llvm::Twine& path_twine) override {
    if (handle_budget_->terminal_error) return handle_budget_->terminal_error;
    llvm::SmallString<kVfsMaximumPathByteLength> path;
    path_twine.toVector(path);
    auto metadata = imported_status(path);
    if (!metadata) return metadata.getError();
    return llvm_status(path, *metadata);
  }

  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>> openFileForRead(
      const llvm::Twine& path_twine) override {
    if (handle_budget_->terminal_error) return handle_budget_->terminal_error;
    llvm::SmallString<kVfsMaximumPathByteLength> path;
    path_twine.toVector(path);
    auto metadata = imported_status(path);
    if (!metadata) return metadata.getError();
    if (metadata->kind != 1U) {
      return std::make_error_code(std::errc::is_a_directory);
    }
    if (handle_budget_->live_handle_count >= kVfsMaximumLiveHandleCount) {
      return std::make_error_code(std::errc::too_many_files_open);
    }
    std::string stable_path(path);
    auto stable_status = llvm_status(stable_path, *metadata);
    auto open_result = imported_open(path, handle_budget_);
    if (!open_result) return open_result.getError();
    ImportedVfsOpenGuard open_guard(open_result->handle, handle_budget_);
    if (open_result->file_byte_length != metadata->file_byte_length) {
      return std::make_error_code(std::errc::protocol_error);
    }
    auto file = std::make_unique<ImportedVfsFile>(
        std::move(stable_path), std::move(stable_status), *open_result,
        handle_budget_);
    ++handle_budget_->live_handle_count;
    open_guard.release();
    return std::unique_ptr<llvm::vfs::File>(std::move(file));
  }

  llvm::vfs::directory_iterator dir_begin(
      const llvm::Twine& directory_twine, std::error_code& error) override {
    if (handle_budget_->terminal_error) {
      error = handle_budget_->terminal_error;
      return {};
    }
    llvm::SmallString<kVfsMaximumPathByteLength> directory;
    directory_twine.toVector(directory);
    std::uint32_t path_pointer = 0;
    if ((error = imported_path_pointer(directory, path_pointer))) return {};

    alignas(4) std::uint32_t entry_count = 0;
    std::uint32_t count_pointer = 0;
    if (!wasm_pointer(&entry_count, count_pointer)) {
      error = std::make_error_code(std::errc::value_too_large);
      return {};
    }
    const auto count_status = bg_vfs_directory_count(
        path_pointer, static_cast<std::uint32_t>(directory.size()),
        count_pointer);
    if (count_status != static_cast<std::int32_t>(ImportedVfsStatus::kOk)) {
      error = imported_vfs_error(count_status);
      return {};
    }
    if (entry_count > kVfsMaximumDirectoryEntryCount) {
      error = std::make_error_code(std::errc::value_too_large);
      return {};
    }

    if (entry_count == 0U) {
      error.clear();
      return {};
    }
    auto iterator = std::make_shared<ImportedVfsDirectoryIterator>(
        std::string(directory), entry_count, handle_budget_);
    error = iterator->initialize();
    if (error) return {};
    return llvm::vfs::directory_iterator(std::move(iterator));
  }

  std::error_code setCurrentWorkingDirectory(
      const llvm::Twine& path_twine) override {
    llvm::SmallString<kVfsMaximumPathByteLength> path;
    path_twine.toVector(path);
    return path == "/" ? std::error_code{}
                       : std::make_error_code(std::errc::operation_not_permitted);
  }

  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override {
    return std::string("/");
  }

  std::error_code makeAbsolute(
      llvm::SmallVectorImpl<char>& path) const override {
    if (path.empty()) return std::make_error_code(std::errc::invalid_argument);
    if (path.front() != '/') path.insert(path.begin(), '/');
    return valid_canonical_path(llvm::StringRef(path.data(), path.size()))
               ? std::error_code{}
               : std::make_error_code(std::errc::invalid_argument);
  }

 private:
  std::shared_ptr<ImportedVfsHandleBudget> handle_budget_ =
      std::make_shared<ImportedVfsHandleBudget>();
};

}  // namespace

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> imported_closed_vfs() {
  return llvm::makeIntrusiveRefCnt<ImportedVfsFileSystem>();
}

}  // namespace browsergrad::cpp_cute
