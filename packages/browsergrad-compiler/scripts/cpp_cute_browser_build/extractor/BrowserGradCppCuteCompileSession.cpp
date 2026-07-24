#include "BrowserGradCppCuteCompileSession.h"

#include "BrowserGradCppCuteCanonicalJson.h"
#include "BrowserGradCppCuteSha256.h"
#include "BrowserGradCppCuteVirtualPath.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace browsergrad::cpp_cute {
namespace {

constexpr std::size_t kMaximumDecodeScratchByteLength = 16U * 1024U * 1024U;
constexpr std::uint32_t kMaximumResultByteLength = 32U * 1024U * 1024U;
constexpr std::string_view kProfileSchema =
    "browsergrad.compiler.cpp-cute.frontend-profile";
constexpr std::string_view kRequestSchema =
    "browsergrad.compiler.cpp-cute.frontend-request";
constexpr std::string_view kArtifactSchema =
    "browsergrad.compiler.cpp-cute.frontend-artifact";
constexpr std::string_view kContractSchema =
    "browsergrad.compiler.cpp-cute.compilation-contract";
constexpr std::string_view kRuntimeAbiId =
    "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1";
constexpr std::string_view kRuntimeAbiManifestSha256 =
    "e41d149bbb65329bd0a9bf0355ddc6fea8eae24070660e26ab5dc743787f6f37";
constexpr std::string_view kSemanticAdapterManifestSha256 =
    "e5aa795c4feebd523ed72b95be03b102d497f2e0313ee9c99fadf1309cde6150";
constexpr std::string_view kTemporalPolicyId =
    "browsergrad.compiler.cpp-cute.temporal-macros.reject@1";
constexpr std::string_view kWarningRegistryId =
    "browsergrad.compiler.cpp-cute.clang-warning-registry@1";
constexpr std::string_view kDiagnosticNormalizationManifestSha256 =
    "6de153792fb09711a9f71ee470433f58ce4183a09fe4e8eb987c9d9bf6a46997";
constexpr std::int64_t kProfileMajor = 2;
constexpr std::int64_t kProfileMinor = 6;
constexpr std::int64_t kContractMajor = 1;
constexpr std::int64_t kContractMinor = 2;
constexpr std::int64_t kRequestMajor = 1;
constexpr std::int64_t kRequestMinor = 0;
constexpr std::int64_t kArtifactMajor = 3;
constexpr std::int64_t kArtifactMinor = 0;

struct DecodeError final {
  CompileSessionDecodeStatus status;
  CompileSessionDecodeFailure failure;
};

[[noreturn]] void reject(CompileSessionDecodeStatus status,
                         CompileSessionRegion region,
                         CompileSessionDecodeReason reason,
                         std::uint32_t offset) {
  throw DecodeError{status, {region, reason, offset}};
}

struct BudgetState final {
  std::size_t live = 0;
};

class BudgetResource final : public std::pmr::memory_resource {
 public:
  explicit BudgetResource(BudgetState& state) noexcept : state_(state) {}

 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (bytes > kMaximumDecodeScratchByteLength - state_.live) {
      throw std::bad_alloc();
    }
    void* result = std::pmr::new_delete_resource()->allocate(bytes, alignment);
    state_.live += bytes;
    return result;
  }

  void do_deallocate(void* pointer, std::size_t bytes,
                     std::size_t alignment) override {
    std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
    state_.live -= bytes;
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

  BudgetState& state_;
};

using PmrString = std::pmr::string;

struct OwnedCompilerOption final {
  explicit OwnedCompilerOption(std::pmr::memory_resource* memory)
      : name_or_id(memory), value_or_disposition(memory),
        include_root_id(memory), virtual_path(memory) {}
  CompilerOptionKind kind = CompilerOptionKind::kDefine;
  std::uint32_t ordinal = 0;
  PmrString name_or_id;
  PmrString value_or_disposition;
  PmrString include_root_id;
  PmrString virtual_path;
  bool has_value = false;
};

struct OwnedSemanticPass final {
  explicit OwnedSemanticPass(std::pmr::memory_resource* memory)
      : pass_id(memory), domain(memory), role(memory), invocation_mode(memory),
        target_triple(memory), auxiliary_target_triple(memory),
        device_architecture(memory) {}
  std::uint32_t ordinal = 0;
  PmrString pass_id;
  PmrString domain;
  PmrString role;
  PmrString invocation_mode;
  PmrString target_triple;
  PmrString auxiliary_target_triple;
  PmrString device_architecture;
};

struct OwnedIncludeRoot final {
  explicit OwnedIncludeRoot(std::pmr::memory_resource* memory)
      : include_root_id(memory), mode(memory), virtual_path(memory),
        manifest_sha256(memory), owner_kind(memory), dependency_id(memory) {}
  PmrString include_root_id;
  PmrString mode;
  PmrString virtual_path;
  PmrString manifest_sha256;
  PmrString owner_kind;
  PmrString dependency_id;
};

struct OwnedSourceFile final {
  explicit OwnedSourceFile(std::pmr::memory_resource* memory)
      : file_id(memory), role(memory), virtual_path(memory),
        content_sha256(memory), byte_length(memory), include_root_id(memory) {}
  PmrString file_id;
  PmrString role;
  PmrString virtual_path;
  PmrString content_sha256;
  PmrString byte_length;
  PmrString include_root_id;
  std::uint64_t byte_length_value = 0;
  bool has_include_root = false;
};

struct OwnedEntryRequest final {
  explicit OwnedEntryRequest(std::pmr::memory_resource* memory)
      : request_id(memory), kind(memory), declaration_kind(memory),
        virtual_path(memory), begin_byte(memory), end_byte(memory),
        token_sha256(memory) {}
  PmrString request_id;
  PmrString kind;
  PmrString declaration_kind;
  PmrString virtual_path;
  PmrString begin_byte;
  PmrString end_byte;
  PmrString token_sha256;
  std::uint64_t begin_value = 0;
  std::uint64_t end_value = 0;
};

struct OwnedDependency final {
  explicit OwnedDependency(std::pmr::memory_resource* memory)
      : dependency_id(memory), kind(memory), header_set_sha256(memory) {}
  PmrString dependency_id;
  PmrString kind;
  PmrString header_set_sha256;
};

struct SessionStorage final {
  SessionStorage()
      : memory(budget), profile_id(&memory), profile_hash(&memory),
        compilation_contract_hash(&memory), request_id(&memory),
        request_hash(&memory), main_virtual_path(&memory),
        extractor_binary_sha256(&memory), compiler_binary_sha256(&memory),
        compiler_version(&memory),
        compiler_resource_virtual_path(&memory),
        compiler_resource_sha256(&memory),
        cuda_toolkit_root_virtual_path(&memory), source_roots(&memory),
        options(&memory), passes(&memory), include_roots(&memory),
        dependencies(&memory), source_files(&memory), entry(&memory) {}

  BudgetState budget;
  BudgetResource memory;
  PmrString profile_id;
  PmrString profile_hash;
  PmrString compilation_contract_hash;
  PmrString request_id;
  PmrString request_hash;
  PmrString main_virtual_path;
  PmrString extractor_binary_sha256;
  PmrString compiler_binary_sha256;
  PmrString compiler_version;
  PmrString compiler_resource_virtual_path;
  PmrString compiler_resource_sha256;
  PmrString cuda_toolkit_root_virtual_path;
  std::pmr::vector<PmrString> source_roots;
  std::pmr::vector<OwnedCompilerOption> options;
  std::pmr::vector<OwnedSemanticPass> passes;
  std::pmr::vector<OwnedIncludeRoot> include_roots;
  std::pmr::vector<OwnedDependency> dependencies;
  std::pmr::vector<OwnedSourceFile> source_files;
  OwnedEntryRequest entry;
  std::array<std::uint64_t, 20U> profile_semantic_limits{};
  std::array<std::uint64_t, 20U> request_semantic_limits{};
  std::uint32_t maximum_output_byte_length = 0;
  std::uint64_t runtime_maximum_linear_bytes = 0;
  std::uint64_t runtime_stack_bytes = 0;
  std::uint64_t runtime_compiler_working_bytes = 0;
  std::uint64_t runtime_vfs_live_open_bytes = 0;
};

enum class JsonKind : std::uint8_t {
  kObject,
  kArray,
  kString,
  kInteger,
  kBoolean,
  kNull,
};

constexpr std::uint32_t kNoNode = std::numeric_limits<std::uint32_t>::max();

struct JsonNode final {
  JsonKind kind = JsonKind::kNull;
  std::uint32_t begin = 0;
  std::uint32_t end = 0;
  std::uint32_t content_begin = 0;
  std::uint32_t content_end = 0;
  std::uint32_t key_begin = 0;
  std::uint32_t key_end = 0;
  std::uint32_t first_child = kNoNode;
  std::uint32_t next_sibling = kNoNode;
  std::int64_t integer = 0;
  bool boolean = false;
};

struct ByteRegion final {
  const std::uint8_t* bytes = nullptr;
  std::uint32_t length = 0;
};

class JsonDocument final {
 public:
  JsonDocument(ByteRegion region, std::pmr::memory_resource* memory)
      : region_(region), nodes_(memory) {
    if (region.bytes == nullptr || region.length == 0U) throw std::bad_alloc();
    parse_value(1U, nullptr, nullptr);
    if (cursor_ != region_.length || nodes_.empty()) throw std::bad_alloc();
  }

  const JsonNode& root() const { return nodes_[0]; }
  const JsonNode& node(std::uint32_t index) const { return nodes_[index]; }
  ByteRegion region() const noexcept { return region_; }

  const JsonNode* field(const JsonNode& object, std::string_view key) const {
    if (object.kind != JsonKind::kObject) return nullptr;
    std::uint32_t child = object.first_child;
    while (child != kNoNode) {
      const JsonNode& candidate = nodes_[child];
      const std::string_view raw_key(
          reinterpret_cast<const char*>(region_.bytes + candidate.key_begin),
          candidate.key_end - candidate.key_begin);
      if (raw_key == key) return &candidate;
      child = candidate.next_sibling;
    }
    return nullptr;
  }

  std::uint32_t child_count(const JsonNode& node) const {
    std::uint32_t result = 0;
    std::uint32_t child = node.first_child;
    while (child != kNoNode) {
      ++result;
      child = nodes_[child].next_sibling;
    }
    return result;
  }

  const JsonNode* element(const JsonNode& array, std::uint32_t index) const {
    if (array.kind != JsonKind::kArray) return nullptr;
    std::uint32_t child = array.first_child;
    while (child != kNoNode && index != 0U) {
      child = nodes_[child].next_sibling;
      --index;
    }
    return child == kNoNode ? nullptr : &nodes_[child];
  }

  std::string_view raw(const JsonNode& node) const {
    return {reinterpret_cast<const char*>(region_.bytes + node.begin),
            node.end - node.begin};
  }

  std::string_view raw_string(const JsonNode& node) const {
    if (node.kind != JsonKind::kString) return {};
    return {reinterpret_cast<const char*>(region_.bytes + node.content_begin),
            node.content_end - node.content_begin};
  }

 private:
  static bool digit(std::uint8_t byte) {
    return byte >= static_cast<std::uint8_t>('0') &&
           byte <= static_cast<std::uint8_t>('9');
  }

  std::uint32_t append_node(JsonKind kind, std::uint32_t begin) {
    if (nodes_.size() >= std::numeric_limits<std::uint32_t>::max()) {
      throw std::bad_alloc();
    }
    nodes_.push_back(JsonNode{});
    JsonNode& node = nodes_.back();
    node.kind = kind;
    node.begin = begin;
    return static_cast<std::uint32_t>(nodes_.size() - 1U);
  }

  void link_child(std::uint32_t parent, std::uint32_t child,
                  std::uint32_t* previous) {
    if (nodes_[parent].first_child == kNoNode) nodes_[parent].first_child = child;
    if (*previous != kNoNode) nodes_[*previous].next_sibling = child;
    *previous = child;
  }

  std::pair<std::uint32_t, std::uint32_t> parse_string_span() {
    if (cursor_ >= region_.length || region_.bytes[cursor_] != '"') {
      throw std::bad_alloc();
    }
    const std::uint32_t begin = ++cursor_;
    while (cursor_ < region_.length) {
      if (region_.bytes[cursor_] == '"') {
        const std::uint32_t end = cursor_++;
        return {begin, end};
      }
      if (region_.bytes[cursor_] == '\\') {
        cursor_ += region_.bytes[cursor_ + 1U] == 'u' ? 6U : 2U;
      } else {
        ++cursor_;
      }
    }
    throw std::bad_alloc();
  }

  std::uint32_t parse_value(std::uint32_t depth, std::uint32_t* key_begin,
                            std::uint32_t* key_end) {
    if (depth > kRuntimeV1MaxNestingDepth || cursor_ >= region_.length) {
      throw std::bad_alloc();
    }
    const std::uint32_t begin = cursor_;
    const std::uint8_t first = region_.bytes[cursor_];
    JsonKind kind = JsonKind::kInteger;
    if (first == '{') kind = JsonKind::kObject;
    else if (first == '[') kind = JsonKind::kArray;
    else if (first == '"') kind = JsonKind::kString;
    else if (first == 't' || first == 'f') kind = JsonKind::kBoolean;
    else if (first == 'n') kind = JsonKind::kNull;
    const std::uint32_t index = append_node(kind, begin);
    if (key_begin != nullptr) {
      nodes_[index].key_begin = *key_begin;
      nodes_[index].key_end = *key_end;
    }

    if (kind == JsonKind::kObject) {
      ++cursor_;
      std::uint32_t previous = kNoNode;
      if (region_.bytes[cursor_] != '}') {
        while (true) {
          const auto key = parse_string_span();
          if (region_.bytes[cursor_++] != ':') throw std::bad_alloc();
          const std::uint32_t child = parse_value(depth + 1U,
                                                  const_cast<std::uint32_t*>(&key.first),
                                                  const_cast<std::uint32_t*>(&key.second));
          link_child(index, child, &previous);
          if (region_.bytes[cursor_] == '}') break;
          if (region_.bytes[cursor_++] != ',') throw std::bad_alloc();
        }
      }
      ++cursor_;
    } else if (kind == JsonKind::kArray) {
      ++cursor_;
      std::uint32_t previous = kNoNode;
      if (region_.bytes[cursor_] != ']') {
        while (true) {
          const std::uint32_t child = parse_value(depth + 1U, nullptr, nullptr);
          link_child(index, child, &previous);
          if (region_.bytes[cursor_] == ']') break;
          if (region_.bytes[cursor_++] != ',') throw std::bad_alloc();
        }
      }
      ++cursor_;
    } else if (kind == JsonKind::kString) {
      const auto span = parse_string_span();
      nodes_[index].content_begin = span.first;
      nodes_[index].content_end = span.second;
    } else if (kind == JsonKind::kBoolean) {
      nodes_[index].boolean = first == 't';
      cursor_ += first == 't' ? 4U : 5U;
    } else if (kind == JsonKind::kNull) {
      cursor_ += 4U;
    } else {
      if (region_.bytes[cursor_] == '-') ++cursor_;
      while (cursor_ < region_.length && digit(region_.bytes[cursor_])) ++cursor_;
      const char* number_begin = reinterpret_cast<const char*>(region_.bytes + begin);
      const char* number_end = reinterpret_cast<const char*>(region_.bytes + cursor_);
      if (std::from_chars(number_begin, number_end, nodes_[index].integer).ec !=
          std::errc{}) {
        throw std::bad_alloc();
      }
    }
    nodes_[index].end = cursor_;
    return index;
  }

  ByteRegion region_;
  std::pmr::vector<JsonNode> nodes_;
  std::uint32_t cursor_ = 0;
};

std::uint32_t offset_of(const JsonNode& node) { return node.begin; }

bool is_string(const JsonNode* node) {
  return node != nullptr && node->kind == JsonKind::kString;
}

bool string_equals(const JsonDocument& document, const JsonNode* node,
                   std::string_view expected) {
  return is_string(node) && document.raw_string(*node) == expected;
}

void copy_string(const JsonDocument& document, const JsonNode& node,
                 PmrString& output, CompileSessionRegion region) {
  if (node.kind != JsonKind::kString) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, offset_of(node));
  }
  output.clear();
  const ByteRegion bytes = document.region();
  std::uint32_t cursor = node.content_begin;
  while (cursor < node.content_end) {
    const std::uint8_t byte = bytes.bytes[cursor++];
    if (byte != '\\') {
      output.push_back(static_cast<char>(byte));
      continue;
    }
    const std::uint8_t escape = bytes.bytes[cursor++];
    switch (escape) {
      case '"': output.push_back('"'); break;
      case '\\': output.push_back('\\'); break;
      case 'b': output.push_back('\b'); break;
      case 'f': output.push_back('\f'); break;
      case 'n': output.push_back('\n'); break;
      case 'r': output.push_back('\r'); break;
      case 't': output.push_back('\t'); break;
      case 'u': {
        auto hex = [](std::uint8_t value) -> std::uint8_t {
          return value <= '9' ? value - '0' : value - 'a' + 10U;
        };
        cursor += 2U;
        output.push_back(static_cast<char>(
            (hex(bytes.bytes[cursor]) << 4U) | hex(bytes.bytes[cursor + 1U])));
        cursor += 2U;
        break;
      }
      default:
        reject(CompileSessionDecodeStatus::kInternalError, region,
               CompileSessionDecodeReason::kSchema, cursor - 1U);
    }
  }
}

const JsonNode& required_field(const JsonDocument& document,
                               const JsonNode& object, std::string_view key,
                               CompileSessionRegion region) {
  const JsonNode* result = document.field(object, key);
  if (result == nullptr) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, object.begin);
  }
  return *result;
}

void closed_object(const JsonDocument& document, const JsonNode& object,
                   std::initializer_list<std::string_view> keys,
                   CompileSessionRegion region) {
  if (object.kind != JsonKind::kObject ||
      document.child_count(object) != keys.size()) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, object.begin);
  }
  for (std::string_view key : keys) {
    (void)required_field(document, object, key, region);
  }
}

void require_literal(const JsonDocument& document, const JsonNode& node,
                     std::string_view value, CompileSessionRegion region,
                     CompileSessionDecodeStatus status =
                         CompileSessionDecodeStatus::kInvalidFrame,
                     CompileSessionDecodeReason reason =
                         CompileSessionDecodeReason::kSchema) {
  if (!string_equals(document, &node, value)) {
    reject(status, region, reason, node.begin);
  }
}

std::uint64_t positive_integer(const JsonNode& node,
                               CompileSessionRegion region,
                               std::uint64_t maximum =
                                   9'007'199'254'740'991ULL) {
  if (node.kind != JsonKind::kInteger || node.integer <= 0 ||
      static_cast<std::uint64_t>(node.integer) > maximum) {
    reject(CompileSessionDecodeStatus::kResourceLimit, region,
           CompileSessionDecodeReason::kLimit, node.begin);
  }
  return static_cast<std::uint64_t>(node.integer);
}

bool lowercase_sha256(std::string_view value) {
  if (value.size() != 64U) return false;
  return std::all_of(value.begin(), value.end(), [](char byte) {
    return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
  });
}

void require_sha256(const JsonDocument& document, const JsonNode& node,
                    CompileSessionRegion region) {
  if (!is_string(&node) || !lowercase_sha256(document.raw_string(node))) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, node.begin);
  }
}

bool canonical_u64(std::string_view value, std::uint64_t* parsed) {
  if (value.empty() || (value.size() > 1U && value.front() == '0')) return false;
  for (char byte : value) if (byte < '0' || byte > '9') return false;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), *parsed);
  return result.ec == std::errc{} && result.ptr == value.data() + value.size();
}

class Utf16StringIterator final {
 public:
  explicit Utf16StringIterator(std::string_view value) : value_(value) {}

  bool next(std::uint16_t& unit) {
    if (pending_ != 0U) {
      unit = pending_;
      pending_ = 0U;
      return true;
    }
    if (cursor_ == value_.size()) return false;
    const auto first = static_cast<std::uint8_t>(value_[cursor_]);
    std::uint32_t code_point = first;
    std::size_t length = 1U;
    if (first >= 0xc2U && first <= 0xdfU) {
      length = 2U;
      code_point = first & 0x1fU;
    } else if (first >= 0xe0U && first <= 0xefU) {
      length = 3U;
      code_point = first & 0x0fU;
    } else if (first >= 0xf0U) {
      length = 4U;
      code_point = first & 0x07U;
    }
    for (std::size_t index = 1U; index < length; ++index) {
      code_point = (code_point << 6U) |
                   (static_cast<std::uint8_t>(value_[cursor_ + index]) & 0x3fU);
    }
    cursor_ += length;
    if (code_point <= 0xffffU) {
      unit = static_cast<std::uint16_t>(code_point);
      return true;
    }
    code_point -= 0x1'0000U;
    unit = static_cast<std::uint16_t>(0xd800U + (code_point >> 10U));
    pending_ = static_cast<std::uint16_t>(0xdc00U + (code_point & 0x3ffU));
    return true;
  }

 private:
  std::string_view value_;
  std::size_t cursor_ = 0U;
  std::uint16_t pending_ = 0U;
};

int compare_canonical_strings(std::string_view left, std::string_view right) {
  Utf16StringIterator left_units(left);
  Utf16StringIterator right_units(right);
  while (true) {
    std::uint16_t left_unit = 0U;
    std::uint16_t right_unit = 0U;
    const bool has_left = left_units.next(left_unit);
    const bool has_right = right_units.next(right_unit);
    if (!has_left || !has_right) {
      return has_left == has_right ? 0 : has_left ? 1 : -1;
    }
    if (left_unit != right_unit) return left_unit < right_unit ? -1 : 1;
  }
}

class HashWriter final {
 public:
  void text(std::string_view value) {
    if (!healthy_ || !hash_.update(
            reinterpret_cast<const std::uint8_t*>(value.data()), value.size())) {
      healthy_ = false;
    }
  }

  std::string finish() {
    Sha256Digest digest{};
    if (!healthy_ || !hash_.finalize(digest)) {
      reject(CompileSessionDecodeStatus::kInternalError,
             CompileSessionRegion::kNone, CompileSessionDecodeReason::kHash, 0U);
    }
    const Sha256LowercaseHex hex = sha256_lowercase_hex(digest);
    return std::string(hex.data(), 64U);
  }

 private:
  Sha256 hash_;
  bool healthy_ = true;
};

void hash_raw(HashWriter& writer, const JsonDocument& document,
              const JsonNode& node) {
  writer.text(document.raw(node));
}

std::string hash_profile(const JsonDocument& profile) {
  HashWriter writer;
  writer.text("{\"domain\":\"browsergrad.compiler.cpp-cute.frontend-profile.v2\",\"profile\":");
  hash_raw(writer, profile, profile.root());
  writer.text("}");
  return writer.finish();
}

std::string prefixed_digest(std::string_view prefix, std::string digest) {
  return std::string(prefix) + digest;
}

}  // namespace

struct DecodedCompileSession::Impl final {
  SessionStorage storage;
};

namespace {

constexpr std::array<std::string_view, 20U> kSemanticLimitKeys = {{
    "maxSourceFiles", "maxSourceBytes", "maxHeaderFiles", "maxHeaderBytes",
    "maxIncludeDepth", "maxMacroExpansions", "maxPreprocessedTokens",
    "maxAstNodes", "maxConstexprSteps", "maxTemplateInstantiations",
    "maxTemplateDepth", "maxDeclarations", "maxTypes", "maxConstants",
    "maxLayouts", "maxTensors", "maxOperations", "maxTargetIntrinsics",
    "maxDiagnostics", "maxOutputBytes",
}};

constexpr std::array<std::string_view, 20U> kCanonicalSemanticLimitKeys = {{
    "maxAstNodes", "maxConstants", "maxConstexprSteps", "maxDeclarations",
    "maxDiagnostics", "maxHeaderBytes", "maxHeaderFiles", "maxIncludeDepth",
    "maxLayouts", "maxMacroExpansions", "maxOperations", "maxOutputBytes",
    "maxPreprocessedTokens", "maxSourceBytes", "maxSourceFiles",
    "maxTargetIntrinsics", "maxTemplateDepth", "maxTemplateInstantiations",
    "maxTensors", "maxTypes",
}};

constexpr std::array<std::string_view, 24U> kAllLimitKeys = {{
    "maxSourceFiles", "maxSourceBytes", "maxHeaderFiles", "maxHeaderBytes",
    "maxIncludeDepth", "maxMacroExpansions", "maxPreprocessedTokens",
    "maxAstNodes", "maxConstexprSteps", "maxTemplateInstantiations",
    "maxTemplateDepth", "maxDeclarations", "maxTypes", "maxConstants",
    "maxLayouts", "maxTensors", "maxOperations", "maxTargetIntrinsics",
    "maxDiagnostics", "maxOutputBytes", "maxWallTimeMs", "maxCpuTimeMs",
    "maxMemoryBytes", "maxProcesses",
}};

constexpr std::array<std::uint64_t, 24U> kMaximumProfileLimits = {{
    10'000ULL, 64ULL * 1024ULL * 1024ULL, 100'000ULL,
    512ULL * 1024ULL * 1024ULL, 1'024ULL, 10'000'000ULL,
    100'000'000ULL, 20'000'000ULL, 100'000'000ULL, 5'000'000ULL,
    4'096ULL, 5'000'000ULL, 5'000'000ULL, 5'000'000ULL, 1'000'000ULL,
    1'000'000ULL, 5'000'000ULL, 1'000'000ULL, 1'000'000ULL,
    64ULL * 1024ULL * 1024ULL, 30ULL * 60ULL * 1'000ULL,
    30ULL * 60ULL * 1'000ULL, 16ULL * 1024ULL * 1024ULL * 1024ULL,
    1'024ULL,
}};

void closed_exact(const JsonDocument& document, const JsonNode& object,
                  std::span<const std::string_view> keys,
                  CompileSessionRegion region) {
  if (object.kind != JsonKind::kObject ||
      document.child_count(object) != keys.size()) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, object.begin);
  }
  for (std::string_view key : keys) {
    (void)required_field(document, object, key, region);
  }
}

void validate_version(const JsonDocument& document, const JsonNode& version,
                      std::int64_t major, std::int64_t minor,
                      CompileSessionRegion region) {
  closed_object(document, version, {"major", "minor"}, region);
  const JsonNode& major_node = required_field(document, version, "major", region);
  const JsonNode& minor_node = required_field(document, version, "minor", region);
  if (major_node.kind != JsonKind::kInteger ||
      minor_node.kind != JsonKind::kInteger || major_node.integer != major ||
      minor_node.integer != minor) {
    reject(CompileSessionDecodeStatus::kAbiMismatch, region,
           CompileSessionDecodeReason::kUnsupportedVersion, version.begin);
  }
}

void require_string(const JsonNode& node, CompileSessionRegion region) {
  if (node.kind != JsonKind::kString) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, node.begin);
  }
}

bool identifier(std::string_view value) {
  if (value.empty()) return false;
  const auto first = static_cast<unsigned char>(value.front());
  if (!((first >= 'A' && first <= 'Z') || (first >= 'a' && first <= 'z') ||
        first == '_')) return false;
  for (char raw : value.substr(1U)) {
    const auto byte = static_cast<unsigned char>(raw);
    if (!((byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') ||
          (byte >= '0' && byte <= '9') || byte == '_')) return false;
  }
  return true;
}

bool dependency_id(std::string_view value) {
  if (value.empty() || value.size() > 128U || value.front() < 'a' ||
      value.front() > 'z') return false;
  return std::all_of(value.begin() + 1, value.end(), [](char byte) {
    return (byte >= 'a' && byte <= 'z') || (byte >= '0' && byte <= '9') ||
           byte == '.' || byte == '_' || byte == '-';
  });
}

bool supported_warning(std::string_view value) {
  constexpr std::array<std::string_view, 6U> policies = {{
      "clang.deprecated-declarations", "clang.sign-compare",
      "clang.unknown-pragmas", "clang.unused-function",
      "clang.unused-parameter", "clang.unused-variable",
  }};
  return std::find(policies.begin(), policies.end(), value) != policies.end();
}

std::string_view decoded_ascii(const JsonDocument& document,
                               const JsonNode& node,
                               CompileSessionRegion region) {
  require_string(node, region);
  const std::string_view value = document.raw_string(node);
  if (value.find('\\') != std::string_view::npos) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, node.begin);
  }
  return value;
}

void validate_string_array(const JsonDocument& document, const JsonNode& array,
                           CompileSessionRegion region, bool nonempty = false) {
  if (array.kind != JsonKind::kArray ||
      (nonempty && document.child_count(array) == 0U)) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, array.begin);
  }
  std::string_view previous;
  for (std::uint32_t index = 0; index < document.child_count(array); ++index) {
    const JsonNode& item = *document.element(array, index);
    const std::string_view value = decoded_ascii(document, item, region);
    if (index != 0U && value <= previous) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, item.begin);
    }
    previous = value;
  }
}

void validate_browser_deployment(const JsonDocument& document,
                                 const JsonNode& deployment,
                                 SessionStorage& storage,
                                 CompileSessionRegion region) {
  if (deployment.kind != JsonKind::kObject) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, deployment.begin);
  }
  const JsonNode* mode = document.field(deployment, "mode");
  if (!string_equals(document, mode, "browser-local")) {
    reject(CompileSessionDecodeStatus::kAbiMismatch, region,
           CompileSessionDecodeReason::kUnsupportedDeployment,
           mode == nullptr ? deployment.begin : mode->begin);
  }
  closed_object(document, deployment,
                {"mode", "contractId", "assetSetSha256",
                 "buildProvenanceLockSha256", "extractor", "worker",
                 "compilerRuntime", "assetLimits"}, region);
  require_literal(document,
                  required_field(document, deployment, "contractId", region),
                  "browsergrad.compiler.cpp-cute.browser-worker@1", region,
                  CompileSessionDecodeStatus::kAbiMismatch,
                  CompileSessionDecodeReason::kRuntimeAbi);
  require_sha256(document,
                 required_field(document, deployment, "assetSetSha256", region),
                 region);
  require_sha256(document, required_field(
      document, deployment, "buildProvenanceLockSha256", region), region);

  const JsonNode& extractor = required_field(document, deployment, "extractor", region);
  closed_object(document, extractor,
                {"id", "version", "buildId", "binarySha256",
                 "semanticAdapterManifestSha256"}, region);
  for (std::string_view key : {"id", "version", "buildId"}) {
    require_string(required_field(document, extractor, key, region), region);
  }
  require_sha256(document, required_field(document, extractor, "binarySha256", region), region);
  copy_string(document,
              required_field(document, extractor, "binarySha256", region),
              storage.extractor_binary_sha256, region);
  const JsonNode& adapter = required_field(
      document, extractor, "semanticAdapterManifestSha256", region);
  require_sha256(document, adapter, region);
  require_literal(document, adapter, kSemanticAdapterManifestSha256, region,
                  CompileSessionDecodeStatus::kAbiMismatch,
                  CompileSessionDecodeReason::kSemanticAdapter);

  const JsonNode& worker = required_field(document, deployment, "worker", region);
  closed_object(document, worker,
                {"protocolId", "buildId", "moduleSha256", "moduleByteLength",
                 "moduleFormat", "construction", "isolation", "threading",
                 "cancellation", "network", "assetDelivery"}, region);
  require_literal(document, required_field(document, worker, "protocolId", region),
                  "browsergrad.compiler.cpp-cute.browser-worker@1", region);
  require_string(required_field(document, worker, "buildId", region), region);
  require_sha256(document, required_field(document, worker, "moduleSha256", region), region);
  (void)positive_integer(required_field(document, worker, "moduleByteLength", region),
                         region, 64U * 1024U * 1024U);
  for (const auto& pair : std::array<std::pair<std::string_view, std::string_view>, 7U>{{
           {"moduleFormat", "self-contained-es-module"},
           {"construction", "host-verified-blob-url"},
           {"isolation", "dedicated-worker"}, {"threading", "single-thread"},
           {"cancellation", "terminate-worker"}, {"network", "forbidden"},
           {"assetDelivery", "host-verified-transfer"},
       }}) {
    require_literal(document, required_field(document, worker, pair.first, region),
                    pair.second, region);
  }

  const JsonNode& runtime = required_field(document, deployment, "compilerRuntime", region);
  closed_object(document, runtime,
                {"runtimeAbiId", "runtimeAbiManifestSha256", "wasmAddressBits",
                 "requiredWasmFeatures", "moduleHandoff", "workerSideFetch",
                 "memory", "virtualFileSystem"}, region);
  require_literal(document, required_field(document, runtime, "runtimeAbiId", region),
                  kRuntimeAbiId, region, CompileSessionDecodeStatus::kAbiMismatch,
                  CompileSessionDecodeReason::kRuntimeAbi);
  const JsonNode& runtime_hash = required_field(
      document, runtime, "runtimeAbiManifestSha256", region);
  require_sha256(document, runtime_hash, region);
  require_literal(document, runtime_hash, kRuntimeAbiManifestSha256, region,
                  CompileSessionDecodeStatus::kAbiMismatch,
                  CompileSessionDecodeReason::kRuntimeAbi);
  const JsonNode& address_bits = required_field(document, runtime, "wasmAddressBits", region);
  if (address_bits.kind != JsonKind::kInteger || address_bits.integer != 32) {
    reject(CompileSessionDecodeStatus::kAbiMismatch, region,
           CompileSessionDecodeReason::kRuntimeAbi, address_bits.begin);
  }
  const JsonNode& features = required_field(document, runtime, "requiredWasmFeatures", region);
  constexpr std::array<std::string_view, 4U> expected_features = {{
      "bulk-memory", "mutable-globals", "nontrapping-fptoint", "sign-extension",
  }};
  if (features.kind != JsonKind::kArray || document.child_count(features) != 4U) {
    reject(CompileSessionDecodeStatus::kAbiMismatch, region,
           CompileSessionDecodeReason::kRuntimeAbi, features.begin);
  }
  for (std::uint32_t index = 0; index < expected_features.size(); ++index) {
    require_literal(document, *document.element(features, index), expected_features[index],
                    region, CompileSessionDecodeStatus::kAbiMismatch,
                    CompileSessionDecodeReason::kRuntimeAbi);
  }
  require_literal(document, required_field(document, runtime, "moduleHandoff", region),
                  "host-verified-module-or-bytes", region);
  require_literal(document, required_field(document, runtime, "workerSideFetch", region),
                  "forbidden", region);
  const JsonNode& memory = required_field(document, runtime, "memory", region);
  closed_object(document, memory,
                {"sharing", "ownership", "initialPages", "maximumPages",
                 "stackByteLength", "maxCompilerWorkingByteLength"}, region);
  require_literal(document, required_field(document, memory, "sharing", region),
                  "unshared", region);
  require_literal(document, required_field(document, memory, "ownership", region),
                  "worker", region);
  const std::uint64_t initial = positive_integer(
      required_field(document, memory, "initialPages", region), region, 32768U);
  const std::uint64_t maximum = positive_integer(
      required_field(document, memory, "maximumPages", region), region, 32768U);
  if (initial != 4096U || maximum != 16384U) {
    reject(CompileSessionDecodeStatus::kAbiMismatch, region,
           CompileSessionDecodeReason::kRuntimeAbi, memory.begin);
  }
  storage.runtime_maximum_linear_bytes = maximum * 65'536ULL;
  const std::uint64_t stack = positive_integer(
      required_field(document, memory, "stackByteLength", region), region,
      256U * 1024U * 1024U);
  if (stack != 16U * 1024U * 1024U) {
    reject(CompileSessionDecodeStatus::kAbiMismatch, region,
           CompileSessionDecodeReason::kRuntimeAbi, memory.begin);
  }
  storage.runtime_stack_bytes = stack;
  storage.runtime_compiler_working_bytes = positive_integer(required_field(
      document, memory, "maxCompilerWorkingByteLength", region), region,
      536'870'912ULL);
  const JsonNode& runtime_vfs = required_field(
      document, runtime, "virtualFileSystem", region);
  closed_object(document, runtime_vfs,
                {"storage", "maxRetainedHostPackByteLength",
                 "maxAggregateLiveOpenByteLength", "maxIndexedNodes",
                 "maxIndexLogicalByteLength"}, region);
  require_literal(document, required_field(document, runtime_vfs, "storage", region),
                  "host-backed-lazy", region);
  const std::uint64_t retained = positive_integer(required_field(
      document, runtime_vfs, "maxRetainedHostPackByteLength", region), region,
      4ULL * 1024ULL * 1024ULL * 1024ULL);
  const std::uint64_t aggregate = positive_integer(required_field(
      document, runtime_vfs, "maxAggregateLiveOpenByteLength", region), region,
      402'653'184ULL);
  storage.runtime_vfs_live_open_bytes = aggregate;
  (void)positive_integer(required_field(
      document, runtime_vfs, "maxIndexedNodes", region), region, 262'144ULL);
  (void)positive_integer(required_field(
      document, runtime_vfs, "maxIndexLogicalByteLength", region), region,
      134'217'728ULL);

  const JsonNode& asset_limits = required_field(document, deployment, "assetLimits", region);
  constexpr std::array<std::string_view, 7U> asset_keys = {{
      "maxAssets", "maxAssetCompressedByteLength", "maxAssetUnpackedByteLength",
      "maxAssetFileContentByteLength", "maxTotalCompressedByteLength",
      "maxTotalUnpackedByteLength", "maxTotalFileContentByteLength",
  }};
  closed_exact(document, asset_limits, asset_keys, region);
  std::array<std::uint64_t, 7U> asset_values{};
  for (std::size_t index = 0; index < asset_keys.size(); ++index) {
    asset_values[index] = positive_integer(
        required_field(document, asset_limits, asset_keys[index], region), region,
        4ULL * 1024ULL * 1024ULL * 1024ULL);
  }
  if (asset_values[1] > asset_values[4] || asset_values[2] > asset_values[5] ||
      asset_values[3] > asset_values[6] || retained > asset_values[5] ||
      aggregate > asset_values[6]) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, asset_limits.begin);
  }
}

void validate_language(const JsonDocument& document, const JsonNode& language,
                       SessionStorage& storage, CompileSessionRegion region) {
  closed_object(document, language,
                {"cxxStandard", "cudaCompatibility", "preprocessing",
                 "diagnostics", "semanticPasses", "options"}, region);
  require_literal(document, required_field(document, language, "cxxStandard", region),
                  "c++17", region);
  require_string(required_field(document, language, "cudaCompatibility", region), region);

  const JsonNode& preprocessing = required_field(document, language, "preprocessing", region);
  closed_object(document, preprocessing, {"temporalMacros"}, region);
  const JsonNode& temporal = required_field(document, preprocessing, "temporalMacros", region);
  closed_object(document, temporal, {"policyId", "mode"}, region);
  require_literal(document, required_field(document, temporal, "policyId", region),
                  kTemporalPolicyId, region);
  require_literal(document, required_field(document, temporal, "mode", region),
                  "reject", region);

  const JsonNode& diagnostics = required_field(document, language, "diagnostics", region);
  closed_object(document, diagnostics,
                {"warningRegistryId", "baseline",
                 "normalizationManifestSha256"}, region);
  require_literal(document,
                  required_field(document, diagnostics, "warningRegistryId", region),
                  kWarningRegistryId, region);
  require_literal(document, required_field(document, diagnostics, "baseline", region),
                  "compiler-default", region);
  const JsonNode& normalization = required_field(
      document, diagnostics, "normalizationManifestSha256", region);
  require_sha256(document, normalization, region);
  require_literal(document, normalization,
                  kDiagnosticNormalizationManifestSha256, region,
                  CompileSessionDecodeStatus::kAbiMismatch,
                  CompileSessionDecodeReason::kSemanticAdapter);

  const JsonNode& passes = required_field(document, language, "semanticPasses", region);
  if (passes.kind != JsonKind::kArray || document.child_count(passes) != 2U) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, passes.begin);
  }
  constexpr std::array<std::array<std::string_view, 4U>, 2U> pass_literals = {{
      {{"cuda-device-sema", "device", "semantic-extraction", "cuda-device-only"}},
      {{"cuda-host-sema", "host", "validation", "cuda-host-only"}},
  }};
  for (std::uint32_t index = 0; index < 2U; ++index) {
    const JsonNode& pass = *document.element(passes, index);
    closed_object(document, pass,
                  {"ordinal", "passId", "domain", "role", "invocationMode",
                   "targetTriple", "auxiliaryTargetTriple", "deviceArchitecture"},
                  region);
    const JsonNode& ordinal = required_field(document, pass, "ordinal", region);
    if (ordinal.kind != JsonKind::kInteger || ordinal.integer != index) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, ordinal.begin);
    }
    const auto& expected = pass_literals[index];
    require_literal(document, required_field(document, pass, "passId", region),
                    expected[0], region);
    require_literal(document, required_field(document, pass, "domain", region),
                    expected[1], region);
    require_literal(document, required_field(document, pass, "role", region),
                    expected[2], region);
    require_literal(document,
                    required_field(document, pass, "invocationMode", region),
                    expected[3], region);
    storage.passes.emplace_back(&storage.memory);
    OwnedSemanticPass& output = storage.passes.back();
    output.ordinal = index;
    copy_string(document, required_field(document, pass, "passId", region),
                output.pass_id, region);
    copy_string(document, required_field(document, pass, "domain", region),
                output.domain, region);
    copy_string(document, required_field(document, pass, "role", region),
                output.role, region);
    copy_string(document, required_field(document, pass, "invocationMode", region),
                output.invocation_mode, region);
    copy_string(document, required_field(document, pass, "targetTriple", region),
                output.target_triple, region);
    copy_string(document,
                required_field(document, pass, "auxiliaryTargetTriple", region),
                output.auxiliary_target_triple, region);
    copy_string(document,
                required_field(document, pass, "deviceArchitecture", region),
                output.device_architecture, region);
    if (!output.device_architecture.starts_with("sm_") ||
        output.device_architecture.size() < 5U) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, pass.begin);
    }
  }

  const JsonNode& options = required_field(document, language, "options", region);
  if (options.kind != JsonKind::kArray || document.child_count(options) > 4096U) {
    reject(CompileSessionDecodeStatus::kResourceLimit, region,
           CompileSessionDecodeReason::kLimit, options.begin);
  }
  std::pmr::vector<PmrString> singletons(&storage.memory);
  for (std::uint32_t index = 0; index < document.child_count(options); ++index) {
    const JsonNode& option = *document.element(options, index);
    if (option.kind != JsonKind::kObject) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, option.begin);
    }
    const JsonNode& kind_node = required_field(document, option, "kind", region);
    const std::string_view kind = decoded_ascii(document, kind_node, region);
    storage.options.emplace_back(&storage.memory);
    OwnedCompilerOption& output = storage.options.back();
    output.ordinal = index;
    PmrString singleton(&storage.memory);
    if (kind == "define") {
      closed_object(document, option, {"kind", "name", "value"}, region);
      output.kind = CompilerOptionKind::kDefine;
      copy_string(document, required_field(document, option, "name", region),
                  output.name_or_id, region);
      if (!identifier(output.name_or_id) || output.name_or_id.front() == '_' ||
          output.name_or_id.find("__") != std::string_view::npos ||
          output.name_or_id == "defined" || output.name_or_id == "__DATE__" ||
          output.name_or_id == "__TIME__" || output.name_or_id == "__TIMESTAMP__") {
        reject(CompileSessionDecodeStatus::kInvalidFrame, region,
               CompileSessionDecodeReason::kSchema, option.begin);
      }
      const JsonNode& value = required_field(document, option, "value", region);
      if (value.kind == JsonKind::kString) {
        copy_string(document, value, output.value_or_disposition, region);
        if (output.value_or_disposition.size() > 1024U) {
          reject(CompileSessionDecodeStatus::kResourceLimit, region,
                 CompileSessionDecodeReason::kLimit, value.begin);
        }
        output.has_value = true;
      } else if (value.kind != JsonKind::kNull) {
        reject(CompileSessionDecodeStatus::kInvalidFrame, region,
               CompileSessionDecodeReason::kSchema, value.begin);
      }
      singleton = "macro:";
      singleton += output.name_or_id;
    } else if (kind == "undefine") {
      closed_object(document, option, {"kind", "name"}, region);
      output.kind = CompilerOptionKind::kUndefine;
      copy_string(document, required_field(document, option, "name", region),
                  output.name_or_id, region);
      if (!identifier(output.name_or_id) || output.name_or_id.front() == '_' ||
          output.name_or_id.find("__") != std::string_view::npos ||
          output.name_or_id == "defined") {
        reject(CompileSessionDecodeStatus::kInvalidFrame, region,
               CompileSessionDecodeReason::kSchema, option.begin);
      }
      singleton = "macro:";
      singleton += output.name_or_id;
    } else if (kind == "frontend-option") {
      closed_object(document, option, {"kind", "id", "value"}, region);
      output.kind = CompilerOptionKind::kFrontendOption;
      copy_string(document, required_field(document, option, "id", region),
                  output.name_or_id, region);
      const JsonNode& value = required_field(document, option, "value", region);
      if (output.name_or_id == "syntax-only") {
        if (value.kind != JsonKind::kNull) {
          reject(CompileSessionDecodeStatus::kInvalidFrame, region,
                 CompileSessionDecodeReason::kSchema, value.begin);
        }
      } else if (output.name_or_id == "error-limit") {
        copy_string(document, value, output.value_or_disposition, region);
        if (output.value_or_disposition.empty() ||
            output.value_or_disposition.size() > 6U ||
            output.value_or_disposition.front() == '0' ||
            !std::all_of(output.value_or_disposition.begin(),
                         output.value_or_disposition.end(),
                         [](char byte) { return byte >= '0' && byte <= '9'; })) {
          reject(CompileSessionDecodeStatus::kInvalidFrame, region,
                 CompileSessionDecodeReason::kSchema, value.begin);
        }
        output.has_value = true;
      } else {
        reject(CompileSessionDecodeStatus::kInvalidFrame, region,
               CompileSessionDecodeReason::kSchema, option.begin);
      }
      singleton = "frontend:";
      singleton += output.name_or_id;
    } else if (kind == "warning-policy") {
      closed_object(document, option, {"kind", "id", "disposition"}, region);
      output.kind = CompilerOptionKind::kWarningPolicy;
      copy_string(document, required_field(document, option, "id", region),
                  output.name_or_id, region);
      copy_string(document,
                  required_field(document, option, "disposition", region),
                  output.value_or_disposition, region);
      if (!supported_warning(output.name_or_id) ||
          (output.value_or_disposition != "ignore" &&
           output.value_or_disposition != "warn" &&
           output.value_or_disposition != "error")) {
        reject(CompileSessionDecodeStatus::kInvalidFrame, region,
               CompileSessionDecodeReason::kSchema, option.begin);
      }
      output.has_value = true;
      singleton = "warning:";
      singleton += output.name_or_id;
    } else if (kind == "forced-include") {
      closed_object(document, option,
                    {"kind", "includeRootId", "virtualPath"}, region);
      output.kind = CompilerOptionKind::kForcedInclude;
      copy_string(document,
                  required_field(document, option, "includeRootId", region),
                  output.include_root_id, region);
      copy_string(document,
                  required_field(document, option, "virtualPath", region),
                  output.virtual_path, region);
      if (!dependency_id(output.include_root_id) ||
          !cpp_cute_valid_canonical_virtual_path(output.virtual_path)) {
        reject(CompileSessionDecodeStatus::kInvalidFrame, region,
               CompileSessionDecodeReason::kSchema, option.begin);
      }
      singleton = "forced-include:";
      singleton += output.include_root_id;
      singleton += ':';
      singleton += output.virtual_path;
    } else {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, option.begin);
    }
    if (std::find(singletons.begin(), singletons.end(), singleton) !=
        singletons.end()) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, option.begin);
    }
    singletons.push_back(std::move(singleton));
  }
  for (const std::string_view required : {
           std::string_view("frontend:syntax-only"),
           std::string_view("frontend:error-limit")}) {
    if (std::find(singletons.begin(), singletons.end(), required) ==
        singletons.end()) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, options.begin);
    }
  }
}

void validate_target(const JsonDocument& document, const JsonNode& target,
                     const SessionStorage& storage,
                     CompileSessionRegion region) {
  closed_object(document, target, {"host", "device"}, region);
  const JsonNode& host = required_field(document, target, "host", region);
  const JsonNode& device = required_field(document, target, "device", region);
  closed_object(document, host,
                {"triple", "endianness", "pointerBits", "dataLayout"}, region);
  closed_object(document, device,
                {"triple", "architecture", "endianness", "pointerBits",
                 "dataLayout"}, region);
  require_literal(document, required_field(document, host, "endianness", region),
                  "little", region);
  require_literal(document, required_field(document, device, "endianness", region),
                  "little", region);
  const JsonNode& host_bits = required_field(document, host, "pointerBits", region);
  const JsonNode& device_bits = required_field(document, device, "pointerBits", region);
  if (host_bits.kind != JsonKind::kInteger || host_bits.integer != 64 ||
      device_bits.kind != JsonKind::kInteger || device_bits.integer != 64) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, target.begin);
  }
  require_literal(document, required_field(document, device, "triple", region),
                  "nvptx64-nvidia-cuda", region);
  for (const JsonNode* field : {
           &required_field(document, host, "triple", region),
           &required_field(document, host, "dataLayout", region),
           &required_field(document, device, "architecture", region),
           &required_field(document, device, "dataLayout", region)}) {
    require_string(*field, region);
  }
  if (storage.passes.size() != 2U ||
      storage.passes[0].target_triple != document.raw_string(
          required_field(document, device, "triple", region)) ||
      storage.passes[0].auxiliary_target_triple != document.raw_string(
          required_field(document, host, "triple", region)) ||
      storage.passes[1].target_triple != document.raw_string(
          required_field(document, host, "triple", region)) ||
      storage.passes[1].auxiliary_target_triple != document.raw_string(
          required_field(document, device, "triple", region)) ||
      storage.passes[0].device_architecture != document.raw_string(
          required_field(document, device, "architecture", region)) ||
      storage.passes[1].device_architecture != document.raw_string(
          required_field(document, device, "architecture", region))) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, target.begin);
  }
}

void validate_toolchain(const JsonDocument& document, const JsonNode& toolchain,
                        SessionStorage& storage, CompileSessionRegion region) {
  closed_object(document, toolchain, {"compiler", "dependencies"}, region);
  const JsonNode& compiler = required_field(document, toolchain, "compiler", region);
  closed_object(document, compiler,
                {"id", "version", "buildId", "binarySha256",
                 "resourceDirectoryVirtualPath", "resourceDirectorySha256"},
                region);
  require_literal(document, required_field(document, compiler, "id", region),
                  "clang", region, CompileSessionDecodeStatus::kAbiMismatch,
                  CompileSessionDecodeReason::kCompilerIdentity);
  const JsonNode& compiler_version =
      required_field(document, compiler, "version", region);
  require_literal(document, compiler_version, "22.1.8", region,
                  CompileSessionDecodeStatus::kAbiMismatch,
                  CompileSessionDecodeReason::kCompilerIdentity);
  copy_string(document, compiler_version, storage.compiler_version, region);
  require_string(required_field(document, compiler, "buildId", region), region);
  require_sha256(document, required_field(document, compiler, "binarySha256", region), region);
  require_sha256(document,
                 required_field(document, compiler, "resourceDirectorySha256", region),
                 region);
  copy_string(document,
              required_field(document, compiler,
                             "resourceDirectoryVirtualPath", region),
              storage.compiler_resource_virtual_path, region);
  if (!cpp_cute_valid_canonical_virtual_path(
          storage.compiler_resource_virtual_path) ||
      storage.compiler_resource_virtual_path == "/" ||
      storage.compiler_resource_virtual_path.size() +
              std::string_view("/include").size() >
          4096U) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema,
           required_field(document, compiler,
                          "resourceDirectoryVirtualPath", region)
               .begin);
  }
  copy_string(document, required_field(document, compiler, "binarySha256", region),
              storage.compiler_binary_sha256, region);
  copy_string(document,
              required_field(document, compiler, "resourceDirectorySha256", region),
              storage.compiler_resource_sha256, region);
  if (storage.compiler_binary_sha256 != storage.extractor_binary_sha256) {
    reject(CompileSessionDecodeStatus::kAbiMismatch, region,
           CompileSessionDecodeReason::kCompilerIdentity, compiler.begin);
  }

  const JsonNode& dependencies = required_field(document, toolchain, "dependencies", region);
  if (dependencies.kind != JsonKind::kArray ||
      document.child_count(dependencies) == 0U) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, dependencies.begin);
  }
  std::array<std::uint32_t, 6U> kinds{};
  constexpr std::array<std::string_view, 6U> kind_names = {{
      "cuda-toolkit", "cutlass", "cccl", "cxx-standard-library",
      "c-system-headers", "linux-sysroot",
  }};
  for (std::uint32_t index = 0; index < document.child_count(dependencies); ++index) {
    const JsonNode& dependency = *document.element(dependencies, index);
    closed_object(document, dependency,
                  {"dependencyId", "kind", "version", "revision",
                   "headerSetSha256"}, region);
    storage.dependencies.emplace_back(&storage.memory);
    OwnedDependency& output = storage.dependencies.back();
    copy_string(document,
                required_field(document, dependency, "dependencyId", region),
                output.dependency_id, region);
    copy_string(document, required_field(document, dependency, "kind", region),
                output.kind, region);
    if (!dependency_id(output.dependency_id) ||
        (index != 0U && output.dependency_id <=
                            storage.dependencies[index - 1U].dependency_id)) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, dependency.begin);
    }
    const auto found = std::find(kind_names.begin(), kind_names.end(), output.kind);
    if (found == kind_names.end()) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, dependency.begin);
    }
    const std::size_t kind_index = static_cast<std::size_t>(found - kind_names.begin());
    if (++kinds[kind_index] > 1U) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, dependency.begin);
    }
    require_string(required_field(document, dependency, "version", region), region);
    require_string(required_field(document, dependency, "revision", region), region);
    require_sha256(document,
                   required_field(document, dependency, "headerSetSha256", region),
                   region);
    copy_string(document,
                required_field(document, dependency, "headerSetSha256", region),
                output.header_set_sha256, region);
  }
  if (kinds[0] != 1U || kinds[1] != 1U || kinds[3] != 1U ||
      kinds[4] + kinds[5] != 1U) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, dependencies.begin);
  }
}

const OwnedIncludeRoot* find_include_root(const SessionStorage& storage,
                                          std::string_view id) {
  const auto found = std::find_if(
      storage.include_roots.begin(), storage.include_roots.end(),
      [id](const OwnedIncludeRoot& root) { return root.include_root_id == id; });
  return found == storage.include_roots.end() ? nullptr : &*found;
}

std::int32_t include_root_search_rank(const SessionStorage& storage,
                                      const OwnedIncludeRoot& root) {
  if (root.owner_kind == "source") return 0;
  if (root.owner_kind == "compiler-resource-directory") return 3;
  const auto dependency = std::find_if(
      storage.dependencies.begin(), storage.dependencies.end(),
      [&root](const OwnedDependency& candidate) {
        return candidate.dependency_id == root.dependency_id;
      });
  if (dependency == storage.dependencies.end()) return -1;
  if (dependency->kind == "cuda-toolkit" || dependency->kind == "cutlass" ||
      dependency->kind == "cccl") {
    return 1;
  }
  return dependency->kind == "cxx-standard-library" ? 2 : 4;
}

void validate_virtual_file_system(const JsonDocument& document,
                                  const JsonNode& vfs, SessionStorage& storage,
                                  CompileSessionRegion region) {
  closed_object(document, vfs, {"sourceRoots", "includeRoots"}, region);
  const JsonNode& source_roots = required_field(document, vfs, "sourceRoots", region);
  if (source_roots.kind != JsonKind::kArray ||
      document.child_count(source_roots) == 0U ||
      document.child_count(source_roots) > 256U) {
    reject(CompileSessionDecodeStatus::kResourceLimit, region,
           CompileSessionDecodeReason::kLimit, source_roots.begin);
  }
  for (std::uint32_t index = 0; index < document.child_count(source_roots); ++index) {
    const JsonNode& root = *document.element(source_roots, index);
    PmrString value(&storage.memory);
    copy_string(document, root, value, region);
    if (!cpp_cute_valid_canonical_virtual_path(value) ||
        std::find(storage.source_roots.begin(), storage.source_roots.end(), value) !=
            storage.source_roots.end()) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, root.begin);
    }
    storage.source_roots.push_back(std::move(value));
  }

  const JsonNode& include_roots = required_field(document, vfs, "includeRoots", region);
  if (include_roots.kind != JsonKind::kArray ||
      document.child_count(include_roots) == 0U ||
      document.child_count(include_roots) > 256U) {
    reject(CompileSessionDecodeStatus::kResourceLimit, region,
           CompileSessionDecodeReason::kLimit, include_roots.begin);
  }
  std::uint32_t compiler_resource_roots = 0U;
  for (std::uint32_t index = 0; index < document.child_count(include_roots); ++index) {
    const JsonNode& root = *document.element(include_roots, index);
    closed_object(document, root,
                  {"includeRootId", "mode", "virtualPath", "manifestSha256",
                   "owner"}, region);
    storage.include_roots.emplace_back(&storage.memory);
    OwnedIncludeRoot& output = storage.include_roots.back();
    copy_string(document, required_field(document, root, "includeRootId", region),
                output.include_root_id, region);
    copy_string(document, required_field(document, root, "mode", region),
                output.mode, region);
    copy_string(document, required_field(document, root, "virtualPath", region),
                output.virtual_path, region);
    copy_string(document, required_field(document, root, "manifestSha256", region),
                output.manifest_sha256, region);
    if (!dependency_id(output.include_root_id) ||
        (output.mode != "quote" && output.mode != "system") ||
        !cpp_cute_valid_canonical_virtual_path(output.virtual_path) ||
        !lowercase_sha256(output.manifest_sha256)) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, root.begin);
    }
    for (std::uint32_t previous = 0; previous < index; ++previous) {
      if (storage.include_roots[previous].include_root_id == output.include_root_id ||
          storage.include_roots[previous].virtual_path == output.virtual_path) {
        reject(CompileSessionDecodeStatus::kInvalidFrame, region,
               CompileSessionDecodeReason::kSchema, root.begin);
      }
    }
    const JsonNode& owner = required_field(document, root, "owner", region);
    const JsonNode& owner_kind = required_field(document, owner, "kind", region);
    copy_string(document, owner_kind, output.owner_kind, region);
    if (output.owner_kind == "source" ||
        output.owner_kind == "compiler-resource-directory") {
      closed_object(document, owner, {"kind"}, region);
      if (output.owner_kind == "source") {
        if (output.mode != "quote") {
          reject(CompileSessionDecodeStatus::kInvalidFrame, region,
                 CompileSessionDecodeReason::kSchema, root.begin);
        }
        const std::size_t containers = static_cast<std::size_t>(std::count_if(
            storage.source_roots.begin(), storage.source_roots.end(),
            [&output](const PmrString& source) {
              return cpp_cute_virtual_path_contains(source,
                                                     output.virtual_path);
            }));
        if (containers != 1U) {
          reject(CompileSessionDecodeStatus::kInvalidFrame, region,
                 CompileSessionDecodeReason::kSchema, root.begin);
        }
      } else {
        ++compiler_resource_roots;
        PmrString expected_virtual_path(
            storage.compiler_resource_virtual_path, &storage.memory);
        expected_virtual_path += "/include";
        if (output.mode != "system" ||
            output.virtual_path != expected_virtual_path ||
            output.manifest_sha256 != storage.compiler_resource_sha256) {
          reject(CompileSessionDecodeStatus::kInvalidFrame, region,
                 CompileSessionDecodeReason::kSchema, root.begin);
        }
      }
    } else if (output.owner_kind == "dependency") {
      closed_object(document, owner, {"kind", "dependencyId"}, region);
      copy_string(document,
                  required_field(document, owner, "dependencyId", region),
                  output.dependency_id, region);
      const auto dependency = std::find_if(
          storage.dependencies.begin(), storage.dependencies.end(),
          [&output](const OwnedDependency& candidate) {
            return candidate.dependency_id == output.dependency_id;
          });
      if (dependency == storage.dependencies.end() ||
          dependency->header_set_sha256 != output.manifest_sha256 ||
          output.mode != "system") {
        reject(CompileSessionDecodeStatus::kInvalidFrame, region,
               CompileSessionDecodeReason::kSchema, root.begin);
      }
      if (dependency->kind == "cuda-toolkit") {
        constexpr std::string_view kIncludeSuffix = "/include";
        if (!storage.cuda_toolkit_root_virtual_path.empty() ||
            output.virtual_path.size() <= kIncludeSuffix.size() ||
            !std::string_view(output.virtual_path).ends_with(kIncludeSuffix)) {
          reject(CompileSessionDecodeStatus::kInvalidFrame, region,
                 CompileSessionDecodeReason::kSchema, root.begin);
        }
        storage.cuda_toolkit_root_virtual_path.assign(
            output.virtual_path.begin(),
            output.virtual_path.end() - kIncludeSuffix.size());
      }
    } else {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, owner.begin);
    }
  }
  if (compiler_resource_roots != 1U) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, include_roots.begin);
  }
  for (const OwnedDependency& dependency : storage.dependencies) {
    if (std::none_of(storage.include_roots.begin(), storage.include_roots.end(),
                     [&dependency](const OwnedIncludeRoot& root) {
                       return root.owner_kind == "dependency" &&
                              root.dependency_id == dependency.dependency_id;
                     })) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, include_roots.begin);
    }
  }
  std::int32_t prior_search_rank = -1;
  for (const OwnedIncludeRoot& root : storage.include_roots) {
    const std::int32_t search_rank = include_root_search_rank(storage, root);
    if (search_rank < 0 || search_rank < prior_search_rank) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, include_roots.begin);
    }
    prior_search_rank = search_rank;
  }
  for (const OwnedCompilerOption& option : storage.options) {
    if (option.kind != CompilerOptionKind::kForcedInclude) continue;
    const OwnedIncludeRoot* root = find_include_root(storage, option.include_root_id);
    if (root == nullptr || root->virtual_path == option.virtual_path ||
        !cpp_cute_virtual_path_contains(root->virtual_path,
                                        option.virtual_path)) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, include_roots.begin);
    }
  }
}

void validate_compatibility(const JsonDocument& document,
                            const JsonNode& compatibility,
                            CompileSessionRegion region) {
  closed_object(document, compatibility,
                {"supportedSourceFeatures", "unsupportedSourceFeatures",
                 "unsupportedIntrinsicFamilies"}, region);
  const JsonNode& supported = required_field(
      document, compatibility, "supportedSourceFeatures", region);
  const JsonNode& unsupported = required_field(
      document, compatibility, "unsupportedSourceFeatures", region);
  validate_string_array(document, supported, region, true);
  validate_string_array(document, unsupported, region);
  validate_string_array(document, required_field(
      document, compatibility, "unsupportedIntrinsicFamilies", region), region);
  bool temporal_unsupported = false;
  bool temporal_supported = false;
  for (std::uint32_t index = 0; index < document.child_count(unsupported); ++index) {
    temporal_unsupported |= string_equals(document, document.element(unsupported, index),
                                           "cxx:temporal-macros@1");
  }
  for (std::uint32_t index = 0; index < document.child_count(supported); ++index) {
    temporal_supported |= string_equals(document, document.element(supported, index),
                                         "cxx:temporal-macros@1");
  }
  if (!temporal_unsupported || temporal_supported) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, compatibility.begin);
  }
}

void validate_profile_limits(const JsonDocument& document,
                             const JsonNode& limits, SessionStorage& storage,
                             CompileSessionRegion region) {
  closed_exact(document, limits, kAllLimitKeys, region);
  for (std::size_t index = 0; index < kSemanticLimitKeys.size(); ++index) {
    storage.profile_semantic_limits[index] = positive_integer(
        required_field(document, limits, kSemanticLimitKeys[index], region), region,
        kMaximumProfileLimits[index]);
  }
  std::array<std::uint64_t, 4U> operational{};
  for (std::size_t index = 20U; index < kAllLimitKeys.size(); ++index) {
    operational[index - 20U] = positive_integer(
        required_field(document, limits, kAllLimitKeys[index], region), region,
        kMaximumProfileLimits[index]);
  }
  if (operational[1] > operational[0] * operational[3]) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, limits.begin);
  }
  if (storage.profile_semantic_limits[19] > kMaximumResultByteLength) {
    reject(CompileSessionDecodeStatus::kResourceLimit, region,
           CompileSessionDecodeReason::kLimit,
           required_field(document, limits, "maxOutputBytes", region).begin);
  }
  const std::uint64_t reserved = storage.runtime_stack_bytes +
                                 storage.runtime_compiler_working_bytes +
                                 kRuntimeV1MaxDocumentByteLength +
                                 storage.profile_semantic_limits[19];
  if (operational[2] > storage.runtime_maximum_linear_bytes ||
      reserved > operational[2] ||
      storage.profile_semantic_limits[1] + storage.profile_semantic_limits[3] >
          storage.runtime_vfs_live_open_bytes) {
    reject(CompileSessionDecodeStatus::kResourceLimit, region,
           CompileSessionDecodeReason::kLimit, limits.begin);
  }
}

void validate_profile(const JsonDocument& document, SessionStorage& storage) {
  constexpr CompileSessionRegion region = CompileSessionRegion::kProfile;
  const JsonNode& root = document.root();
  closed_object(document, root,
                {"schema", "version", "profileId", "deployment", "language",
                 "target", "toolchain", "virtualFileSystem", "compatibility",
                 "extractionLimits"}, region);
  require_literal(document, required_field(document, root, "schema", region),
                  kProfileSchema, region);
  validate_version(document, required_field(document, root, "version", region),
                   kProfileMajor, kProfileMinor, region);
  copy_string(document, required_field(document, root, "profileId", region),
              storage.profile_id, region);
  if (!storage.profile_id.starts_with("browsergrad.compiler.cpp-cute.") ||
      storage.profile_id.find('@') == std::string_view::npos) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema,
           required_field(document, root, "profileId", region).begin);
  }
  validate_browser_deployment(
      document, required_field(document, root, "deployment", region), storage,
      region);
  validate_language(document, required_field(document, root, "language", region),
                    storage, region);
  validate_target(document, required_field(document, root, "target", region),
                  storage, region);
  validate_toolchain(document, required_field(document, root, "toolchain", region),
                     storage, region);
  validate_virtual_file_system(
      document, required_field(document, root, "virtualFileSystem", region),
      storage, region);
  validate_compatibility(
      document, required_field(document, root, "compatibility", region), region);
  validate_profile_limits(
      document, required_field(document, root, "extractionLimits", region),
      storage, region);
}

void emit_member(HashWriter& writer, const JsonDocument& document,
                 std::string_view key, const JsonNode& value, bool& first) {
  if (!first) writer.text(",");
  first = false;
  writer.text("\"");
  writer.text(key);
  writer.text("\":");
  hash_raw(writer, document, value);
}

std::string hash_compilation_contract(const JsonDocument& profile) {
  const JsonNode& root = profile.root();
  const JsonNode& deployment = *profile.field(root, "deployment");
  const JsonNode& extractor = *profile.field(deployment, "extractor");
  const JsonNode& toolchain = *profile.field(root, "toolchain");
  const JsonNode& compiler = *profile.field(toolchain, "compiler");
  const JsonNode& limits = *profile.field(root, "extractionLimits");
  HashWriter writer;
  writer.text("{\"contract\":{");
  bool first = true;
  emit_member(writer, profile, "compatibility", *profile.field(root, "compatibility"), first);
  writer.text(",\"compiler\":{");
  bool compiler_first = true;
  for (std::string_view key : {"buildId", "id", "resourceDirectorySha256",
                               "resourceDirectoryVirtualPath", "version"}) {
    emit_member(writer, profile, key, *profile.field(compiler, key), compiler_first);
  }
  writer.text("}");
  first = false;
  emit_member(writer, profile, "dependencies", *profile.field(toolchain, "dependencies"), first);
  writer.text(",\"extractionLimits\":{");
  bool limit_first = true;
  for (std::string_view key : kCanonicalSemanticLimitKeys) {
    emit_member(writer, profile, key, *profile.field(limits, key), limit_first);
  }
  writer.text("}");
  emit_member(writer, profile, "language", *profile.field(root, "language"), first);
  writer.text(",\"schema\":\"");
  writer.text(kContractSchema);
  writer.text("\"");
  writer.text(",\"semanticAdapterManifestSha256\":");
  hash_raw(writer, profile, *profile.field(extractor, "semanticAdapterManifestSha256"));
  emit_member(writer, profile, "target", *profile.field(root, "target"), first);
  writer.text(",\"version\":{\"major\":");
  writer.text(std::to_string(kContractMajor));
  writer.text(",\"minor\":");
  writer.text(std::to_string(kContractMinor));
  writer.text("}");
  emit_member(writer, profile, "virtualFileSystem",
              *profile.field(root, "virtualFileSystem"), first);
  writer.text("},\"domain\":\"browsergrad.compiler.cpp-cute.compilation-contract.v1\"}");
  return writer.finish();
}

std::string hash_source_file(const JsonDocument& request,
                             const JsonNode& file) {
  HashWriter writer;
  writer.text("{\"domain\":\"browsergrad.compiler.cpp-cute.source-file.v1\",\"file\":{");
  bool first = true;
  for (std::string_view key : {"byteLength", "contentSha256", "includeRootId",
                               "role", "virtualPath"}) {
    emit_member(writer, request, key, *request.field(file, key), first);
  }
  writer.text("}}");
  return writer.finish();
}

std::string hash_entry_request(const JsonDocument& request,
                               const JsonNode& entry) {
  HashWriter writer;
  writer.text("{\"domain\":\"browsergrad.compiler.cpp-cute.entry-request.v1\",\"request\":{");
  bool first = true;
  for (std::string_view key : {"anchor", "declarationKind", "kind"}) {
    emit_member(writer, request, key, *request.field(entry, key), first);
  }
  writer.text("}}");
  return writer.finish();
}

std::string hash_request(const JsonDocument& request) {
  const JsonNode& root = request.root();
  HashWriter writer;
  writer.text("{\"domain\":\"browsergrad.compiler.cpp-cute.frontend-request.v1\",\"request\":{");
  bool first = true;
  for (std::string_view key : {"compilationContractHash", "entryRequests",
                               "expectedArtifact", "files", "limits",
                               "mainVirtualPath", "schema", "version"}) {
    emit_member(writer, request, key, *request.field(root, key), first);
  }
  writer.text("}}");
  return writer.finish();
}

void require_identity(std::string_view actual, std::string_view expected,
                      CompileSessionRegion region, std::uint32_t offset) {
  if (actual != expected) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kIdentityMismatch, offset);
  }
}

void validate_request_limits(const JsonDocument& document,
                             const JsonNode& limits, SessionStorage& storage,
                             CompileSessionRegion region) {
  closed_exact(document, limits, kSemanticLimitKeys, region);
  for (std::size_t index = 0; index < kSemanticLimitKeys.size(); ++index) {
    const JsonNode& node = required_field(document, limits, kSemanticLimitKeys[index], region);
    const std::uint64_t value = positive_integer(node, region);
    if (value > storage.profile_semantic_limits[index]) {
      reject(CompileSessionDecodeStatus::kResourceLimit, region,
             CompileSessionDecodeReason::kLimit, node.begin);
    }
    storage.request_semantic_limits[index] = value;
    if (index == 19U) {
      if (value > kMaximumResultByteLength) {
        reject(CompileSessionDecodeStatus::kResourceLimit, region,
               CompileSessionDecodeReason::kLimit, node.begin);
      }
      storage.maximum_output_byte_length = static_cast<std::uint32_t>(value);
    }
  }
}

void validate_request(const JsonDocument& document, SessionStorage& storage) {
  constexpr CompileSessionRegion region = CompileSessionRegion::kRequest;
  const JsonNode& root = document.root();
  closed_object(document, root,
                {"schema", "version", "requestId", "compilationContractHash",
                 "mainVirtualPath", "files", "entryRequests",
                 "expectedArtifact", "limits"}, region);
  require_literal(document, required_field(document, root, "schema", region),
                  kRequestSchema, region);
  validate_version(document, required_field(document, root, "version", region),
                   kRequestMajor, kRequestMinor, region);
  const JsonNode& contract = required_field(
      document, root, "compilationContractHash", region);
  require_sha256(document, contract, region);
  if (document.raw_string(contract) != storage.compilation_contract_hash) {
    reject(CompileSessionDecodeStatus::kInvalidFrame,
           CompileSessionRegion::kCrossRegion,
           CompileSessionDecodeReason::kContractMismatch, contract.begin);
  }
  copy_string(document, required_field(document, root, "mainVirtualPath", region),
              storage.main_virtual_path, region);
  if (!cpp_cute_valid_canonical_virtual_path(storage.main_virtual_path) ||
      std::none_of(storage.source_roots.begin(), storage.source_roots.end(),
                   [&storage](const PmrString& source_root) {
                     return cpp_cute_virtual_path_contains(
                         source_root, storage.main_virtual_path);
                   })) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema,
           required_field(document, root, "mainVirtualPath", region).begin);
  }
  validate_request_limits(document,
                          required_field(document, root, "limits", region),
                          storage, region);

  const JsonNode& files = required_field(document, root, "files", region);
  if (files.kind != JsonKind::kArray || document.child_count(files) == 0U ||
      document.child_count(files) > storage.request_semantic_limits[0]) {
    reject(CompileSessionDecodeStatus::kResourceLimit, region,
           CompileSessionDecodeReason::kLimit, files.begin);
  }
  std::uint32_t main_count = 0U;
  std::uint64_t total_source_bytes = 0U;
  for (std::uint32_t index = 0; index < document.child_count(files); ++index) {
    const JsonNode& file = *document.element(files, index);
    closed_object(document, file,
                  {"fileId", "role", "virtualPath", "contentSha256",
                   "byteLength", "includeRootId"}, region);
    storage.source_files.emplace_back(&storage.memory);
    OwnedSourceFile& output = storage.source_files.back();
    copy_string(document, required_field(document, file, "fileId", region),
                output.file_id, region);
    copy_string(document, required_field(document, file, "role", region),
                output.role, region);
    copy_string(document, required_field(document, file, "virtualPath", region),
                output.virtual_path, region);
    copy_string(document, required_field(document, file, "contentSha256", region),
                output.content_sha256, region);
    copy_string(document, required_field(document, file, "byteLength", region),
                output.byte_length, region);
    if (!output.file_id.starts_with("bg.cpp.file.sha256.") ||
        output.file_id.size() != 83U || !lowercase_sha256(output.content_sha256) ||
        !cpp_cute_valid_canonical_virtual_path(output.virtual_path) ||
        !canonical_u64(output.byte_length, &output.byte_length_value) ||
        output.byte_length_value == 0U ||
        std::none_of(storage.source_roots.begin(), storage.source_roots.end(),
                     [&output](const PmrString& source_root) {
                       return cpp_cute_virtual_path_contains(
                           source_root, output.virtual_path);
                     }) ||
        (index != 0U && compare_canonical_strings(
                            output.virtual_path,
                            storage.source_files[index - 1U].virtual_path) <= 0)) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, file.begin);
    }
    if (std::any_of(storage.source_files.begin(), storage.source_files.end() - 1,
                    [&output](const OwnedSourceFile& candidate) {
                      return candidate.file_id == output.file_id;
                    })) {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, file.begin);
    }
    const JsonNode& include_root = required_field(document, file, "includeRootId", region);
    if (output.role == "main-source") {
      if (include_root.kind != JsonKind::kNull ||
          output.virtual_path != storage.main_virtual_path) {
        reject(CompileSessionDecodeStatus::kInvalidFrame, region,
               CompileSessionDecodeReason::kSchema, file.begin);
      }
      ++main_count;
    } else if (output.role == "project-header") {
      copy_string(document, include_root, output.include_root_id, region);
      output.has_include_root = true;
      const OwnedIncludeRoot* root = find_include_root(storage, output.include_root_id);
      if (root == nullptr || root->owner_kind != "source" ||
          !cpp_cute_virtual_path_contains(root->virtual_path,
                                          output.virtual_path)) {
        reject(CompileSessionDecodeStatus::kInvalidFrame, region,
               CompileSessionDecodeReason::kSchema, file.begin);
      }
    } else {
      reject(CompileSessionDecodeStatus::kInvalidFrame, region,
             CompileSessionDecodeReason::kSchema, file.begin);
    }
    if (output.byte_length_value >
        std::numeric_limits<std::uint64_t>::max() - total_source_bytes) {
      reject(CompileSessionDecodeStatus::kResourceLimit, region,
             CompileSessionDecodeReason::kLimit, file.begin);
    }
    total_source_bytes += output.byte_length_value;
    const std::string expected = prefixed_digest(
        "bg.cpp.file.sha256.", hash_source_file(document, file));
    require_identity(output.file_id, expected, region,
                     required_field(document, file, "fileId", region).begin);
  }
  if (main_count != 1U || total_source_bytes > storage.request_semantic_limits[1]) {
    reject(total_source_bytes > storage.request_semantic_limits[1]
               ? CompileSessionDecodeStatus::kResourceLimit
               : CompileSessionDecodeStatus::kInvalidFrame,
           region, total_source_bytes > storage.request_semantic_limits[1]
                       ? CompileSessionDecodeReason::kLimit
                       : CompileSessionDecodeReason::kSchema,
           files.begin);
  }

  const JsonNode& entries = required_field(document, root, "entryRequests", region);
  if (entries.kind != JsonKind::kArray || document.child_count(entries) != 1U) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, entries.begin);
  }
  const JsonNode& entry = *document.element(entries, 0U);
  closed_object(document, entry,
                {"requestId", "kind", "declarationKind", "anchor"}, region);
  copy_string(document, required_field(document, entry, "requestId", region),
              storage.entry.request_id, region);
  copy_string(document, required_field(document, entry, "kind", region),
              storage.entry.kind, region);
  copy_string(document, required_field(document, entry, "declarationKind", region),
              storage.entry.declaration_kind, region);
  if (!((storage.entry.kind == "layout" &&
         storage.entry.declaration_kind == "variable") ||
        (storage.entry.kind == "view-copy" &&
         storage.entry.declaration_kind == "function"))) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, entry.begin);
  }
  const JsonNode& anchor = required_field(document, entry, "anchor", region);
  closed_object(document, anchor,
                {"virtualPath", "beginByte", "endByte", "tokenSha256"}, region);
  copy_string(document, required_field(document, anchor, "virtualPath", region),
              storage.entry.virtual_path, region);
  copy_string(document, required_field(document, anchor, "beginByte", region),
              storage.entry.begin_byte, region);
  copy_string(document, required_field(document, anchor, "endByte", region),
              storage.entry.end_byte, region);
  copy_string(document, required_field(document, anchor, "tokenSha256", region),
              storage.entry.token_sha256, region);
  if (storage.entry.virtual_path != storage.main_virtual_path ||
      !canonical_u64(storage.entry.begin_byte, &storage.entry.begin_value) ||
      !canonical_u64(storage.entry.end_byte, &storage.entry.end_value) ||
      storage.entry.begin_value >= storage.entry.end_value ||
      storage.entry.end_value - storage.entry.begin_value > 256U ||
      !lowercase_sha256(storage.entry.token_sha256)) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, anchor.begin);
  }
  const auto main_file = std::find_if(
      storage.source_files.begin(), storage.source_files.end(),
      [](const OwnedSourceFile& file) { return file.role == "main-source"; });
  if (main_file == storage.source_files.end() ||
      storage.entry.end_value > main_file->byte_length_value) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, region,
           CompileSessionDecodeReason::kSchema, anchor.begin);
  }
  const std::string expected_entry = prefixed_digest(
      "bg.cpp.entry-request.sha256.", hash_entry_request(document, entry));
  require_identity(storage.entry.request_id, expected_entry, region,
                   required_field(document, entry, "requestId", region).begin);

  const JsonNode& artifact = required_field(document, root, "expectedArtifact", region);
  closed_object(document, artifact, {"schema", "version"}, region);
  require_literal(document, required_field(document, artifact, "schema", region),
                  kArtifactSchema, region, CompileSessionDecodeStatus::kAbiMismatch,
                  CompileSessionDecodeReason::kUnsupportedVersion);
  validate_version(document,
                   required_field(document, artifact, "version", region),
                   kArtifactMajor, kArtifactMinor, region);
  const JsonNode& request_id = required_field(document, root, "requestId", region);
  copy_string(document, request_id, storage.request_id, region);
  storage.request_hash = hash_request(document);
  require_identity(storage.request_id,
                   prefixed_digest("bg.cpp.frontend-request.sha256.",
                                   std::string(storage.request_hash)),
                   region, request_id.begin);
}

void validate_canonical_region(ByteRegion region, CompileSessionRegion label) {
  if (region.bytes == nullptr || region.length == 0U ||
      region.length > kRuntimeV1MaxDocumentByteLength) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, label,
           CompileSessionDecodeReason::kCanonicalJson, 0U);
  }
  const CanonicalJsonValidation validation = validate_canonical_json(
      region.bytes, region.length, kRuntimeV1CanonicalJsonLimits);
  if (validation.status == CanonicalJsonStatus::kResourceLimit) {
    reject(CompileSessionDecodeStatus::kResourceLimit, label,
           CompileSessionDecodeReason::kLimit,
           validation.error_byte_offset);
  }
  if (validation.status != CanonicalJsonStatus::kValid) {
    reject(CompileSessionDecodeStatus::kInvalidFrame, label,
           CompileSessionDecodeReason::kCanonicalJson,
           validation.error_byte_offset);
  }
}

CompilerOptionView option_view(const OwnedCompilerOption& option) noexcept {
  return {option.kind, option.ordinal, option.name_or_id,
          option.value_or_disposition, option.include_root_id,
          option.virtual_path, option.has_value};
}

SemanticPassView pass_view(const OwnedSemanticPass& pass) noexcept {
  return {pass.ordinal, pass.pass_id, pass.domain, pass.role,
          pass.invocation_mode, pass.target_triple,
          pass.auxiliary_target_triple, pass.device_architecture};
}

IncludeRootView root_view(const OwnedIncludeRoot& root) noexcept {
  return {0U, root.include_root_id, root.mode, root.virtual_path,
          root.manifest_sha256, root.owner_kind, root.dependency_id};
}

SourceFileView file_view(const OwnedSourceFile& file) noexcept {
  return {file.file_id, file.role, file.virtual_path, file.content_sha256,
          file.byte_length, file.include_root_id, file.has_include_root};
}

EntryRequestView entry_view(const OwnedEntryRequest& entry) noexcept {
  return {entry.request_id, entry.kind, entry.declaration_kind,
          entry.virtual_path, entry.begin_byte, entry.end_byte,
          entry.token_sha256};
}

}  // namespace

DecodedCompileSession::DecodedCompileSession(
    std::unique_ptr<const Impl> implementation)
    : implementation_(std::move(implementation)) {}

DecodedCompileSession::~DecodedCompileSession() = default;

std::string_view DecodedCompileSession::profile_id() const noexcept {
  return implementation_->storage.profile_id;
}

std::string_view DecodedCompileSession::profile_hash() const noexcept {
  return implementation_->storage.profile_hash;
}

std::string_view DecodedCompileSession::compilation_contract_hash() const noexcept {
  return implementation_->storage.compilation_contract_hash;
}

std::string_view DecodedCompileSession::compiler_version() const noexcept {
  return implementation_->storage.compiler_version;
}

std::string_view
DecodedCompileSession::compiler_resource_directory_virtual_path() const noexcept {
  return implementation_->storage.compiler_resource_virtual_path;
}

std::string_view
DecodedCompileSession::cuda_toolkit_root_virtual_path() const noexcept {
  return implementation_->storage.cuda_toolkit_root_virtual_path;
}

std::string_view DecodedCompileSession::request_id() const noexcept {
  return implementation_->storage.request_id;
}

std::string_view DecodedCompileSession::request_hash() const noexcept {
  return implementation_->storage.request_hash;
}

std::string_view DecodedCompileSession::main_virtual_path() const noexcept {
  return implementation_->storage.main_virtual_path;
}

std::uint32_t DecodedCompileSession::maximum_output_byte_length() const noexcept {
  return implementation_->storage.maximum_output_byte_length;
}

std::uint64_t DecodedCompileSession::request_semantic_limit(
    CompileSemanticLimit limit) const noexcept {
  return implementation_->storage.request_semantic_limits[
      static_cast<std::size_t>(limit)];
}

std::size_t DecodedCompileSession::compiler_option_count() const noexcept {
  return implementation_->storage.options.size();
}

CompilerOptionView DecodedCompileSession::compiler_option(
    std::size_t index) const noexcept {
  if (index >= implementation_->storage.options.size()) return {};
  return option_view(implementation_->storage.options[index]);
}

std::size_t DecodedCompileSession::semantic_pass_count() const noexcept {
  return implementation_->storage.passes.size();
}

SemanticPassView DecodedCompileSession::semantic_pass(
    std::size_t index) const noexcept {
  if (index >= implementation_->storage.passes.size()) return {};
  return pass_view(implementation_->storage.passes[index]);
}

std::size_t DecodedCompileSession::include_root_count() const noexcept {
  return implementation_->storage.include_roots.size();
}

IncludeRootView DecodedCompileSession::include_root(
    std::size_t index) const noexcept {
  if (index >= implementation_->storage.include_roots.size()) return {};
  IncludeRootView result = root_view(implementation_->storage.include_roots[index]);
  result.ordinal = static_cast<std::uint32_t>(index);
  return result;
}

std::size_t DecodedCompileSession::source_file_count() const noexcept {
  return implementation_->storage.source_files.size();
}

SourceFileView DecodedCompileSession::source_file(
    std::size_t index) const noexcept {
  if (index >= implementation_->storage.source_files.size()) return {};
  return file_view(implementation_->storage.source_files[index]);
}

EntryRequestView DecodedCompileSession::entry_request() const noexcept {
  return entry_view(implementation_->storage.entry);
}

CompileSessionDecodeResult decode_compile_session(
    const ValidatedInputFrameRegions& regions) noexcept {
  CompileSessionDecodeResult result;
  try {
    const ByteRegion profile_region{
        regions.profile_bytes(), regions.profile_byte_length()};
    const ByteRegion request_region{
        regions.request_bytes(), regions.request_byte_length()};
    validate_canonical_region(profile_region, CompileSessionRegion::kProfile);
    validate_canonical_region(request_region, CompileSessionRegion::kRequest);

    auto implementation = std::make_unique<DecodedCompileSession::Impl>();
    BudgetResource scratch(implementation->storage.budget);
    JsonDocument profile(profile_region, &scratch);
    JsonDocument request(request_region, &scratch);
    validate_profile(profile, implementation->storage);
    implementation->storage.profile_hash = hash_profile(profile);
    implementation->storage.compilation_contract_hash =
        hash_compilation_contract(profile);
    validate_request(request, implementation->storage);

    std::unique_ptr<const DecodedCompileSession::Impl> immutable(
        implementation.release());
    result.session = std::unique_ptr<DecodedCompileSession>(
        new DecodedCompileSession(std::move(immutable)));
    result.status = CompileSessionDecodeStatus::kReady;
    result.failure = {};
    return result;
  } catch (const DecodeError& error) {
    result.status = error.status;
    result.failure = error.failure;
    return result;
  } catch (const std::bad_alloc&) {
    result.status = CompileSessionDecodeStatus::kResourceLimit;
    result.failure = {CompileSessionRegion::kNone,
                      CompileSessionDecodeReason::kAllocation, 0U};
    return result;
  } catch (...) {
    result.status = CompileSessionDecodeStatus::kInternalError;
    result.failure = {CompileSessionRegion::kNone,
                      CompileSessionDecodeReason::kSchema, 0U};
    return result;
  }
}

}  // namespace browsergrad::cpp_cute
