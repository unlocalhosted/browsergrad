#pragma once

#include <cstdint>

namespace browsergrad::cpp_cute {

struct CanonicalJsonLimits {
  std::uint32_t max_document_byte_length = 0;
  std::uint32_t max_depth = 0;
  std::uint32_t max_nodes = 0;
  std::uint32_t max_decoded_string_byte_length = 0;
  std::uint32_t max_array_length = 0;
  std::uint32_t max_object_property_count = 0;
};

inline constexpr std::uint32_t kRuntimeV1MaxDocumentByteLength = 4194304U;
inline constexpr std::uint32_t kRuntimeV1MaxNestingDepth = 128U;
inline constexpr std::uint32_t kRuntimeV1MaxNodeCount = 1000000U;
inline constexpr std::uint32_t kRuntimeV1MaxStringByteLength = 4194304U;
inline constexpr std::uint32_t kRuntimeV1MaxArrayElementCount = 65536U;
inline constexpr std::uint32_t kRuntimeV1MaxObjectPropertyCount = 512U;

inline constexpr CanonicalJsonLimits kRuntimeV1CanonicalJsonLimits = {
    kRuntimeV1MaxDocumentByteLength,
    kRuntimeV1MaxNestingDepth,
    kRuntimeV1MaxNodeCount,
    kRuntimeV1MaxStringByteLength,
    kRuntimeV1MaxArrayElementCount,
    kRuntimeV1MaxObjectPropertyCount,
};

enum class CanonicalJsonStatus {
  kValid,
  kInvalid,
  kResourceLimit,
};

struct CanonicalJsonValidation {
  CanonicalJsonStatus status = CanonicalJsonStatus::kInvalid;
  std::uint32_t error_byte_offset = 0;
  std::uint32_t node_count = 0;
  std::uint32_t decoded_string_byte_length = 0;
  std::uint32_t maximum_depth = 0;
};

/**
 * Validates BrowserGrad semantic canonical JSON directly from immutable UTF-8.
 *
 * The validator allocates nothing and consults no locale or ambient state.
 * Object keys use lexicographic UTF-16 code-unit order, matching the shared
 * TypeScript canonicalizer. JSON numbers are canonical safe integers only.
 */
CanonicalJsonValidation validate_canonical_json(
    const std::uint8_t* bytes, std::uint32_t byte_length,
    const CanonicalJsonLimits& limits);

}  // namespace browsergrad::cpp_cute
