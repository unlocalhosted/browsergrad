#include "BrowserGradCppCuteArtifactJson.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "BrowserGradCppCuteSha256.h"

namespace browsergrad::cpp_cute::artifact_json {
namespace {

constexpr std::uint32_t kMaximumJsonDepth = 128U;
constexpr std::uint32_t kMaximumJsonNodes = 1000000U;
constexpr std::uint32_t kMaximumJsonArrayLength = 65536U;
constexpr std::uint32_t kMaximumJsonObjectPropertyCount = 512U;

struct SerializationState final {
  std::size_t maximum_byte_length = 0U;
  std::uint32_t node_count = 0U;
};

void append_checked(std::string& output, const std::string_view bytes,
                    const std::size_t maximum) {
  if (bytes.size() > maximum - output.size()) throw ArtifactResourceLimit();
  output.append(bytes);
}

void append_checked(std::string& output, const char byte,
                    const std::size_t maximum) {
  if (output.size() == maximum) throw ArtifactResourceLimit();
  output.push_back(byte);
}

void serialize_json_string(const std::string_view input, std::string& output,
                           const std::size_t maximum) {
  append_checked(output, '"', maximum);
  constexpr std::array<char, 16U> kHex = {'0', '1', '2', '3', '4', '5',
                                          '6', '7', '8', '9', 'a', 'b',
                                          'c', 'd', 'e', 'f'};
  for (const unsigned char byte : input) {
    switch (byte) {
      case '"':
        append_checked(output, "\\\"", maximum);
        break;
      case '\\':
        append_checked(output, "\\\\", maximum);
        break;
      case '\b':
        append_checked(output, "\\b", maximum);
        break;
      case '\f':
        append_checked(output, "\\f", maximum);
        break;
      case '\n':
        append_checked(output, "\\n", maximum);
        break;
      case '\r':
        append_checked(output, "\\r", maximum);
        break;
      case '\t':
        append_checked(output, "\\t", maximum);
        break;
      default:
        if (byte < 0x20U) {
          std::array<char, 6U> encoded = {
              '\\', 'u', '0', '0', kHex[byte >> 4U], kHex[byte & 0x0fU]};
          append_checked(output,
                         std::string_view(encoded.data(), encoded.size()),
                         maximum);
        } else {
          append_checked(output, static_cast<char>(byte), maximum);
        }
    }
  }
  append_checked(output, '"', maximum);
}

void serialize_json(const Json& input, std::string& output,
                    SerializationState& state, const std::uint32_t depth) {
  if (depth > kMaximumJsonDepth || state.node_count == kMaximumJsonNodes) {
    throw ArtifactResourceLimit();
  }
  ++state.node_count;
  if (std::holds_alternative<std::nullptr_t>(input.value)) {
    append_checked(output, "null", state.maximum_byte_length);
    return;
  }
  if (const auto* value = std::get_if<bool>(&input.value)) {
    append_checked(output, *value ? "true" : "false",
                   state.maximum_byte_length);
    return;
  }
  if (const auto* value = std::get_if<std::int64_t>(&input.value)) {
    std::array<char, 32U> encoded{};
    const auto converted =
        std::to_chars(encoded.data(), encoded.data() + encoded.size(), *value);
    if (converted.ec != std::errc{}) throw InvalidObservation();
    append_checked(output, std::string_view(encoded.data(), converted.ptr),
                   state.maximum_byte_length);
    return;
  }
  if (const auto* value = std::get_if<std::string>(&input.value)) {
    serialize_json_string(*value, output, state.maximum_byte_length);
    return;
  }
  if (const auto* values = std::get_if<Json::Array>(&input.value)) {
    if (values->size() > kMaximumJsonArrayLength) {
      throw ArtifactResourceLimit();
    }
    append_checked(output, '[', state.maximum_byte_length);
    for (std::size_t index = 0U; index < values->size(); ++index) {
      if (index != 0U) append_checked(output, ',', state.maximum_byte_length);
      serialize_json((*values)[index], output, state, depth + 1U);
    }
    append_checked(output, ']', state.maximum_byte_length);
    return;
  }
  const auto& values = std::get<Json::Object>(input.value);
  if (values.size() > kMaximumJsonObjectPropertyCount) {
    throw ArtifactResourceLimit();
  }
  append_checked(output, '{', state.maximum_byte_length);
  std::size_t index = 0U;
  for (const auto& [key, value] : values) {
    if (index != 0U) append_checked(output, ',', state.maximum_byte_length);
    serialize_json_string(key, output, state.maximum_byte_length);
    append_checked(output, ':', state.maximum_byte_length);
    serialize_json(value, output, state, depth + 1U);
    ++index;
  }
  append_checked(output, '}', state.maximum_byte_length);
}

std::string sha256_hex(const std::string_view bytes) {
  Sha256 hash;
  Sha256Digest digest{};
  if (!hash.update(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                   bytes.size()) ||
      !hash.finalize(digest)) {
    throw InvalidObservation();
  }
  const Sha256LowercaseHex encoded = sha256_lowercase_hex(digest);
  return std::string(encoded.data(), 64U);
}

}  // namespace

Json object(
    const std::initializer_list<std::pair<std::string, Json>> properties) {
  Json::Object result;
  for (const auto& property : properties) {
    if (!result.emplace(property.first, property.second).second) {
      throw InvalidObservation();
    }
  }
  return Json(std::move(result));
}

Json array(const std::initializer_list<Json> elements) {
  return Json(Json::Array(elements));
}

std::string canonical_json(const Json& input,
                           const std::size_t maximum_byte_length) {
  if (maximum_byte_length == 0U) throw ArtifactResourceLimit();
  std::string output;
  output.reserve(std::min<std::size_t>(maximum_byte_length, 64U * 1024U));
  SerializationState state{maximum_byte_length, 0U};
  serialize_json(input, output, state, 1U);
  return output;
}

std::string hash_json(const Json& input,
                      const std::size_t maximum_byte_length) {
  return sha256_hex(canonical_json(input, maximum_byte_length));
}

std::string stable_id(const std::string_view kind, const Json& value,
                      const std::size_t maximum_byte_length) {
  if (kind.empty()) throw InvalidObservation();
  const std::string digest = hash_json(
      object({{"domain", std::string("browsergrad.compiler.cpp-cute.") +
                             std::string(kind) + "-id.v1"},
              {"value", value}}),
      maximum_byte_length);
  return std::string("bg.cpp.") + std::string(kind) + ".sha256." + digest;
}

}  // namespace browsergrad::cpp_cute::artifact_json
