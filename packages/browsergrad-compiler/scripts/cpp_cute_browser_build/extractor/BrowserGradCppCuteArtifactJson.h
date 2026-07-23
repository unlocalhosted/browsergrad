#pragma once

#include <cstdint>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace browsergrad::cpp_cute::artifact_json {

class InvalidObservation final : public std::runtime_error {
 public:
  InvalidObservation() : std::runtime_error("invalid producer observation") {}
};

class ArtifactResourceLimit final : public std::runtime_error {
 public:
  ArtifactResourceLimit() : std::runtime_error("artifact resource limit") {}
};

struct Json final {
  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json, std::less<>>;
  using Value = std::variant<std::nullptr_t, bool, std::int64_t, std::string,
                             Array, Object>;

  Json() noexcept : value(nullptr) {}
  Json(std::nullptr_t) noexcept : value(nullptr) {}
  Json(bool input) noexcept : value(input) {}
  Json(std::int64_t input) noexcept : value(input) {}
  Json(std::uint32_t input) noexcept
      : value(static_cast<std::int64_t>(input)) {}
  Json(std::string input) : value(std::move(input)) {}
  Json(std::string_view input) : value(std::string(input)) {}
  Json(const char* input) : value(std::string(input)) {}
  Json(Array input) : value(std::move(input)) {}
  Json(Object input) : value(std::move(input)) {}

  Value value;
};

Json object(std::initializer_list<std::pair<std::string, Json>> properties);
Json array(std::initializer_list<Json> elements);

std::string canonical_json(const Json& input, std::size_t maximum_byte_length);
std::string hash_json(const Json& input, std::size_t maximum_byte_length);
std::string stable_id(std::string_view kind, const Json& value,
                      std::size_t maximum_byte_length);

}  // namespace browsergrad::cpp_cute::artifact_json
