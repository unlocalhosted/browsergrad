#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "extractor/BrowserGradCppCuteCanonicalJson.h"

namespace {

using browsergrad::cpp_cute::CanonicalJsonLimits;
using browsergrad::cpp_cute::CanonicalJsonStatus;
using browsergrad::cpp_cute::kRuntimeV1CanonicalJsonLimits;
using browsergrad::cpp_cute::validate_canonical_json;

#define BG_CHECK(condition)                                                   \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::fprintf(stderr, "canonical JSON check failed at line %d: %s\n",  \
                   __LINE__, #condition);                                     \
      return 1;                                                               \
    }                                                                         \
  } while (false)

CanonicalJsonStatus validate(const std::vector<std::uint8_t>& bytes,
                             CanonicalJsonLimits limits = kRuntimeV1CanonicalJsonLimits) {
  return validate_canonical_json(bytes.data(),
                                 static_cast<std::uint32_t>(bytes.size()),
                                 limits)
      .status;
}

CanonicalJsonStatus validate(const char* text,
                             CanonicalJsonLimits limits = kRuntimeV1CanonicalJsonLimits) {
  const std::string value(text);
  return validate_canonical_json(
             reinterpret_cast<const std::uint8_t*>(value.data()),
             static_cast<std::uint32_t>(value.size()), limits)
      .status;
}

int run_cases(const char* canonical_fixture_path) {
  for (const char* valid : {
           "null",
           "true",
           "false",
           "0",
           "-9007199254740991",
           "9007199254740991",
           "[]",
           "{}",
           "[0,true,null,\"x\"]",
           "{\"a\":1,\"z\":2,\"é\":3,\"😀\":4,\"\":5}",
           "\"\\b\\t\\n\\f\\r\\u0000\\\"\\\\\"",
       }) {
    BG_CHECK(validate(valid) == CanonicalJsonStatus::kValid);
  }

  for (const char* invalid : {
           "",
           " null",
           "null ",
           "nullx",
           "01",
           "-0",
           "1.0",
           "1e0",
           "9007199254740992",
           "-9007199254740992",
           "{\"b\":1,\"a\":2}",
           "{\"a\":1,\"a\":2}",
           "{\"a\" :1}",
           "[1,]",
           "[,1]",
           "\"\\/\"",
           "\"\\u0061\"",
           "\"\\u001F\"",
           "\"\\u0008\"",
           "\"\\ud800\"",
       }) {
    BG_CHECK(validate(invalid) == CanonicalJsonStatus::kInvalid);
  }

  BG_CHECK(validate(std::vector<std::uint8_t>{0xefU, 0xbbU, 0xbfU, 'n', 'u', 'l', 'l'}) ==
           CanonicalJsonStatus::kInvalid);
  BG_CHECK(validate(std::vector<std::uint8_t>{'"', 0xc0U, 0xafU, '"'}) ==
           CanonicalJsonStatus::kInvalid);
  BG_CHECK(validate(std::vector<std::uint8_t>{'"', 0xedU, 0xa0U, 0x80U, '"'}) ==
           CanonicalJsonStatus::kInvalid);
  BG_CHECK(validate(std::vector<std::uint8_t>{'"', 0xf4U, 0x90U, 0x80U, 0x80U, '"'}) ==
           CanonicalJsonStatus::kInvalid);

  CanonicalJsonLimits limits = kRuntimeV1CanonicalJsonLimits;
  limits.max_document_byte_length = 3U;
  BG_CHECK(validate("null", limits) == CanonicalJsonStatus::kResourceLimit);
  limits = kRuntimeV1CanonicalJsonLimits;
  limits.max_depth = 2U;
  BG_CHECK(validate("[[0]]", limits) == CanonicalJsonStatus::kResourceLimit);
  limits = kRuntimeV1CanonicalJsonLimits;
  limits.max_depth = 257U;
  BG_CHECK(validate("null", limits) == CanonicalJsonStatus::kInvalid);
  limits = kRuntimeV1CanonicalJsonLimits;
  limits.max_nodes = 2U;
  BG_CHECK(validate("[0,1]", limits) == CanonicalJsonStatus::kResourceLimit);
  limits = kRuntimeV1CanonicalJsonLimits;
  limits.max_decoded_string_byte_length = 1U;
  BG_CHECK(validate("\"é\"", limits) == CanonicalJsonStatus::kResourceLimit);
  limits = kRuntimeV1CanonicalJsonLimits;
  limits.max_array_length = 1U;
  BG_CHECK(validate("[0,1]", limits) == CanonicalJsonStatus::kResourceLimit);
  limits = kRuntimeV1CanonicalJsonLimits;
  limits.max_object_property_count = 1U;
  BG_CHECK(validate("{\"a\":0,\"b\":1}", limits) ==
           CanonicalJsonStatus::kResourceLimit);

  std::ifstream input(canonical_fixture_path, std::ios::binary);
  BG_CHECK(input.good());
  const std::vector<std::uint8_t> fixture{
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  BG_CHECK(!fixture.empty());
  const auto validation = validate_canonical_json(
      fixture.data(), static_cast<std::uint32_t>(fixture.size()),
      kRuntimeV1CanonicalJsonLimits);
  BG_CHECK(validation.status == CanonicalJsonStatus::kValid);
  BG_CHECK(validation.error_byte_offset == fixture.size());
  BG_CHECK(validation.node_count > 1U);
  BG_CHECK(validation.maximum_depth > 1U);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  return run_cases(argv[1]);
}
