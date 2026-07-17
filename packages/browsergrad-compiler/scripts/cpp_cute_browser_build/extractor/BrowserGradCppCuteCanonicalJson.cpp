#include "BrowserGradCppCuteCanonicalJson.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace browsergrad::cpp_cute {
namespace {

constexpr std::uint64_t kMaximumSafeInteger = 9'007'199'254'740'991ULL;
constexpr std::uint32_t kMaximumParserDepth = 256U;

struct StringSpan {
  std::uint32_t begin = 0;
  std::uint32_t end = 0;
};

struct DecodedCodePoint {
  std::uint32_t value = 0;
  std::uint32_t byte_length = 0;
};

bool is_decimal_digit(std::uint8_t value) {
  return value >= static_cast<std::uint8_t>('0') &&
         value <= static_cast<std::uint8_t>('9');
}

bool is_lower_hex_digit(std::uint8_t value) {
  return is_decimal_digit(value) ||
         (value >= static_cast<std::uint8_t>('a') &&
          value <= static_cast<std::uint8_t>('f'));
}

std::uint32_t lower_hex_value(std::uint8_t value) {
  return is_decimal_digit(value)
             ? static_cast<std::uint32_t>(value - '0')
             : static_cast<std::uint32_t>(value - 'a' + 10U);
}

bool decode_utf8(const std::uint8_t* bytes, std::uint32_t remaining,
                 DecodedCodePoint* decoded) {
  if (remaining == 0U || decoded == nullptr) return false;
  const std::uint8_t first = bytes[0];
  if (first <= 0x7fU) {
    decoded->value = first;
    decoded->byte_length = 1U;
    return true;
  }
  if (first >= 0xc2U && first <= 0xdfU && remaining >= 2U &&
      (bytes[1] & 0xc0U) == 0x80U) {
    decoded->value = (static_cast<std::uint32_t>(first & 0x1fU) << 6U) |
                     static_cast<std::uint32_t>(bytes[1] & 0x3fU);
    decoded->byte_length = 2U;
    return true;
  }
  if (first >= 0xe0U && first <= 0xefU && remaining >= 3U &&
      (bytes[1] & 0xc0U) == 0x80U && (bytes[2] & 0xc0U) == 0x80U) {
    if ((first == 0xe0U && bytes[1] < 0xa0U) ||
        (first == 0xedU && bytes[1] >= 0xa0U)) {
      return false;
    }
    decoded->value = (static_cast<std::uint32_t>(first & 0x0fU) << 12U) |
                     (static_cast<std::uint32_t>(bytes[1] & 0x3fU) << 6U) |
                     static_cast<std::uint32_t>(bytes[2] & 0x3fU);
    decoded->byte_length = 3U;
    return true;
  }
  if (first >= 0xf0U && first <= 0xf4U && remaining >= 4U &&
      (bytes[1] & 0xc0U) == 0x80U && (bytes[2] & 0xc0U) == 0x80U &&
      (bytes[3] & 0xc0U) == 0x80U) {
    if ((first == 0xf0U && bytes[1] < 0x90U) ||
        (first == 0xf4U && bytes[1] > 0x8fU)) {
      return false;
    }
    decoded->value = (static_cast<std::uint32_t>(first & 0x07U) << 18U) |
                     (static_cast<std::uint32_t>(bytes[1] & 0x3fU) << 12U) |
                     (static_cast<std::uint32_t>(bytes[2] & 0x3fU) << 6U) |
                     static_cast<std::uint32_t>(bytes[3] & 0x3fU);
    decoded->byte_length = 4U;
    return true;
  }
  return false;
}

class Utf16Iterator final {
 public:
  Utf16Iterator(const std::uint8_t* bytes, StringSpan span)
      : bytes_(bytes), cursor_(span.begin), end_(span.end) {}

  bool next(std::uint16_t* unit) {
    if (unit == nullptr) return false;
    if (pending_low_surrogate_ != 0U) {
      *unit = pending_low_surrogate_;
      pending_low_surrogate_ = 0U;
      return true;
    }
    if (cursor_ == end_) return false;
    const std::uint8_t first = bytes_[cursor_];
    if (first == static_cast<std::uint8_t>('\\')) {
      const std::uint8_t escape = bytes_[cursor_ + 1U];
      cursor_ += 2U;
      switch (escape) {
        case '"': *unit = '"'; return true;
        case '\\': *unit = '\\'; return true;
        case 'b': *unit = 0x08U; return true;
        case 'f': *unit = 0x0cU; return true;
        case 'n': *unit = 0x0aU; return true;
        case 'r': *unit = 0x0dU; return true;
        case 't': *unit = 0x09U; return true;
        case 'u': {
          const std::uint16_t value = static_cast<std::uint16_t>(
              (lower_hex_value(bytes_[cursor_ + 2U]) << 4U) |
              lower_hex_value(bytes_[cursor_ + 3U]));
          cursor_ += 4U;
          *unit = value;
          return true;
        }
        default: return false;
      }
    }
    DecodedCodePoint decoded;
    if (!decode_utf8(bytes_ + cursor_, end_ - cursor_, &decoded)) return false;
    cursor_ += decoded.byte_length;
    if (decoded.value <= 0xffffU) {
      *unit = static_cast<std::uint16_t>(decoded.value);
      return true;
    }
    const std::uint32_t supplementary = decoded.value - 0x1'0000U;
    *unit = static_cast<std::uint16_t>(0xd800U + (supplementary >> 10U));
    pending_low_surrogate_ =
        static_cast<std::uint16_t>(0xdc00U + (supplementary & 0x3ffU));
    return true;
  }

 private:
  const std::uint8_t* bytes_;
  std::uint32_t cursor_;
  std::uint32_t end_;
  std::uint16_t pending_low_surrogate_ = 0U;
};

int compare_utf16(const std::uint8_t* bytes, StringSpan left,
                  StringSpan right) {
  Utf16Iterator left_units(bytes, left);
  Utf16Iterator right_units(bytes, right);
  while (true) {
    std::uint16_t left_unit = 0U;
    std::uint16_t right_unit = 0U;
    const bool has_left = left_units.next(&left_unit);
    const bool has_right = right_units.next(&right_unit);
    if (!has_left || !has_right) {
      return has_left == has_right ? 0 : has_left ? 1 : -1;
    }
    if (left_unit != right_unit) return left_unit < right_unit ? -1 : 1;
  }
}

class Parser final {
 public:
  Parser(const std::uint8_t* bytes, std::uint32_t byte_length,
         const CanonicalJsonLimits& limits)
      : bytes_(bytes), byte_length_(byte_length), limits_(limits) {}

  CanonicalJsonValidation run() {
    if (!valid_limits() || bytes_ == nullptr || byte_length_ == 0U) {
      return result_;
    }
    if (byte_length_ > limits_.max_document_byte_length) {
      resource(0U);
      return result_;
    }
    if (!parse_value(1U) || cursor_ != byte_length_) return result_;
    result_.status = CanonicalJsonStatus::kValid;
    result_.error_byte_offset = byte_length_;
    return result_;
  }

 private:
  bool valid_limits() const {
    return limits_.max_document_byte_length > 0U && limits_.max_depth > 0U &&
           limits_.max_depth <= kMaximumParserDepth &&
           limits_.max_nodes > 0U &&
           limits_.max_decoded_string_byte_length > 0U &&
           limits_.max_array_length > 0U &&
           limits_.max_object_property_count > 0U;
  }

  bool parse_value(std::uint32_t depth) {
    if (depth > limits_.max_depth) return resource(cursor_);
    if (result_.node_count == limits_.max_nodes) return resource(cursor_);
    ++result_.node_count;
    result_.maximum_depth = std::max(result_.maximum_depth, depth);
    if (cursor_ == byte_length_) return invalid(cursor_);
    switch (bytes_[cursor_]) {
      case '{': return parse_object(depth);
      case '[': return parse_array(depth);
      case '"': {
        StringSpan ignored;
        return parse_string(&ignored);
      }
      case 't': return consume_literal("true", 4U);
      case 'f': return consume_literal("false", 5U);
      case 'n': return consume_literal("null", 4U);
      default: return parse_number();
    }
  }

  bool parse_object(std::uint32_t depth) {
    ++cursor_;
    if (cursor_ < byte_length_ && bytes_[cursor_] == '}') {
      ++cursor_;
      return true;
    }
    std::uint32_t property_count = 0U;
    StringSpan previous_key;
    bool has_previous_key = false;
    while (cursor_ < byte_length_) {
      if (property_count == limits_.max_object_property_count) {
        return resource(cursor_);
      }
      StringSpan key;
      if (!parse_string(&key)) return false;
      if (has_previous_key && compare_utf16(bytes_, previous_key, key) >= 0) {
        return invalid(key.begin);
      }
      previous_key = key;
      has_previous_key = true;
      ++property_count;
      if (cursor_ == byte_length_ || bytes_[cursor_] != ':') {
        return invalid(cursor_);
      }
      ++cursor_;
      if (!parse_value(depth + 1U)) return false;
      if (cursor_ == byte_length_) return invalid(cursor_);
      if (bytes_[cursor_] == '}') {
        ++cursor_;
        return true;
      }
      if (bytes_[cursor_] != ',') return invalid(cursor_);
      ++cursor_;
    }
    return invalid(cursor_);
  }

  bool parse_array(std::uint32_t depth) {
    ++cursor_;
    if (cursor_ < byte_length_ && bytes_[cursor_] == ']') {
      ++cursor_;
      return true;
    }
    std::uint32_t element_count = 0U;
    while (cursor_ < byte_length_) {
      if (element_count == limits_.max_array_length) return resource(cursor_);
      ++element_count;
      if (!parse_value(depth + 1U)) return false;
      if (cursor_ == byte_length_) return invalid(cursor_);
      if (bytes_[cursor_] == ']') {
        ++cursor_;
        return true;
      }
      if (bytes_[cursor_] != ',') return invalid(cursor_);
      ++cursor_;
    }
    return invalid(cursor_);
  }

  bool parse_string(StringSpan* span) {
    if (span == nullptr || cursor_ == byte_length_ || bytes_[cursor_] != '"') {
      return invalid(cursor_);
    }
    ++cursor_;
    span->begin = cursor_;
    std::uint64_t decoded_byte_length = 0U;
    while (cursor_ < byte_length_) {
      const std::uint8_t first = bytes_[cursor_];
      if (first == '"') {
        span->end = cursor_;
        ++cursor_;
        const std::uint64_t cumulative =
            static_cast<std::uint64_t>(result_.decoded_string_byte_length) +
            decoded_byte_length;
        if (cumulative > limits_.max_decoded_string_byte_length) {
          return resource(span->begin);
        }
        result_.decoded_string_byte_length =
            static_cast<std::uint32_t>(cumulative);
        return true;
      }
      if (first == '\\') {
        if (byte_length_ - cursor_ < 2U) return invalid(cursor_);
        const std::uint8_t escape = bytes_[cursor_ + 1U];
        if (escape == '"' || escape == '\\' || escape == 'b' ||
            escape == 'f' || escape == 'n' || escape == 'r' ||
            escape == 't') {
          cursor_ += 2U;
          ++decoded_byte_length;
          continue;
        }
        if (escape != 'u' || byte_length_ - cursor_ < 6U ||
            bytes_[cursor_ + 2U] != '0' || bytes_[cursor_ + 3U] != '0' ||
            !is_lower_hex_digit(bytes_[cursor_ + 4U]) ||
            !is_lower_hex_digit(bytes_[cursor_ + 5U])) {
          return invalid(cursor_);
        }
        const std::uint32_t control =
            (lower_hex_value(bytes_[cursor_ + 4U]) << 4U) |
            lower_hex_value(bytes_[cursor_ + 5U]);
        if (control >= 0x20U || control == 0x08U || control == 0x09U ||
            control == 0x0aU || control == 0x0cU || control == 0x0dU) {
          return invalid(cursor_);
        }
        cursor_ += 6U;
        ++decoded_byte_length;
        continue;
      }
      if (first < 0x20U) return invalid(cursor_);
      DecodedCodePoint decoded;
      if (!decode_utf8(bytes_ + cursor_, byte_length_ - cursor_, &decoded)) {
        return invalid(cursor_);
      }
      if (decoded.value == '"' || decoded.value == '\\') return invalid(cursor_);
      cursor_ += decoded.byte_length;
      decoded_byte_length += decoded.byte_length;
    }
    return invalid(cursor_);
  }

  bool parse_number() {
    const std::uint32_t start = cursor_;
    bool negative = false;
    if (cursor_ < byte_length_ && bytes_[cursor_] == '-') {
      negative = true;
      ++cursor_;
    }
    if (cursor_ == byte_length_ || !is_decimal_digit(bytes_[cursor_])) {
      return invalid(start);
    }
    if (bytes_[cursor_] == '0') {
      ++cursor_;
      if (negative ||
          (cursor_ < byte_length_ && is_decimal_digit(bytes_[cursor_]))) {
        return invalid(start);
      }
      return true;
    }
    std::uint64_t magnitude = 0U;
    while (cursor_ < byte_length_ && is_decimal_digit(bytes_[cursor_])) {
      const std::uint32_t digit = bytes_[cursor_] - '0';
      if (magnitude > (kMaximumSafeInteger - digit) / 10U) {
        return invalid(start);
      }
      magnitude = magnitude * 10U + digit;
      ++cursor_;
    }
    if (magnitude == 0U || magnitude > kMaximumSafeInteger) {
      return invalid(start);
    }
    if (cursor_ < byte_length_ &&
        (bytes_[cursor_] == '.' || bytes_[cursor_] == 'e' ||
         bytes_[cursor_] == 'E')) {
      return invalid(start);
    }
    return true;
  }

  bool consume_literal(const char* literal, std::uint32_t length) {
    if (literal == nullptr || length > byte_length_ - cursor_) {
      return invalid(cursor_);
    }
    for (std::uint32_t index = 0U; index < length; ++index) {
      if (bytes_[cursor_ + index] !=
          static_cast<std::uint8_t>(literal[index])) {
        return invalid(cursor_);
      }
    }
    cursor_ += length;
    return true;
  }

  bool invalid(std::uint32_t offset) {
    if (result_.status != CanonicalJsonStatus::kResourceLimit) {
      result_.status = CanonicalJsonStatus::kInvalid;
      result_.error_byte_offset = std::min(offset, byte_length_);
    }
    return false;
  }

  bool resource(std::uint32_t offset) {
    result_.status = CanonicalJsonStatus::kResourceLimit;
    result_.error_byte_offset = std::min(offset, byte_length_);
    return false;
  }

  const std::uint8_t* bytes_;
  std::uint32_t byte_length_;
  const CanonicalJsonLimits& limits_;
  std::uint32_t cursor_ = 0U;
  CanonicalJsonValidation result_;
};

}  // namespace

CanonicalJsonValidation validate_canonical_json(
    const std::uint8_t* bytes, std::uint32_t byte_length,
    const CanonicalJsonLimits& limits) {
  return Parser(bytes, byte_length, limits).run();
}

}  // namespace browsergrad::cpp_cute
