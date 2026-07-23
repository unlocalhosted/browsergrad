#include "BrowserGradCppCuteVirtualPath.h"

namespace browsergrad::cpp_cute {
namespace {

bool valid_utf8(const std::string_view value) noexcept {
  std::size_t index = 0U;
  while (index < value.size()) {
    const auto first = static_cast<unsigned char>(value[index]);
    if (first <= 0x7fU) {
      ++index;
      continue;
    }
    const auto continuation = [&value](const std::size_t offset) {
      return (static_cast<unsigned char>(value[offset]) & 0xc0U) == 0x80U;
    };
    if (first >= 0xc2U && first <= 0xdfU) {
      if (index + 1U >= value.size() || !continuation(index + 1U)) return false;
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

}  // namespace

bool cpp_cute_valid_canonical_virtual_path(
    const std::string_view path) noexcept {
  if (path.empty() || path.size() > kCppCuteMaximumVirtualPathByteLength ||
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
    const std::string_view segment =
        path.substr(segment_begin, index - segment_begin);
    if (segment.empty() || segment == "." || segment == "..") return false;
    segment_begin = index + 1U;
  }
  return true;
}

bool cpp_cute_normalize_virtual_path(
    const std::string_view path, char* const output,
    const std::size_t output_capacity, std::size_t& output_size) noexcept {
  output_size = 0U;
  if (output == nullptr || output_capacity == 0U || path.empty() ||
      path.size() > kCppCuteMaximumVirtualPathByteLength) {
    return false;
  }
  output[0] = '/';
  output_size = 1U;
  std::size_t segment_begin = path.front() == '/' ? 1U : 0U;
  for (std::size_t index = segment_begin; index <= path.size(); ++index) {
    if (index < path.size() && path[index] != '/') continue;
    const std::string_view segment =
        path.substr(segment_begin, index - segment_begin);
    segment_begin = index + 1U;
    if (segment.empty() || segment == ".") continue;
    if (segment == "..") {
      if (output_size == 1U) {
        output_size = 0U;
        return false;
      }
      std::size_t separator = output_size - 1U;
      while (separator > 0U && output[separator] != '/') --separator;
      output_size = separator == 0U ? 1U : separator;
      continue;
    }
    const std::size_t separator_size = output_size > 1U ? 1U : 0U;
    const std::size_t capacity =
        output_capacity < kCppCuteMaximumVirtualPathByteLength
            ? output_capacity
            : kCppCuteMaximumVirtualPathByteLength;
    if (output_size > capacity ||
        separator_size > capacity - output_size ||
        segment.size() > capacity - output_size - separator_size) {
      output_size = 0U;
      return false;
    }
    if (separator_size != 0U) output[output_size++] = '/';
    for (const char byte : segment) output[output_size++] = byte;
  }
  if (!cpp_cute_valid_canonical_virtual_path(
          std::string_view(output, output_size))) {
    output_size = 0U;
    return false;
  }
  return true;
}

bool cpp_cute_virtual_path_contains(const std::string_view root,
                                    const std::string_view candidate) noexcept {
  return root == "/" ? candidate.starts_with('/')
                     : candidate == root ||
                           (candidate.size() > root.size() &&
                            candidate.starts_with(root) &&
                            candidate[root.size()] == '/');
}

}  // namespace browsergrad::cpp_cute
