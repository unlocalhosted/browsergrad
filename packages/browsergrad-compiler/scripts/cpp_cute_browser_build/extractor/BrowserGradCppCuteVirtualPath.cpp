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

bool cpp_cute_normalize_virtual_path(const std::string_view path,
                                     std::string& output) noexcept {
  try {
    output.clear();
    if (path.empty() || path.size() > kCppCuteMaximumVirtualPathByteLength) {
      return false;
    }
    output.reserve(path.size() + (path.front() == '/' ? 0U : 1U));
    output.push_back('/');
    std::size_t segment_begin = path.front() == '/' ? 1U : 0U;
    for (std::size_t index = segment_begin; index <= path.size(); ++index) {
      if (index < path.size() && path[index] != '/') continue;
      const std::string_view segment =
          path.substr(segment_begin, index - segment_begin);
      segment_begin = index + 1U;
      if (segment.empty() || segment == ".") continue;
      if (segment == "..") {
        if (output.size() == 1U) {
          output.clear();
          return false;
        }
        const std::size_t separator = output.rfind('/');
        output.resize(separator == 0U ? 1U : separator);
        continue;
      }
      if (output.size() > 1U) output.push_back('/');
      output.append(segment);
      if (output.size() > kCppCuteMaximumVirtualPathByteLength) {
        output.clear();
        return false;
      }
    }
    if (!cpp_cute_valid_canonical_virtual_path(output)) {
      output.clear();
      return false;
    }
    return true;
  } catch (...) {
    output.clear();
    return false;
  }
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
