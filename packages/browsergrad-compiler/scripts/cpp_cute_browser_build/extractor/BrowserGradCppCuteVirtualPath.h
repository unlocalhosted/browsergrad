#pragma once

#include <cstddef>
#include <string_view>

namespace browsergrad::cpp_cute {

inline constexpr std::size_t kCppCuteMaximumVirtualPathByteLength = 4'096U;

/** Validates one absolute canonical UTF-8 path in the closed virtual VFS. */
bool cpp_cute_valid_canonical_virtual_path(std::string_view path) noexcept;

/** Returns true when candidate is root or a path below root. */
bool cpp_cute_virtual_path_contains(std::string_view root,
                                    std::string_view candidate) noexcept;

}  // namespace browsergrad::cpp_cute
