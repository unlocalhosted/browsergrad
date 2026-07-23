#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace browsergrad::cpp_cute {

inline constexpr std::size_t kCppCuteMaximumVirtualPathByteLength = 4'096U;

/** Validates one absolute canonical UTF-8 path in the closed virtual VFS. */
bool cpp_cute_valid_canonical_virtual_path(std::string_view path) noexcept;

/**
 * Lexically normalizes one compiler-originated absolute or root-relative path
 * into the closed VFS namespace. Dot segments and repeated separators are
 * removed; attempts to escape above the virtual root fail closed.
 */
bool cpp_cute_normalize_virtual_path(std::string_view path,
                                     std::string& output) noexcept;

/** Returns true when candidate is root or a path below root. */
bool cpp_cute_virtual_path_contains(std::string_view root,
                                    std::string_view candidate) noexcept;

}  // namespace browsergrad::cpp_cute
