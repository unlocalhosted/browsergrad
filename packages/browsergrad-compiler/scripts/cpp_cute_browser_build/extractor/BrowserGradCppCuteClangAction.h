#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace browsergrad::cpp_cute {

struct SourceAnchor {
  std::string virtual_path;
  std::uint32_t begin_byte = 0;
  std::uint32_t end_byte = 0;
};

struct LayoutTrace {
  bool selected = false;
  bool resolved_layout_type = false;
  std::string canonical_usr;
  std::string canonical_name;
  std::string canonical_type;
  std::string initializer_callee;
  std::uint32_t identity_begin_byte = 0;
  std::uint32_t identity_end_byte = 0;
};

/** Review-only instrumentation over the exact imported VFS. */
bool run_layout_trace_for_review(const std::vector<std::string>& command_line,
                                 const SourceAnchor& anchor,
                                 LayoutTrace& trace);

}  // namespace browsergrad::cpp_cute
