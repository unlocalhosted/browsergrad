#pragma once

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace browsergrad::cpp_cute {

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> imported_closed_vfs();

}  // namespace browsergrad::cpp_cute
