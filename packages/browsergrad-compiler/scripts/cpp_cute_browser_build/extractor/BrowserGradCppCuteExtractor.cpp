#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace browsergrad::cpp_cute {
namespace {

constexpr std::uint32_t kRuntimeAbiVersion = 0x0001'0000U;
constexpr std::uint32_t kInputFrameMaximumByteLength = 4U * 1024U * 1024U;
constexpr std::uint32_t kInputFrameHeaderByteLength = 64U;
constexpr std::uint32_t kInputFrameAlignment = 8U;
constexpr std::array<std::uint8_t, 8> kInputFrameMagic = {
    'B', 'G', 'C', 'C', 'A', 'B', 'I', '1'};

enum class WireCompileStatus : std::int32_t {
  kArtifactReady = 0,
  kIdle = 1,
  kInputAllocated = 2,
  kInvalidState = 100,
  kInvalidArgument = 101,
  kInvalidFrame = 102,
  kAbiMismatch = 103,
  kVfsError = 104,
  kResourceLimit = 105,
  kInternalError = 106,
};

enum class RuntimePhase {
  kIdle,
  kInputAllocated,
  kFailed,
};

/**
 * This is intentionally distinct from the wire status. The checked-in tracer
 * is a review implementation, not an artifact producer. Until the custom VFS,
 * explicit device/host passes, and canonical artifact-v3 writer exist, a
 * structurally valid frame fails closed as the ABI's non-artifact internal
 * error and can never surface kArtifactReady.
 */
enum class ReviewOnlyBlocker {
  kCustomVfsUnavailable,
  kCudaDualPassUnavailable,
  kCanonicalArtifactV3Unavailable,
};

struct RuntimeState {
  RuntimePhase phase = RuntimePhase::kIdle;
  WireCompileStatus status = WireCompileStatus::kIdle;
  std::uint8_t* input = nullptr;
  std::uint32_t input_byte_length = 0;
  ReviewOnlyBlocker blocker = ReviewOnlyBlocker::kCustomVfsUnavailable;
};

RuntimeState g_runtime;

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

std::uint16_t read_u16_le(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t read_u32_le(const std::uint8_t* bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

bool all_zero(const std::uint8_t* begin, const std::uint8_t* end) {
  for (const auto* cursor = begin; cursor != end; ++cursor) {
    if (*cursor != 0U) return false;
  }
  return true;
}

bool validate_frame_envelope(const std::uint8_t* bytes,
                             std::uint32_t byte_length) {
  if (bytes == nullptr || byte_length < kInputFrameHeaderByteLength ||
      byte_length > kInputFrameMaximumByteLength) {
    return false;
  }
  for (std::size_t index = 0; index < kInputFrameMagic.size(); ++index) {
    if (bytes[index] != kInputFrameMagic[index]) return false;
  }
  if (read_u16_le(bytes + 8U) != 1U || read_u16_le(bytes + 10U) != 0U ||
      read_u32_le(bytes + 12U) != kInputFrameHeaderByteLength ||
      read_u32_le(bytes + 16U) != byte_length ||
      read_u32_le(bytes + 20U) != 0U ||
      read_u32_le(bytes + 24U) != kInputFrameHeaderByteLength ||
      !all_zero(bytes + 40U, bytes + kInputFrameHeaderByteLength)) {
    return false;
  }

  const std::uint64_t profile_offset = read_u32_le(bytes + 24U);
  const std::uint64_t profile_byte_length = read_u32_le(bytes + 28U);
  const std::uint64_t request_offset = read_u32_le(bytes + 32U);
  const std::uint64_t request_byte_length = read_u32_le(bytes + 36U);
  if (profile_byte_length == 0U || request_byte_length == 0U ||
      request_offset % kInputFrameAlignment != 0U) {
    return false;
  }
  const std::uint64_t profile_end = profile_offset + profile_byte_length;
  const std::uint64_t expected_request_offset =
      align_up(profile_end, kInputFrameAlignment);
  const std::uint64_t request_end = request_offset + request_byte_length;
  const std::uint64_t expected_total =
      align_up(request_end, kInputFrameAlignment);
  if (profile_end > byte_length || request_end > byte_length ||
      request_offset != expected_request_offset || expected_total != byte_length) {
    return false;
  }
  return all_zero(bytes + profile_end, bytes + request_offset) &&
         all_zero(bytes + request_end, bytes + byte_length);
}

const clang::Expr* strip_transparent_expression(const clang::Expr* expression) {
  const clang::Expr* current = expression;
  while (current != nullptr) {
    current = current->IgnoreParenImpCasts();
    if (const auto* cleanup = llvm::dyn_cast<clang::ExprWithCleanups>(current)) {
      current = cleanup->getSubExpr();
      continue;
    }
    if (const auto* materialized =
            llvm::dyn_cast<clang::MaterializeTemporaryExpr>(current)) {
      current = materialized->getSubExpr();
      continue;
    }
    if (const auto* bound = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(current)) {
      current = bound->getSubExpr();
      continue;
    }
    return current;
  }
  return nullptr;
}

bool is_resolved_cute_layout(clang::QualType type) {
  const auto* record = type.getCanonicalType()->getAsCXXRecordDecl();
  const auto* specialization =
      llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(record);
  return specialization != nullptr &&
         specialization->getSpecializedTemplate()->getQualifiedNameAsString() ==
             "cute::Layout";
}

class LayoutTraceVisitor final
    : public clang::RecursiveASTVisitor<LayoutTraceVisitor> {
 public:
  LayoutTraceVisitor(clang::ASTContext& context, SourceAnchor anchor,
                     LayoutTrace& trace)
      : context_(context), anchor_(std::move(anchor)), trace_(trace) {}

  bool VisitVarDecl(clang::VarDecl* declaration) {
    if (trace_.selected || declaration == nullptr ||
        declaration->getIdentifier() == nullptr ||
        declaration->isThisDeclarationADefinition(context_) ==
            clang::VarDecl::DeclarationOnly) {
      return true;
    }
    const auto& source_manager = context_.getSourceManager();
    const clang::SourceLocation name_location =
        source_manager.getSpellingLoc(declaration->getLocation());
    if (name_location.isInvalid() ||
        source_manager.getFilename(name_location) !=
            llvm::StringRef(anchor_.virtual_path)) {
      return true;
    }
    const clang::SourceLocation name_end = clang::Lexer::getLocForEndOfToken(
        name_location, 0U, source_manager, context_.getLangOpts());
    if (name_end.isInvalid()) return true;
    const auto begin = source_manager.getFileOffset(name_location);
    const auto end = source_manager.getFileOffset(name_end);
    if (begin != anchor_.begin_byte || end != anchor_.end_byte) return true;

    trace_.selected = true;
    trace_.identity_begin_byte = begin;
    trace_.identity_end_byte = end;
    trace_.canonical_name = declaration->getQualifiedNameAsString();
    trace_.canonical_type = declaration->getType().getCanonicalType().getAsString(
        context_.getPrintingPolicy());
    trace_.resolved_layout_type = is_resolved_cute_layout(declaration->getType());
    llvm::SmallString<128> usr;
    if (clang::index::generateUSRForDecl(declaration, usr)) {
      trace_.canonical_usr.clear();
    } else {
      trace_.canonical_usr = std::string(usr);
    }
    if (const auto* call = llvm::dyn_cast_or_null<clang::CallExpr>(
            strip_transparent_expression(declaration->getInit()))) {
      if (const auto* callee = call->getDirectCallee()) {
        trace_.initializer_callee = callee->getQualifiedNameAsString();
      }
    }
    return true;
  }

 private:
  clang::ASTContext& context_;
  SourceAnchor anchor_;
  LayoutTrace& trace_;
};

class LayoutTraceConsumer final : public clang::ASTConsumer {
 public:
  LayoutTraceConsumer(clang::ASTContext& context, SourceAnchor anchor,
                      LayoutTrace& trace)
      : visitor_(context, std::move(anchor), trace) {}

  void HandleTranslationUnit(clang::ASTContext& context) override {
    visitor_.TraverseDecl(context.getTranslationUnitDecl());
  }

 private:
  LayoutTraceVisitor visitor_;
};

class LayoutTraceAction final : public clang::ASTFrontendAction {
 public:
  LayoutTraceAction(SourceAnchor anchor, LayoutTrace& trace)
      : anchor_(std::move(anchor)), trace_(trace) {}

  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance& compiler, llvm::StringRef) override {
    return std::make_unique<LayoutTraceConsumer>(
        compiler.getASTContext(), anchor_, trace_);
  }

 private:
  SourceAnchor anchor_;
  LayoutTrace& trace_;
};

/**
 * Review-only semantic tracer. The caller must provide an already closed VFS;
 * there is deliberately no physical-filesystem fallback here. The production
 * C ABI does not call this function until the Worker VFS bridge and explicit
 * CUDA device/host ToolInvocation pair are implemented.
 */
[[maybe_unused]] bool run_layout_trace_for_review(
    const std::vector<std::string>& command_line,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> closed_vfs,
    const SourceAnchor& anchor, LayoutTrace& trace) {
  if (!closed_vfs) return false;
  clang::FileSystemOptions file_system_options;
  clang::FileManager files(file_system_options, std::move(closed_vfs));
  auto action = std::make_unique<LayoutTraceAction>(anchor, trace);
  clang::tooling::ToolInvocation invocation(command_line, std::move(action),
                                            &files);
  return invocation.run();
}

void release_input() {
  std::free(g_runtime.input);
  g_runtime.input = nullptr;
  g_runtime.input_byte_length = 0;
}

std::int32_t wire_status(WireCompileStatus status) {
  return static_cast<std::int32_t>(status);
}

}  // namespace
}  // namespace browsergrad::cpp_cute

#if defined(__EMSCRIPTEN__)
#define BG_CPP_CUTE_EXPORT \
  __attribute__((used)) __attribute__((visibility("default")))
#else
#define BG_CPP_CUTE_EXPORT
#endif

extern "C" {

BG_CPP_CUTE_EXPORT std::uint32_t bg_cpp_cute_abi_version(void) {
  return browsergrad::cpp_cute::kRuntimeAbiVersion;
}

BG_CPP_CUTE_EXPORT std::uint32_t bg_cpp_cute_alloc(
    std::uint32_t byte_length) {
  using namespace browsergrad::cpp_cute;
  if (g_runtime.phase != RuntimePhase::kIdle || byte_length == 0U ||
      byte_length > kInputFrameMaximumByteLength) {
    g_runtime.status = WireCompileStatus::kInvalidArgument;
    g_runtime.phase = RuntimePhase::kFailed;
    return 0U;
  }
  auto* allocation = static_cast<std::uint8_t*>(std::malloc(byte_length));
  if (allocation == nullptr) {
    g_runtime.status = WireCompileStatus::kResourceLimit;
    g_runtime.phase = RuntimePhase::kFailed;
    return 0U;
  }
  const auto pointer = reinterpret_cast<std::uintptr_t>(allocation);
  if (pointer > std::numeric_limits<std::uint32_t>::max()) {
    std::free(allocation);
    g_runtime.status = WireCompileStatus::kInternalError;
    g_runtime.phase = RuntimePhase::kFailed;
    return 0U;
  }
  g_runtime.input = allocation;
  g_runtime.input_byte_length = byte_length;
  g_runtime.status = WireCompileStatus::kInputAllocated;
  g_runtime.phase = RuntimePhase::kInputAllocated;
  return static_cast<std::uint32_t>(pointer);
}

BG_CPP_CUTE_EXPORT std::int32_t bg_cpp_cute_compile(
    std::uint32_t input_pointer, std::uint32_t input_length) {
  using namespace browsergrad::cpp_cute;
  if (g_runtime.phase != RuntimePhase::kInputAllocated ||
      g_runtime.input == nullptr) {
    g_runtime.status = WireCompileStatus::kInvalidState;
    g_runtime.phase = RuntimePhase::kFailed;
    return wire_status(g_runtime.status);
  }
  const auto expected_pointer = reinterpret_cast<std::uintptr_t>(g_runtime.input);
  if (input_pointer != expected_pointer ||
      input_length != g_runtime.input_byte_length) {
    g_runtime.status = WireCompileStatus::kInvalidArgument;
    g_runtime.phase = RuntimePhase::kFailed;
    return wire_status(g_runtime.status);
  }
  if (!validate_frame_envelope(g_runtime.input, g_runtime.input_byte_length)) {
    g_runtime.status = WireCompileStatus::kInvalidFrame;
    g_runtime.phase = RuntimePhase::kFailed;
    return wire_status(g_runtime.status);
  }

  // Fail closed. No review trace is a canonical frontend artifact, and status
  // zero remains unreachable until the three typed blockers above are removed.
  g_runtime.blocker = ReviewOnlyBlocker::kCustomVfsUnavailable;
  g_runtime.status = WireCompileStatus::kInternalError;
  g_runtime.phase = RuntimePhase::kFailed;
  return wire_status(g_runtime.status);
}

BG_CPP_CUTE_EXPORT void bg_cpp_cute_free(std::uint32_t pointer,
                                         std::uint32_t byte_length) {
  using namespace browsergrad::cpp_cute;
  if (g_runtime.input == nullptr ||
      pointer != reinterpret_cast<std::uintptr_t>(g_runtime.input) ||
      byte_length != g_runtime.input_byte_length) {
    g_runtime.status = WireCompileStatus::kInvalidArgument;
    g_runtime.phase = RuntimePhase::kFailed;
    return;
  }
  const bool was_allocated = g_runtime.phase == RuntimePhase::kInputAllocated;
  release_input();
  if (was_allocated) {
    g_runtime.phase = RuntimePhase::kIdle;
    g_runtime.status = WireCompileStatus::kIdle;
  }
}

BG_CPP_CUTE_EXPORT void bg_cpp_cute_reset(void) {
  using namespace browsergrad::cpp_cute;
  release_input();
  g_runtime = RuntimeState{};
}

BG_CPP_CUTE_EXPORT std::uint32_t bg_cpp_cute_result_length(void) {
  return 0U;
}

BG_CPP_CUTE_EXPORT std::uint32_t bg_cpp_cute_result_pointer(void) {
  return 0U;
}

BG_CPP_CUTE_EXPORT std::int32_t bg_cpp_cute_status(void) {
  return browsergrad::cpp_cute::wire_status(
      browsergrad::cpp_cute::g_runtime.status);
}

}  // extern "C"

#undef BG_CPP_CUTE_EXPORT
