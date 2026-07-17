#include "BrowserGradCppCuteClangAction.h"

#include "BrowserGradCppCuteImportedVfs.h"

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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <utility>

namespace browsergrad::cpp_cute {
namespace {

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

}  // namespace

bool run_layout_trace_for_review(const std::vector<std::string>& command_line,
                                 const SourceAnchor& anchor,
                                 LayoutTrace& trace) {
  clang::FileSystemOptions file_system_options;
  clang::FileManager files(file_system_options, imported_closed_vfs());
  auto action = std::make_unique<LayoutTraceAction>(anchor, trace);
  clang::tooling::ToolInvocation invocation(command_line, std::move(action),
                                            &files);
  return invocation.run();
}

}  // namespace browsergrad::cpp_cute
