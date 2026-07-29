#include "BrowserGradCppCuteClangAction.h"

#include <array>
#include <charconv>
#include <limits>
#include <memory>
#include <optional>
#include <system_error>
#include <utility>

#include "BrowserGradCppCuteImportedVfs.h"
#include "BrowserGradCppCuteMetrics.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/Sema/TemplateInstCallback.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

namespace browsergrad::cpp_cute {
namespace {

ViewCopyScalarKind view_copy_scalar_kind(
    const clang::QualType pointee,
    const clang::ASTContext& context) noexcept {
  const clang::QualType scalar = pointee.getUnqualifiedType();
  const ViewCopyScalarKind kind = [&scalar]() noexcept {
    if (scalar->isSpecificBuiltinType(clang::BuiltinType::Float)) {
      return ViewCopyScalarKind::kFloat32;
    }
    if (scalar->isSpecificBuiltinType(clang::BuiltinType::Int)) {
      return ViewCopyScalarKind::kSignedInt32;
    }
    if (scalar->isSpecificBuiltinType(clang::BuiltinType::UInt)) {
      return ViewCopyScalarKind::kUnsignedInt32;
    }
    return ViewCopyScalarKind::kUnsupported;
  }();
  if (kind == ViewCopyScalarKind::kUnsupported) return kind;
  if (context.getTypeSize(scalar) != 32U ||
      context.getTypeAlign(scalar) != 32U) {
    return ViewCopyScalarKind::kUnsupported;
  }
  return kind;
}

const clang::Expr* strip_transparent_expression(const clang::Expr* expression) {
  const clang::Expr* current = expression;
  while (current != nullptr) {
    current = current->IgnoreParenImpCasts();
    if (const auto* cleanup =
            llvm::dyn_cast<clang::ExprWithCleanups>(current)) {
      current = cleanup->getSubExpr();
      continue;
    }
    if (const auto* materialized =
            llvm::dyn_cast<clang::MaterializeTemporaryExpr>(current)) {
      current = materialized->getSubExpr();
      continue;
    }
    if (const auto* bound =
            llvm::dyn_cast<clang::CXXBindTemporaryExpr>(current)) {
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

bool decode_resolved_cute_tensor(
    clang::QualType type, clang::QualType& engine_type,
    clang::QualType& layout_type,
    const clang::ClassTemplateDecl*& tensor_template) {
  const auto* record = type.getCanonicalType()->getAsCXXRecordDecl();
  const auto* specialization =
      llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(record);
  if (specialization == nullptr ||
      specialization->getSpecializedTemplate()->getQualifiedNameAsString() !=
          "cute::Tensor") {
    return false;
  }
  const clang::TemplateArgumentList& arguments =
      specialization->getTemplateArgs();
  if (arguments.size() != 2U ||
      arguments[0].getKind() != clang::TemplateArgument::Type ||
      arguments[1].getKind() != clang::TemplateArgument::Type) {
    return false;
  }
  engine_type = arguments[0].getAsType().getCanonicalType();
  layout_type = arguments[1].getAsType().getCanonicalType();
  tensor_template = specialization->getSpecializedTemplate();
  return true;
}

bool decode_cute_view_iterator(clang::QualType engine_type,
                               clang::QualType& iterator_type) {
  engine_type = engine_type.getCanonicalType();
  if (engine_type->isPointerType()) {
    iterator_type = engine_type;
    return true;
  }
  const auto* record = engine_type->getAsCXXRecordDecl();
  const auto* specialization =
      llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(record);
  if (specialization == nullptr) return false;
  const std::string name =
      specialization->getSpecializedTemplate()->getQualifiedNameAsString();
  if (name != "cute::ViewEngine" && name != "cute::ConstViewEngine") {
    return false;
  }
  const clang::TemplateArgumentList& arguments =
      specialization->getTemplateArgs();
  if (arguments.size() != 1U ||
      arguments[0].getKind() != clang::TemplateArgument::Type) {
    return false;
  }
  iterator_type = arguments[0].getAsType().getCanonicalType();
  return iterator_type->isPointerType();
}

std::string canonical_usr(const clang::Decl* declaration) {
  llvm::SmallString<128> usr;
  if (declaration == nullptr) return {};
  const clang::Decl* canonical = declaration->getCanonicalDecl();
  if (canonical == nullptr ||
      clang::index::generateUSRForDecl(canonical, usr)) {
    return {};
  }
  return std::string(usr);
}

const clang::ValueDecl* referenced_value(const clang::Expr* expression) {
  const auto* reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
      strip_transparent_expression(expression));
  return reference == nullptr ? nullptr : reference->getDecl();
}

bool is_cutlass_cute_header_path(const std::string_view path) {
  constexpr std::string_view kPrefix = "/toolchain/cutlass/include/cute/";
  return path.starts_with(kPrefix) && path.size() > kPrefix.size();
}

bool checked_multiply(const std::int64_t left, const std::int64_t right,
                      std::int64_t& output) noexcept {
  if (left < 0 ||
      (right > 0 &&
       left > std::numeric_limits<std::int64_t>::max() / right) ||
      (right < 0 && left > 0 &&
       right < std::numeric_limits<std::int64_t>::min() / left)) {
    return false;
  }
  output = left * right;
  return true;
}

bool checked_add(const std::int64_t left, const std::int64_t right,
                 std::int64_t& output) noexcept {
  if (right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) {
    return false;
  }
  if (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right) {
    return false;
  }
  output = left + right;
  return true;
}

bool integral_template_argument(const clang::TemplateArgument& argument,
                                std::int64_t& output) {
  if (argument.getKind() != clang::TemplateArgument::Integral) return false;
  llvm::SmallString<32> digits;
  argument.getAsIntegral().toString(digits, 10U);
  const std::string_view text(digits.data(), digits.size());
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), output);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

bool decode_static_hierarchy(clang::QualType type,
                             LayoutIntegerHierarchy& output,
                             std::uint32_t& node_count,
                             const std::uint32_t depth = 1U) {
  constexpr std::uint32_t kMaximumDepth = 64U;
  constexpr std::uint32_t kMaximumNodes = 4096U;
  if (depth > kMaximumDepth || ++node_count > kMaximumNodes) return false;
  const auto* record = type.getCanonicalType()->getAsCXXRecordDecl();
  const auto* specialization =
      llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(record);
  if (specialization == nullptr) return false;
  const std::string name =
      specialization->getSpecializedTemplate()->getQualifiedNameAsString();
  const clang::TemplateArgumentList& arguments =
      specialization->getTemplateArgs();
  if (name == "cute::C" || name == "cute::integral_constant") {
    const unsigned value_index = name == "cute::C" ? 0U : 1U;
    if (arguments.size() <= value_index) return false;
    output = LayoutIntegerHierarchy{};
    return integral_template_argument(arguments[value_index], output.value);
  }
  if (name != "cute::tuple" && name != "cute::detail::packed_tuple") {
    return false;
  }
  output = LayoutIntegerHierarchy{};
  output.tuple = true;
  std::size_t element_count = 0U;
  for (unsigned index = 0U; index < arguments.size(); ++index) {
    const clang::TemplateArgument& argument = arguments[index];
    element_count += argument.getKind() == clang::TemplateArgument::Pack
                         ? argument.pack_size()
                         : 1U;
  }
  output.elements.reserve(element_count);
  const auto append_element = [&](const clang::TemplateArgument& argument) {
    if (argument.getKind() != clang::TemplateArgument::Type) return false;
    LayoutIntegerHierarchy child;
    if (!decode_static_hierarchy(argument.getAsType(), child, node_count,
                                 depth + 1U)) {
      return false;
    }
    output.elements.push_back(std::move(child));
    return true;
  };
  for (unsigned index = 0U; index < arguments.size(); ++index) {
    const clang::TemplateArgument& argument = arguments[index];
    if (argument.getKind() == clang::TemplateArgument::Pack) {
      for (const clang::TemplateArgument& packed : argument.pack_elements()) {
        if (!append_element(packed)) return false;
      }
    } else if (!append_element(argument)) {
      return false;
    }
  }
  return !output.elements.empty();
}

void flatten_hierarchy(const LayoutIntegerHierarchy& hierarchy,
                       std::vector<std::int64_t>& values) {
  if (!hierarchy.tuple) {
    values.push_back(hierarchy.value);
    return;
  }
  for (const LayoutIntegerHierarchy& element : hierarchy.elements) {
    flatten_hierarchy(element, values);
  }
}

bool decode_static_affine_layout(clang::QualType type, LayoutTrace& trace) {
  const auto* record = type.getCanonicalType()->getAsCXXRecordDecl();
  const auto* specialization =
      llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(record);
  if (specialization == nullptr ||
      specialization->getSpecializedTemplate()->getQualifiedNameAsString() !=
          "cute::Layout") {
    return false;
  }
  const clang::TemplateArgumentList& arguments =
      specialization->getTemplateArgs();
  if (arguments.size() != 2U ||
      arguments[0].getKind() != clang::TemplateArgument::Type ||
      arguments[1].getKind() != clang::TemplateArgument::Type) {
    return false;
  }
  std::uint32_t node_count = 0U;
  LayoutIntegerHierarchy shape;
  LayoutIntegerHierarchy stride;
  if (!decode_static_hierarchy(arguments[0].getAsType(), shape, node_count) ||
      !decode_static_hierarchy(arguments[1].getAsType(), stride, node_count)) {
    return false;
  }
  std::vector<std::int64_t> shapes;
  std::vector<std::int64_t> strides;
  flatten_hierarchy(shape, shapes);
  flatten_hierarchy(stride, strides);
  if (shapes.empty() || shapes.size() != strides.size() ||
      std::any_of(shapes.begin(), shapes.end(),
                  [](const std::int64_t value) { return value <= 0; })) {
    return false;
  }
  std::int64_t size = 1;
  std::int64_t cosize = 1;
  for (std::size_t index = 0U; index < shapes.size(); ++index) {
    if (!checked_multiply(size, shapes[index], size)) return false;
    std::int64_t contribution = 0;
    if (!checked_multiply(shapes[index] - 1, strides[index], contribution) ||
        !checked_add(cosize, contribution, cosize)) {
      return false;
    }
  }
  if (shapes.size() > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  trace.resolved_static_affine_layout = true;
  trace.rank =
      shape.tuple ? static_cast<std::uint32_t>(shape.elements.size()) : 1U;
  trace.leaf_rank = static_cast<std::uint32_t>(shapes.size());
  trace.size = size;
  trace.cosize = cosize;
  trace.shape = std::move(shape);
  trace.stride = std::move(stride);
  return true;
}

class SemanticTraceVisitor final
    : public clang::RecursiveASTVisitor<SemanticTraceVisitor> {
 public:
  SemanticTraceVisitor(clang::ASTContext& context, SourceAnchor anchor,
                       LayoutTrace& layout_trace,
                       ViewCopyTrace& view_copy_trace, ClangPassReview& review)
      : context_(context),
        anchor_(std::move(anchor)),
        layout_trace_(layout_trace),
        view_copy_trace_(view_copy_trace),
        review_(review) {}

  bool VisitDecl(clang::Decl* declaration) {
    return declaration == nullptr ||
           record_ast_node(declaration->getLocation());
  }

  bool VisitStmt(clang::Stmt* statement) {
    return statement == nullptr || record_ast_node(statement->getBeginLoc());
  }

  bool VisitVarDecl(clang::VarDecl* declaration) {
    if (declaration == nullptr || declaration->getIdentifier() == nullptr ||
        declaration->isThisDeclarationADefinition(context_) ==
            clang::VarDecl::DeclarationOnly) {
      return true;
    }
    if (anchor_.kind == SourceAnchorKind::kLayoutVariable) {
      capture_layout(declaration);
    } else if (!llvm::isa<clang::ParmVarDecl>(declaration) &&
               selected_function_ != nullptr &&
               declaration->getParentFunctionOrMethod() == selected_function_) {
      capture_tensor(declaration);
    }
    return true;
  }

  bool VisitFunctionDecl(clang::FunctionDecl* declaration) {
    if (anchor_.kind != SourceAnchorKind::kViewCopyFunction ||
        declaration == nullptr || declaration->getIdentifier() == nullptr) {
      return true;
    }
    std::uint32_t identity_begin = 0U;
    std::uint32_t identity_end = 0U;
    if (!identity_range(*declaration, identity_begin, identity_end) ||
        identity_begin != anchor_.begin_byte ||
        identity_end != anchor_.end_byte) {
      return true;
    }
    if (view_copy_trace_.selected) {
      view_copy_trace_.ambiguous = true;
      return true;
    }
    view_copy_trace_.selected = true;
    selected_function_ = declaration;
    view_copy_trace_.identity_begin_byte = identity_begin;
    view_copy_trace_.identity_end_byte = identity_end;
    if (!source_range(declaration->getSourceRange(),
                      view_copy_trace_.declaration_begin_byte,
                      view_copy_trace_.declaration_end_byte)) {
      return true;
    }
    view_copy_trace_.canonical_usr = canonical_usr(declaration);
    view_copy_trace_.canonical_name = declaration->getQualifiedNameAsString();
    view_copy_trace_.canonical_type =
        declaration->getType().getCanonicalType().getAsString(
            context_.getPrintingPolicy());
    view_copy_trace_.cuda_host = declaration->hasAttr<clang::CUDAHostAttr>();
    view_copy_trace_.cuda_device =
        declaration->hasAttr<clang::CUDADeviceAttr>();
    view_copy_trace_.cuda_global =
        declaration->hasAttr<clang::CUDAGlobalAttr>();
    view_copy_trace_.force_inline =
        declaration->hasAttr<clang::AlwaysInlineAttr>();
    const auto* prototype =
        declaration->getType()->getAs<clang::FunctionProtoType>();
    view_copy_trace_.resolved_function =
        declaration->doesThisDeclarationHaveABody() &&
        declaration->getReturnType()->isVoidType() && prototype != nullptr &&
        !prototype->isVariadic() && declaration->getNumParams() == 2U &&
        !view_copy_trace_.cuda_host && view_copy_trace_.cuda_device &&
        !view_copy_trace_.cuda_global &&
        !view_copy_trace_.canonical_usr.empty();
    view_copy_trace_.parameters.reserve(declaration->getNumParams());
    for (std::uint32_t ordinal = 0U; ordinal < declaration->getNumParams();
         ++ordinal) {
      const clang::ParmVarDecl* parameter = declaration->getParamDecl(ordinal);
      ViewCopyParameterTrace trace;
      trace.ordinal = ordinal;
      trace.canonical_usr = canonical_usr(parameter);
      trace.canonical_name = parameter->getQualifiedNameAsString();
      trace.canonical_type =
          parameter->getType().getCanonicalType().getAsString(
              context_.getPrintingPolicy());
      static_cast<void>(source_range(parameter->getSourceRange(),
                                     trace.declaration_begin_byte,
                                     trace.declaration_end_byte));
      static_cast<void>(identity_range(*parameter, trace.identity_begin_byte,
                                       trace.identity_end_byte));
      const clang::QualType parameter_type =
          parameter->getType().getCanonicalType();
      if (const auto* pointer = parameter_type->getAs<clang::PointerType>()) {
        const clang::QualType pointee =
            pointer->getPointeeType().getCanonicalType();
        trace.resolved_pointer = true;
        trace.scalar_kind = view_copy_scalar_kind(pointee, context_);
        trace.pointee_const = pointee.isConstQualified();
      }
      view_copy_trace_.parameters.push_back(std::move(trace));
    }
    return true;
  }

  bool VisitCallExpr(clang::CallExpr* expression) {
    if (anchor_.kind != SourceAnchorKind::kViewCopyFunction ||
        selected_function_ == nullptr || expression == nullptr) {
      return true;
    }
    const clang::FunctionDecl* callee = expression->getDirectCallee();
    if (callee == nullptr ||
        callee->getQualifiedNameAsString() != "cute::copy") {
      return true;
    }
    if (view_copy_trace_.resolved_copy ||
        !view_copy_trace_.copy_callee_name.empty()) {
      view_copy_trace_.ambiguous = true;
      return true;
    }
    view_copy_trace_.copy_callee_name = callee->getQualifiedNameAsString();
    view_copy_trace_.copy_callee_usr = canonical_usr(callee);
    const clang::FunctionTemplateDecl* primary_template =
        callee->getPrimaryTemplate();
    view_copy_trace_.copy_callee_path = declaration_path(
        primary_template == nullptr
            ? static_cast<const clang::Decl*>(callee)
            : static_cast<const clang::Decl*>(primary_template));
    if (!source_range(expression->getSourceRange(),
                      view_copy_trace_.copy_begin_byte,
                      view_copy_trace_.copy_end_byte) ||
        expression->getNumArgs() != 2U ||
        !expression->getType()->isVoidType()) {
      return true;
    }
    const auto* source_declaration = llvm::dyn_cast_or_null<clang::VarDecl>(
        referenced_value(expression->getArg(0U)));
    const auto* destination_declaration =
        llvm::dyn_cast_or_null<clang::VarDecl>(
            referenced_value(expression->getArg(1U)));
    const auto source = tensor_ordinal(source_declaration);
    const auto destination = tensor_ordinal(destination_declaration);
    if (!source.has_value() || !destination.has_value() ||
        *source == *destination) {
      return true;
    }
    view_copy_trace_.source_tensor_ordinal = *source;
    view_copy_trace_.destination_tensor_ordinal = *destination;
    const ViewCopyTensorTrace& source_tensor =
        view_copy_trace_.tensors[*source];
    const ViewCopyTensorTrace& destination_tensor =
        view_copy_trace_.tensors[*destination];
    view_copy_trace_.resolved_copy =
        !view_copy_trace_.copy_callee_usr.empty() &&
        source_tensor.engine_pointee_const &&
        !destination_tensor.engine_pointee_const;
    return true;
  }

  void finalize() {
    if (anchor_.kind != SourceAnchorKind::kViewCopyFunction) return;
    if (view_copy_trace_.parameters.size() != 2U ||
        view_copy_trace_.tensors.size() != 2U) {
      view_copy_trace_.resolved_function = false;
      return;
    }
    const ViewCopyParameterTrace& source = view_copy_trace_.parameters[0U];
    const ViewCopyParameterTrace& destination = view_copy_trace_.parameters[1U];
    if (!source.resolved_pointer ||
        source.scalar_kind == ViewCopyScalarKind::kUnsupported ||
        !source.pointee_const || !destination.resolved_pointer ||
        destination.scalar_kind != source.scalar_kind ||
        destination.pointee_const ||
        source.canonical_usr.empty() || destination.canonical_usr.empty() ||
        view_copy_trace_.ambiguous || !view_copy_trace_.resolved_copy) {
      view_copy_trace_.resolved_function = false;
      return;
    }
    for (const ViewCopyTensorTrace& tensor : view_copy_trace_.tensors) {
      if (!tensor.resolved_tensor_type ||
          !tensor.resolved_static_affine_layout ||
          !tensor.initializer_parameter_bound || tensor.canonical_usr.empty() ||
          tensor.initializer_callee_usr.empty() ||
          !is_cutlass_cute_header_path(tensor.tensor_template_path) ||
          !is_cutlass_cute_header_path(tensor.layout_template_path) ||
          !is_cutlass_cute_header_path(tensor.initializer_callee_path)) {
        view_copy_trace_.resolved_function = false;
        return;
      }
    }
    if (!is_cutlass_cute_header_path(view_copy_trace_.copy_callee_path)) {
      view_copy_trace_.resolved_function = false;
      return;
    }
    const ViewCopyTensorTrace& source_tensor =
        view_copy_trace_.tensors[view_copy_trace_.source_tensor_ordinal];
    const ViewCopyTensorTrace& destination_tensor =
        view_copy_trace_.tensors[view_copy_trace_.destination_tensor_ordinal];
    if (source_tensor.engine_parameter_ordinal != 0U ||
        destination_tensor.engine_parameter_ordinal != 1U) {
      view_copy_trace_.resolved_function = false;
    }
  }

 private:
  clang::ASTContext& context_;
  SourceAnchor anchor_;
  LayoutTrace& layout_trace_;
  ViewCopyTrace& view_copy_trace_;
  ClangPassReview& review_;
  const clang::FunctionDecl* selected_function_ = nullptr;
  std::vector<const clang::VarDecl*> tensor_declarations_;

  bool record_ast_node(const clang::SourceLocation location) {
    if (record_frontend_ast_node()) return true;
    report_limit(location);
    return false;
  }

  void report_limit(const clang::SourceLocation location) {
    if (review_.frontend_work_limit_exceeded) return;
    review_.frontend_work_limit_exceeded = true;
    clang::DiagnosticsEngine& diagnostics = context_.getDiagnostics();
    const unsigned id =
        diagnostics.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                    "BrowserGrad frontend work limit exceeded");
    diagnostics.Report(location, id);
  }

  bool source_range(const clang::SourceRange range, std::uint32_t& begin_byte,
                    std::uint32_t& end_byte) const {
    const clang::SourceManager& source_manager = context_.getSourceManager();
    const clang::SourceLocation begin =
        source_manager.getExpansionLoc(range.getBegin());
    const clang::SourceLocation end_token =
        source_manager.getExpansionLoc(range.getEnd());
    if (begin.isInvalid() || end_token.isInvalid() ||
        source_manager.getFilename(begin) !=
            llvm::StringRef(anchor_.virtual_path) ||
        source_manager.getFilename(end_token) !=
            llvm::StringRef(anchor_.virtual_path)) {
      return false;
    }
    const clang::SourceLocation end = clang::Lexer::getLocForEndOfToken(
        end_token, 0U, source_manager, context_.getLangOpts());
    if (end.isInvalid()) return false;
    const std::uint64_t begin_offset = source_manager.getFileOffset(begin);
    const std::uint64_t end_offset = source_manager.getFileOffset(end);
    if (begin_offset >= end_offset ||
        end_offset > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    begin_byte = static_cast<std::uint32_t>(begin_offset);
    end_byte = static_cast<std::uint32_t>(end_offset);
    return true;
  }

  bool identity_range(const clang::NamedDecl& declaration,
                      std::uint32_t& begin_byte,
                      std::uint32_t& end_byte) const {
    return source_range(clang::SourceRange(declaration.getLocation()),
                        begin_byte, end_byte);
  }

  std::string declaration_path(const clang::Decl* declaration) const {
    if (declaration == nullptr) return {};
    const clang::SourceManager& source_manager = context_.getSourceManager();
    const clang::SourceLocation location =
        source_manager.getSpellingLoc(declaration->getLocation());
    if (location.isInvalid()) return {};
    return source_manager.getFilename(location).str();
  }

  void capture_layout(clang::VarDecl* declaration) {
    if (layout_trace_.selected) return;
    std::uint32_t begin = 0U;
    std::uint32_t end = 0U;
    if (!identity_range(*declaration, begin, end) ||
        begin != anchor_.begin_byte || end != anchor_.end_byte) {
      return;
    }
    layout_trace_.selected = true;
    layout_trace_.identity_begin_byte = begin;
    layout_trace_.identity_end_byte = end;
    layout_trace_.canonical_name = declaration->getQualifiedNameAsString();
    layout_trace_.canonical_type =
        declaration->getType().getCanonicalType().getAsString(
            context_.getPrintingPolicy());
    layout_trace_.resolved_layout_type =
        is_resolved_cute_layout(declaration->getType());
    if (layout_trace_.resolved_layout_type) {
      static_cast<void>(
          decode_static_affine_layout(declaration->getType(), layout_trace_));
    }
    layout_trace_.canonical_usr = canonical_usr(declaration);
    if (const auto* call = llvm::dyn_cast_or_null<clang::CallExpr>(
            strip_transparent_expression(declaration->getInit()))) {
      if (const auto* callee = call->getDirectCallee()) {
        layout_trace_.initializer_callee = callee->getQualifiedNameAsString();
      }
    }
  }

  void capture_tensor(clang::VarDecl* declaration) {
    clang::QualType engine_type;
    clang::QualType iterator_type;
    clang::QualType layout_type;
    const clang::ClassTemplateDecl* tensor_template = nullptr;
    if (!decode_resolved_cute_tensor(declaration->getType(), engine_type,
                                     layout_type, tensor_template) ||
        !decode_cute_view_iterator(engine_type, iterator_type)) {
      return;
    }
    if (view_copy_trace_.tensors.size() >= 2U) {
      view_copy_trace_.ambiguous = true;
      return;
    }
    ViewCopyTensorTrace trace;
    trace.resolved_tensor_type = true;
    trace.canonical_usr = canonical_usr(declaration);
    trace.canonical_name = declaration->getQualifiedNameAsString();
    trace.canonical_type =
        declaration->getType().getCanonicalType().getAsString(
            context_.getPrintingPolicy());
    trace.tensor_template_path = declaration_path(tensor_template);
    static_cast<void>(source_range(declaration->getSourceRange(),
                                   trace.declaration_begin_byte,
                                   trace.declaration_end_byte));
    static_cast<void>(identity_range(*declaration, trace.identity_begin_byte,
                                     trace.identity_end_byte));
    trace.layout_canonical_type =
        layout_type.getAsString(context_.getPrintingPolicy());
    const auto* layout_record = layout_type->getAsCXXRecordDecl();
    const auto* layout_specialization =
        llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(
            layout_record);
    trace.layout_template_path =
        layout_specialization == nullptr
            ? std::string{}
            : declaration_path(layout_specialization->getSpecializedTemplate());
    LayoutTrace layout;
    layout.resolved_layout_type = is_resolved_cute_layout(layout_type);
    if (layout.resolved_layout_type &&
        decode_static_affine_layout(layout_type, layout)) {
      trace.resolved_static_affine_layout = true;
      trace.rank = layout.rank;
      trace.leaf_rank = layout.leaf_rank;
      trace.size = layout.size;
      trace.cosize = layout.cosize;
      trace.shape = std::move(layout.shape);
      trace.stride = std::move(layout.stride);
    }
    if (const auto* pointer = iterator_type->getAs<clang::PointerType>()) {
      trace.engine_pointee_const =
          pointer->getPointeeType().getCanonicalType().isConstQualified();
    }
    if (const auto* call = llvm::dyn_cast_or_null<clang::CallExpr>(
            strip_transparent_expression(declaration->getInit()))) {
      const clang::FunctionDecl* callee = call->getDirectCallee();
      if (callee != nullptr) {
        trace.initializer_callee_name = callee->getQualifiedNameAsString();
        trace.initializer_callee_usr = canonical_usr(callee);
        const clang::FunctionTemplateDecl* primary_template =
            callee->getPrimaryTemplate();
        trace.initializer_callee_path = declaration_path(
            primary_template == nullptr
                ? static_cast<const clang::Decl*>(callee)
                : static_cast<const clang::Decl*>(primary_template));
      }
      if (callee != nullptr &&
          trace.initializer_callee_name == "cute::make_tensor" &&
          call->getNumArgs() >= 1U) {
        const auto* parameter = llvm::dyn_cast_or_null<clang::ParmVarDecl>(
            referenced_value(call->getArg(0U)));
        if (parameter != nullptr) {
          for (std::size_t ordinal = 0U;
               ordinal < view_copy_trace_.parameters.size(); ++ordinal) {
            if (selected_function_->getParamDecl(ordinal)->getCanonicalDecl() ==
                parameter->getCanonicalDecl()) {
              trace.engine_parameter_ordinal =
                  static_cast<std::uint32_t>(ordinal);
              trace.initializer_parameter_bound =
                  context_.hasSameType(iterator_type, parameter->getType());
              break;
            }
          }
        }
      }
    }
    tensor_declarations_.push_back(declaration->getCanonicalDecl());
    view_copy_trace_.tensors.push_back(std::move(trace));
  }

  std::optional<std::uint32_t> tensor_ordinal(
      const clang::VarDecl* declaration) const {
    if (declaration == nullptr) return std::nullopt;
    const clang::VarDecl* canonical = declaration->getCanonicalDecl();
    for (std::size_t ordinal = 0U; ordinal < tensor_declarations_.size();
         ++ordinal) {
      if (tensor_declarations_[ordinal] == canonical) {
        return static_cast<std::uint32_t>(ordinal);
      }
    }
    return std::nullopt;
  }
};

class FrontendWorkTemplateCallbacks final
    : public clang::TemplateInstantiationCallback {
 public:
  FrontendWorkTemplateCallbacks(clang::DiagnosticsEngine& diagnostics,
                                ClangPassReview& review)
      : diagnostics_(diagnostics), review_(review) {}

  void initialize(const clang::Sema&) override {}
  void finalize(const clang::Sema&) override {}

  void atTemplateBegin(
      const clang::Sema&,
      const clang::Sema::CodeSynthesisContext& context) override {
    if (!context.isInstantiationRecord() ||
        begin_frontend_template_instantiation()) {
      return;
    }
    report_limit(context.PointOfInstantiation);
  }

  void atTemplateEnd(
      const clang::Sema&,
      const clang::Sema::CodeSynthesisContext& context) override {
    if (!context.isInstantiationRecord() ||
        review_.frontend_work_limit_exceeded) {
      return;
    }
    if (!end_frontend_template_instantiation()) {
      report_limit(context.PointOfInstantiation);
    }
  }

 private:
  clang::DiagnosticsEngine& diagnostics_;
  ClangPassReview& review_;

  void report_limit(const clang::SourceLocation location) {
    if (review_.frontend_work_limit_exceeded) return;
    review_.frontend_work_limit_exceeded = true;
    const unsigned id = diagnostics_.getCustomDiagID(
        clang::DiagnosticsEngine::Fatal,
        "BrowserGrad frontend work limit exceeded");
    diagnostics_.Report(location, id);
  }
};

class SemanticTraceConsumer final : public clang::SemaConsumer {
 public:
  SemanticTraceConsumer(clang::ASTContext& context, SourceAnchor anchor,
                        LayoutTrace& layout_trace,
                        ViewCopyTrace& view_copy_trace, ClangPassReview& review)
      : context_(context),
        visitor_(context, std::move(anchor), layout_trace, view_copy_trace,
                 review),
        review_(review) {}

  void InitializeSema(clang::Sema& sema) override {
    sema.TemplateInstCallbacks.push_back(
        std::make_unique<FrontendWorkTemplateCallbacks>(
            context_.getDiagnostics(), review_));
  }

  void HandleTranslationUnit(clang::ASTContext& context) override {
    visitor_.TraverseDecl(context.getTranslationUnitDecl());
    visitor_.finalize();
  }

 private:
  clang::ASTContext& context_;
  SemanticTraceVisitor visitor_;
  ClangPassReview& review_;
};

class IncludeObservationCallbacks final : public clang::PPCallbacks {
 public:
  IncludeObservationCallbacks(clang::SourceManager& source_manager,
                              const clang::LangOptions& language_options,
                              std::shared_ptr<ImportedVfsObserver> observer,
                              ClangPassReview& review)
      : source_manager_(source_manager),
        language_options_(language_options),
        observer_(std::move(observer)),
        review_(review) {}

  void FileChanged(clang::SourceLocation location,
                   clang::PPCallbacks::FileChangeReason reason,
                   clang::SrcMgr::CharacteristicKind, clang::FileID) override {
    if (reason != clang::PPCallbacks::EnterFile ||
        review_.frontend_work_limit_exceeded) {
      return;
    }
    clang::SourceLocation current = source_manager_.getExpansionLoc(location);
    if (current.isInvalid()) return;
    clang::FileID file = source_manager_.getFileID(current);
    std::uint64_t depth = 0U;
    while (file.isValid()) {
      const clang::SourceLocation include = source_manager_.getIncludeLoc(file);
      if (include.isInvalid()) break;
      ++depth;
      if (depth == std::numeric_limits<std::uint64_t>::max()) break;
      file =
          source_manager_.getFileID(source_manager_.getExpansionLoc(include));
    }
    if (!record_frontend_include_depth(depth)) report_limit(location);
  }

  void MacroExpands(const clang::Token& macro_name,
                    const clang::MacroDefinition&, clang::SourceRange,
                    const clang::MacroArgs*) override {
    if (!review_.frontend_work_limit_exceeded &&
        !record_frontend_macro_expansion()) {
      report_limit(macro_name.getLocation());
    }
  }

  void InclusionDirective(clang::SourceLocation hash_location,
                          const clang::Token&, llvm::StringRef file_name,
                          bool is_angled, clang::CharSourceRange filename_range,
                          clang::OptionalFileEntryRef file, llvm::StringRef,
                          llvm::StringRef, const clang::Module*, bool,
                          clang::SrcMgr::CharacteristicKind) override {
    if (!file.has_value() || observer_ == nullptr ||
        hash_location.isInvalid() || filename_range.isInvalid()) {
      return;
    }
    const clang::SourceLocation hash =
        source_manager_.getFileLoc(hash_location);
    clang::CharSourceRange file_range = clang::Lexer::makeFileCharRange(
        filename_range, source_manager_, language_options_);
    if (file_range.isInvalid()) {
      file_range = clang::Lexer::getAsCharRange(
          source_manager_.getExpansionRange(filename_range), source_manager_,
          language_options_);
    }
    const clang::SourceLocation filename_begin = file_range.getBegin();
    const clang::SourceLocation filename_end = file_range.getEnd();
    if (hash.isInvalid() || file_range.isInvalid() ||
        filename_begin.isInvalid() || filename_end.isInvalid() ||
        source_manager_.getFileID(hash) !=
            source_manager_.getFileID(filename_begin) ||
        source_manager_.getFileID(hash) !=
            source_manager_.getFileID(filename_end)) {
      static_cast<void>(observer_->record_invalid_include_source_range());
      return;
    }
    const clang::FileID including_file_id = source_manager_.getFileID(hash);
    const clang::OptionalFileEntryRef including_file =
        source_manager_.getFileEntryRefForID(including_file_id);
    if (!including_file.has_value()) {
      // Clang reports the synthetic command-line include directive as a
      // callback from a pseudo buffer. The exact forced-include edge was
      // already recorded from its closed compiler option and has no source
      // directive range or including source file.
      return;
    }
    const llvm::StringRef resolved_file = file->getName();
    const std::uint64_t begin = source_manager_.getFileOffset(hash);
    const std::uint64_t end = source_manager_.getFileOffset(filename_end);
    ImportedVfsIncludeEdgeObservation edge;
    edge.kind = is_angled ? ImportedVfsIncludeKind::kSourceAngle
                          : ImportedVfsIncludeKind::kSourceQuote;
    std::array<char, kCppCuteMaximumVirtualPathByteLength> canonical_path{};
    std::size_t canonical_path_size = 0U;
    const llvm::StringRef including_file_name = including_file->getName();
    if (!cpp_cute_normalize_virtual_path(
            std::string_view(including_file_name.data(),
                             including_file_name.size()),
            canonical_path.data(), canonical_path.size(),
            canonical_path_size)) {
      edge.including_file_path = including_file_name.str();
    } else {
      edge.including_file_path.assign(canonical_path.data(),
                                      canonical_path_size);
    }
    canonical_path.fill(0);
    canonical_path_size = 0U;
    if (!cpp_cute_normalize_virtual_path(
            std::string_view(resolved_file.data(), resolved_file.size()),
            canonical_path.data(), canonical_path.size(),
            canonical_path_size)) {
      edge.resolved_file_path = resolved_file.str();
    } else {
      edge.resolved_file_path.assign(canonical_path.data(),
                                     canonical_path_size);
    }
    edge.spelling = file_name.str();
    edge.directive_start_byte_offset = begin;
    edge.directive_end_byte_offset = end;
    static_cast<void>(observer_->record_resolved_include_edge(std::move(edge)));
  }

 private:
  clang::SourceManager& source_manager_;
  const clang::LangOptions& language_options_;
  std::shared_ptr<ImportedVfsObserver> observer_;
  ClangPassReview& review_;

  void report_limit(const clang::SourceLocation location) {
    if (review_.frontend_work_limit_exceeded) return;
    review_.frontend_work_limit_exceeded = true;
    clang::DiagnosticsEngine& diagnostics = source_manager_.getDiagnostics();
    const unsigned id =
        diagnostics.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                    "BrowserGrad frontend work limit exceeded");
    diagnostics.Report(location, id);
  }
};

RawDiagnosticSeverity diagnostic_severity(
    const clang::DiagnosticsEngine::Level level) noexcept {
  switch (level) {
    case clang::DiagnosticsEngine::Ignored:
      return RawDiagnosticSeverity::kIgnored;
    case clang::DiagnosticsEngine::Note:
      return RawDiagnosticSeverity::kNote;
    case clang::DiagnosticsEngine::Remark:
      return RawDiagnosticSeverity::kRemark;
    case clang::DiagnosticsEngine::Warning:
      return RawDiagnosticSeverity::kWarning;
    case clang::DiagnosticsEngine::Error:
      return RawDiagnosticSeverity::kError;
    case clang::DiagnosticsEngine::Fatal:
      return RawDiagnosticSeverity::kFatal;
  }
  return RawDiagnosticSeverity::kFatal;
}

RawDiagnosticStage diagnostic_stage(
    const std::uint32_t diagnostic_id) noexcept {
  if (diagnostic_id >= clang::diag::DIAG_START_LEX &&
      diagnostic_id < clang::diag::DIAG_START_PARSE) {
    return RawDiagnosticStage::kPreprocessor;
  }
  if (diagnostic_id >= clang::diag::DIAG_START_PARSE &&
      diagnostic_id < clang::diag::DIAG_START_AST) {
    return RawDiagnosticStage::kParser;
  }
  if (diagnostic_id >= clang::diag::DIAG_START_SEMA &&
      diagnostic_id < clang::diag::DIAG_START_ANALYSIS) {
    return RawDiagnosticStage::kSemaCuda;
  }
  return RawDiagnosticStage::kParser;
}

struct CustomDiagnosticRegistry final {
  std::uint32_t temporal_consultation_id = 0U;
  std::uint32_t temporal_mutation_id = 0U;
  bool bound = false;
};

class BoundedDiagnosticCapture final : public clang::DiagnosticConsumer {
 public:
  BoundedDiagnosticCapture(
      const std::uint32_t maximum_count,
      const std::uint32_t maximum_retained_byte_length,
      std::shared_ptr<const CustomDiagnosticRegistry> custom_registry,
      std::vector<ClangDiagnosticObservation>& observations,
      bool& failed) noexcept
      : maximum_count_(maximum_count),
        maximum_retained_byte_length_(maximum_retained_byte_length),
        custom_registry_(std::move(custom_registry)),
        observations_(observations),
        failed_(failed) {}

  void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                        const clang::Diagnostic& information) override {
    clang::DiagnosticConsumer::HandleDiagnostic(level, information);
    if (failed_ || level == clang::DiagnosticsEngine::Ignored) return;
    try {
      if (observations_.size() >= maximum_count_) {
        failed_ = true;
        return;
      }
      llvm::SmallString<256> rendered;
      information.FormatDiagnostic(rendered);
      constexpr std::size_t kMaximumOneMessageByteLength = 4096U;
      if (rendered.empty() || rendered.size() > kMaximumOneMessageByteLength ||
          rendered.size() > maximum_retained_byte_length_ - retained_bytes_) {
        failed_ = true;
        return;
      }
      ClangDiagnosticObservation observation;
      observation.raw_diagnostic_id = information.getID();
      observation.stage = diagnostic_stage(observation.raw_diagnostic_id);
      observation.severity = diagnostic_severity(level);
      if (level == clang::DiagnosticsEngine::Error ||
          level == clang::DiagnosticsEngine::Fatal) {
        ++error_count_;
      }
      observation.rendered_message.assign(rendered.begin(), rendered.end());
      if (custom_registry_ != nullptr && custom_registry_->bound &&
          observation.raw_diagnostic_id ==
              custom_registry_->temporal_consultation_id) {
        observation.custom = true;
        observation.custom_code = CustomDiagnosticCode::kTemporalMacroForbidden;
        observation.stage = RawDiagnosticStage::kPreprocessor;
      } else if (custom_registry_ != nullptr && custom_registry_->bound &&
                 observation.raw_diagnostic_id ==
                     custom_registry_->temporal_mutation_id) {
        observation.custom = true;
        observation.custom_code =
            CustomDiagnosticCode::kTemporalMacroMutationForbidden;
        observation.stage = RawDiagnosticStage::kPreprocessor;
      }
      const clang::SourceLocation original = information.getLocation();
      if (original.isValid()) {
        clang::SourceManager& source_manager = information.getSourceManager();
        const clang::SourceLocation location =
            source_manager.getSpellingLoc(original);
        if (location.isValid() &&
            source_manager.isWrittenInMainFile(location)) {
          const llvm::StringRef path = source_manager.getFilename(location);
          if (!path.empty()) {
            observation.has_source_location = true;
            observation.virtual_path = path.str();
            observation.byte_offset = source_manager.getFileOffset(location);
          }
        } else if (location.isValid()) {
          const llvm::StringRef path = source_manager.getFilename(location);
          if (!path.empty()) {
            observation.has_source_location = true;
            observation.virtual_path = path.str();
            observation.byte_offset = source_manager.getFileOffset(location);
          }
        }
      }
      retained_bytes_ += rendered.size();
      observations_.push_back(std::move(observation));
    } catch (...) {
      failed_ = true;
    }
  }

  std::uint32_t error_count() const noexcept { return error_count_; }

 private:
  std::size_t maximum_count_ = 0U;
  std::size_t maximum_retained_byte_length_ = 0U;
  std::size_t retained_bytes_ = 0U;
  std::shared_ptr<const CustomDiagnosticRegistry> custom_registry_;
  std::vector<ClangDiagnosticObservation>& observations_;
  bool& failed_;
  std::uint32_t error_count_ = 0U;
};

class LayoutTraceAction final : public clang::ASTFrontendAction {
 public:
  LayoutTraceAction(SourceAnchor anchor,
                    std::shared_ptr<ImportedVfsObserver> observer,
                    std::shared_ptr<CustomDiagnosticRegistry> custom_registry,
                    ClangPassReview& review)
      : anchor_(std::move(anchor)),
        observer_(std::move(observer)),
        custom_registry_(std::move(custom_registry)),
        review_(review) {}

  bool BeginSourceFileAction(clang::CompilerInstance& compiler) override {
    if (!clang::ASTFrontendAction::BeginSourceFileAction(compiler)) {
      return false;
    }
    CppCutePreprocessorPolicyInstallation policy =
        install_cpp_cute_preprocessor_policy(compiler);
    review_.policy_install_status = policy.status;
    if (!policy) return false;
    if (custom_registry_ == nullptr ||
        policy.consultation_diagnostic_id == 0U ||
        policy.mutation_diagnostic_id == 0U) {
      return false;
    }
    custom_registry_->temporal_consultation_id =
        policy.consultation_diagnostic_id;
    custom_registry_->temporal_mutation_id = policy.mutation_diagnostic_id;
    custom_registry_->bound = true;
    policy_state_ = std::move(policy.state);
    compiler.getPreprocessor().addPPCallbacks(
        std::make_unique<IncludeObservationCallbacks>(
            compiler.getSourceManager(), compiler.getLangOpts(), observer_,
            review_));
    clang::DiagnosticsEngine& diagnostics = compiler.getDiagnostics();
    compiler.getPreprocessor().setTokenWatcher(
        [&review = review_, &diagnostics](const clang::Token& token) {
          if (token.is(clang::tok::eof) ||
              review.frontend_work_limit_exceeded ||
              record_frontend_preprocessed_token()) {
            return;
          }
          review.frontend_work_limit_exceeded = true;
          const unsigned id = diagnostics.getCustomDiagID(
              clang::DiagnosticsEngine::Fatal,
              "BrowserGrad frontend work limit exceeded");
          diagnostics.Report(token.getLocation(), id);
        });
    return true;
  }

  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance& compiler, llvm::StringRef) override {
    return std::make_unique<SemanticTraceConsumer>(
        compiler.getASTContext(), anchor_, review_.layout_trace,
        review_.view_copy_trace, review_);
  }

  void EndSourceFileAction() override {
    if (policy_state_ != nullptr) {
      review_.policy_failed = policy_state_->failed();
      review_.policy_violation_count = policy_state_->violation_count();
    }
    clang::ASTFrontendAction::EndSourceFileAction();
  }

 private:
  SourceAnchor anchor_;
  std::shared_ptr<ImportedVfsObserver> observer_;
  std::shared_ptr<CustomDiagnosticRegistry> custom_registry_;
  std::shared_ptr<CppCutePreprocessorPolicyState> policy_state_;
  ClangPassReview& review_;
};

}  // namespace

bool run_cpp_cute_clang_pass_for_review(
    const std::vector<std::string>& command_line, const SourceAnchor& anchor,
    const std::span<const ClangForcedIncludeObservation> forced_includes,
    const ImportedVfsObservationLimits observation_limits,
    const std::uint32_t maximum_diagnostic_count,
    const std::uint32_t maximum_diagnostic_byte_length,
    ClangPassReview& review) {
  review = ClangPassReview{};
  if (maximum_diagnostic_count == 0U || maximum_diagnostic_byte_length == 0U) {
    review.diagnostic_capture_failed = true;
    return false;
  }
  auto observer = std::make_shared<ImportedVfsObserver>(observation_limits);
  for (const ClangForcedIncludeObservation forced : forced_includes) {
    ImportedVfsIncludeEdgeObservation edge;
    edge.kind = ImportedVfsIncludeKind::kCompilerForced;
    edge.resolved_file_path = forced.virtual_path;
    edge.compiler_option_ordinal = forced.compiler_option_ordinal;
    if (observer->record_resolved_include_edge(std::move(edge))) break;
  }
  clang::FileSystemOptions file_system_options;
  auto files = llvm::makeIntrusiveRefCnt<clang::FileManager>(
      file_system_options, imported_closed_vfs(observer));
  auto custom_registry = std::make_shared<CustomDiagnosticRegistry>();
  auto action = std::make_unique<LayoutTraceAction>(anchor, observer,
                                                    custom_registry, review);
  clang::tooling::ToolInvocation invocation(command_line, std::move(action),
                                            files.get());
  BoundedDiagnosticCapture diagnostic_capture(
      maximum_diagnostic_count, maximum_diagnostic_byte_length, custom_registry,
      review.diagnostics, review.diagnostic_capture_failed);
  invocation.setDiagnosticConsumer(&diagnostic_capture);
  if (!begin_frontend_work_semantic_pass()) {
    review.frontend_work_limit_exceeded = true;
    return false;
  }
  review.invocation_succeeded = invocation.run();
  review.clang_error_count = diagnostic_capture.error_count();
  if (!review.frontend_work_limit_exceeded &&
      !complete_frontend_work_semantic_pass()) {
    review.frontend_work_limit_exceeded = true;
  }
  const std::error_code observation_error =
      observer->snapshot(review.vfs_observation);
  review.vfs_failed = static_cast<bool>(observation_error);
  review.vfs_failure = observer->failure();
  return !observation_error && review.invocation_succeeded;
}

}  // namespace browsergrad::cpp_cute
