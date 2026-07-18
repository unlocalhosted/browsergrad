#include "BrowserGradCppCutePreprocessorPolicy.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string_view>
#include <utility>

namespace browsergrad::cpp_cute {

class CppCuteTemporalMacroCallbacks final : public clang::PPCallbacks {
 public:
  CppCuteTemporalMacroCallbacks(
      clang::Preprocessor& preprocessor,
      std::shared_ptr<CppCutePreprocessorPolicyState> state,
      const unsigned consultation_diagnostic_id,
      const unsigned mutation_diagnostic_id)
      : preprocessor_(preprocessor),
        state_(std::move(state)),
        consultation_diagnostic_id_(consultation_diagnostic_id),
        mutation_diagnostic_id_(mutation_diagnostic_id) {}

  void MacroExpands(const clang::Token& macro_name_token,
                    const clang::MacroDefinition&, clang::SourceRange,
                    const clang::MacroArgs*) override {
    report(macro_name_token, TemporalMacroUse::kExpansion);
  }

  void Defined(const clang::Token& macro_name_token,
               const clang::MacroDefinition&, clang::SourceRange) override {
    report(macro_name_token, TemporalMacroUse::kDefined);
  }

  void Ifdef(clang::SourceLocation, const clang::Token& macro_name_token,
             const clang::MacroDefinition&) override {
    report(macro_name_token, TemporalMacroUse::kIfdef);
  }

  void Ifndef(clang::SourceLocation, const clang::Token& macro_name_token,
              const clang::MacroDefinition&) override {
    report(macro_name_token, TemporalMacroUse::kIfndef);
  }

  void Elifdef(clang::SourceLocation, const clang::Token& macro_name_token,
               const clang::MacroDefinition&) override {
    report(macro_name_token, TemporalMacroUse::kElifdef);
  }

  void Elifndef(clang::SourceLocation, const clang::Token& macro_name_token,
                const clang::MacroDefinition&) override {
    report(macro_name_token, TemporalMacroUse::kElifndef);
  }

  void MacroDefined(const clang::Token& macro_name_token,
                    const clang::MacroDirective*) override {
    report_source_mutation(macro_name_token, TemporalMacroUse::kDefine);
  }

  void MacroUndefined(const clang::Token& macro_name_token,
                      const clang::MacroDefinition&,
                      const clang::MacroDirective*) override {
    report_source_mutation(macro_name_token, TemporalMacroUse::kUndefine);
  }

 private:
  std::optional<TemporalMacroKind> macro_kind(
      const clang::Token& token) const noexcept {
    const clang::IdentifierInfo* identifier = token.getIdentifierInfo();
    if (identifier == nullptr) return std::nullopt;
    const llvm::StringRef name = identifier->getName();
    return classify_temporal_macro(
        std::string_view(name.data(), name.size()));
  }

  bool is_source_directive(const clang::Token& token) const noexcept {
    if (token.getLocation().isInvalid()) return false;
    const clang::SourceManager& source_manager =
        preprocessor_.getSourceManager();
    const clang::SourceLocation spelling_location =
        source_manager.getSpellingLoc(token.getLocation());
    if (spelling_location.isInvalid()) return false;
    const clang::FileID predefines_file = preprocessor_.getPredefinesFileID();
    return !predefines_file.isValid() ||
           source_manager.getFileID(spelling_location) != predefines_file;
  }

  void report(const clang::Token& token, TemporalMacroUse use) {
    const auto macro = macro_kind(token);
    if (!macro.has_value()) return;
    state_->record(*macro, use);
    const std::string_view name = temporal_macro_name(*macro);
    preprocessor_.getDiagnostics()
        .Report(token.getLocation(), consultation_diagnostic_id_)
        << llvm::StringRef(name.data(), name.size());
  }

  void report_source_mutation(const clang::Token& token,
                              TemporalMacroUse use) {
    if (!is_source_directive(token)) return;
    const auto macro = macro_kind(token);
    if (!macro.has_value()) return;
    state_->record(*macro, use);
    const std::string_view name = temporal_macro_name(*macro);
    preprocessor_.getDiagnostics()
        .Report(token.getLocation(), mutation_diagnostic_id_)
        << llvm::StringRef(name.data(), name.size());
  }

  clang::Preprocessor& preprocessor_;
  std::shared_ptr<CppCutePreprocessorPolicyState> state_;
  unsigned consultation_diagnostic_id_ = 0U;
  unsigned mutation_diagnostic_id_ = 0U;
};

CppCutePreprocessorPolicyInstallation install_cpp_cute_preprocessor_policy(
    clang::CompilerInstance& compiler) {
  if (!compiler.hasDiagnostics()) {
    return CppCutePreprocessorPolicyInstallation{
        CppCutePreprocessorPolicyInstallStatus::kMissingDiagnostics,
        nullptr,
    };
  }
  if (!compiler.hasPreprocessor()) {
    return CppCutePreprocessorPolicyInstallation{
        CppCutePreprocessorPolicyInstallStatus::kMissingPreprocessor,
        nullptr,
    };
  }

  auto state = std::make_shared<CppCutePreprocessorPolicyState>();
  clang::Preprocessor& preprocessor = compiler.getPreprocessor();
  const unsigned consultation_diagnostic_id =
      preprocessor.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          kTemporalMacroForbiddenDiagnosticMessage);
  const unsigned mutation_diagnostic_id =
      preprocessor.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          kTemporalMacroMutationForbiddenDiagnosticMessage);
  // PPCallbacks observes a builtin macro before Clang materializes its value,
  // but it cannot cancel Clang's error-recovery expansion. Pin that rejected
  // recovery path so neither the wall clock nor source-file mtime is observed.
  compiler.getPreprocessorOpts().SourceDateEpoch =
      kTemporalMacroRejectedRecoveryEpoch;
  preprocessor.addPPCallbacks(std::make_unique<CppCuteTemporalMacroCallbacks>(
      preprocessor, state, consultation_diagnostic_id,
      mutation_diagnostic_id));
  return CppCutePreprocessorPolicyInstallation{
      CppCutePreprocessorPolicyInstallStatus::kInstalled,
      std::move(state),
      consultation_diagnostic_id,
      mutation_diagnostic_id,
  };
}

}  // namespace browsergrad::cpp_cute
