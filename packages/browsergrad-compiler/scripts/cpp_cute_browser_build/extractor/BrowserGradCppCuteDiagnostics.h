#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace browsergrad::cpp_cute {

enum class RawDiagnosticStage : std::uint8_t {
  kPreprocessor,
  kParser,
  kSemaNameLookup,
  kSemaOverloadResolution,
  kSemaTemplateInstantiation,
  kSemaCuda,
  kArtifactExtractor,
};

enum class RawDiagnosticSeverity : std::uint8_t {
  kIgnored,
  kRemark,
  kNote,
  kWarning,
  kError,
  kFatal,
};

enum class CustomDiagnosticCode : std::uint8_t {
  kTemporalMacroForbidden,
  kTemporalMacroMutationForbidden,
  kDiagnosticResourceLimit,
  kDiagnosticNormalizationFailed,
  kSemanticExtractionFailed,
  kHostDeviceSurfaceDivergence,
};

enum class DiagnosticPhase : std::uint8_t {
  kPreprocessing,
  kParsing,
  kNameLookup,
  kOverloadResolution,
  kTemplateInstantiation,
  kCudaSema,
  kArtifactExtraction,
};

enum class DiagnosticSeverity : std::uint8_t {
  kNone,
  kRemark,
  kNote,
  kWarning,
  kError,
  kFatal,
};

enum class DiagnosticCategory : std::uint8_t {
  kPreprocessing,
  kParsing,
  kNameLookup,
  kOverloadResolution,
  kTemplateInstantiation,
  kCudaSema,
  kArtifactExtraction,
  kPolicy,
  kResourceLimit,
};

enum class DiagnosticSubjectKind : std::uint8_t {
  kCompiler,
  kFile,
  kDeclaration,
  kType,
  kExpression,
  kFact,
};

struct RawDiagnosticSubject {
  DiagnosticSubjectKind kind = DiagnosticSubjectKind::kCompiler;
  std::string_view entity_id;
};

struct RawDiagnosticRelatedLocation {
  std::string_view span_id;
  std::string_view rendered_message;
};

struct RawDiagnosticLocation {
  bool has_source = false;
  std::string_view primary_span_id;
  std::span<const RawDiagnosticRelatedLocation> related;
};

struct RawDiagnosticFixIt {
  std::string_view virtual_path;
  std::uint64_t begin_byte = 0U;
  std::uint64_t end_byte = 0U;
  std::string_view replacement;
};

struct RawDiagnosticInput {
  RawDiagnosticStage stage = RawDiagnosticStage::kPreprocessor;
  RawDiagnosticSeverity severity = RawDiagnosticSeverity::kIgnored;
  bool custom = false;
  std::uint32_t raw_diagnostic_id = 0U;
  CustomDiagnosticCode custom_code = CustomDiagnosticCode::kTemporalMacroForbidden;
  std::string_view rendered_message;
  RawDiagnosticLocation location;
  RawDiagnosticSubject subject;
  std::span<const RawDiagnosticFixIt> fix_its;
};

struct NormalizedDiagnosticRelatedLocation {
  std::string span_id;
  std::string rendered_message;
};

struct NormalizedDiagnosticLocation {
  bool has_source = false;
  std::string primary_span_id;
  std::vector<NormalizedDiagnosticRelatedLocation> related;
};

struct NormalizedDiagnosticSubject {
  DiagnosticSubjectKind kind = DiagnosticSubjectKind::kCompiler;
  std::string entity_id;
};

struct NormalizedDiagnostic {
  std::string diagnostic_id;
  DiagnosticPhase phase = DiagnosticPhase::kArtifactExtraction;
  DiagnosticSeverity severity = DiagnosticSeverity::kFatal;
  DiagnosticCategory category = DiagnosticCategory::kArtifactExtraction;
  std::string code;
  std::string rendered_message;
  NormalizedDiagnosticLocation location;
  NormalizedDiagnosticSubject subject;
  std::string parent_diagnostic_id;
  bool blocking = true;
};

struct DiagnosticNormalizerConfig {
  std::string_view compilation_contract_hash;
  std::string_view owner_pass_id;
  std::uint32_t maximum_unique_diagnostics = 0U;
  std::uint32_t maximum_retained_normalized_bytes = 0U;
  std::span<const std::string_view> opened_span_ids;
  std::span<const std::string_view> opened_virtual_paths;
};

std::uint32_t cpp_cute_maximum_retained_normalized_diagnostic_bytes() noexcept;

enum class DiagnosticNormalizationStatus : std::uint8_t {
  kReady,
  kEmitted,
  kOmitted,
  kDuplicate,
  kInvalidInput,
  kResourceLimit,
  kPoisoned,
};

struct DiagnosticNormalizationResult {
  DiagnosticNormalizationStatus status =
      DiagnosticNormalizationStatus::kInvalidInput;
  std::size_t diagnostic_index = 0U;
};

/**
 * Stateful, bounded normalizer for one exact semantic pass.
 *
 * Invalid or over-budget input emits one closed fatal diagnostic when the
 * configuration permits it, then permanently poisons the instance. Exact
 * duplicate projections collapse without consuming diagnostic or note budget.
 */
class CppCuteDiagnosticNormalizer final {
 public:
  explicit CppCuteDiagnosticNormalizer(
      const DiagnosticNormalizerConfig& config);
  ~CppCuteDiagnosticNormalizer();

  CppCuteDiagnosticNormalizer(const CppCuteDiagnosticNormalizer&) = delete;
  CppCuteDiagnosticNormalizer& operator=(
      const CppCuteDiagnosticNormalizer&) = delete;
  CppCuteDiagnosticNormalizer(CppCuteDiagnosticNormalizer&&) = delete;
  CppCuteDiagnosticNormalizer& operator=(CppCuteDiagnosticNormalizer&&) =
      delete;

  bool configured() const noexcept;
  bool poisoned() const noexcept;
  DiagnosticNormalizationStatus terminal_status() const noexcept;
  std::string_view policy_manifest_id() const noexcept;
  std::size_t diagnostic_count() const noexcept;
  const NormalizedDiagnostic* diagnostic(std::size_t index) const noexcept;
  std::uint32_t retained_normalized_byte_length() const noexcept;

  DiagnosticNormalizationResult normalize(
      const RawDiagnosticInput& input) noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
};

}  // namespace browsergrad::cpp_cute
