#include "extractor/BrowserGradCppCuteDiagnostics.h"

#include <array>
#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>

namespace {

using namespace browsergrad::cpp_cute;

static_assert(!std::is_copy_constructible_v<CppCuteDiagnosticNormalizer>);
static_assert(!std::is_move_constructible_v<CppCuteDiagnosticNormalizer>);

#define BG_CHECK(condition)                                                  \
  do {                                                                       \
    if (!(condition)) {                                                      \
      std::fprintf(stderr, "diagnostics check failed at line %d: %s\n",    \
                   __LINE__, #condition);                                    \
      return 1;                                                              \
    }                                                                        \
  } while (false)

constexpr std::string_view kContractHash =
    "2222222222222222222222222222222222222222222222222222222222222222";
constexpr std::string_view kSpanOne =
    "bg.cpp.span.sha256.1111111111111111111111111111111111111111111111111111111111111111";
constexpr std::string_view kSpanTwo =
    "bg.cpp.span.sha256.3333333333333333333333333333333333333333333333333333333333333333";
constexpr std::string_view kMainPath = "/workspace/src/main.cu";

DiagnosticNormalizerConfig config(const std::uint32_t maximum = 64U) {
  static constexpr std::array spans = {kSpanOne, kSpanTwo};
  static constexpr std::array paths = {kMainPath};
  return {
      kContractHash,
      "cuda-device-sema",
      maximum,
      256U * 1024U,
      spans,
      paths,
  };
}

RawDiagnosticInput root() {
  RawDiagnosticInput input;
  input.stage = RawDiagnosticStage::kParser;
  input.severity = RawDiagnosticSeverity::kError;
  input.raw_diagnostic_id = 1234U;
  input.rendered_message = "expected expression";
  input.location.has_source = true;
  input.location.primary_span_id = kSpanOne;
  input.subject.kind = DiagnosticSubjectKind::kCompiler;
  return input;
}

int run(std::string_view expected_root_id, std::string_view expected_note_id,
        std::string_view expected_custom_id) {
  CppCuteDiagnosticNormalizer normalizer(config());
  BG_CHECK(normalizer.configured());
  BG_CHECK(!normalizer.poisoned());
  BG_CHECK(normalizer.terminal_status() == DiagnosticNormalizationStatus::kReady);
  BG_CHECK(normalizer.policy_manifest_id() ==
           "bg.cpp.diagnostic-normalization.sha256.21d60795bc3da39c9003316f0e6c489c98f7db94cf24a0707bc0fae94001a10a");

  RawDiagnosticInput primary = root();
  const DiagnosticNormalizationResult first = normalizer.normalize(primary);
  BG_CHECK(first.status == DiagnosticNormalizationStatus::kEmitted);
  BG_CHECK(first.diagnostic_index == 0U);
  BG_CHECK(normalizer.diagnostic_count() == 1U);
  const NormalizedDiagnostic* first_output = normalizer.diagnostic(0U);
  BG_CHECK(first_output != nullptr);
  BG_CHECK(first_output->diagnostic_id == expected_root_id);
  BG_CHECK(first_output->phase == DiagnosticPhase::kParsing);
  BG_CHECK(first_output->severity == DiagnosticSeverity::kError);
  BG_CHECK(first_output->category == DiagnosticCategory::kParsing);
  BG_CHECK(first_output->code == "clang:diag-1234");
  BG_CHECK(first_output->blocking);
  BG_CHECK(first_output->parent_diagnostic_id.empty());

  const DiagnosticNormalizationResult duplicate = normalizer.normalize(primary);
  BG_CHECK(duplicate.status == DiagnosticNormalizationStatus::kDuplicate);
  BG_CHECK(duplicate.diagnostic_index == 0U);
  BG_CHECK(normalizer.diagnostic_count() == 1U);

  const std::array related = {
      RawDiagnosticRelatedLocation{kSpanOne, "instantiated here"},
  };
  RawDiagnosticInput note;
  note.stage = RawDiagnosticStage::kParser;
  note.severity = RawDiagnosticSeverity::kNote;
  note.raw_diagnostic_id = 1235U;
  note.rendered_message = "while parsing template";
  note.location = {true, kSpanTwo, related};
  note.subject.kind = DiagnosticSubjectKind::kCompiler;
  const DiagnosticNormalizationResult note_result = normalizer.normalize(note);
  BG_CHECK(note_result.status == DiagnosticNormalizationStatus::kEmitted);
  BG_CHECK(note_result.diagnostic_index == 1U);
  const NormalizedDiagnostic* note_output = normalizer.diagnostic(1U);
  BG_CHECK(note_output != nullptr);
  BG_CHECK(note_output->diagnostic_id == expected_note_id);
  BG_CHECK(note_output->parent_diagnostic_id == expected_root_id);
  BG_CHECK(note_output->location.related.size() == 1U);
  BG_CHECK(note_output->location.related[0].rendered_message ==
           "instantiated here");
  BG_CHECK(!note_output->blocking);
  BG_CHECK(normalizer.normalize(note).status ==
           DiagnosticNormalizationStatus::kDuplicate);

  RawDiagnosticInput ignored = root();
  ignored.severity = RawDiagnosticSeverity::kIgnored;
  BG_CHECK(normalizer.normalize(ignored).status ==
           DiagnosticNormalizationStatus::kOmitted);
  BG_CHECK(normalizer.diagnostic_count() == 2U);

  RawDiagnosticInput custom;
  custom.stage = RawDiagnosticStage::kPreprocessor;
  custom.severity = RawDiagnosticSeverity::kError;
  custom.custom = true;
  custom.custom_code = CustomDiagnosticCode::kTemporalMacroForbidden;
  custom.rendered_message = "temporal macro is forbidden";
  custom.location = {true, kSpanOne, {}};
  custom.subject.kind = DiagnosticSubjectKind::kCompiler;
  const DiagnosticNormalizationResult custom_result = normalizer.normalize(custom);
  BG_CHECK(custom_result.status == DiagnosticNormalizationStatus::kEmitted);
  const NormalizedDiagnostic* custom_output =
      normalizer.diagnostic(custom_result.diagnostic_index);
  BG_CHECK(custom_output != nullptr);
  BG_CHECK(custom_output->diagnostic_id == expected_custom_id);
  BG_CHECK(custom_output->code ==
           "browsergrad.cpp-cute:temporal-macro-forbidden");
  BG_CHECK(custom_output->category == DiagnosticCategory::kPolicy);

  const std::array fix_its = {
      RawDiagnosticFixIt{kMainPath, 0U, 1U, "x"},
  };
  RawDiagnosticInput with_fix_it = root();
  with_fix_it.fix_its = fix_its;
  BG_CHECK(normalizer.normalize(with_fix_it).status ==
           DiagnosticNormalizationStatus::kDuplicate);

  CppCuteDiagnosticNormalizer resource_limited(config(2U));
  BG_CHECK(resource_limited.normalize(root()).status ==
           DiagnosticNormalizationStatus::kEmitted);
  RawDiagnosticInput second = root();
  second.raw_diagnostic_id = 1236U;
  second.rendered_message = "expected identifier";
  const DiagnosticNormalizationResult limited =
      resource_limited.normalize(second);
  BG_CHECK(limited.status == DiagnosticNormalizationStatus::kResourceLimit);
  BG_CHECK(resource_limited.poisoned());
  BG_CHECK(resource_limited.diagnostic_count() == 2U);
  BG_CHECK(resource_limited.diagnostic(1U)->code ==
           "browsergrad.cpp-cute:diagnostic-resource-limit");
  BG_CHECK(resource_limited.normalize(second).status ==
           DiagnosticNormalizationStatus::kPoisoned);

  CppCuteDiagnosticNormalizer orphan(config());
  BG_CHECK(orphan.normalize(note).status ==
           DiagnosticNormalizationStatus::kInvalidInput);
  BG_CHECK(orphan.poisoned());
  BG_CHECK(orphan.diagnostic_count() == 1U);
  BG_CHECK(orphan.diagnostic(0U)->code ==
           "browsergrad.cpp-cute:diagnostic-normalization-failed");

  const std::array overlapping = {
      RawDiagnosticFixIt{kMainPath, 0U, 4U, "x"},
      RawDiagnosticFixIt{kMainPath, 3U, 5U, "y"},
  };
  CppCuteDiagnosticNormalizer invalid_fix_it(config());
  RawDiagnosticInput overlap = root();
  overlap.fix_its = overlapping;
  BG_CHECK(invalid_fix_it.normalize(overlap).status ==
           DiagnosticNormalizationStatus::kInvalidInput);
  BG_CHECK(invalid_fix_it.poisoned());

  CppCuteDiagnosticNormalizer message_limit(config());
  std::string oversized(4'097U, 'x');
  RawDiagnosticInput too_large = root();
  too_large.rendered_message = oversized;
  BG_CHECK(message_limit.normalize(too_large).status ==
           DiagnosticNormalizationStatus::kResourceLimit);

  static constexpr std::array duplicate_spans = {kSpanOne, kSpanOne};
  DiagnosticNormalizerConfig invalid_config = config();
  invalid_config.opened_span_ids = duplicate_spans;
  CppCuteDiagnosticNormalizer not_configured(invalid_config);
  BG_CHECK(!not_configured.configured());
  BG_CHECK(not_configured.normalize(root()).status ==
           DiagnosticNormalizationStatus::kInvalidInput);

  BG_CHECK(normalizer.retained_normalized_byte_length() > 0U);
  BG_CHECK(normalizer.diagnostic(999U) == nullptr);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr,
                 "usage: diagnostics-native-test ROOT_ID NOTE_ID CUSTOM_ID\n");
    return 2;
  }
  return run(argv[1], argv[2], argv[3]);
}
