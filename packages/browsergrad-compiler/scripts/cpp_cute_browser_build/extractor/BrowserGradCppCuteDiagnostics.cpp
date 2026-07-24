#include "BrowserGradCppCuteDiagnostics.h"

#include "BrowserGradCppCuteSha256.h"
#include "BrowserGradCppCuteVirtualPath.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace browsergrad::cpp_cute {
namespace {

struct GeneratedStagePolicy {
  RawDiagnosticStage raw;
  DiagnosticPhase phase;
  DiagnosticCategory category;
};

struct GeneratedSeverityPolicy {
  RawDiagnosticSeverity raw;
  DiagnosticSeverity normalized;
  bool emit;
  bool blocking;
  bool parent_required;
};

struct GeneratedCustomPolicy {
  CustomDiagnosticCode raw;
  RawDiagnosticStage stage;
  DiagnosticSeverity severity;
  DiagnosticCategory category;
  std::string_view code;
};

#include "BrowserGradCppCuteDiagnosticsPolicy.inc"

static_assert(kGeneratedStagePolicies.size() == 7U);
static_assert(kGeneratedSeverityPolicies.size() == 6U);
static_assert(kGeneratedCustomPolicies.size() == 6U);

constexpr std::size_t kMaximumOpenedIdentityCount = 1'000'000U;
constexpr std::size_t kMaximumOpenedIdentityBytes = 64U * 1024U * 1024U;

struct TransparentStringHash {
  using is_transparent = void;

  std::size_t operator()(const std::string_view value) const noexcept {
    return std::hash<std::string_view>{}(value);
  }

  std::size_t operator()(const std::string& value) const noexcept {
    return (*this)(std::string_view(value));
  }
};

struct StoredDiagnostic {
  NormalizedDiagnostic diagnostic;
  std::string canonical_projection;
  std::size_t retained_bytes = 0U;
};

bool lowercase_sha256(std::string_view value) noexcept {
  if (value.size() != 64U) return false;
  return std::all_of(value.begin(), value.end(), [](const char byte) {
    return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
  });
}

bool stable_id(std::string_view value, std::string_view kind) noexcept {
  constexpr std::string_view first = "bg.cpp.";
  constexpr std::string_view last = ".sha256.";
  if (!value.starts_with(first)) return false;
  value.remove_prefix(first.size());
  if (!value.starts_with(kind)) return false;
  value.remove_prefix(kind.size());
  if (!value.starts_with(last)) return false;
  value.remove_prefix(last.size());
  return lowercase_sha256(value);
}

bool valid_utf8(std::string_view value) noexcept {
  std::size_t index = 0U;
  while (index < value.size()) {
    const auto first = static_cast<unsigned char>(value[index]);
    if (first <= 0x7fU) {
      ++index;
      continue;
    }
    const auto continuation = [&value](const std::size_t at) {
      return at < value.size() &&
             (static_cast<unsigned char>(value[at]) & 0xc0U) == 0x80U;
    };
    if (first >= 0xc2U && first <= 0xdfU) {
      if (!continuation(index + 1U)) return false;
      index += 2U;
      continue;
    }
    if (first >= 0xe0U && first <= 0xefU) {
      if (index + 2U >= value.size() || !continuation(index + 1U) ||
          !continuation(index + 2U)) {
        return false;
      }
      const auto second = static_cast<unsigned char>(value[index + 1U]);
      if ((first == 0xe0U && second < 0xa0U) ||
          (first == 0xedU && second >= 0xa0U)) {
        return false;
      }
      index += 3U;
      continue;
    }
    if (first >= 0xf0U && first <= 0xf4U) {
      if (index + 3U >= value.size() || !continuation(index + 1U) ||
          !continuation(index + 2U) || !continuation(index + 3U)) {
        return false;
      }
      const auto second = static_cast<unsigned char>(value[index + 1U]);
      if ((first == 0xf0U && second < 0x90U) ||
          (first == 0xf4U && second >= 0x90U)) {
        return false;
      }
      index += 4U;
      continue;
    }
    return false;
  }
  return true;
}

bool valid_rendered_text(std::string_view value,
                         const std::size_t maximum) noexcept {
  if (value.empty() || value.size() > maximum || !valid_utf8(value)) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](const char value) {
    const auto byte = static_cast<unsigned char>(value);
    return byte == 0x09U || byte == 0x0aU || byte >= 0x20U;
  });
}

const GeneratedStagePolicy* stage_policy(
    const RawDiagnosticStage stage) noexcept {
  const auto found = std::find_if(
      kGeneratedStagePolicies.begin(), kGeneratedStagePolicies.end(),
      [stage](const GeneratedStagePolicy& candidate) {
        return candidate.raw == stage;
      });
  return found == kGeneratedStagePolicies.end() ? nullptr : &*found;
}

const GeneratedSeverityPolicy* severity_policy(
    const RawDiagnosticSeverity severity) noexcept {
  const auto found = std::find_if(
      kGeneratedSeverityPolicies.begin(), kGeneratedSeverityPolicies.end(),
      [severity](const GeneratedSeverityPolicy& candidate) {
        return candidate.raw == severity;
      });
  return found == kGeneratedSeverityPolicies.end() ? nullptr : &*found;
}

const GeneratedCustomPolicy* custom_policy(
    const CustomDiagnosticCode code) noexcept {
  const auto found = std::find_if(
      kGeneratedCustomPolicies.begin(), kGeneratedCustomPolicies.end(),
      [code](const GeneratedCustomPolicy& candidate) {
        return candidate.raw == code;
      });
  return found == kGeneratedCustomPolicies.end() ? nullptr : &*found;
}

std::string_view phase_name(const DiagnosticPhase phase) noexcept {
  switch (phase) {
    case DiagnosticPhase::kPreprocessing: return "preprocessing";
    case DiagnosticPhase::kParsing: return "parsing";
    case DiagnosticPhase::kNameLookup: return "name-lookup";
    case DiagnosticPhase::kOverloadResolution: return "overload-resolution";
    case DiagnosticPhase::kTemplateInstantiation: return "template-instantiation";
    case DiagnosticPhase::kCudaSema: return "cuda-sema";
    case DiagnosticPhase::kArtifactExtraction: return "artifact-extraction";
  }
  return {};
}

std::string_view severity_name(const DiagnosticSeverity severity) noexcept {
  switch (severity) {
    case DiagnosticSeverity::kRemark: return "remark";
    case DiagnosticSeverity::kNote: return "note";
    case DiagnosticSeverity::kWarning: return "warning";
    case DiagnosticSeverity::kError: return "error";
    case DiagnosticSeverity::kFatal: return "fatal";
    case DiagnosticSeverity::kNone: return {};
  }
  return {};
}

std::string_view subject_kind_name(const DiagnosticSubjectKind kind) noexcept {
  switch (kind) {
    case DiagnosticSubjectKind::kCompiler: return "compiler";
    case DiagnosticSubjectKind::kFile: return "file";
    case DiagnosticSubjectKind::kDeclaration: return "declaration";
    case DiagnosticSubjectKind::kType: return "type";
    case DiagnosticSubjectKind::kExpression: return "expression";
    case DiagnosticSubjectKind::kFact: return "fact";
  }
  return {};
}

std::string_view subject_entity_kind(
    const DiagnosticSubjectKind kind) noexcept {
  switch (kind) {
    case DiagnosticSubjectKind::kFile: return "file";
    case DiagnosticSubjectKind::kDeclaration: return "declaration";
    case DiagnosticSubjectKind::kType: return "type";
    case DiagnosticSubjectKind::kExpression: return "expression";
    case DiagnosticSubjectKind::kFact: return "fact";
    case DiagnosticSubjectKind::kCompiler: return {};
  }
  return {};
}

std::string_view subject_entity_field(
    const DiagnosticSubjectKind kind) noexcept {
  switch (kind) {
    case DiagnosticSubjectKind::kFile: return "fileId";
    case DiagnosticSubjectKind::kDeclaration: return "declarationId";
    case DiagnosticSubjectKind::kType: return "typeId";
    case DiagnosticSubjectKind::kExpression: return "expressionId";
    case DiagnosticSubjectKind::kFact: return "factId";
    case DiagnosticSubjectKind::kCompiler: return {};
  }
  return {};
}

bool checked_add(std::size_t& total, const std::size_t value,
                 const std::size_t maximum) noexcept {
  if (value > maximum || total > maximum - value) return false;
  total += value;
  return true;
}

void append_json_string(std::string& output, const std::string_view value) {
  output.push_back('"');
  for (const char character : value) {
    switch (character) {
      case '"': output.append("\\\""); break;
      case '\\': output.append("\\\\"); break;
      case '\t': output.append("\\t"); break;
      case '\n': output.append("\\n"); break;
      default: output.push_back(character); break;
    }
  }
  output.push_back('"');
}

void append_subject(std::string& output,
                    const NormalizedDiagnosticSubject& subject) {
  output.push_back('{');
  const std::string_view kind = subject_kind_name(subject.kind);
  if (subject.kind == DiagnosticSubjectKind::kCompiler) {
    output.append("\"kind\":\"compiler\"");
  } else {
    const std::string_view field = subject_entity_field(subject.kind);
    if (field < std::string_view("kind")) {
      append_json_string(output, field);
      output.push_back(':');
      append_json_string(output, subject.entity_id);
      output.append(",\"kind\":");
      append_json_string(output, kind);
    } else {
      output.append("\"kind\":");
      append_json_string(output, kind);
      output.push_back(',');
      append_json_string(output, field);
      output.push_back(':');
      append_json_string(output, subject.entity_id);
    }
  }
  output.push_back('}');
}

void append_location(std::string& output,
                     const NormalizedDiagnosticLocation& location) {
  if (!location.has_source) {
    output.append("{\"kind\":\"none\"}");
    return;
  }
  output.append("{\"kind\":\"source\",\"primarySpanId\":");
  append_json_string(output, location.primary_span_id);
  output.append(",\"related\":[");
  for (std::size_t index = 0U; index < location.related.size(); ++index) {
    if (index != 0U) output.push_back(',');
    output.append("{\"message\":");
    append_json_string(output, location.related[index].rendered_message);
    output.append(",\"spanId\":");
    append_json_string(output, location.related[index].span_id);
    output.push_back('}');
  }
  output.append("]}");
}

std::string canonical_projection(
    const std::string_view compilation_contract_hash,
    const std::string_view owner_pass_id,
    const NormalizedDiagnostic& diagnostic) {
  std::string output;
  output.reserve(512U + diagnostic.rendered_message.size() +
                 diagnostic.location.related.size() * 128U);
  output.append("{\"compilationContractHash\":");
  append_json_string(output, compilation_contract_hash);
  output.append(",\"diagnostic\":{\"code\":");
  append_json_string(output, diagnostic.code);
  output.append(",\"location\":");
  append_location(output, diagnostic.location);
  output.append(",\"parentDiagnosticId\":");
  if (diagnostic.parent_diagnostic_id.empty()) {
    output.append("null");
  } else {
    append_json_string(output, diagnostic.parent_diagnostic_id);
  }
  output.append(",\"phase\":");
  append_json_string(output, phase_name(diagnostic.phase));
  output.append(",\"renderedMessage\":");
  append_json_string(output, diagnostic.rendered_message);
  output.append(",\"severity\":");
  append_json_string(output, severity_name(diagnostic.severity));
  output.append(",\"subject\":");
  append_subject(output, diagnostic.subject);
  output.append("},\"domain\":");
  append_json_string(output, kDiagnosticStableIdDomain);
  output.append(",\"ownerPassId\":");
  append_json_string(output, owner_pass_id);
  output.push_back('}');
  return output;
}

std::string diagnostic_id(const std::string_view projection) {
  Sha256 hash;
  Sha256Digest digest{};
  if (!hash.update(reinterpret_cast<const std::uint8_t*>(projection.data()),
                   projection.size()) ||
      !hash.finalize(digest)) {
    return {};
  }
  const Sha256LowercaseHex hex = sha256_lowercase_hex(digest);
  std::string output(kDiagnosticStableIdPrefix);
  output.append(hex.data(), 64U);
  return output;
}

std::string clang_code(const std::uint32_t raw_id) {
  std::array<char, 10U> digits{};
  const auto encoded = std::to_chars(digits.data(),
                                     digits.data() + digits.size(), raw_id);
  if (encoded.ec != std::errc{}) return {};
  std::string result(kClangDiagnosticCodePrefix);
  result.append(digits.data(), encoded.ptr);
  return result;
}

std::size_t retained_size(const NormalizedDiagnostic& diagnostic,
                          const std::string_view projection) noexcept {
  std::size_t result = sizeof(NormalizedDiagnostic) + projection.size();
  const auto add = [&result](const std::size_t value) {
    return checked_add(result, value,
                       std::numeric_limits<std::size_t>::max());
  };
  if (!add(diagnostic.diagnostic_id.size() * 2U) ||
      !add(diagnostic.code.size()) ||
      !add(diagnostic.rendered_message.size()) ||
      !add(diagnostic.location.primary_span_id.size()) ||
      !add(diagnostic.subject.entity_id.size()) ||
      !add(diagnostic.parent_diagnostic_id.size()) ||
      !add(diagnostic.location.related.size() *
           sizeof(NormalizedDiagnosticRelatedLocation))) {
    return std::numeric_limits<std::size_t>::max();
  }
  for (const auto& related : diagnostic.location.related) {
    if (!add(related.span_id.size()) ||
        !add(related.rendered_message.size())) {
      return std::numeric_limits<std::size_t>::max();
    }
  }
  return result;
}

}  // namespace

std::uint32_t
cpp_cute_maximum_retained_normalized_diagnostic_bytes() noexcept {
  return kMaximumRetainedNormalizedBytes;
}

struct CppCuteDiagnosticNormalizer::Impl final {
  explicit Impl(const DiagnosticNormalizerConfig& config) {
    if (!lowercase_sha256(config.compilation_contract_hash) ||
        (config.owner_pass_id != "cuda-device-sema" &&
         config.owner_pass_id != "cuda-host-sema") ||
        config.maximum_unique_diagnostics == 0U ||
        config.maximum_unique_diagnostics > kMaximumUniqueDiagnostics ||
        config.maximum_retained_normalized_bytes == 0U ||
        config.maximum_retained_normalized_bytes >
            kMaximumRetainedNormalizedBytes ||
        config.opened_span_ids.size() > kMaximumOpenedIdentityCount ||
        config.opened_virtual_paths.size() > kMaximumOpenedIdentityCount) {
      return;
    }
    compilation_contract_hash = config.compilation_contract_hash;
    owner_pass_id = config.owner_pass_id;
    maximum_unique_diagnostics = config.maximum_unique_diagnostics;
    maximum_retained_bytes = config.maximum_retained_normalized_bytes;
    for (const CustomDiagnosticCode terminal_code : {
             CustomDiagnosticCode::kDiagnosticResourceLimit,
             CustomDiagnosticCode::kDiagnosticNormalizationFailed,
         }) {
      const GeneratedCustomPolicy* policy = custom_policy(terminal_code);
      if (policy == nullptr) return;
      NormalizedDiagnostic terminal;
      terminal.phase = DiagnosticPhase::kArtifactExtraction;
      terminal.severity = policy->severity;
      terminal.category = policy->category;
      terminal.code = policy->code;
      terminal.rendered_message =
          terminal_code == CustomDiagnosticCode::kDiagnosticResourceLimit
              ? "diagnostic normalization exceeded a configured resource limit"
              : "diagnostic normalization rejected invalid producer input";
      terminal.subject.kind = DiagnosticSubjectKind::kCompiler;
      terminal.blocking = true;
      const std::string projection = canonical_projection(
          compilation_contract_hash, owner_pass_id, terminal);
      terminal.diagnostic_id = diagnostic_id(projection);
      const std::size_t cost = retained_size(terminal, projection);
      if (terminal.diagnostic_id.empty() ||
          cost == std::numeric_limits<std::size_t>::max()) {
        return;
      }
      terminal_reserve_bytes = std::max(terminal_reserve_bytes, cost);
    }
    if (maximum_retained_bytes < terminal_reserve_bytes) return;

    std::size_t registry_bytes = 0U;
    opened_span_ids.reserve(config.opened_span_ids.size());
    for (const std::string_view id : config.opened_span_ids) {
      if (!stable_id(id, "span") ||
          !checked_add(registry_bytes, id.size(),
                       kMaximumOpenedIdentityBytes) ||
          !opened_span_ids.emplace(id).second) {
        return;
      }
    }
    opened_virtual_paths.reserve(config.opened_virtual_paths.size());
    for (const std::string_view path : config.opened_virtual_paths) {
      if (!cpp_cute_valid_canonical_virtual_path(path) ||
          !checked_add(registry_bytes, path.size(),
                       kMaximumOpenedIdentityBytes) ||
          !opened_virtual_paths.emplace(path).second) {
        return;
      }
    }
    configured = true;
  }

  bool known_span(const std::string_view id) const {
    return opened_span_ids.find(id) != opened_span_ids.end();
  }

  bool known_path(const std::string_view path) const {
    return opened_virtual_paths.find(path) != opened_virtual_paths.end();
  }

  bool valid_subject(const RawDiagnosticSubject& subject) const noexcept {
    if (subject.kind == DiagnosticSubjectKind::kCompiler) {
      return subject.entity_id.empty();
    }
    const std::string_view entity_kind = subject_entity_kind(subject.kind);
    return !entity_kind.empty() && stable_id(subject.entity_id, entity_kind);
  }

  DiagnosticNormalizationStatus validate_location(
      const RawDiagnosticInput& input,
      NormalizedDiagnosticLocation& output) const {
    if (!input.location.has_source) {
      return input.location.primary_span_id.empty() &&
                     input.location.related.empty() &&
                     input.subject.kind == DiagnosticSubjectKind::kCompiler
                 ? DiagnosticNormalizationStatus::kReady
                 : DiagnosticNormalizationStatus::kInvalidInput;
    }
    if (!known_span(input.location.primary_span_id)) {
      return DiagnosticNormalizationStatus::kInvalidInput;
    }
    if (input.location.related.size() > kMaximumRelatedLocationCount) {
      return DiagnosticNormalizationStatus::kResourceLimit;
    }
    output.has_source = true;
    output.primary_span_id = input.location.primary_span_id;
    output.related.reserve(input.location.related.size());
    std::size_t aggregate = 0U;
    for (const RawDiagnosticRelatedLocation& related :
         input.location.related) {
      if (!known_span(related.span_id) ||
          !valid_rendered_text(related.rendered_message,
                               kMaximumRelatedMessageBytes)) {
        return related.rendered_message.size() > kMaximumRelatedMessageBytes
                   ? DiagnosticNormalizationStatus::kResourceLimit
                   : DiagnosticNormalizationStatus::kInvalidInput;
      }
      if (!checked_add(aggregate, related.rendered_message.size(),
                       kMaximumAggregateRelatedMessageBytes)) {
        return DiagnosticNormalizationStatus::kResourceLimit;
      }
      output.related.push_back(
          {std::string(related.span_id),
           std::string(related.rendered_message)});
    }
    return DiagnosticNormalizationStatus::kReady;
  }

  DiagnosticNormalizationStatus validate_fix_its(
      const std::span<const RawDiagnosticFixIt> fix_its) const noexcept {
    if (fix_its.size() > kMaximumFixItsPerDiagnostic) {
      return DiagnosticNormalizationStatus::kResourceLimit;
    }
    std::size_t aggregate = 0U;
    for (std::size_t index = 0U; index < fix_its.size(); ++index) {
      const RawDiagnosticFixIt& current = fix_its[index];
      if (!cpp_cute_valid_canonical_virtual_path(current.virtual_path) ||
          !known_path(current.virtual_path) ||
          current.begin_byte > current.end_byte ||
          !valid_utf8(current.replacement)) {
        return DiagnosticNormalizationStatus::kInvalidInput;
      }
      if (current.replacement.size() > kMaximumReplacementBytesPerFixIt ||
          !checked_add(aggregate, current.replacement.size(),
                       kMaximumAggregateReplacementBytes)) {
        return DiagnosticNormalizationStatus::kResourceLimit;
      }
      for (std::size_t previous = 0U; previous < index; ++previous) {
        const RawDiagnosticFixIt& candidate = fix_its[previous];
        if (candidate.virtual_path == current.virtual_path &&
            candidate.begin_byte < current.end_byte &&
            current.begin_byte < candidate.end_byte) {
          return DiagnosticNormalizationStatus::kInvalidInput;
        }
      }
    }
    return DiagnosticNormalizationStatus::kReady;
  }

  DiagnosticNormalizationResult poison(
      const DiagnosticNormalizationStatus status) noexcept {
    if (poisoned) return {DiagnosticNormalizationStatus::kPoisoned, 0U};
    poisoned = true;
    terminal_status = status;
    const CustomDiagnosticCode code =
        status == DiagnosticNormalizationStatus::kResourceLimit
            ? CustomDiagnosticCode::kDiagnosticResourceLimit
            : CustomDiagnosticCode::kDiagnosticNormalizationFailed;
    const std::string_view message =
        status == DiagnosticNormalizationStatus::kResourceLimit
            ? "diagnostic normalization exceeded a configured resource limit"
            : "diagnostic normalization rejected invalid producer input";
    const GeneratedCustomPolicy* policy = custom_policy(code);
    if (policy == nullptr || diagnostics.size() >= maximum_unique_diagnostics) {
      return {status, 0U};
    }
    try {
      NormalizedDiagnostic diagnostic;
      diagnostic.phase = DiagnosticPhase::kArtifactExtraction;
      diagnostic.severity = policy->severity;
      diagnostic.category = policy->category;
      diagnostic.code = policy->code;
      diagnostic.rendered_message = message;
      diagnostic.subject.kind = DiagnosticSubjectKind::kCompiler;
      diagnostic.blocking = true;
      std::string projection = canonical_projection(
          compilation_contract_hash, owner_pass_id, diagnostic);
      diagnostic.diagnostic_id = diagnostic_id(projection);
      const std::size_t cost = retained_size(diagnostic, projection);
      if (diagnostic.diagnostic_id.empty() || cost > maximum_retained_bytes ||
          retained_bytes > maximum_retained_bytes - cost) {
        return {status, 0U};
      }
      const std::size_t index = diagnostics.size();
      duplicate_index.emplace(diagnostic.diagnostic_id, index);
      retained_bytes += cost;
      diagnostics.push_back(
          {std::move(diagnostic), std::move(projection), cost});
      return {status, index};
    } catch (...) {
      return {status, 0U};
    }
  }

  DiagnosticNormalizationResult normalize_impl(
      const RawDiagnosticInput& input) {
    if (!configured) {
      return {DiagnosticNormalizationStatus::kInvalidInput, 0U};
    }
    if (poisoned) {
      return {DiagnosticNormalizationStatus::kPoisoned, 0U};
    }
    const GeneratedStagePolicy* stage = stage_policy(input.stage);
    const GeneratedSeverityPolicy* severity = severity_policy(input.severity);
    if (stage == nullptr || severity == nullptr) {
      return poison(DiagnosticNormalizationStatus::kInvalidInput);
    }
    if (!severity->emit) {
      return {DiagnosticNormalizationStatus::kOmitted, 0U};
    }

    const GeneratedCustomPolicy* custom = nullptr;
    if (input.custom) {
      custom = custom_policy(input.custom_code);
      if (custom == nullptr || custom->stage != input.stage ||
          custom->severity != severity->normalized) {
        return poison(DiagnosticNormalizationStatus::kInvalidInput);
      }
    }
    const std::size_t message_limit =
        severity->parent_required ? kMaximumRenderedNoteBytes
                                  : kMaximumRenderedMessageBytes;
    if (input.rendered_message.size() > message_limit) {
      return poison(DiagnosticNormalizationStatus::kResourceLimit);
    }
    if (!valid_rendered_text(input.rendered_message, message_limit) ||
        !valid_subject(input.subject)) {
      return poison(DiagnosticNormalizationStatus::kInvalidInput);
    }
    const DiagnosticNormalizationStatus fix_it_status =
        validate_fix_its(input.fix_its);
    if (fix_it_status != DiagnosticNormalizationStatus::kReady) {
      return poison(fix_it_status);
    }

    NormalizedDiagnostic diagnostic;
    diagnostic.phase = stage->phase;
    diagnostic.severity = severity->normalized;
    diagnostic.category = custom == nullptr ? stage->category : custom->category;
    diagnostic.code = custom == nullptr ? clang_code(input.raw_diagnostic_id)
                                        : std::string(custom->code);
    diagnostic.rendered_message = input.rendered_message;
    diagnostic.subject = {input.subject.kind,
                          std::string(input.subject.entity_id)};
    diagnostic.blocking = custom == nullptr ? severity->blocking : true;
    if (diagnostic.code.empty()) {
      return poison(DiagnosticNormalizationStatus::kInvalidInput);
    }
    const DiagnosticNormalizationStatus location_status =
        validate_location(input, diagnostic.location);
    if (location_status != DiagnosticNormalizationStatus::kReady) {
      return poison(location_status);
    }
    if (severity->parent_required) {
      if (last_root_diagnostic_id.empty()) {
        return poison(DiagnosticNormalizationStatus::kInvalidInput);
      }
      diagnostic.parent_diagnostic_id = last_root_diagnostic_id;
    }

    std::string projection = canonical_projection(
        compilation_contract_hash, owner_pass_id, diagnostic);
    diagnostic.diagnostic_id = diagnostic_id(projection);
    if (diagnostic.diagnostic_id.empty()) {
      return poison(DiagnosticNormalizationStatus::kInvalidInput);
    }
    const auto duplicate = duplicate_index.find(diagnostic.diagnostic_id);
    if (duplicate != duplicate_index.end()) {
      const StoredDiagnostic& stored = diagnostics[duplicate->second];
      if (stored.canonical_projection != projection) {
        return poison(DiagnosticNormalizationStatus::kInvalidInput);
      }
      if (!severity->parent_required) {
        last_root_diagnostic_id = diagnostic.diagnostic_id;
        note_count = 0U;
        note_bytes = 0U;
      }
      return {DiagnosticNormalizationStatus::kDuplicate,
              duplicate->second};
    }

    if (severity->parent_required &&
        (note_count >= kMaximumNotesPerRoot ||
         input.rendered_message.size() >
             kMaximumAggregateNoteBytes - note_bytes)) {
      return poison(DiagnosticNormalizationStatus::kResourceLimit);
    }
    if (diagnostics.size() + 1U >= maximum_unique_diagnostics) {
      return poison(DiagnosticNormalizationStatus::kResourceLimit);
    }
    const std::size_t cost = retained_size(diagnostic, projection);
    if (cost == std::numeric_limits<std::size_t>::max() ||
        cost > maximum_retained_bytes ||
        retained_bytes > maximum_retained_bytes - cost ||
        maximum_retained_bytes - retained_bytes - cost <
            terminal_reserve_bytes) {
      return poison(DiagnosticNormalizationStatus::kResourceLimit);
    }

    const std::size_t index = diagnostics.size();
    duplicate_index.emplace(diagnostic.diagnostic_id, index);
    retained_bytes += cost;
    diagnostics.push_back(
        {std::move(diagnostic), std::move(projection), cost});
    if (severity->parent_required) {
      ++note_count;
      note_bytes += input.rendered_message.size();
    } else {
      last_root_diagnostic_id = diagnostics.back().diagnostic.diagnostic_id;
      note_count = 0U;
      note_bytes = 0U;
    }
    return {DiagnosticNormalizationStatus::kEmitted, index};
  }

  bool configured = false;
  bool poisoned = false;
  DiagnosticNormalizationStatus terminal_status =
      DiagnosticNormalizationStatus::kReady;
  std::string compilation_contract_hash;
  std::string owner_pass_id;
  std::uint32_t maximum_unique_diagnostics = 0U;
  std::size_t maximum_retained_bytes = 0U;
  std::size_t terminal_reserve_bytes = 0U;
  std::size_t retained_bytes = 0U;
  std::unordered_set<std::string, TransparentStringHash, std::equal_to<>>
      opened_span_ids;
  std::unordered_set<std::string, TransparentStringHash, std::equal_to<>>
      opened_virtual_paths;
  std::vector<StoredDiagnostic> diagnostics;
  std::unordered_map<std::string, std::size_t> duplicate_index;
  std::string last_root_diagnostic_id;
  std::uint32_t note_count = 0U;
  std::size_t note_bytes = 0U;
};

CppCuteDiagnosticNormalizer::CppCuteDiagnosticNormalizer(
    const DiagnosticNormalizerConfig& config)
    : implementation_(std::make_unique<Impl>(config)) {}

CppCuteDiagnosticNormalizer::~CppCuteDiagnosticNormalizer() = default;

bool CppCuteDiagnosticNormalizer::configured() const noexcept {
  return implementation_->configured;
}

bool CppCuteDiagnosticNormalizer::poisoned() const noexcept {
  return implementation_->poisoned;
}

DiagnosticNormalizationStatus
CppCuteDiagnosticNormalizer::terminal_status() const noexcept {
  return implementation_->terminal_status;
}

std::string_view CppCuteDiagnosticNormalizer::policy_manifest_id() const noexcept {
  return kDiagnosticNormalizationManifestId;
}

std::size_t CppCuteDiagnosticNormalizer::diagnostic_count() const noexcept {
  return implementation_->diagnostics.size();
}

const NormalizedDiagnostic* CppCuteDiagnosticNormalizer::diagnostic(
    const std::size_t index) const noexcept {
  if (index >= implementation_->diagnostics.size()) return nullptr;
  return &implementation_->diagnostics[index].diagnostic;
}

std::uint32_t
CppCuteDiagnosticNormalizer::retained_normalized_byte_length() const noexcept {
  return static_cast<std::uint32_t>(implementation_->retained_bytes);
}

DiagnosticNormalizationResult CppCuteDiagnosticNormalizer::normalize(
    const RawDiagnosticInput& input) noexcept {
  try {
    return implementation_->normalize_impl(input);
  } catch (...) {
    return implementation_->poison(DiagnosticNormalizationStatus::kInvalidInput);
  }
}

}  // namespace browsergrad::cpp_cute
