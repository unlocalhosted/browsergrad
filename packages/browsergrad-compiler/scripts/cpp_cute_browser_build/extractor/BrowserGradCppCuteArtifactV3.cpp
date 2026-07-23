#include "BrowserGradCppCuteArtifactV3.h"

#include "BrowserGradCppCuteArtifactWriter.h"
#include "BrowserGradCppCuteCompilePlan.h"
#include "BrowserGradCppCuteCompileSession.h"
#include "BrowserGradCppCuteMetrics.h"
#include "BrowserGradCppCuteProducer.h"

namespace browsergrad::cpp_cute {
namespace {

WireCompileStatus wire_status_for_decode(
    const CompileSessionDecodeStatus status) noexcept {
  switch (status) {
    case CompileSessionDecodeStatus::kInvalidFrame:
      return WireCompileStatus::kInvalidFrame;
    case CompileSessionDecodeStatus::kAbiMismatch:
      return WireCompileStatus::kAbiMismatch;
    case CompileSessionDecodeStatus::kResourceLimit:
      return WireCompileStatus::kResourceLimit;
    case CompileSessionDecodeStatus::kInternalError:
    case CompileSessionDecodeStatus::kReady:
      return WireCompileStatus::kInternalError;
  }
  return WireCompileStatus::kInternalError;
}

WireCompileStatus wire_status_for_plan(const CompilePlanStatus status) noexcept {
  switch (status) {
    case CompilePlanStatus::kResourceLimit:
      return WireCompileStatus::kResourceLimit;
    case CompilePlanStatus::kInvalidSessionData:
    case CompilePlanStatus::kInvocationRejected:
      return WireCompileStatus::kInvalidFrame;
    case CompilePlanStatus::kInternalError:
    case CompilePlanStatus::kReady:
      return WireCompileStatus::kInternalError;
  }
  return WireCompileStatus::kInternalError;
}

ArtifactV3CompileResult result_for_producer(
    const ProducerReviewResult& producer,
    const DecodedCompileSession& session,
    ArtifactV3ResultSink& result_sink) noexcept {
  const auto write_artifact = [&]() -> ArtifactV3CompileResult {
    switch (write_cpp_cute_artifact_v3(producer, session, result_sink)) {
      case ArtifactV3WriteStatus::kReady:
        return {WireCompileStatus::kArtifactReady, std::nullopt};
      case ArtifactV3WriteStatus::kInvalidObservation:
        return {WireCompileStatus::kInvalidFrame,
                ReviewOnlyBlocker::kCanonicalArtifactV3Unavailable};
      case ArtifactV3WriteStatus::kResourceLimit:
        return {WireCompileStatus::kResourceLimit,
                ReviewOnlyBlocker::kCanonicalArtifactV3Unavailable};
      case ArtifactV3WriteStatus::kInternalError:
        report_native_diagnostic(
            NativeDiagnosticCode::kArtifactWriterInternal);
        return {WireCompileStatus::kInternalError,
                ReviewOnlyBlocker::kCanonicalArtifactV3Unavailable};
    }
    return {WireCompileStatus::kInternalError,
            ReviewOnlyBlocker::kCanonicalArtifactV3Unavailable};
  };
  switch (producer.status) {
    case ProducerReviewStatus::kReviewComplete: {
      if (producer.completed_pass_count != 2U ||
          !producer.shared_surface_converged) {
        return {WireCompileStatus::kInternalError,
                ReviewOnlyBlocker::kCudaDualPassUnavailable};
      }
      return write_artifact();
    }
    case ProducerReviewStatus::kReviewCompleteWithBlockingDiagnostics:
      return write_artifact();
    case ProducerReviewStatus::kInvalidPlan:
      return {WireCompileStatus::kInvalidFrame,
              ReviewOnlyBlocker::kCudaDualPassUnavailable};
    case ProducerReviewStatus::kResourceLimit:
      return {WireCompileStatus::kResourceLimit,
              ReviewOnlyBlocker::kCudaDualPassUnavailable};
    case ProducerReviewStatus::kVfsError:
      return {WireCompileStatus::kVfsError,
              ReviewOnlyBlocker::kCudaDualPassUnavailable};
    case ProducerReviewStatus::kInternalError:
      report_native_diagnostic(
          NativeDiagnosticCode::kArtifactProducerInternal);
      return {WireCompileStatus::kInternalError,
              ReviewOnlyBlocker::kCudaDualPassUnavailable};
  }
  return {WireCompileStatus::kInternalError,
          ReviewOnlyBlocker::kCudaDualPassUnavailable};
}

}  // namespace

ArtifactV3CompileResult build_artifact_v3(
    const ValidatedInputFrameRegions& input,
    ArtifactV3ResultSink& result_sink) {
  CompileSessionDecodeResult decoded = decode_compile_session(input);
  if (decoded.status != CompileSessionDecodeStatus::kReady ||
      !decoded.session) {
    if (decoded.status == CompileSessionDecodeStatus::kInternalError ||
        decoded.status == CompileSessionDecodeStatus::kReady) {
      report_native_diagnostic(
          NativeDiagnosticCode::kArtifactDecodeInternal);
    }
    return {wire_status_for_decode(decoded.status),
            ReviewOnlyBlocker::kCudaDualPassUnavailable};
  }

  PrepareCppCuteCompilePlanResult prepared =
      prepare_cpp_cute_compile_plan(*decoded.session);
  if (prepared.status != CompilePlanStatus::kReady || !prepared.plan) {
    if (prepared.status == CompilePlanStatus::kInternalError ||
        prepared.status == CompilePlanStatus::kReady) {
      report_native_diagnostic(
          NativeDiagnosticCode::kArtifactPlanInternal);
    }
    return {wire_status_for_plan(prepared.status),
            ReviewOnlyBlocker::kCudaDualPassUnavailable};
  }
  if (!result_sink.bind_invocation_maximum_byte_length(
          prepared.plan->maximum_output_byte_length())) {
    report_native_diagnostic(
        NativeDiagnosticCode::kArtifactSinkBindInternal);
    return {WireCompileStatus::kInternalError,
            ReviewOnlyBlocker::kCudaDualPassUnavailable};
  }

  const FrontendWorkLimitsV1 frontend_limits{
      decoded.session->request_semantic_limit(CompileSemanticLimit::kIncludeDepth),
      decoded.session->request_semantic_limit(CompileSemanticLimit::kMacroExpansions),
      decoded.session->request_semantic_limit(CompileSemanticLimit::kPreprocessedTokens),
      decoded.session->request_semantic_limit(CompileSemanticLimit::kAstNodes),
      decoded.session->request_semantic_limit(CompileSemanticLimit::kConstexprSteps),
      decoded.session->request_semantic_limit(CompileSemanticLimit::kTemplateInstantiations),
      decoded.session->request_semantic_limit(CompileSemanticLimit::kTemplateDepth),
  };
  if (!begin_frontend_work_invocation(frontend_limits)) {
    report_native_diagnostic(
        NativeDiagnosticCode::kArtifactFrontendBeginInternal);
    return {WireCompileStatus::kInternalError,
            ReviewOnlyBlocker::kCudaDualPassUnavailable};
  }

  const ProducerReviewResult producer =
      run_cpp_cute_producer_review(*prepared.plan, *decoded.session);
  if (producer.status == ProducerReviewStatus::kResourceLimit) {
    fail_frontend_work_invocation();
    return {WireCompileStatus::kResourceLimit,
            ReviewOnlyBlocker::kCudaDualPassUnavailable};
  }
  if (producer.completed_pass_count == 0U ||
      !complete_frontend_work_invocation(producer.completed_pass_count)) {
    report_native_diagnostic(
        NativeDiagnosticCode::kArtifactFrontendCompleteInternal);
    fail_frontend_work_invocation();
    return {WireCompileStatus::kInternalError,
            ReviewOnlyBlocker::kCudaDualPassUnavailable};
  }
  ArtifactV3CompileResult result =
      result_for_producer(producer, *decoded.session, result_sink);
  if (result.status != WireCompileStatus::kArtifactReady) {
    fail_frontend_work_invocation();
  }
  return result;
}

}  // namespace browsergrad::cpp_cute
