#pragma once

#include <cstddef>
#include <cstdint>

namespace browsergrad::cpp_cute {

/**
 * ABI 1.1 module-global requested-byte allocator record.
 *
 * Storage is owned by the Wasm module for its complete lifetime. The Worker
 * may copy this record only between synchronous ABI calls. It must never write
 * through the exported pointer.
 */
struct alignas(8) AllocatorMetricsRecordV1 {
  std::uint8_t magic[8];
  std::uint32_t version;
  std::uint32_t byte_length;
  std::uint64_t current_live_global_requested_byte_length;
  std::uint64_t peak_live_global_requested_byte_length;
  std::uint64_t cumulative_global_allocated_requested_byte_length;
  std::uint64_t cumulative_global_freed_requested_byte_length;
  std::uint64_t successful_allocation_count;
  std::uint64_t free_count;
  std::uint64_t failed_allocation_count;
};

static_assert(sizeof(AllocatorMetricsRecordV1) == 72U);
static_assert(alignof(AllocatorMetricsRecordV1) == 8U);
static_assert(offsetof(AllocatorMetricsRecordV1, magic) == 0U);
static_assert(offsetof(AllocatorMetricsRecordV1, version) == 8U);
static_assert(offsetof(AllocatorMetricsRecordV1, byte_length) == 12U);
static_assert(offsetof(
                  AllocatorMetricsRecordV1,
                  current_live_global_requested_byte_length) == 16U);
static_assert(offsetof(
                  AllocatorMetricsRecordV1,
                  peak_live_global_requested_byte_length) == 24U);
static_assert(offsetof(
                  AllocatorMetricsRecordV1,
                  cumulative_global_allocated_requested_byte_length) == 32U);
static_assert(offsetof(
                  AllocatorMetricsRecordV1,
                  cumulative_global_freed_requested_byte_length) == 40U);
static_assert(offsetof(
                  AllocatorMetricsRecordV1,
                  successful_allocation_count) == 48U);
static_assert(offsetof(AllocatorMetricsRecordV1, free_count) == 56U);
static_assert(offsetof(
                  AllocatorMetricsRecordV1,
                  failed_allocation_count) == 64U);

std::uint32_t allocator_metrics_pointer();

/**
 * False is sticky for the module lifetime. Counter overflow, allocator-hook
 * re-entry, or requested-size metadata corruption permanently forbids an
 * artifact-ready result. Reset deliberately cannot recover this state.
 */
bool allocator_metrics_healthy();

/**
 * ABI 1.2 exact frontend-work record for one synchronous compilation.
 *
 * Counters aggregate the CUDA device pass followed by the CUDA host pass.
 * The host may admit the values only when phase is kComplete, flags is
 * kHealthy, and completed_semantic_passes is one or two. Accepted artifacts
 * require both passes; a rejected artifact may terminate after the first
 * blocking pass. A limit or counter overflow moves the record to kFailed
 * before a successful artifact can be returned.
 */
enum class FrontendWorkPhaseV1 : std::uint32_t {
  kIdle = 0U,
  kCollecting = 1U,
  kComplete = 2U,
  kFailed = 3U,
};

inline constexpr std::uint32_t kFrontendWorkHealthyFlagV1 = 1U;

struct alignas(8) FrontendWorkMetricsRecordV1 {
  std::uint8_t magic[8];
  std::uint32_t version;
  std::uint32_t byte_length;
  std::uint32_t phase;
  std::uint32_t flags;
  std::uint64_t generation;
  std::uint64_t include_depth;
  std::uint64_t macro_expansions;
  std::uint64_t preprocessed_tokens;
  std::uint64_t ast_nodes;
  std::uint64_t constexpr_steps;
  std::uint64_t template_instantiations;
  std::uint64_t template_depth;
  std::uint64_t completed_semantic_passes;
};

static_assert(sizeof(FrontendWorkMetricsRecordV1) == 96U);
static_assert(alignof(FrontendWorkMetricsRecordV1) == 8U);
static_assert(offsetof(FrontendWorkMetricsRecordV1, magic) == 0U);
static_assert(offsetof(FrontendWorkMetricsRecordV1, version) == 8U);
static_assert(offsetof(FrontendWorkMetricsRecordV1, byte_length) == 12U);
static_assert(offsetof(FrontendWorkMetricsRecordV1, phase) == 16U);
static_assert(offsetof(FrontendWorkMetricsRecordV1, flags) == 20U);
static_assert(offsetof(FrontendWorkMetricsRecordV1, generation) == 24U);
static_assert(offsetof(FrontendWorkMetricsRecordV1, include_depth) == 32U);
static_assert(offsetof(FrontendWorkMetricsRecordV1, macro_expansions) == 40U);
static_assert(offsetof(FrontendWorkMetricsRecordV1, preprocessed_tokens) == 48U);
static_assert(offsetof(FrontendWorkMetricsRecordV1, ast_nodes) == 56U);
static_assert(offsetof(FrontendWorkMetricsRecordV1, constexpr_steps) == 64U);
static_assert(offsetof(FrontendWorkMetricsRecordV1,
                       template_instantiations) == 72U);
static_assert(offsetof(FrontendWorkMetricsRecordV1, template_depth) == 80U);
static_assert(offsetof(FrontendWorkMetricsRecordV1,
                       completed_semantic_passes) == 88U);

struct FrontendWorkLimitsV1 {
  std::uint64_t max_include_depth = 0U;
  std::uint64_t max_macro_expansions = 0U;
  std::uint64_t max_preprocessed_tokens = 0U;
  std::uint64_t max_ast_nodes = 0U;
  std::uint64_t max_constexpr_steps = 0U;
  std::uint64_t max_template_instantiations = 0U;
  std::uint64_t max_template_depth = 0U;
};

std::uint32_t frontend_work_metrics_pointer();
bool begin_frontend_work_invocation(FrontendWorkLimitsV1 limits) noexcept;
bool begin_frontend_work_semantic_pass() noexcept;
bool record_frontend_include_depth(std::uint64_t depth) noexcept;
bool record_frontend_macro_expansion() noexcept;
bool record_frontend_preprocessed_token() noexcept;
bool record_frontend_ast_node() noexcept;
bool begin_frontend_template_instantiation() noexcept;
bool end_frontend_template_instantiation() noexcept;
bool complete_frontend_work_semantic_pass() noexcept;
bool complete_frontend_work_invocation(
    std::uint64_t expected_semantic_passes) noexcept;
void fail_frontend_work_invocation() noexcept;
void reset_frontend_work_metrics() noexcept;
bool frontend_work_metrics_ready() noexcept;

#if defined(BG_CPP_CUTE_METRICS_TESTING)
const FrontendWorkMetricsRecordV1&
frontend_work_metrics_record_for_testing() noexcept;
#endif

/** Called only by the pinned ExprConstant.cpp instrumentation patch. */
extern "C" bool browsergrad_cpp_cute_constexpr_step_hook() noexcept;

}  // namespace browsergrad::cpp_cute
