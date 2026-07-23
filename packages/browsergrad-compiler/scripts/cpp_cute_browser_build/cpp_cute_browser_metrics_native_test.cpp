#define BG_CPP_CUTE_METRICS_TESTING 1
#include "extractor/BrowserGradCppCuteMetrics.cpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {

using namespace browsergrad::cpp_cute;

#define BG_CHECK(condition)                                                   \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::fprintf(stderr, "metrics check failed at line %d: %s\n",         \
                   __LINE__, #condition);                                     \
      return 1;                                                               \
    }                                                                         \
  } while (false)

void* fail_malloc(std::size_t) { return nullptr; }
void* fail_calloc(std::size_t, std::size_t) { return nullptr; }
void* fail_realloc(void*, std::size_t) { return nullptr; }
void* same_pointer_realloc(void* pointer, std::size_t) { return pointer; }

void* moved_realloc(void* pointer, std::size_t byte_length) {
  void* replacement = std::malloc(byte_length);
  if (replacement != nullptr) std::free(pointer);
  return replacement;
}

void reset_state() {
  g_test_builtin_allocator = TestBuiltinAllocator{
      std::malloc,
      std::calloc,
      std::realloc,
      std::free,
      test_memalign,
  };
  if (g_allocations.entries != nullptr) std::free(g_allocations.entries);
  g_allocations = AllocationTable{};
  g_metrics = AllocatorMetricsRecordV1{
      {'B', 'G', 'R', 'T', 'M', 'E', 'T', '1'},
      1U,
      72U,
      0U,
      0U,
      0U,
      0U,
      0U,
      0U,
      0U,
  };
  g_metrics_healthy = true;
  g_metrics_failure_reason = AllocatorMetricsFailureReason::kNone;
  g_allocator_hook_active = false;
  g_frontend_work = FrontendWorkMetricsRecordV1{
      {'B', 'G', 'F', 'W', 'K', '0', '0', '1'},
      1U,
      96U,
      static_cast<std::uint32_t>(FrontendWorkPhaseV1::kIdle),
      kFrontendWorkHealthyFlagV1,
      0U,
      0U,
      0U,
      0U,
      0U,
      0U,
      0U,
      0U,
      0U,
  };
  g_frontend_work_limits = FrontendWorkLimitsV1{};
  g_frontend_work_active_template_depth = 0U;
  g_frontend_work_pass_active = false;
}

bool record_invariants_hold() {
  return g_metrics.cumulative_global_freed_requested_byte_length <=
             g_metrics.cumulative_global_allocated_requested_byte_length &&
         g_metrics.current_live_global_requested_byte_length ==
             g_metrics.cumulative_global_allocated_requested_byte_length -
                 g_metrics.cumulative_global_freed_requested_byte_length &&
         g_metrics.free_count <= g_metrics.successful_allocation_count &&
         g_metrics.current_live_global_requested_byte_length <=
             g_metrics.peak_live_global_requested_byte_length &&
         g_metrics.peak_live_global_requested_byte_length <=
             g_metrics.cumulative_global_allocated_requested_byte_length;
}

int run_metrics_tests() {
  reset_state();
  BG_CHECK(sizeof(g_metrics) == 72U);
  BG_CHECK(alignof(decltype(g_metrics)) == 8U);
  BG_CHECK(std::memcmp(g_metrics.magic, "BGRTMET1", 8U) == 0);
  BG_CHECK(g_metrics.version == 1U);
  BG_CHECK(g_metrics.byte_length == 72U);
  BG_CHECK(allocator_metrics_failure_reason() ==
           AllocatorMetricsFailureReason::kNone);
  BG_CHECK(sizeof(g_frontend_work) == 96U);
  BG_CHECK(alignof(decltype(g_frontend_work)) == 8U);
  BG_CHECK(std::memcmp(g_frontend_work.magic, "BGFWK001", 8U) == 0);
  BG_CHECK(g_frontend_work.version == 1U);
  BG_CHECK(g_frontend_work.byte_length == 96U);
  BG_CHECK(g_frontend_work.phase ==
           static_cast<std::uint32_t>(FrontendWorkPhaseV1::kIdle));
  BG_CHECK(g_frontend_work.flags == kFrontendWorkHealthyFlagV1);
  BG_CHECK(browsergrad_cpp_cute_constexpr_step_hook());

  constexpr FrontendWorkLimitsV1 frontend_limits{
      4U, 8U, 16U, 16U, 8U, 8U, 4U,
  };
  BG_CHECK(begin_frontend_work_invocation(frontend_limits));
  BG_CHECK(g_frontend_work.generation == 1U);
  BG_CHECK(begin_frontend_work_semantic_pass());
  BG_CHECK(record_frontend_include_depth(2U));
  BG_CHECK(record_frontend_include_depth(1U));
  BG_CHECK(record_frontend_macro_expansion());
  BG_CHECK(record_frontend_preprocessed_token());
  BG_CHECK(record_frontend_preprocessed_token());
  BG_CHECK(record_frontend_ast_node());
  BG_CHECK(browsergrad_cpp_cute_constexpr_step_hook());
  BG_CHECK(begin_frontend_template_instantiation());
  BG_CHECK(begin_frontend_template_instantiation());
  BG_CHECK(end_frontend_template_instantiation());
  BG_CHECK(end_frontend_template_instantiation());
  BG_CHECK(complete_frontend_work_semantic_pass());
  BG_CHECK(begin_frontend_work_semantic_pass());
  BG_CHECK(record_frontend_include_depth(3U));
  BG_CHECK(record_frontend_macro_expansion());
  BG_CHECK(record_frontend_preprocessed_token());
  BG_CHECK(record_frontend_ast_node());
  BG_CHECK(browsergrad_cpp_cute_constexpr_step_hook());
  BG_CHECK(complete_frontend_work_semantic_pass());
  BG_CHECK(complete_frontend_work_invocation(2U));
  BG_CHECK(frontend_work_metrics_ready());
  BG_CHECK(g_frontend_work.include_depth == 3U);
  BG_CHECK(g_frontend_work.macro_expansions == 2U);
  BG_CHECK(g_frontend_work.preprocessed_tokens == 3U);
  BG_CHECK(g_frontend_work.ast_nodes == 2U);
  BG_CHECK(g_frontend_work.constexpr_steps == 2U);
  BG_CHECK(g_frontend_work.template_instantiations == 2U);
  BG_CHECK(g_frontend_work.template_depth == 2U);
  BG_CHECK(g_frontend_work.completed_semantic_passes == 2U);
  reset_frontend_work_metrics();
  BG_CHECK(g_frontend_work.generation == 1U);
  BG_CHECK(g_frontend_work.phase ==
           static_cast<std::uint32_t>(FrontendWorkPhaseV1::kIdle));
  BG_CHECK(!frontend_work_metrics_ready());

  constexpr FrontendWorkLimitsV1 single_token_limit{
      1U, 1U, 1U, 1U, 1U, 1U, 1U,
  };
  BG_CHECK(begin_frontend_work_invocation(single_token_limit));
  BG_CHECK(begin_frontend_work_semantic_pass());
  BG_CHECK(record_frontend_preprocessed_token());
  BG_CHECK(!record_frontend_preprocessed_token());
  BG_CHECK(g_frontend_work.phase ==
           static_cast<std::uint32_t>(FrontendWorkPhaseV1::kFailed));
  BG_CHECK(g_frontend_work.flags == 0U);
  BG_CHECK(!frontend_work_metrics_ready());
  reset_frontend_work_metrics();
  BG_CHECK(g_frontend_work.generation == 2U);

  BG_CHECK(begin_frontend_work_invocation(single_token_limit));
  BG_CHECK(begin_frontend_work_semantic_pass());
  BG_CHECK(begin_frontend_template_instantiation());
  BG_CHECK(!begin_frontend_template_instantiation());
  BG_CHECK(!frontend_work_metrics_ready());
  reset_frontend_work_metrics();

  void* pointer = metrics_malloc(32U);
  BG_CHECK(pointer != nullptr);
  BG_CHECK(g_metrics.current_live_global_requested_byte_length == 32U);
  BG_CHECK(g_metrics.peak_live_global_requested_byte_length == 32U);
  BG_CHECK(g_metrics.cumulative_global_allocated_requested_byte_length == 32U);
  BG_CHECK(g_metrics.successful_allocation_count == 1U);
  BG_CHECK(g_allocations.size == 1U);

  pointer = metrics_realloc(pointer, 48U);
  BG_CHECK(pointer != nullptr);
  BG_CHECK(g_metrics.current_live_global_requested_byte_length == 48U);
  BG_CHECK(g_metrics.peak_live_global_requested_byte_length == 48U);
  BG_CHECK(g_metrics.cumulative_global_allocated_requested_byte_length == 80U);
  BG_CHECK(g_metrics.cumulative_global_freed_requested_byte_length == 32U);
  BG_CHECK(g_metrics.successful_allocation_count == 2U);
  BG_CHECK(g_metrics.free_count == 1U);

  g_test_builtin_allocator.realloc_function = fail_realloc;
  BG_CHECK(metrics_realloc(pointer, 64U) == nullptr);
  BG_CHECK(g_metrics.current_live_global_requested_byte_length == 48U);
  BG_CHECK(g_metrics.successful_allocation_count == 2U);
  BG_CHECK(g_metrics.free_count == 1U);
  BG_CHECK(g_metrics.failed_allocation_count == 1U);

  g_test_builtin_allocator.realloc_function = same_pointer_realloc;
  BG_CHECK(metrics_realloc(pointer, 24U) == pointer);
  BG_CHECK(g_metrics.current_live_global_requested_byte_length == 24U);
  BG_CHECK(g_metrics.cumulative_global_allocated_requested_byte_length == 104U);
  BG_CHECK(g_metrics.cumulative_global_freed_requested_byte_length == 80U);
  BG_CHECK(g_metrics.successful_allocation_count == 3U);
  BG_CHECK(g_metrics.free_count == 2U);
  g_test_builtin_allocator.realloc_function = std::realloc;
  BG_CHECK(metrics_free_impl(pointer));
  BG_CHECK(g_metrics.current_live_global_requested_byte_length == 0U);
  BG_CHECK(g_metrics.cumulative_global_freed_requested_byte_length == 104U);
  BG_CHECK(g_metrics.free_count == 3U);
  BG_CHECK(record_invariants_hold());

  pointer = metrics_malloc(8U);
  BG_CHECK(pointer != nullptr);
  g_test_builtin_allocator.realloc_function = moved_realloc;
  void* moved = metrics_realloc(pointer, 16U);
  BG_CHECK(moved != nullptr);
  BG_CHECK(moved != pointer);
  g_test_builtin_allocator.realloc_function = std::realloc;
  BG_CHECK(metrics_realloc(moved, 0U) == nullptr);
  BG_CHECK(g_allocations.size == 0U);
  BG_CHECK(record_invariants_hold());

  const std::uint64_t successful_before_zero =
      g_metrics.successful_allocation_count;
  void* zero = metrics_malloc(0U);
  if (zero == nullptr) {
    BG_CHECK(g_metrics.successful_allocation_count == successful_before_zero);
  } else {
    BG_CHECK(g_metrics.successful_allocation_count ==
             successful_before_zero + 1U);
    BG_CHECK(metrics_free_impl(zero));
  }
  const std::uint64_t failed_before_overflow =
      g_metrics.failed_allocation_count;
  BG_CHECK(metrics_calloc(std::numeric_limits<std::size_t>::max(), 2U) ==
           nullptr);
  BG_CHECK(g_metrics.failed_allocation_count == failed_before_overflow + 1U);

  void* aligned = metrics_memalign(64U, 128U, 128U);
  BG_CHECK(aligned != nullptr);
  BG_CHECK(reinterpret_cast<std::uintptr_t>(aligned) % 64U == 0U);
  BG_CHECK(metrics_free_impl(aligned));
  const std::uint64_t failed_before_alignment =
      g_metrics.failed_allocation_count;
  BG_CHECK(metrics_memalign(3U, 16U, 16U) == nullptr);
  BG_CHECK(g_metrics.failed_allocation_count == failed_before_alignment + 1U);
  BG_CHECK(record_invariants_hold());

  reset_state();
  std::array<void*, 128U> allocations{};
  std::uint64_t expected_live = 0U;
  for (std::size_t index = 0U; index < allocations.size(); ++index) {
    const std::size_t byte_length = index + 1U;
    allocations[index] = metrics_malloc(byte_length);
    BG_CHECK(allocations[index] != nullptr);
    expected_live += byte_length;
  }
  BG_CHECK(g_allocations.size == allocations.size());
  BG_CHECK(g_allocations.capacity >= 256U);
  BG_CHECK(g_metrics.current_live_global_requested_byte_length == expected_live);
  for (std::size_t index = 0U; index < allocations.size(); index += 2U) {
    BG_CHECK(metrics_free_impl(allocations[index]));
  }
  for (std::size_t index = 1U; index < allocations.size(); index += 2U) {
    BG_CHECK(metrics_free_impl(allocations[index]));
  }
  BG_CHECK(g_allocations.size == 0U);
  BG_CHECK(g_metrics.current_live_global_requested_byte_length == 0U);
  BG_CHECK(g_metrics.cumulative_global_allocated_requested_byte_length ==
           g_metrics.cumulative_global_freed_requested_byte_length);
  BG_CHECK(g_metrics.successful_allocation_count == allocations.size());
  BG_CHECK(g_metrics.free_count == allocations.size());
  BG_CHECK(record_invariants_hold());

  reset_state();
  g_test_builtin_allocator.calloc_function = fail_calloc;
  BG_CHECK(metrics_malloc(9U) == nullptr);
  BG_CHECK(g_allocations.capacity == 0U);
  BG_CHECK(g_metrics.successful_allocation_count == 0U);
  BG_CHECK(g_metrics.failed_allocation_count == 1U);
  BG_CHECK(allocator_metrics_healthy());

  reset_state();
  pointer = metrics_malloc(1U);
  BG_CHECK(pointer != nullptr);
  BG_CHECK(metrics_free_impl(pointer));
  g_test_builtin_allocator.malloc_function = fail_malloc;
  BG_CHECK(metrics_malloc(9U) == nullptr);
  BG_CHECK(g_metrics.current_live_global_requested_byte_length == 0U);
  BG_CHECK(g_metrics.successful_allocation_count == 1U);
  BG_CHECK(g_metrics.free_count == 1U);
  BG_CHECK(g_metrics.failed_allocation_count == 1U);

  reset_state();
  g_metrics.successful_allocation_count =
      std::numeric_limits<std::uint64_t>::max();
  BG_CHECK(metrics_malloc(1U) == nullptr);
  BG_CHECK(!allocator_metrics_healthy());
  BG_CHECK(allocator_metrics_failure_reason() ==
           AllocatorMetricsFailureReason::kCreationCounterOverflow);
  BG_CHECK(g_metrics.successful_allocation_count ==
           std::numeric_limits<std::uint64_t>::max());

  reset_state();
  int untracked = 0;
  BG_CHECK(!metrics_free_impl(&untracked));
  BG_CHECK(!allocator_metrics_healthy());
  BG_CHECK(allocator_metrics_failure_reason() ==
           AllocatorMetricsFailureReason::kUntrackedFree);
  g_metrics.successful_allocation_count =
      std::numeric_limits<std::uint64_t>::max();
  BG_CHECK(metrics_malloc(1U) == nullptr);
  BG_CHECK(allocator_metrics_failure_reason() ==
           AllocatorMetricsFailureReason::kUntrackedFree);
  reset_state();
  return 0;
}

}  // namespace

int main() { return run_metrics_tests(); }
