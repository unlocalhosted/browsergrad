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

}  // namespace browsergrad::cpp_cute
