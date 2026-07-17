#include "BrowserGradCppCuteMetrics.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

#if defined(__EMSCRIPTEN__)
#include <emscripten/heap.h>
#endif

#if defined(__EMSCRIPTEN__) && defined(BG_CPP_CUTE_METRICS_TESTING)
#error "allocator test backend must never be enabled in the Wasm producer"
#endif

namespace browsergrad::cpp_cute {
namespace {

constexpr std::size_t kInitialAllocationTableCapacity = 64U;
[[maybe_unused]] constexpr std::size_t kWasmPageByteLength = 64U * 1024U;

constinit AllocatorMetricsRecordV1 g_metrics = {
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
constinit bool g_metrics_healthy = true;
constinit bool g_allocator_hook_active = false;

struct AllocationEntry {
  void* pointer = nullptr;
  std::size_t requested_byte_length = 0U;
};

struct AllocationTable {
  AllocationEntry* entries = nullptr;
  std::size_t capacity = 0U;
  std::size_t size = 0U;
};

constinit AllocationTable g_allocations;

void poison_metrics() { g_metrics_healthy = false; }

#if defined(BG_CPP_CUTE_METRICS_TESTING)
struct TestBuiltinAllocator {
  void* (*malloc_function)(std::size_t);
  void* (*calloc_function)(std::size_t, std::size_t);
  void* (*realloc_function)(void*, std::size_t);
  void (*free_function)(void*);
  void* (*memalign_function)(std::size_t, std::size_t);
};

void* test_memalign(std::size_t alignment, std::size_t byte_length) {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, byte_length) != 0) return nullptr;
  return pointer;
}

constinit TestBuiltinAllocator g_test_builtin_allocator = {
    std::malloc,
    std::calloc,
    std::realloc,
    std::free,
    test_memalign,
};
#endif

void* builtin_malloc(std::size_t byte_length) {
#if defined(BG_CPP_CUTE_METRICS_TESTING)
  return g_test_builtin_allocator.malloc_function(byte_length);
#elif defined(__EMSCRIPTEN__)
  return emscripten_builtin_malloc(byte_length);
#else
  return std::malloc(byte_length);
#endif
}

void* builtin_calloc(std::size_t count, std::size_t byte_length) {
#if defined(BG_CPP_CUTE_METRICS_TESTING)
  return g_test_builtin_allocator.calloc_function(count, byte_length);
#elif defined(__EMSCRIPTEN__)
  return emscripten_builtin_calloc(count, byte_length);
#else
  return std::calloc(count, byte_length);
#endif
}

void* builtin_realloc(void* pointer, std::size_t byte_length) {
#if defined(BG_CPP_CUTE_METRICS_TESTING)
  return g_test_builtin_allocator.realloc_function(pointer, byte_length);
#elif defined(__EMSCRIPTEN__)
  return emscripten_builtin_realloc(pointer, byte_length);
#else
  return std::realloc(pointer, byte_length);
#endif
}

void builtin_free(void* pointer) {
#if defined(BG_CPP_CUTE_METRICS_TESTING)
  g_test_builtin_allocator.free_function(pointer);
#elif defined(__EMSCRIPTEN__)
  emscripten_builtin_free(pointer);
#else
  std::free(pointer);
#endif
}

void* builtin_memalign(std::size_t alignment, std::size_t byte_length) {
#if defined(BG_CPP_CUTE_METRICS_TESTING)
  return g_test_builtin_allocator.memalign_function(alignment, byte_length);
#elif defined(__EMSCRIPTEN__)
  return emscripten_builtin_memalign(alignment, byte_length);
#else
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, byte_length) != 0) return nullptr;
  return pointer;
#endif
}

bool checked_add(std::uint64_t left,
                 std::uint64_t right,
                 std::uint64_t* result) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
  *result = left + right;
  return true;
}

bool checked_multiply(std::size_t left,
                      std::size_t right,
                      std::size_t* result) {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

[[maybe_unused]] bool checked_align_up(std::size_t value,
                                      std::size_t alignment,
                                      std::size_t* result) {
  if (alignment == 0U || value >
      std::numeric_limits<std::size_t>::max() - (alignment - 1U)) {
    return false;
  }
  *result = (value + alignment - 1U) & ~(alignment - 1U);
  return true;
}

bool is_power_of_two(std::size_t value) {
  return value != 0U && (value & (value - 1U)) == 0U;
}

bool valid_posix_alignment(std::size_t alignment) {
  return is_power_of_two(alignment) && alignment % sizeof(void*) == 0U;
}

std::size_t hash_pointer(void* pointer) {
  std::uintptr_t value = reinterpret_cast<std::uintptr_t>(pointer) >> 3U;
  value ^= value >> 16U;
  value *= static_cast<std::uintptr_t>(0x7feb352dU);
  value ^= value >> 15U;
  value *= static_cast<std::uintptr_t>(0x846ca68bU);
  value ^= value >> 16U;
  return static_cast<std::size_t>(value);
}

bool table_shape_valid() {
  return (g_allocations.capacity == 0U && g_allocations.entries == nullptr &&
          g_allocations.size == 0U) ||
         (g_allocations.entries != nullptr &&
          is_power_of_two(g_allocations.capacity) &&
          g_allocations.size < g_allocations.capacity);
}

AllocationEntry* find_entry(void* pointer) {
  if (pointer == nullptr || !table_shape_valid()) {
    if (pointer != nullptr) poison_metrics();
    return nullptr;
  }
  if (g_allocations.capacity == 0U) return nullptr;
  const std::size_t mask = g_allocations.capacity - 1U;
  std::size_t index = hash_pointer(pointer) & mask;
  for (std::size_t probe = 0U; probe < g_allocations.capacity; ++probe) {
    AllocationEntry& entry = g_allocations.entries[index];
    if (entry.pointer == nullptr) return nullptr;
    if (entry.pointer == pointer) return &entry;
    index = (index + 1U) & mask;
  }
  poison_metrics();
  return nullptr;
}

bool insert_without_growth(AllocationEntry* entries,
                           std::size_t capacity,
                           void* pointer,
                           std::size_t requested_byte_length) {
  const std::size_t mask = capacity - 1U;
  std::size_t index = hash_pointer(pointer) & mask;
  for (std::size_t probe = 0U; probe < capacity; ++probe) {
    AllocationEntry& entry = entries[index];
    if (entry.pointer == nullptr) {
      entry.pointer = pointer;
      entry.requested_byte_length = requested_byte_length;
      return true;
    }
    if (entry.pointer == pointer) return false;
    index = (index + 1U) & mask;
  }
  return false;
}

bool resize_allocation_table(std::size_t capacity) {
  if (!is_power_of_two(capacity) ||
      capacity > std::numeric_limits<std::size_t>::max() /
                     sizeof(AllocationEntry)) {
    return false;
  }
  auto* replacement = static_cast<AllocationEntry*>(
      builtin_calloc(capacity, sizeof(AllocationEntry)));
  if (replacement == nullptr) return false;
  for (std::size_t index = 0U; index < g_allocations.capacity; ++index) {
    const AllocationEntry entry = g_allocations.entries[index];
    if (entry.pointer != nullptr &&
        !insert_without_growth(replacement, capacity, entry.pointer,
                               entry.requested_byte_length)) {
      builtin_free(replacement);
      poison_metrics();
      return false;
    }
  }
  builtin_free(g_allocations.entries);
  g_allocations.entries = replacement;
  g_allocations.capacity = capacity;
  return true;
}

bool ensure_allocation_table_capacity() {
  if (!table_shape_valid()) {
    poison_metrics();
    return false;
  }
  const std::size_t required = g_allocations.size + 1U;
  if (required < g_allocations.size) return false;
  if (g_allocations.capacity != 0U &&
      required <= g_allocations.capacity - g_allocations.capacity / 4U) {
    return true;
  }
  if (g_allocations.capacity >
      std::numeric_limits<std::size_t>::max() / 2U) {
    return false;
  }
  const std::size_t replacement_capacity =
      g_allocations.capacity == 0U
          ? kInitialAllocationTableCapacity
          : g_allocations.capacity * 2U;
  return resize_allocation_table(replacement_capacity);
}

bool insert_allocation(void* pointer, std::size_t requested_byte_length) {
  if (pointer == nullptr || g_allocations.capacity == 0U ||
      g_allocations.size >= g_allocations.capacity ||
      !insert_without_growth(g_allocations.entries, g_allocations.capacity,
                             pointer, requested_byte_length)) {
    poison_metrics();
    return false;
  }
  ++g_allocations.size;
  return true;
}

bool erase_allocation(void* pointer) {
  if (pointer == nullptr || g_allocations.capacity == 0U) {
    poison_metrics();
    return false;
  }
  const std::size_t mask = g_allocations.capacity - 1U;
  std::size_t index = hash_pointer(pointer) & mask;
  for (std::size_t probe = 0U; probe < g_allocations.capacity; ++probe) {
    AllocationEntry& entry = g_allocations.entries[index];
    if (entry.pointer == nullptr) {
      poison_metrics();
      return false;
    }
    if (entry.pointer == pointer) {
      entry = AllocationEntry{};
      --g_allocations.size;
      std::size_t cursor = (index + 1U) & mask;
      while (g_allocations.entries[cursor].pointer != nullptr) {
        const AllocationEntry displaced = g_allocations.entries[cursor];
        g_allocations.entries[cursor] = AllocationEntry{};
        --g_allocations.size;
        if (!insert_allocation(displaced.pointer,
                               displaced.requested_byte_length)) {
          return false;
        }
        cursor = (cursor + 1U) & mask;
      }
      return true;
    }
    index = (index + 1U) & mask;
  }
  poison_metrics();
  return false;
}

bool preflight_creation(std::size_t requested_byte_length) {
  if (!g_metrics_healthy) return false;
  const std::uint64_t requested = requested_byte_length;
  std::uint64_t ignored = 0U;
  if (!checked_add(g_metrics.current_live_global_requested_byte_length,
                   requested, &ignored) ||
      !checked_add(
          g_metrics.cumulative_global_allocated_requested_byte_length,
          requested, &ignored) ||
      !checked_add(g_metrics.successful_allocation_count, 1U, &ignored)) {
    poison_metrics();
    return false;
  }
  return true;
}

void commit_creation(std::size_t requested_byte_length) {
  const std::uint64_t requested = requested_byte_length;
  g_metrics.current_live_global_requested_byte_length += requested;
  g_metrics.cumulative_global_allocated_requested_byte_length += requested;
  ++g_metrics.successful_allocation_count;
  if (g_metrics.current_live_global_requested_byte_length >
      g_metrics.peak_live_global_requested_byte_length) {
    g_metrics.peak_live_global_requested_byte_length =
        g_metrics.current_live_global_requested_byte_length;
  }
}

bool preflight_release(std::size_t requested_byte_length) {
  const std::uint64_t requested = requested_byte_length;
  std::uint64_t cumulative_freed = 0U;
  if (!g_metrics_healthy ||
      g_metrics.current_live_global_requested_byte_length < requested ||
      !checked_add(g_metrics.cumulative_global_freed_requested_byte_length,
                   requested, &cumulative_freed) ||
      cumulative_freed >
          g_metrics.cumulative_global_allocated_requested_byte_length ||
      g_metrics.free_count == std::numeric_limits<std::uint64_t>::max()) {
    poison_metrics();
    return false;
  }
  return true;
}

void commit_release(std::size_t requested_byte_length) {
  const std::uint64_t requested = requested_byte_length;
  g_metrics.current_live_global_requested_byte_length -= requested;
  g_metrics.cumulative_global_freed_requested_byte_length += requested;
  ++g_metrics.free_count;
}

bool preflight_reallocation(std::size_t old_requested_byte_length,
                            std::size_t new_requested_byte_length) {
  const std::uint64_t old_requested = old_requested_byte_length;
  const std::uint64_t new_requested = new_requested_byte_length;
  std::uint64_t ignored = 0U;
  if (!g_metrics_healthy ||
      g_metrics.current_live_global_requested_byte_length < old_requested ||
      !checked_add(
          g_metrics.cumulative_global_allocated_requested_byte_length,
          new_requested, &ignored) ||
      !checked_add(g_metrics.cumulative_global_freed_requested_byte_length,
                   old_requested, &ignored) ||
      !checked_add(g_metrics.successful_allocation_count, 1U, &ignored) ||
      !checked_add(g_metrics.free_count, 1U, &ignored)) {
    poison_metrics();
    return false;
  }
  const std::uint64_t without_old =
      g_metrics.current_live_global_requested_byte_length - old_requested;
  if (!checked_add(without_old, new_requested, &ignored)) {
    poison_metrics();
    return false;
  }
  return true;
}

void commit_reallocation(std::size_t old_requested_byte_length,
                         std::size_t new_requested_byte_length) {
  const std::uint64_t old_requested = old_requested_byte_length;
  const std::uint64_t new_requested = new_requested_byte_length;
  g_metrics.current_live_global_requested_byte_length =
      g_metrics.current_live_global_requested_byte_length - old_requested +
      new_requested;
  g_metrics.cumulative_global_allocated_requested_byte_length += new_requested;
  g_metrics.cumulative_global_freed_requested_byte_length += old_requested;
  ++g_metrics.successful_allocation_count;
  ++g_metrics.free_count;
  if (g_metrics.current_live_global_requested_byte_length >
      g_metrics.peak_live_global_requested_byte_length) {
    g_metrics.peak_live_global_requested_byte_length =
        g_metrics.current_live_global_requested_byte_length;
  }
}

void record_failed_request(std::size_t requested_byte_length) {
  if (requested_byte_length == 0U || !g_metrics_healthy) return;
  if (g_metrics.failed_allocation_count ==
      std::numeric_limits<std::uint64_t>::max()) {
    poison_metrics();
    return;
  }
  ++g_metrics.failed_allocation_count;
}

class AllocatorHookGuard final {
 public:
  AllocatorHookGuard() {
    if (!g_metrics_healthy || g_allocator_hook_active) {
      poison_metrics();
      return;
    }
    g_allocator_hook_active = true;
    active_ = true;
  }

  ~AllocatorHookGuard() {
    if (active_) g_allocator_hook_active = false;
  }

  bool active() const { return active_; }

 private:
  bool active_ = false;
};

[[maybe_unused]] void reject_failed_request(
    std::size_t requested_byte_length, int error) {
  AllocatorHookGuard guard;
  if (guard.active()) record_failed_request(requested_byte_length);
  errno = error;
}

enum class BuiltinAllocationKind {
  kMalloc,
  kCalloc,
  kMemalign,
};

void* tracked_creation(std::size_t requested_byte_length,
                       std::size_t builtin_byte_length,
                       std::size_t alignment,
                       BuiltinAllocationKind kind) {
  AllocatorHookGuard guard;
  if (!guard.active()) {
    errno = ENOMEM;
    return nullptr;
  }
  if (!preflight_creation(requested_byte_length) ||
      !ensure_allocation_table_capacity()) {
    if (g_metrics_healthy) record_failed_request(requested_byte_length);
    errno = ENOMEM;
    return nullptr;
  }

  void* pointer = nullptr;
  switch (kind) {
    case BuiltinAllocationKind::kMalloc:
      pointer = builtin_malloc(builtin_byte_length);
      break;
    case BuiltinAllocationKind::kCalloc:
      pointer = builtin_calloc(1U, builtin_byte_length);
      break;
    case BuiltinAllocationKind::kMemalign:
      pointer = builtin_memalign(alignment, builtin_byte_length);
      break;
  }
  if (pointer == nullptr) {
    record_failed_request(requested_byte_length);
    return nullptr;
  }
  if (find_entry(pointer) != nullptr || !g_metrics_healthy) {
    poison_metrics();
    errno = ENOMEM;
    return nullptr;
  }
  if (!insert_allocation(pointer, requested_byte_length)) {
    builtin_free(pointer);
    errno = ENOMEM;
    return nullptr;
  }
  commit_creation(requested_byte_length);
  return pointer;
}

void* metrics_malloc(std::size_t byte_length) {
  return tracked_creation(byte_length, byte_length, 0U,
                          BuiltinAllocationKind::kMalloc);
}

[[maybe_unused]] void* metrics_calloc(std::size_t count,
                                      std::size_t byte_length) {
  std::size_t requested_byte_length = 0U;
  if (!checked_multiply(count, byte_length, &requested_byte_length)) {
    reject_failed_request(1U, ENOMEM);
    return nullptr;
  }
  return tracked_creation(requested_byte_length, requested_byte_length, 0U,
                          BuiltinAllocationKind::kCalloc);
}

bool metrics_free_impl(void* pointer) {
  if (pointer == nullptr) return true;
  AllocatorHookGuard guard;
  if (!guard.active()) return false;
  AllocationEntry* entry = find_entry(pointer);
  if (entry == nullptr || !g_metrics_healthy) {
    poison_metrics();
    return false;
  }
  const std::size_t requested_byte_length = entry->requested_byte_length;
  if (!preflight_release(requested_byte_length)) return false;
  builtin_free(pointer);
  if (!erase_allocation(pointer)) return false;
  commit_release(requested_byte_length);
  return true;
}

[[maybe_unused]] void metrics_free(void* pointer) {
  static_cast<void>(metrics_free_impl(pointer));
}

[[maybe_unused]] void* metrics_realloc(void* pointer,
                                       std::size_t byte_length) {
  if (pointer == nullptr) return metrics_malloc(byte_length);
  AllocatorHookGuard guard;
  if (!guard.active()) {
    errno = ENOMEM;
    return nullptr;
  }
  AllocationEntry* entry = find_entry(pointer);
  if (entry == nullptr || !g_metrics_healthy) {
    poison_metrics();
    errno = EINVAL;
    return nullptr;
  }
  const std::size_t old_requested_byte_length = entry->requested_byte_length;
  if (byte_length == 0U) {
    if (!preflight_release(old_requested_byte_length)) return nullptr;
    builtin_free(pointer);
    if (!erase_allocation(pointer)) return nullptr;
    commit_release(old_requested_byte_length);
    return nullptr;
  }
  if (!preflight_reallocation(old_requested_byte_length, byte_length)) {
    errno = ENOMEM;
    return nullptr;
  }
  void* replacement = builtin_realloc(pointer, byte_length);
  if (replacement == nullptr) {
    record_failed_request(byte_length);
    return nullptr;
  }
  if (replacement == pointer) {
    entry->requested_byte_length = byte_length;
  } else {
    if (find_entry(replacement) != nullptr || !g_metrics_healthy) {
      poison_metrics();
      errno = ENOMEM;
      return nullptr;
    }
    if (!erase_allocation(pointer) ||
        !insert_allocation(replacement, byte_length)) {
      poison_metrics();
      errno = ENOMEM;
      return nullptr;
    }
  }
  commit_reallocation(old_requested_byte_length, byte_length);
  return replacement;
}

[[maybe_unused]] void* metrics_memalign(
    std::size_t alignment,
    std::size_t builtin_byte_length,
    std::size_t requested_byte_length) {
  if (!valid_posix_alignment(alignment)) {
    AllocatorHookGuard guard;
    if (guard.active()) record_failed_request(requested_byte_length);
    errno = EINVAL;
    return nullptr;
  }
  return tracked_creation(requested_byte_length, builtin_byte_length, alignment,
                          BuiltinAllocationKind::kMemalign);
}

}  // namespace

std::uint32_t allocator_metrics_pointer() {
  const std::uintptr_t pointer = reinterpret_cast<std::uintptr_t>(&g_metrics);
  constexpr std::uintptr_t kMaximumRecordStart =
      std::numeric_limits<std::uint32_t>::max() -
      (sizeof(AllocatorMetricsRecordV1) - 1U);
  if (pointer == 0U ||
      pointer > kMaximumRecordStart ||
      pointer % alignof(AllocatorMetricsRecordV1) != 0U) {
    poison_metrics();
    return 0U;
  }
  return static_cast<std::uint32_t>(pointer);
}

bool allocator_metrics_healthy() { return g_metrics_healthy; }

}  // namespace browsergrad::cpp_cute

#if defined(__EMSCRIPTEN__)

#define BG_CPP_CUTE_ALLOCATOR_OVERRIDE \
  __attribute__((used)) __attribute__((noinline))

extern "C" {

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void* malloc(std::size_t byte_length) {
  return browsergrad::cpp_cute::metrics_malloc(byte_length);
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void* __libc_malloc(
    std::size_t byte_length) {
  return browsergrad::cpp_cute::metrics_malloc(byte_length);
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void* calloc(
    std::size_t count, std::size_t byte_length) {
  return browsergrad::cpp_cute::metrics_calloc(count, byte_length);
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void* __libc_calloc(
    std::size_t count, std::size_t byte_length) {
  return browsergrad::cpp_cute::metrics_calloc(count, byte_length);
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void free(void* pointer) {
  browsergrad::cpp_cute::metrics_free(pointer);
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void __libc_free(void* pointer) {
  browsergrad::cpp_cute::metrics_free(pointer);
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void* realloc(
    void* pointer, std::size_t byte_length) {
  return browsergrad::cpp_cute::metrics_realloc(pointer, byte_length);
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void* __libc_realloc(
    void* pointer, std::size_t byte_length) {
  return browsergrad::cpp_cute::metrics_realloc(pointer, byte_length);
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void* reallocarray(
    void* pointer, std::size_t count, std::size_t byte_length) {
  std::size_t requested_byte_length = 0U;
  if (!browsergrad::cpp_cute::checked_multiply(
          count, byte_length, &requested_byte_length)) {
    browsergrad::cpp_cute::reject_failed_request(1U, ENOMEM);
    return nullptr;
  }
  return browsergrad::cpp_cute::metrics_realloc(pointer,
                                                 requested_byte_length);
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void* aligned_alloc(
    std::size_t alignment, std::size_t byte_length) {
  if (alignment == 0U || byte_length % alignment != 0U) {
    browsergrad::cpp_cute::reject_failed_request(byte_length, EINVAL);
    return nullptr;
  }
  return browsergrad::cpp_cute::metrics_memalign(
      alignment, byte_length, byte_length);
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void* memalign(
    std::size_t alignment, std::size_t byte_length) {
  return browsergrad::cpp_cute::metrics_memalign(
      alignment, byte_length, byte_length);
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE int posix_memalign(
    void** output, std::size_t alignment, std::size_t byte_length) {
  if (output == nullptr ||
      !browsergrad::cpp_cute::valid_posix_alignment(alignment)) {
    browsergrad::cpp_cute::reject_failed_request(byte_length, EINVAL);
    return EINVAL;
  }
  void* pointer = browsergrad::cpp_cute::metrics_memalign(
      alignment, byte_length, byte_length);
  if (pointer == nullptr) return errno == EINVAL ? EINVAL : ENOMEM;
  *output = pointer;
  return 0;
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void* valloc(std::size_t byte_length) {
  return browsergrad::cpp_cute::metrics_memalign(
      browsergrad::cpp_cute::kWasmPageByteLength, byte_length, byte_length);
}

BG_CPP_CUTE_ALLOCATOR_OVERRIDE void* pvalloc(std::size_t byte_length) {
  std::size_t rounded_byte_length = 0U;
  if (!browsergrad::cpp_cute::checked_align_up(
          byte_length == 0U ? 1U : byte_length,
          browsergrad::cpp_cute::kWasmPageByteLength,
          &rounded_byte_length)) {
    browsergrad::cpp_cute::reject_failed_request(byte_length, ENOMEM);
    return nullptr;
  }
  return browsergrad::cpp_cute::metrics_memalign(
      browsergrad::cpp_cute::kWasmPageByteLength,
      rounded_byte_length,
      byte_length);
}

}  // extern "C"

#undef BG_CPP_CUTE_ALLOCATOR_OVERRIDE

#endif
