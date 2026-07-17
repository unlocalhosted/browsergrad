#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace browsergrad::cpp_cute {

using Sha256Digest = std::array<std::uint8_t, 32U>;
using Sha256LowercaseHex = std::array<char, 65U>;

/**
 * Allocation-free incremental SHA-256 state.
 *
 * The object is deliberately noncopyable and nonmovable: one instance owns one
 * linear hash computation and may be finalized exactly once. Any invalid call
 * poisons the instance permanently, returns false, and zeroes a requested
 * digest output so callers cannot mistake stale bytes for verified identity.
 */
class Sha256 final {
 public:
  Sha256() noexcept;

  Sha256(const Sha256&) = delete;
  Sha256& operator=(const Sha256&) = delete;
  Sha256(Sha256&&) = delete;
  Sha256& operator=(Sha256&&) = delete;

  /** Accepts another byte range. A null pointer is valid only for zero bytes. */
  bool update(const std::uint8_t* bytes, std::size_t byte_length) noexcept;

  /** Finalizes once. On failure, digest is zeroed and the instance is poisoned. */
  bool finalize(Sha256Digest& digest) noexcept;

  /** False after any misuse or length overflow. */
  bool healthy() const noexcept;

  /** True only after one successful finalization and before any later misuse. */
  bool finalized() const noexcept;

 private:
  enum class State : std::uint8_t {
    kActive,
    kFinalized,
    kFailed,
  };

  void fail() noexcept;
  void transform(const std::uint8_t* block) noexcept;

  std::array<std::uint32_t, 8U> state_{};
  std::array<std::uint8_t, 64U> buffer_{};
  std::uint64_t total_byte_length_ = 0U;
  std::size_t buffered_byte_length_ = 0U;
  State lifecycle_state_ = State::kActive;
};

/** Returns 64 lowercase hexadecimal characters followed by one NUL byte. */
Sha256LowercaseHex sha256_lowercase_hex(
    const Sha256Digest& digest) noexcept;

}  // namespace browsergrad::cpp_cute
