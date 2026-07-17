#include "BrowserGradCppCuteSha256.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace browsergrad::cpp_cute {
namespace {

constexpr std::array<std::uint32_t, 64U> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

constexpr std::uint32_t rotate_right(std::uint32_t value,
                                     std::uint32_t bits) noexcept {
  return (value >> bits) | (value << (32U - bits));
}

std::uint32_t load_big_endian_u32(const std::uint8_t* bytes) noexcept {
  return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
         (static_cast<std::uint32_t>(bytes[1]) << 16U) |
         (static_cast<std::uint32_t>(bytes[2]) << 8U) |
         static_cast<std::uint32_t>(bytes[3]);
}

}  // namespace

Sha256::Sha256() noexcept
    : state_{
          0x6a09e667U,
          0xbb67ae85U,
          0x3c6ef372U,
          0xa54ff53aU,
          0x510e527fU,
          0x9b05688cU,
          0x1f83d9abU,
          0x5be0cd19U,
      } {}

bool Sha256::update(const std::uint8_t* bytes,
                    std::size_t byte_length) noexcept {
  if (lifecycle_state_ != State::kActive) {
    fail();
    return false;
  }
  if (byte_length == 0U) return true;
  if (bytes == nullptr) {
    fail();
    return false;
  }

  constexpr std::uint64_t kMaximumMessageByteLength =
      std::numeric_limits<std::uint64_t>::max() / 8U;
  if (byte_length > kMaximumMessageByteLength - total_byte_length_) {
    fail();
    return false;
  }
  total_byte_length_ += static_cast<std::uint64_t>(byte_length);

  std::size_t offset = 0U;
  if (buffered_byte_length_ != 0U) {
    const std::size_t available = buffer_.size() - buffered_byte_length_;
    const std::size_t copied = byte_length < available ? byte_length : available;
    for (std::size_t index = 0U; index < copied; ++index) {
      buffer_[buffered_byte_length_ + index] = bytes[index];
    }
    buffered_byte_length_ += copied;
    offset += copied;
    if (buffered_byte_length_ == buffer_.size()) {
      transform(buffer_.data());
      buffered_byte_length_ = 0U;
    }
  }

  while (byte_length - offset >= buffer_.size()) {
    transform(bytes + offset);
    offset += buffer_.size();
  }
  while (offset < byte_length) {
    buffer_[buffered_byte_length_] = bytes[offset];
    ++buffered_byte_length_;
    ++offset;
  }
  return true;
}

bool Sha256::finalize(Sha256Digest& digest) noexcept {
  if (lifecycle_state_ != State::kActive) {
    digest.fill(0U);
    fail();
    return false;
  }

  const std::uint64_t bit_length = total_byte_length_ * 8U;
  buffer_[buffered_byte_length_] = 0x80U;
  ++buffered_byte_length_;

  if (buffered_byte_length_ > 56U) {
    while (buffered_byte_length_ < buffer_.size()) {
      buffer_[buffered_byte_length_] = 0U;
      ++buffered_byte_length_;
    }
    transform(buffer_.data());
    buffered_byte_length_ = 0U;
  }
  while (buffered_byte_length_ < 56U) {
    buffer_[buffered_byte_length_] = 0U;
    ++buffered_byte_length_;
  }
  for (std::size_t index = 0U; index < 8U; ++index) {
    buffer_[63U - index] =
        static_cast<std::uint8_t>(bit_length >> (index * 8U));
  }
  transform(buffer_.data());

  for (std::size_t word = 0U; word < state_.size(); ++word) {
    digest[word * 4U] = static_cast<std::uint8_t>(state_[word] >> 24U);
    digest[word * 4U + 1U] = static_cast<std::uint8_t>(state_[word] >> 16U);
    digest[word * 4U + 2U] = static_cast<std::uint8_t>(state_[word] >> 8U);
    digest[word * 4U + 3U] = static_cast<std::uint8_t>(state_[word]);
  }
  buffer_.fill(0U);
  buffered_byte_length_ = 0U;
  lifecycle_state_ = State::kFinalized;
  return true;
}

bool Sha256::healthy() const noexcept {
  return lifecycle_state_ != State::kFailed;
}

bool Sha256::finalized() const noexcept {
  return lifecycle_state_ == State::kFinalized;
}

void Sha256::fail() noexcept { lifecycle_state_ = State::kFailed; }

void Sha256::transform(const std::uint8_t* block) noexcept {
  std::array<std::uint32_t, 64U> words{};
  for (std::size_t index = 0U; index < 16U; ++index) {
    words[index] = load_big_endian_u32(block + index * 4U);
  }
  for (std::size_t index = 16U; index < words.size(); ++index) {
    const std::uint32_t previous15 = words[index - 15U];
    const std::uint32_t previous2 = words[index - 2U];
    const std::uint32_t sigma0 = rotate_right(previous15, 7U) ^
                                 rotate_right(previous15, 18U) ^
                                 (previous15 >> 3U);
    const std::uint32_t sigma1 = rotate_right(previous2, 17U) ^
                                 rotate_right(previous2, 19U) ^
                                 (previous2 >> 10U);
    words[index] = words[index - 16U] + sigma0 + words[index - 7U] + sigma1;
  }

  std::uint32_t a = state_[0];
  std::uint32_t b = state_[1];
  std::uint32_t c = state_[2];
  std::uint32_t d = state_[3];
  std::uint32_t e = state_[4];
  std::uint32_t f = state_[5];
  std::uint32_t g = state_[6];
  std::uint32_t h = state_[7];

  for (std::size_t index = 0U; index < words.size(); ++index) {
    const std::uint32_t sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                               rotate_right(e, 25U);
    const std::uint32_t choice = (e & f) ^ (~e & g);
    const std::uint32_t temporary1 =
        h + sum1 + choice + kRoundConstants[index] + words[index];
    const std::uint32_t sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                               rotate_right(a, 22U);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temporary2 = sum0 + majority;

    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }

  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

Sha256LowercaseHex sha256_lowercase_hex(
    const Sha256Digest& digest) noexcept {
  constexpr std::array<char, 16U> kHexDigits = {
      '0', '1', '2', '3', '4', '5', '6', '7',
      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
  };
  Sha256LowercaseHex encoded{};
  for (std::size_t index = 0U; index < digest.size(); ++index) {
    encoded[index * 2U] = kHexDigits[digest[index] >> 4U];
    encoded[index * 2U + 1U] = kHexDigits[digest[index] & 0x0fU];
  }
  encoded[64U] = '\0';
  return encoded;
}

}  // namespace browsergrad::cpp_cute
