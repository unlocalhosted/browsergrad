#include "extractor/BrowserGradCppCuteSha256.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

namespace {

using namespace browsergrad::cpp_cute;

#define BG_CHECK(condition)                                                   \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::fprintf(stderr, "sha256 check failed at line %d: %s\n",         \
                   __LINE__, #condition);                                     \
      return 1;                                                               \
    }                                                                         \
  } while (false)

bool digest_equals(const Sha256Digest& digest, const char* expected) {
  const Sha256LowercaseHex encoded = sha256_lowercase_hex(digest);
  return encoded[64U] == '\0' && std::strcmp(encoded.data(), expected) == 0;
}

bool digest_bytes(const std::uint8_t* bytes, std::size_t byte_length,
                  Sha256Digest& digest) {
  Sha256 hash;
  return hash.update(bytes, byte_length) && hash.finalize(digest) &&
         hash.healthy() && hash.finalized();
}

struct FixedDigestVector {
  std::size_t byte_length;
  const char* expected;
};

int run_sha256_tests() {
  static_assert(!std::is_copy_constructible_v<Sha256>);
  static_assert(!std::is_copy_assignable_v<Sha256>);
  static_assert(!std::is_move_constructible_v<Sha256>);
  static_assert(!std::is_move_assignable_v<Sha256>);

  Sha256Digest digest{};
  BG_CHECK(digest_bytes(nullptr, 0U, digest));
  BG_CHECK(digest_equals(
      digest,
      "e3b0c44298fc1c149afbf4c8996fb924"
      "27ae41e4649b934ca495991b7852b855"));

  constexpr std::array<std::uint8_t, 3U> kAbc = {'a', 'b', 'c'};
  BG_CHECK(digest_bytes(kAbc.data(), kAbc.size(), digest));
  BG_CHECK(digest_equals(
      digest,
      "ba7816bf8f01cfea414140de5dae2223"
      "b00361a396177a9cb410ff61f20015ad"));

  constexpr char kLongVector[] =
      "abcdbcdecdefdefgefghfghighijhijk"
      "ijkljklmklmnlmnomnopnopq";
  BG_CHECK(digest_bytes(
      reinterpret_cast<const std::uint8_t*>(kLongVector),
      sizeof(kLongVector) - 1U, digest));
  BG_CHECK(digest_equals(
      digest,
      "248d6a61d20638b8e5c026930c3e6039"
      "a33ce45964ff2167f6ecedd419db06c1"));

  std::array<std::uint8_t, 65U> boundary_as{};
  boundary_as.fill(static_cast<std::uint8_t>('a'));
  constexpr std::array<FixedDigestVector, 5U> kBoundaryVectors = {{
      {55U,
       "9f4390f8d30c2dd92ec9f095b65e2b9a"
       "e9b0a925a5258e241c9f1e910f734318"},
      {57U,
       "f13b2d724659eb3bf47f2dd6af1accc8"
       "7b81f09f59f2b75e5c0bed6589dfe8c6"},
      {63U,
       "7d3e74a05d7db15bce4ad9ec0658ea98"
       "e3f06eeecf16b4c6fff2da457ddc2f34"},
      {64U,
       "ffe054fe7ae0cb6dc65c3af9b61d5209"
       "f439851db43d0ba5997337df154668eb"},
      {65U,
       "635361c48bb9eab14198e76ea8ab7f1a"
       "41685d6ad62aa9146d301d4f17eb0ae0"},
  }};
  for (const FixedDigestVector& vector : kBoundaryVectors) {
    BG_CHECK(digest_bytes(boundary_as.data(), vector.byte_length, digest));
    BG_CHECK(digest_equals(digest, vector.expected));
  }

  constexpr std::array<std::uint8_t, 9U> kBinaryVector = {
      0x00U, 0x01U, 0x7fU, 0x80U, 0xfeU,
      0xffU, 0x00U, 0xa5U, 0x5aU,
  };
  BG_CHECK(digest_bytes(kBinaryVector.data(), kBinaryVector.size(), digest));
  BG_CHECK(digest_equals(
      digest,
      "175711893ed1a4c5e59d1c53e922cf4e"
      "c98eaeffbe6edeed62160407d1601f48"));

  std::array<std::uint8_t, 1000U> thousand_as{};
  thousand_as.fill(static_cast<std::uint8_t>('a'));
  Sha256 million_as;
  for (std::size_t index = 0U; index < 1000U; ++index) {
    BG_CHECK(million_as.update(thousand_as.data(), thousand_as.size()));
  }
  BG_CHECK(million_as.finalize(digest));
  BG_CHECK(digest_equals(
      digest,
      "cdc76e5c9914fb9281a1c7e284d73e67"
      "f1809a48a497200e046d39ccc7112cd0"));

  std::array<std::uint8_t, 4099U> boundary_input{};
  for (std::size_t index = 0U; index < boundary_input.size(); ++index) {
    boundary_input[index] = static_cast<std::uint8_t>(
        (index * 131U + index / 7U + 17U) & 0xffU);
  }
  Sha256Digest whole_digest{};
  BG_CHECK(digest_bytes(boundary_input.data(), boundary_input.size(),
                        whole_digest));
  constexpr std::array<std::size_t, 15U> kChunkSizes = {
      1U, 2U, 3U, 7U, 55U, 56U, 57U, 63U,
      64U, 65U, 127U, 128U, 129U, 1024U, 2048U,
  };
  for (const std::size_t chunk_size : kChunkSizes) {
    Sha256 chunked;
    std::size_t offset = 0U;
    while (offset < boundary_input.size()) {
      const std::size_t remaining = boundary_input.size() - offset;
      const std::size_t current = remaining < chunk_size ? remaining : chunk_size;
      BG_CHECK(chunked.update(boundary_input.data() + offset, current));
      offset += current;
    }
    Sha256Digest chunked_digest{};
    BG_CHECK(chunked.finalize(chunked_digest));
    BG_CHECK(chunked_digest == whole_digest);
  }

  Sha256 double_finalize;
  BG_CHECK(double_finalize.update(kAbc.data(), kAbc.size()));
  BG_CHECK(double_finalize.finalize(digest));
  BG_CHECK(double_finalize.finalized());
  digest.fill(0xffU);
  BG_CHECK(!double_finalize.finalize(digest));
  BG_CHECK(!double_finalize.healthy());
  BG_CHECK(!double_finalize.finalized());
  BG_CHECK(digest == Sha256Digest{});
  BG_CHECK(!double_finalize.update(kAbc.data(), kAbc.size()));

  Sha256 update_after_finalize;
  BG_CHECK(update_after_finalize.finalize(digest));
  BG_CHECK(!update_after_finalize.update(kAbc.data(), kAbc.size()));
  digest.fill(0xffU);
  BG_CHECK(!update_after_finalize.finalize(digest));
  BG_CHECK(digest == Sha256Digest{});

  Sha256 null_input;
  BG_CHECK(!null_input.update(nullptr, 1U));
  BG_CHECK(!null_input.healthy());
  BG_CHECK(!null_input.update(kAbc.data(), kAbc.size()));
  digest.fill(0xffU);
  BG_CHECK(!null_input.finalize(digest));
  BG_CHECK(digest == Sha256Digest{});

  return 0;
}

}  // namespace

int main() { return run_sha256_tests(); }
