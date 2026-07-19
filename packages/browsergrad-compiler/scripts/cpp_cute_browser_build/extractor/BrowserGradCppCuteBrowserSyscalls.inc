namespace {

// These definitions mirror the wasm32 layout in the pinned Emscripten 6.0.3
// wasi/api.h. Keeping the small raw ABI mirror here avoids inheriting the
// header's import attributes on definitions that deliberately close imports.
struct WasiMutableIoVector {
  std::uint8_t* data;
  std::uint32_t byte_length;
};

constexpr std::uint16_t kWasiErrnoNosys = 52;

static_assert(sizeof(void*) == 4);
static_assert(sizeof(WasiMutableIoVector) == 8);
static_assert(offsetof(WasiMutableIoVector, data) == 0);
static_assert(offsetof(WasiMutableIoVector, byte_length) == 4);

}  // namespace

// The public POSIX surface is closed above. Emscripten's libc retains a small
// set of raw backend calls through otherwise unreachable helper objects. Close
// those true backend seams as well: Linux-shaped syscalls return negative
// errno, WASI calls return their unsigned errno, and caller-owned output
// remains untouched.
extern "C" {

int __syscall_unlinkat(int, const char*, int) {
  return -ENOSYS;
}

int __syscall_rmdir(const char*) {
  return -ENOSYS;
}

int __syscall_openat(int, const char*, int, ...) {
  return -ENOSYS;
}

int __syscall_fcntl64(int, int, ...) {
  return -ENOSYS;
}

int __syscall_ioctl(int, int, ...) {
  return -ENOSYS;
}

std::uint16_t __wasi_fd_close(std::uint32_t) {
  return kWasiErrnoNosys;
}

std::uint16_t __wasi_fd_read(std::uint32_t, const WasiMutableIoVector*,
                             std::size_t, std::uint32_t*) {
  return kWasiErrnoNosys;
}

std::uint16_t __wasi_fd_seek(std::uint32_t, std::int64_t, std::uint8_t,
                             std::uint64_t*) {
  return kWasiErrnoNosys;
}

}  // extern "C"
