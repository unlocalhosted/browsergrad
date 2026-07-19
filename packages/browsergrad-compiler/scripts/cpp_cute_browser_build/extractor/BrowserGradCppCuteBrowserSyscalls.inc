namespace {

static_assert(sizeof(void*) == 4);
static_assert(sizeof(__wasi_iovec_t) == 8);
static_assert(offsetof(__wasi_iovec_t, buf) == 0);
static_assert(offsetof(__wasi_iovec_t, buf_len) == 4);

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

__wasi_errno_t __wasi_fd_close(__wasi_fd_t) {
  return __WASI_ERRNO_NOSYS;
}

__wasi_errno_t __wasi_fd_read(__wasi_fd_t, const __wasi_iovec_t*,
                              std::size_t, __wasi_size_t*) {
  return __WASI_ERRNO_NOSYS;
}

__wasi_errno_t __wasi_fd_seek(__wasi_fd_t, __wasi_filedelta_t,
                              __wasi_whence_t, __wasi_filesize_t*) {
  return __WASI_ERRNO_NOSYS;
}

}  // extern "C"
