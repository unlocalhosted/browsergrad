#if !defined(__EMSCRIPTEN__) && !defined(BG_CPP_CUTE_BROWSER_HOST_TESTING)
#error "BrowserGradCppCuteBrowserHost.cpp is only for the browser host boundary"
#endif

#include <cerrno>
#include <csignal>
#include <cstring>
#include <limits>
#include <pwd.h>
#include <spawn.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

int fail_with_enosys() {
  errno = ENOSYS;
  return -1;
}

}  // namespace

// LLVM Support contains portable Unix implementation objects even though this
// product runs in a dedicated browser Worker. These definitions close that
// platform seam inside the Wasm module: they expose no process, user database,
// signal-stack, or resource-accounting host import. APIs whose callers require
// initialized output receive deterministic identity-free values; operations
// that would require ambient host authority fail explicitly with ENOSYS.
extern "C" {

int getpwnam_r(const char*, struct passwd* password, char*, std::size_t,
               struct passwd** result) {
  if (password != nullptr) std::memset(password, 0, sizeof(*password));
  if (result != nullptr) *result = nullptr;
  return ENOSYS;
}

int getpwuid_r(uid_t, struct passwd* password, char*, std::size_t,
               struct passwd** result) {
  if (password != nullptr) std::memset(password, 0, sizeof(*password));
  if (result != nullptr) *result = nullptr;
  return ENOSYS;
}

uid_t getuid() {
  return std::numeric_limits<uid_t>::max();
}

int getrlimit(int, struct rlimit* limits) {
  if (limits == nullptr) {
    errno = EFAULT;
    return -1;
  }
  limits->rlim_cur = RLIM_INFINITY;
  limits->rlim_max = RLIM_INFINITY;
  return 0;
}

int setrlimit(int, const struct rlimit*) {
  return fail_with_enosys();
}

int getrusage(int, struct rusage* usage) {
  if (usage == nullptr) {
    errno = EFAULT;
    return -1;
  }
  std::memset(usage, 0, sizeof(*usage));
  return 0;
}

pid_t getsid(pid_t) {
  return static_cast<pid_t>(fail_with_enosys());
}

pid_t fork() {
  return static_cast<pid_t>(fail_with_enosys());
}

int execve(const char*, char* const[], char* const[]) {
  return fail_with_enosys();
}

int posix_spawn(pid_t* process, const char*,
                const posix_spawn_file_actions_t*, const posix_spawnattr_t*,
                char* const[], char* const[]) {
  if (process != nullptr) *process = static_cast<pid_t>(-1);
  return ENOSYS;
}

int sigaltstack(const stack_t*, stack_t*) {
  return fail_with_enosys();
}

}  // extern "C"
