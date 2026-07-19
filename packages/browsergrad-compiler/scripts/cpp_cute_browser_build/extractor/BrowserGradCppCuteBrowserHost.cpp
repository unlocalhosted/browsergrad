#if !defined(__EMSCRIPTEN__) && !defined(BG_CPP_CUTE_BROWSER_HOST_TESTING)
#error "BrowserGradCppCuteBrowserHost.cpp is only for the browser host boundary"
#endif

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <pwd.h>
#include <sys/mman.h>
#include <spawn.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/wait.h>
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
// signal-stack, clock, entropy, environment, or resource-accounting host
// import. APIs whose callers require initialized output receive deterministic
// identity-free values; operations that would require ambient host authority
// fail explicitly with ENOSYS and do not modify caller output.
extern "C" {

char* getenv(const char*) {
  return nullptr;
}

std::uint16_t __wasi_environ_sizes_get(std::size_t* count,
                                       std::size_t* byte_length) {
  *count = 0;
  *byte_length = 0;
  return 0;
}

std::uint16_t __wasi_environ_get(std::uint8_t**, std::uint8_t*) {
  return 0;
}

int __clock_gettime(clockid_t, struct timespec*) {
  return fail_with_enosys();
}

int clock_gettime(clockid_t clock, struct timespec* time) {
  return __clock_gettime(clock, time);
}

time_t time(time_t*) {
  errno = ENOSYS;
  return static_cast<time_t>(-1);
}

struct tm* __gmtime_r(const time_t*, struct tm*) {
  errno = ENOSYS;
  return nullptr;
}

struct tm* gmtime_r(const time_t* input, struct tm* output) {
  return __gmtime_r(input, output);
}

struct tm* gmtime(const time_t*) {
  errno = ENOSYS;
  return nullptr;
}

struct tm* __localtime_r(const time_t*, struct tm*) {
  errno = ENOSYS;
  return nullptr;
}

struct tm* localtime_r(const time_t* input, struct tm* output) {
  return __localtime_r(input, output);
}

struct tm* localtime(const time_t*) {
  errno = ENOSYS;
  return nullptr;
}

void _tzset_js(long* seconds_west_of_utc, int* observes_daylight_time,
               char* standard_name, char* daylight_name) {
  *seconds_west_of_utc = 0;
  *observes_daylight_time = 0;
  std::memcpy(standard_name, "UTC", sizeof("UTC"));
  std::memcpy(daylight_name, "UTC", sizeof("UTC"));
}

void __tzset() {}

void tzset() {}

int setitimer(int, const struct itimerval*, struct itimerval*) {
  return fail_with_enosys();
}

int getentropy(void*, std::size_t) {
  return fail_with_enosys();
}

#if defined(__EMSCRIPTEN__)
int clock_nanosleep(clockid_t, int, const struct timespec*, struct timespec*) {
  return ENOSYS;
}
#endif

int getpwnam_r(const char*, struct passwd* password, char*, std::size_t,
               struct passwd** result) {
  std::memset(password, 0, sizeof(*password));
  *result = nullptr;
  return ENOSYS;
}

int getpwuid_r(uid_t, struct passwd* password, char*, std::size_t,
               struct passwd** result) {
  std::memset(password, 0, sizeof(*password));
  *result = nullptr;
  return ENOSYS;
}

uid_t getuid() {
  return std::numeric_limits<uid_t>::max();
}

int getrlimit(int, struct rlimit* limits) {
  limits->rlim_cur = RLIM_INFINITY;
  limits->rlim_max = RLIM_INFINITY;
  return 0;
}

int setrlimit(int, const struct rlimit*) {
  return fail_with_enosys();
}

int getrusage(int, struct rusage* usage) {
  std::memset(usage, 0, sizeof(*usage));
  return 0;
}

int posix_madvise(void*, std::size_t, int) {
  // POSIX memory advice cannot change program semantics. The browser runtime
  // owns its linear-memory policy, so accepting the hint as a no-op is exact.
  return 0;
}

int uname(struct utsname* identity) {
  // Expose the compile target, never the runner or end-user host identity.
  std::memset(identity, 0, sizeof(*identity));
  std::memcpy(identity->sysname, "Emscripten", sizeof("Emscripten"));
  std::memcpy(identity->machine, "wasm32", sizeof("wasm32"));
  return 0;
}

pid_t getsid(pid_t) {
  return static_cast<pid_t>(fail_with_enosys());
}

pid_t setsid() {
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

pid_t wait4(pid_t, int*, int, struct rusage*) {
  return static_cast<pid_t>(fail_with_enosys());
}

pid_t waitpid(pid_t, int*, int) {
  return static_cast<pid_t>(fail_with_enosys());
}

int sigaltstack(const stack_t*, stack_t*) {
  return fail_with_enosys();
}

}  // extern "C"
