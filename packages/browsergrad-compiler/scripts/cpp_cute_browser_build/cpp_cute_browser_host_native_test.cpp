#define BG_CPP_CUTE_BROWSER_HOST_TESTING 1
#include "extractor/BrowserGradCppCuteBrowserHost.cpp"

#include <cstdio>

namespace {

#define BG_CHECK(condition)                                                   \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::fprintf(stderr, "browser host check failed at line %d: %s\n",     \
                   __LINE__, #condition);                                     \
      return 1;                                                               \
    }                                                                         \
  } while (false)

int run_browser_host_tests() {
  struct passwd password {};
  struct passwd* result = &password;
  char password_buffer[1] {};
  BG_CHECK(getpwnam_r("browsergrad", &password, password_buffer,
                      sizeof(password_buffer), &result) == ENOSYS);
  BG_CHECK(result == nullptr);
  result = &password;
  BG_CHECK(getpwuid_r(0, &password, password_buffer,
                      sizeof(password_buffer), &result) == ENOSYS);
  BG_CHECK(result == nullptr);
  BG_CHECK(getuid() == std::numeric_limits<uid_t>::max());

  struct rlimit limits {};
  BG_CHECK(getrlimit(RLIMIT_CORE, &limits) == 0);
  BG_CHECK(limits.rlim_cur == RLIM_INFINITY);
  BG_CHECK(limits.rlim_max == RLIM_INFINITY);
  errno = 0;
  BG_CHECK(setrlimit(RLIMIT_CORE, &limits) == -1);
  BG_CHECK(errno == ENOSYS);

  struct rusage usage;
  std::memset(&usage, 0xff, sizeof(usage));
  BG_CHECK(getrusage(RUSAGE_SELF, &usage) == 0);
  const auto* usage_bytes = reinterpret_cast<const unsigned char*>(&usage);
  for (std::size_t index = 0; index < sizeof(usage); ++index) {
    BG_CHECK(usage_bytes[index] == 0U);
  }

  errno = 0;
  BG_CHECK(getsid(0) == static_cast<pid_t>(-1));
  BG_CHECK(errno == ENOSYS);
  errno = 0;
  BG_CHECK(fork() == static_cast<pid_t>(-1));
  BG_CHECK(errno == ENOSYS);
  char command[] = "browsergrad";
  char* arguments[] = {command, nullptr};
  char* environment[] = {nullptr};
  errno = 0;
  BG_CHECK(execve("/forbidden", arguments, environment) == -1);
  BG_CHECK(errno == ENOSYS);
  pid_t process = 7;
  BG_CHECK(posix_spawn(&process, "/forbidden", nullptr, nullptr, arguments,
                       environment) == ENOSYS);
  BG_CHECK(process == static_cast<pid_t>(-1));
  errno = 0;
  BG_CHECK(sigaltstack(nullptr, nullptr) == -1);
  BG_CHECK(errno == ENOSYS);
  return 0;
}

}  // namespace

int main() { return run_browser_host_tests(); }
