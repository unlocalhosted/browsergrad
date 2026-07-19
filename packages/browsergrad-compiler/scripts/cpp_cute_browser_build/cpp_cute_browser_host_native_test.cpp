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
  BG_CHECK(getenv("BROWSERGRAD_FORBIDDEN") == nullptr);

  std::size_t environment_count = 7;
  std::size_t environment_byte_length = 11;
  BG_CHECK(__wasi_environ_sizes_get(&environment_count,
                                    &environment_byte_length) == 0);
  BG_CHECK(environment_count == 0);
  BG_CHECK(environment_byte_length == 0);
  std::uint8_t* environment_entry = reinterpret_cast<std::uint8_t*>(1);
  std::uint8_t environment_byte = 0x5a;
  BG_CHECK(__wasi_environ_get(&environment_entry, &environment_byte) == 0);
  BG_CHECK(environment_entry == reinterpret_cast<std::uint8_t*>(1));
  BG_CHECK(environment_byte == 0x5a);

  long seconds_west_of_utc = 7;
  int observes_daylight_time = 1;
  char standard_name[17];
  char daylight_name[17];
  std::memset(standard_name, 0x5a, sizeof(standard_name));
  std::memset(daylight_name, 0x5a, sizeof(daylight_name));
  _tzset_js(&seconds_west_of_utc, &observes_daylight_time, standard_name,
            daylight_name);
  BG_CHECK(seconds_west_of_utc == 0);
  BG_CHECK(observes_daylight_time == 0);
  BG_CHECK(std::strcmp(standard_name, "UTC") == 0);
  BG_CHECK(std::strcmp(daylight_name, "UTC") == 0);

  struct timespec observed_time;
  std::memset(&observed_time, 0x5a, sizeof(observed_time));
  const struct timespec original_time = observed_time;
  errno = 0;
  BG_CHECK(clock_gettime(CLOCK_REALTIME, &observed_time) == -1);
  BG_CHECK(errno == ENOSYS);
  BG_CHECK(std::memcmp(&observed_time, &original_time, sizeof(observed_time)) == 0);

  time_t written_time = 17;
  errno = 0;
  BG_CHECK(time(&written_time) == static_cast<time_t>(-1));
  BG_CHECK(errno == ENOSYS);
  BG_CHECK(written_time == 17);

  const time_t input_time = 0;
  struct tm calendar;
  std::memset(&calendar, 0x5a, sizeof(calendar));
  const struct tm original_calendar = calendar;
  errno = 0;
  BG_CHECK(gmtime_r(&input_time, &calendar) == nullptr);
  BG_CHECK(errno == ENOSYS);
  BG_CHECK(std::memcmp(&calendar, &original_calendar, sizeof(calendar)) == 0);
  errno = 0;
  BG_CHECK(localtime_r(&input_time, &calendar) == nullptr);
  BG_CHECK(errno == ENOSYS);
  BG_CHECK(std::memcmp(&calendar, &original_calendar, sizeof(calendar)) == 0);

  struct itimerval timer;
  std::memset(&timer, 0x5a, sizeof(timer));
  const struct itimerval original_timer = timer;
  errno = 0;
  BG_CHECK(setitimer(ITIMER_REAL, &timer, nullptr) == -1);
  BG_CHECK(errno == ENOSYS);
  BG_CHECK(std::memcmp(&timer, &original_timer, sizeof(timer)) == 0);

  unsigned char entropy[16];
  std::memset(entropy, 0x5a, sizeof(entropy));
  unsigned char original_entropy[sizeof(entropy)];
  std::memcpy(original_entropy, entropy, sizeof(entropy));
  errno = 0;
  BG_CHECK(getentropy(entropy, sizeof(entropy)) == -1);
  BG_CHECK(errno == ENOSYS);
  BG_CHECK(std::memcmp(entropy, original_entropy, sizeof(entropy)) == 0);

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

  unsigned char advised_memory = 0;
  BG_CHECK(posix_madvise(&advised_memory, sizeof(advised_memory),
                         POSIX_MADV_NORMAL) == 0);
  struct utsname identity;
  std::memset(&identity, 0xff, sizeof(identity));
  BG_CHECK(uname(&identity) == 0);
  BG_CHECK(std::strcmp(identity.sysname, "Emscripten") == 0);
  BG_CHECK(std::strcmp(identity.machine, "wasm32") == 0);
  BG_CHECK(identity.nodename[0] == '\0');
  BG_CHECK(identity.release[0] == '\0');
  BG_CHECK(identity.version[0] == '\0');

  errno = 0;
  BG_CHECK(getsid(0) == static_cast<pid_t>(-1));
  BG_CHECK(errno == ENOSYS);
  errno = 0;
  BG_CHECK(setsid() == static_cast<pid_t>(-1));
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
  int process_status = 7;
  errno = 0;
  BG_CHECK(wait4(1, &process_status, 0, &usage) == static_cast<pid_t>(-1));
  BG_CHECK(errno == ENOSYS);
  BG_CHECK(process_status == 7);
  errno = 0;
  BG_CHECK(waitpid(1, &process_status, 0) == static_cast<pid_t>(-1));
  BG_CHECK(errno == ENOSYS);
  BG_CHECK(process_status == 7);
  errno = 0;
  BG_CHECK(sigaltstack(nullptr, nullptr) == -1);
  BG_CHECK(errno == ENOSYS);
  return 0;
}

}  // namespace

int main() { return run_browser_host_tests(); }
