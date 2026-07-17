import {
  installCppCuteBrowserWorkerEntry,
  type CppCuteBrowserWorkerEntryScope,
} from "./cpp_cute_browser_worker_entry.js";

installCppCuteBrowserWorkerEntry(
  globalThis as unknown as CppCuteBrowserWorkerEntryScope,
);
