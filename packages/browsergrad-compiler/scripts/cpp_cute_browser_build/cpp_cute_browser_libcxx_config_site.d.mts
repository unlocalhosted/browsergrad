export const CPP_CUTE_BROWSER_LIBCXX_CONFIG_SITE_PROFILE:
  "browsergrad.compiler.cpp-cute.libcxx-config-site.linux-x86_64-glibc@1";
export const CPP_CUTE_BROWSER_LIBCXX_MODULE_MAP_PROFILE:
  "browsergrad.compiler.cpp-cute.libcxx-module-map.config-site@1";

export class CppCuteBrowserLibcxxConfigSiteError extends Error {
  readonly code: "BG-COMPILER-CPP-CUTE-BROWSER-LIBCXX-CONFIG-SITE";
}

export interface MaterializedCppCuteBrowserLibcxxConfigSite {
  readonly profile: typeof CPP_CUTE_BROWSER_LIBCXX_CONFIG_SITE_PROFILE;
  readonly templateSha256: string;
  readonly configuredSha256: string;
  readonly bytes: Uint8Array;
}

export function materializeCppCuteBrowserLibcxxConfigSite(
  templateBytes: Uint8Array,
): Readonly<MaterializedCppCuteBrowserLibcxxConfigSite>;

export interface MaterializedCppCuteBrowserLibcxxModuleMap {
  readonly profile: typeof CPP_CUTE_BROWSER_LIBCXX_MODULE_MAP_PROFILE;
  readonly templateSha256: string;
  readonly configuredSha256: string;
  readonly bytes: Uint8Array;
}

export function materializeCppCuteBrowserLibcxxModuleMap(
  templateBytes: Uint8Array,
): Readonly<MaterializedCppCuteBrowserLibcxxModuleMap>;
