export const CPP_CUTE_BROWSER_LIBCXX_CONFIG_SITE_PROFILE:
  "browsergrad.compiler.cpp-cute.libcxx-config-site.linux-x86_64-glibc@1";

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
