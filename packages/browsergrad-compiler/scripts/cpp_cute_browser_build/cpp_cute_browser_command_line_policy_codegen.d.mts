export const CPP_CUTE_COMMAND_LINE_POLICY_INCLUDE_PATH: string;

export function renderCppCuteCommandLinePolicyInclude(manifest: unknown): string;

export function cppCuteCommandLinePolicyIncludeMatches(
  manifest: unknown,
  actualBytes: Uint8Array,
): boolean;
