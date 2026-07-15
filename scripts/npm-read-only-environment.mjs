import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const DEFAULT_READ_ONLY_NPM_CONFIG = resolve(
  fileURLToPath(new URL("npm-read-only.npmrc", import.meta.url)),
);
export const DEFAULT_READ_ONLY_NPM_GLOBAL_CONFIG = resolve(
  fileURLToPath(new URL("npm-read-only-global.npmrc", import.meta.url)),
);

/**
 * Remove publication credentials and OIDC request authority from npm commands
 * that only read public registry state.
 */
export function createReadOnlyNpmEnvironment(options = {}) {
  const env = { ...(options.baseEnvironment ?? process.env) };
  for (const name of Object.keys(env)) {
    const upperName = name.toUpperCase();
    if (
      upperName.includes("TOKEN")
      || upperName === "ACTIONS_ID_TOKEN_REQUEST_URL"
      || (
        upperName.startsWith("NPM_CONFIG_")
        && /(?:AUTH|PASSWORD|USERNAME|OTP|CERT|KEY)/u.test(upperName)
      )
    ) {
      delete env[name];
    }
  }
  return Object.freeze({
    ...env,
    NODE_AUTH_TOKEN: "",
    NPM_TOKEN: "",
    NPM_CONFIG_REGISTRY: "https://registry.npmjs.org/",
    NPM_CONFIG_USERCONFIG: options.userConfig ?? DEFAULT_READ_ONLY_NPM_CONFIG,
    NPM_CONFIG_GLOBALCONFIG:
      options.globalConfig ?? DEFAULT_READ_ONLY_NPM_GLOBAL_CONFIG,
  });
}
