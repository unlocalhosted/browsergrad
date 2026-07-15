import assert from "node:assert/strict";
import { test } from "node:test";

import {
  createReadOnlyNpmEnvironment,
  DEFAULT_READ_ONLY_NPM_CONFIG,
  DEFAULT_READ_ONLY_NPM_GLOBAL_CONFIG,
} from "./npm-read-only-environment.mjs";

test("read-only npm environment removes tokens and OIDC authority", () => {
  const env = createReadOnlyNpmEnvironment({
    baseEnvironment: {
      PATH: "/bin",
      NODE_AUTH_TOKEN: "npm-secret",
      NPM_TOKEN: "npm-fallback",
      GITHUB_TOKEN: "github-secret",
      GH_TOKEN: "gh-secret",
      ACTIONS_ID_TOKEN_REQUEST_URL: "https://oidc.example/",
      ACTIONS_ID_TOKEN_REQUEST_TOKEN: "oidc-secret",
      NPM_CONFIG_USERCONFIG: "/tmp/credentialed-npmrc",
      NPM_CONFIG_GLOBALCONFIG: "/tmp/credentialed-global-npmrc",
      NPM_CONFIG__AUTH: "basic-secret",
      NPM_CONFIG__PASSWORD: "password-secret",
      NPM_CONFIG_USERNAME: "registry-user",
      NPM_CONFIG_CERT: "/tmp/client-cert.pem",
      NPM_CONFIG_KEY: "/tmp/client-key.pem",
      NPM_CONFIG_STRICT_SSL: "true",
    },
  });

  assert.equal(env.PATH, "/bin");
  assert.equal(env.NODE_AUTH_TOKEN, "");
  assert.equal(env.NPM_TOKEN, "");
  assert.equal(env.GITHUB_TOKEN, undefined);
  assert.equal(env.GH_TOKEN, undefined);
  assert.equal(env.ACTIONS_ID_TOKEN_REQUEST_URL, undefined);
  assert.equal(env.ACTIONS_ID_TOKEN_REQUEST_TOKEN, undefined);
  assert.equal(env.NPM_CONFIG_USERCONFIG, DEFAULT_READ_ONLY_NPM_CONFIG);
  assert.equal(env.NPM_CONFIG_GLOBALCONFIG, DEFAULT_READ_ONLY_NPM_GLOBAL_CONFIG);
  assert.equal(env.NPM_CONFIG_REGISTRY, "https://registry.npmjs.org/");
  assert.equal(env.NPM_CONFIG__AUTH, undefined);
  assert.equal(env.NPM_CONFIG__PASSWORD, undefined);
  assert.equal(env.NPM_CONFIG_USERNAME, undefined);
  assert.equal(env.NPM_CONFIG_CERT, undefined);
  assert.equal(env.NPM_CONFIG_KEY, undefined);
  assert.equal(env.NPM_CONFIG_STRICT_SSL, "true");
  assert.equal(Object.isFrozen(env), true);
});

test("read-only npm environment accepts an isolated tokenless user config", () => {
  const env = createReadOnlyNpmEnvironment({
    baseEnvironment: {},
    userConfig: "/tmp/browsergrad-read-only.npmrc",
    globalConfig: "/tmp/browsergrad-read-only-global.npmrc",
  });
  assert.equal(env.NPM_CONFIG_USERCONFIG, "/tmp/browsergrad-read-only.npmrc");
  assert.equal(env.NPM_CONFIG_GLOBALCONFIG, "/tmp/browsergrad-read-only-global.npmrc");
});
