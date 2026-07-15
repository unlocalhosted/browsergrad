import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";

import {
  decodeSingleSlsaV1DsseStatement,
  GITHUB_ACTIONS_BUILD_TYPE,
  GITHUB_HOSTED_BUILDER_ID,
  INTOTO_DSSE_PAYLOAD_TYPE,
  INTOTO_STATEMENT_V1,
  NPM_REGISTRY_URL,
  SLSA_PROVENANCE_V1,
  validateSlsaV1GithubActionsStatement,
  verifyPublishedNpmDependencyProvenance,
  verifyPublishedNpmProvenance,
} from "./verify-npm-provenance.mjs";

const SPEC = "@unlocalhosted/browsergrad-jit@0.9.0";
const NAME = "@unlocalhosted/browsergrad-jit";
const VERSION = "0.9.0";
const REPOSITORY = "https://github.com/unlocalhosted/browsergrad";
const WORKFLOW_PATH = ".github/workflows/release.yml";
const REF = "refs/tags/jit-v0.9.0";
const COMMIT = "a".repeat(40);
const SHA512_HEX = "ab".repeat(64);
const INTEGRITY = `sha512-${Buffer.from(SHA512_HEX, "hex").toString("base64")}`;
const PURL = "pkg:npm/%40unlocalhosted/browsergrad-jit@0.9.0";
const INVOCATION = `${REPOSITORY}/actions/runs/123456789/attempts/2`;
const ATTESTATION_URL =
  "https://registry.npmjs.org/-/npm/v1/attestations/%40unlocalhosted%2Fbrowsergrad-jit@0.9.0";

const EXPECTED = Object.freeze({
  spec: SPEC,
  expectedIntegrity: INTEGRITY,
  repository: REPOSITORY,
  workflowPath: WORKFLOW_PATH,
  ref: REF,
  commit: COMMIT,
});

test("validates one exact npm subject and GitHub-hosted SLSA v1 identity", () => {
  const result = validateSlsaV1GithubActionsStatement(validStatement(), EXPECTED);
  assert.deepEqual(result, {
    spec: SPEC,
    name: NAME,
    version: VERSION,
    integrity: INTEGRITY,
    purl: PURL,
    sha512: SHA512_HEX,
    repository: REPOSITORY,
    workflowPath: WORKFLOW_PATH,
    ref: REF,
    commit: COMMIT,
    builderId: GITHUB_HOSTED_BUILDER_ID,
    invocationId: INVOCATION,
  });
  assert.equal(Object.isFrozen(result), true);
});

test("rejects statement, subject, source, workflow, builder, and invocation mutations", () => {
  const mutations = [
    ["statement type", (value) => { value._type = "https://in-toto.io/Statement/v0.1"; }, /statement\._type/u],
    ["predicate type", (value) => { value.predicateType = "https://slsa.dev/provenance/v0.2"; }, /predicateType/u],
    ["second subject", (value) => { value.subject.push(structuredClone(value.subject[0])); }, /exactly one npm package/u],
    ["subject package", (value) => { value.subject[0].name = "pkg:npm/other@0.9.0"; }, /subject\[0\]\.name/u],
    ["subject version", (value) => { value.subject[0].name = "pkg:npm/%40unlocalhosted/browsergrad-jit@0.9.1"; }, /subject\[0\]\.name/u],
    ["subject digest", (value) => { value.subject[0].digest.sha512 = "00".repeat(64); }, /digest\.sha512/u],
    ["build type", (value) => { value.predicate.buildDefinition.buildType = "foreign"; }, /buildType/u],
    ["repository", (value) => { value.predicate.buildDefinition.externalParameters.workflow.repository = "https://github.com/attacker/repo"; }, /workflow repository/u],
    ["workflow path", (value) => { value.predicate.buildDefinition.externalParameters.workflow.path = ".github/workflows/other.yml"; }, /workflow path/u],
    ["workflow ref", (value) => { value.predicate.buildDefinition.externalParameters.workflow.ref = "refs/heads/main"; }, /workflow ref/u],
    ["second source", (value) => { value.predicate.buildDefinition.resolvedDependencies.push(structuredClone(value.predicate.buildDefinition.resolvedDependencies[0])); }, /exactly one source/u],
    ["source uri", (value) => { value.predicate.buildDefinition.resolvedDependencies[0].uri = `git+${REPOSITORY}@refs/heads/main`; }, /source uri/u],
    ["commit", (value) => { value.predicate.buildDefinition.resolvedDependencies[0].digest.gitCommit = "b".repeat(40); }, /gitCommit/u],
    ["self-hosted builder", (value) => { value.predicate.runDetails.builder.id = "https://github.com/actions/runner/self-hosted"; }, /builder id/u],
    ["foreign invocation", (value) => { value.predicate.runDetails.metadata.invocationId = "https://github.com/attacker/repo/actions/runs/1/attempts/1"; }, /invocationId/u],
    ["run without attempt", (value) => { value.predicate.runDetails.metadata.invocationId = `${REPOSITORY}/actions/runs/123`; }, /invocationId/u],
  ];

  for (const [label, mutate, pattern] of mutations) {
    const statement = validStatement();
    mutate(statement);
    assert.throws(
      () => validateSlsaV1GithubActionsStatement(statement, EXPECTED),
      pattern,
      label,
    );
  }
});

test("rejects non-exact expected identity inputs before trusting a statement", () => {
  for (const [field, value, pattern] of [
    ["spec", "@unlocalhosted/browsergrad-jit@latest", /exact npm package version/u],
    ["expectedIntegrity", `sha256-${Buffer.alloc(32).toString("base64")}`, /canonical sha512 SRI/u],
    ["repository", `${REPOSITORY}.git`, /canonical https/u],
    ["workflowPath", ".github/workflows/../release.yml", /workflowPath/u],
    ["ref", "main", /exact GitHub heads\/tags ref/u],
    ["commit", "A".repeat(40), /lowercase 40-character git SHA/u],
  ]) {
    assert.throws(
      () => validateSlsaV1GithubActionsStatement(validStatement(), {
        ...EXPECTED,
        [field]: value,
      }),
      pattern,
      field,
    );
  }
});

test("decodes exactly one signed SLSA v1 DSSE payload", () => {
  const statement = validStatement();
  const decoded = decodeSingleSlsaV1DsseStatement(validAttestationDocument(statement));
  assert.deepEqual(decoded, statement);
});

test("rejects missing, duplicate, malformed, or unsigned SLSA DSSE payloads", () => {
  const noSlsa = validAttestationDocument();
  noSlsa.attestations = noSlsa.attestations.filter(({ predicateType }) =>
    predicateType !== SLSA_PROVENANCE_V1);
  assert.throws(
    () => decodeSingleSlsaV1DsseStatement(noSlsa),
    /exactly one .*provenance\/v1 attestation, got 0/u,
  );

  const duplicate = validAttestationDocument();
  duplicate.attestations.push(structuredClone(duplicate.attestations[1]));
  assert.throws(
    () => decodeSingleSlsaV1DsseStatement(duplicate),
    /got 2/u,
  );

  const wrongPayloadType = validAttestationDocument();
  wrongPayloadType.attestations[1].bundle.dsseEnvelope.payloadType = "application/json";
  assert.throws(
    () => decodeSingleSlsaV1DsseStatement(wrongPayloadType),
    /payloadType/u,
  );

  const invalidBase64 = validAttestationDocument();
  invalidBase64.attestations[1].bundle.dsseEnvelope.payload = "not base64";
  assert.throws(
    () => decodeSingleSlsaV1DsseStatement(invalidBase64),
    /canonical base64/u,
  );

  const unsigned = validAttestationDocument();
  unsigned.attestations[1].bundle.dsseEnvelope.signatures = [];
  assert.throws(
    () => decodeSingleSlsaV1DsseStatement(unsigned),
    /at least one signature/u,
  );
});

test("live verifier uses only fixed registry reads then fresh no-shell npm install and audit", async () => {
  const fetches = [];
  const commands = [];
  const result = await verifyPublishedNpmProvenance(SPEC, liveExpected(), {
    fetchImpl: async (url, options) => {
      fetches.push({ url: String(url), options });
      return jsonResponse(validMetadata());
    },
    runCommand: async (command, args, options) => {
      commands.push({ command, args, options });
      const manifest = JSON.parse(await readFile(join(options.cwd, "package.json"), "utf8"));
      assert.equal(manifest.private, true);
      return successfulCommand(command, args);
    },
  });

  assert.equal(fetches.length, 1);
  assert.match(fetches[0].url, /^https:\/\/registry\.npmjs\.org\//u);
  for (const fetchCall of fetches) {
    assert.equal(fetchCall.options.redirect, "error");
  }

  assert.equal(commands.length, 3);
  assert.equal(commands[0].command, "npm");
  assert.deepEqual(commands[0].args, ["--version"]);
  assert.equal(commands[1].args[0], "install");
  assert.equal(commands[1].args.at(-1), SPEC);
  assert.ok(commands[1].args.includes("--ignore-scripts"));
  assert.ok(commands[1].args.includes("--omit=optional"));
  assert.ok(commands[1].args.includes("--omit=peer"));
  assert.ok(commands[1].args.includes(`--registry=${NPM_REGISTRY_URL}`));
  assert.equal(commands[2].args[0], "audit");
  assert.equal(commands[2].args[1], "signatures");
  assert.ok(commands[2].args.includes("--json"));
  assert.ok(commands[2].args.includes("--include-attestations"));
  assert.equal(new Set(commands.map(({ options }) => options.cwd)).size, 1);
  for (const { options } of commands) {
    assert.equal(options.shell, false);
    assert.equal(options.env.NODE_AUTH_TOKEN, "");
    assert.equal(options.env.NPM_TOKEN, "");
    assert.equal(options.env.ACTIONS_ID_TOKEN_REQUEST_URL, undefined);
    assert.equal(options.env.ACTIONS_ID_TOKEN_REQUEST_TOKEN, undefined);
    assert.equal(options.env.GITHUB_TOKEN, undefined);
    assert.equal(options.env.NPM_CONFIG_USERCONFIG, join(options.cwd, ".npmrc"));
  }
  assert.equal(existsSync(commands[0].options.cwd), false, "fresh verification directory is removed");
  assert.equal(result.cryptographicallyVerified, true);
  assert.equal(result.registrySignatures, 1);
  assert.equal(result.provenanceAttestations, 1);
});

test("dependency verifier returns the workflow, ref, and commit attested by an allowed release", async () => {
  const statement = validStatement();
  const attestedRef = "refs/tags/jit-v0.8.7";
  const attestedCommit = "b".repeat(40);
  statement.predicate.buildDefinition.externalParameters.workflow.ref = attestedRef;
  statement.predicate.buildDefinition.resolvedDependencies[0] = {
    uri: `git+${REPOSITORY}@${attestedRef}`,
    digest: { gitCommit: attestedCommit },
  };

  const result = await verifyPublishedNpmDependencyProvenance(
    SPEC,
    {
      expectedIntegrity: INTEGRITY,
      repository: REPOSITORY,
      allowedWorkflowPaths: [
        ".github/workflows/manual-release.yml",
        WORKFLOW_PATH,
      ],
    },
    {
      fetchImpl: sequencedFetch(validMetadata()),
      runCommand: successfulCommandForStatement(statement),
    },
  );

  assert.equal(result.repository, REPOSITORY);
  assert.equal(result.workflowPath, WORKFLOW_PATH);
  assert.equal(result.ref, attestedRef);
  assert.equal(result.commit, attestedCommit);
  assert.equal(result.integrity, INTEGRITY);
  assert.equal(result.cryptographicallyVerified, true);
});

test("dependency verifier rejects an attested workflow outside its allowlist", async () => {
  const statement = validStatement();
  statement.predicate.buildDefinition.externalParameters.workflow.path =
    ".github/workflows/untrusted-release.yml";
  let commandCalled = false;

  await assert.rejects(
    verifyPublishedNpmDependencyProvenance(
      SPEC,
      {
        expectedIntegrity: INTEGRITY,
        repository: REPOSITORY,
        allowedWorkflowPaths: [WORKFLOW_PATH],
      },
      {
        fetchImpl: sequencedFetch(validMetadata()),
        runCommand: async (command, args) => {
          commandCalled = true;
          return successfulCommandForStatement(statement)(command, args);
        },
      },
    ),
    /workflow path must be in allowedWorkflowPaths/u,
  );
  assert.equal(commandCalled, true);
});

test("dependency verifier binds approved workflows to their protected ref class", async () => {
  for (const [workflowPath, ref, pattern] of [
    [".github/workflows/publish-npm.yml", "refs/tags/jit-v0.9.0", /publish-npm\.yml provenance must use refs\/heads\/main/u],
    [WORKFLOW_PATH, "refs/heads/main", /release\.yml provenance must use an approved BrowserGrad release tag/u],
  ]) {
    const statement = validStatement();
    statement.predicate.buildDefinition.externalParameters.workflow.path = workflowPath;
    statement.predicate.buildDefinition.externalParameters.workflow.ref = ref;
    statement.predicate.buildDefinition.resolvedDependencies[0].uri = `git+${REPOSITORY}@${ref}`;
    await assert.rejects(
      verifyPublishedNpmDependencyProvenance(
        SPEC,
        {
          expectedIntegrity: INTEGRITY,
          repository: REPOSITORY,
          allowedWorkflowPaths: [workflowPath],
        },
        {
          fetchImpl: sequencedFetch(validMetadata()),
          runCommand: successfulCommandForStatement(statement),
        },
      ),
      pattern,
    );
  }
});

test("live verifier retries bounded transient registry failures", async () => {
  let calls = 0;
  const delays = [];
  const result = await verifyPublishedNpmProvenance(SPEC, liveExpected(), {
    fetchAttempts: 2,
    retryDelayMs: 3,
    sleep: async (delay) => { delays.push(delay); },
    fetchImpl: async () => {
      calls += 1;
      if (calls === 1) {
        return new Response("temporarily unavailable", { status: 503 });
      }
      return jsonResponse(validMetadata());
    },
    runCommand: successfulCommand,
  });
  assert.equal(result.spec, SPEC);
  assert.equal(calls, 2);
  assert.deepEqual(delays, [3]);
});

test("live verifier does not retry permanent registry failures", async () => {
  let calls = 0;
  await assert.rejects(
    verifyPublishedNpmProvenance(SPEC, liveExpected(), {
      fetchAttempts: 3,
      fetchImpl: async () => {
        calls += 1;
        return new Response("not found", { status: 404 });
      },
      sleep: async () => assert.fail("permanent failures must not sleep"),
      runCommand: successfulCommand,
    }),
    /failed after 1 attempt/u,
  );
  assert.equal(calls, 1);
});

test("live verifier rejects registry and provenance descriptor mutations before npm execution", async () => {
  const mutations = [
    ["integrity", (metadata) => { metadata.dist.integrity = `sha512-${Buffer.alloc(64).toString("base64")}`; }, /dist\.integrity/u],
    ["missing descriptor", (metadata) => { delete metadata.dist.attestations; }, /dist\.attestations/u],
    ["predicate", (metadata) => { metadata.dist.attestations.provenance.predicateType = "https://slsa.dev/provenance/v0.2"; }, /provenance predicateType/u],
    ["foreign URL", (metadata) => { metadata.dist.attestations.url = "https://attacker.example/statement"; }, /must stay on/u],
    ["wrong package URL", (metadata) => { metadata.dist.attestations.url = "https://registry.npmjs.org/-/npm/v1/attestations/other@0.9.0"; }, /must identify/u],
    ["missing signatures", (metadata) => { metadata.dist.signatures = []; }, /at least one registry signature/u],
  ];

  for (const [label, mutate, pattern] of mutations) {
    const metadata = validMetadata();
    mutate(metadata);
    let commandCalled = false;
    await assert.rejects(
      verifyPublishedNpmProvenance(SPEC, liveExpected(), {
        fetchAttempts: 1,
        fetchImpl: async () => jsonResponse(metadata),
        runCommand: async () => { commandCalled = true; },
      }),
      pattern,
      label,
    );
    assert.equal(commandCalled, false, label);
  }
});

test("live verifier rejects audit proof failures and still removes the temporary project", async () => {
  let verificationDirectory;
  await assert.rejects(
    verifyPublishedNpmProvenance(SPEC, liveExpected(), {
      fetchImpl: sequencedFetch(validMetadata()),
      runCommand: async (_command, args, options) => {
        verificationDirectory = options.cwd;
        if (args[0] === "--version") {
          return { stdout: "11.12.1\n", stderr: "" };
        }
        if (args[0] === "audit") {
          return {
            stdout: JSON.stringify({ invalid: [{ name: NAME }], missing: [] }),
            stderr: "",
          };
        }
        return { stdout: "installed", stderr: "" };
      },
    }),
    /reported 1 invalid and 0 missing proofs/u,
  );
  assert.equal(existsSync(verificationDirectory), false);
});

test("live verifier rejects npm versions without attestation bundle output", async () => {
  const commands = [];
  await assert.rejects(
    verifyPublishedNpmProvenance(SPEC, liveExpected(), {
      fetchImpl: sequencedFetch(validMetadata()),
      runCommand: async (_command, args) => {
        commands.push(args);
        return { stdout: "11.11.99\n", stderr: "" };
      },
    }),
    /npm >=11\.12\.0 is required/u,
  );
  assert.deepEqual(commands, [["--version"]]);
});

test("live verifier validates identity only from npm's exact verified root bundle", async () => {
  const mutations = [
    ["missing root", (audit) => { audit.verified = []; }, /exactly one verified root entry/u],
    ["wrong location", (audit) => { audit.verified[0].location = "node_modules/other"; }, /exactly one verified root entry/u],
    ["wrong registry", (audit) => { audit.verified[0].registry = "https://registry.example/"; }, /verified npm registry/u],
    ["wrong URL", (audit) => { audit.verified[0].attestations.url = "https://registry.npmjs.org/-/npm/v1/attestations/other@0.9.0"; }, /must identify/u],
    ["missing bundles", (audit) => { delete audit.verified[0].attestationBundles; }, /verified npm attestation bundles/u],
    ["mutated statement", (audit) => {
      const statement = validStatement();
      statement.subject[0].digest.sha512 = "00".repeat(64);
      audit.verified[0].attestationBundles = validAttestationDocument(statement).attestations;
    }, /digest\.sha512/u],
  ];

  for (const [label, mutate, pattern] of mutations) {
    const audit = validAuditOutput();
    mutate(audit);
    await assert.rejects(
      verifyPublishedNpmProvenance(SPEC, liveExpected(), {
        fetchImpl: sequencedFetch(validMetadata()),
        runCommand: commandReturningAudit(audit),
      }),
      pattern,
      label,
    );
  }
});

function validStatement() {
  return {
    _type: INTOTO_STATEMENT_V1,
    subject: [{ name: PURL, digest: { sha512: SHA512_HEX } }],
    predicateType: SLSA_PROVENANCE_V1,
    predicate: {
      buildDefinition: {
        buildType: GITHUB_ACTIONS_BUILD_TYPE,
        externalParameters: {
          workflow: {
            ref: REF,
            repository: REPOSITORY,
            path: WORKFLOW_PATH,
          },
        },
        internalParameters: {
          github: {
            event_name: "push",
            repository_id: "123",
            repository_owner_id: "456",
          },
        },
        resolvedDependencies: [{
          uri: `git+${REPOSITORY}@${REF}`,
          digest: { gitCommit: COMMIT },
        }],
      },
      runDetails: {
        builder: { id: GITHUB_HOSTED_BUILDER_ID },
        metadata: { invocationId: INVOCATION },
      },
    },
  };
}

function validAttestationDocument(statement = validStatement()) {
  return {
    attestations: [
      {
        predicateType: "https://github.com/npm/attestation/tree/main/specs/publish/v0.1",
        bundle: {},
      },
      {
        predicateType: SLSA_PROVENANCE_V1,
        bundle: {
          mediaType: "application/vnd.dev.sigstore.bundle.v0.3+json",
          verificationMaterial: {},
          dsseEnvelope: {
            payload: Buffer.from(JSON.stringify(statement), "utf8").toString("base64"),
            payloadType: INTOTO_DSSE_PAYLOAD_TYPE,
            signatures: [{ sig: Buffer.from("signature").toString("base64"), keyid: "" }],
          },
        },
      },
    ],
  };
}

function validMetadata() {
  return {
    name: NAME,
    version: VERSION,
    dist: {
      integrity: INTEGRITY,
      signatures: [{ keyid: "SHA256:test", sig: "signed" }],
      attestations: {
        url: ATTESTATION_URL,
        provenance: { predicateType: SLSA_PROVENANCE_V1 },
      },
    },
  };
}

function liveExpected() {
  const { spec: _spec, ...expected } = EXPECTED;
  return expected;
}

function jsonResponse(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function sequencedFetch(...values) {
  let index = 0;
  return async () => {
    const value = values[index];
    index += 1;
    return jsonResponse(value);
  };
}

function validAuditOutput(statement = validStatement()) {
  return {
    invalid: [],
    missing: [],
    verified: [{
      name: NAME,
      version: VERSION,
      location: `node_modules/${NAME}`,
      registry: NPM_REGISTRY_URL,
      attestations: {
        url: ATTESTATION_URL,
        provenance: { predicateType: SLSA_PROVENANCE_V1 },
      },
      attestationBundles: validAttestationDocument(statement).attestations,
    }],
  };
}

function successfulCommandForStatement(statement = validStatement()) {
  return commandReturningAudit(validAuditOutput(statement));
}

function commandReturningAudit(audit) {
  return async function runTrustedMockAudit(_command, args) {
    if (args[0] === "--version") {
      return { stdout: "11.12.1\n", stderr: "" };
    }
    if (args[0] === "audit") {
      return { stdout: JSON.stringify(audit), stderr: "" };
    }
    return { stdout: "installed", stderr: "" };
  };
}

async function successfulCommand(_command, args) {
  if (args[0] === "--version") {
    return { stdout: "11.12.1\n", stderr: "" };
  }
  if (args[0] === "audit") {
    return { stdout: JSON.stringify(validAuditOutput()), stderr: "" };
  }
  return { stdout: "installed", stderr: "" };
}
