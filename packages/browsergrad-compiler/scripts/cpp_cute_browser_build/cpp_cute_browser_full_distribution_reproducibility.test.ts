import { createHash } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  realpath,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { canonicalJsonBytes } from
  "@unlocalhosted/browsergrad-semantic-core/schema";
import { afterEach, describe, expect, it } from "vitest";

import {
  cppCuteBrowserBuildProvenancePayloadBase64,
  type CppCuteBrowserBuildProvenanceEnvelopeV1,
} from "../../src/cpp_cute_browser_build_provenance_syntax.js";
import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  createCppCuteBrowserBuildProvenanceSyntaxFixture,
  type CppCuteBrowserBuildProvenanceSyntaxFixture,
} from "../../tests/compiler/support/cpp_cute_browser_build_provenance_syntax_fixtures.js";
import {
  canonicalCppCuteBrowserFullDistributionReproducibilityBytes,
  CppCuteBrowserFullDistributionReproducibilityError,
  observeCppCuteBrowserFullDistributionReproducibility,
  parseCppCuteBrowserFullDistributionReproducibilityArguments,
  requireCppCuteBrowserFullDistributionReproducibilityAuthority,
  verifyCppCuteBrowserFullDistributionReproducibility,
} from "./cpp_cute_browser_full_distribution_reproducibility.mjs";

const DETACHED_OUTPUT =
  "assets/browsergrad-cpp-cute/build-provenance.dsse.json";
const ROOTS: string[] = [];

interface DistributionTree {
  readonly outputRoot: string;
  readonly expectedOutputs: readonly Readonly<{
    outputPath: string;
    sha256: string;
    byteLength: string;
  }>[];
}

interface FullDistributionFixture {
  readonly syntax: CppCuteBrowserBuildProvenanceSyntaxFixture;
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
}

afterEach(async () => {
  await Promise.all(
    ROOTS.splice(0).map(
      (root) => rm(root, { recursive: true, force: true }),
    ),
  );
});

describe("full browser C++/CuTe distribution reproducibility", () => {
  it("accepts only two exact roots and one distinct evidence output", () => {
    expect(
      parseCppCuteBrowserFullDistributionReproducibilityArguments([
        "--second-output-root=/private/tmp/second",
        "--evidence-output=/private/tmp/evidence.json",
        "--first-output-root=/private/tmp/first",
      ]),
    ).toEqual({
      firstOutputRoot: "/private/tmp/first",
      secondOutputRoot: "/private/tmp/second",
      evidenceOutput: "/private/tmp/evidence.json",
    });
    for (const arguments_ of [
      [],
      ["--first-output-root=relative"],
      [
        "--first-output-root=/tmp/same",
        "--second-output-root=/tmp/same",
        "--evidence-output=/tmp/evidence",
      ],
      [
        "--first-output-root=/tmp/first",
        "--second-output-root=/tmp/second",
        "--evidence-output=/tmp/first/evidence",
      ],
    ]) {
      expect(() =>
        parseCppCuteBrowserFullDistributionReproducibilityArguments(
          arguments_,
        ),
      ).toThrow();
    }
  });

  it("reverifies all 25 outputs across two roots while separating detached evidence", async () => {
    const fixture = await createFixture();
    const first = await createDistributionTree(
      "first",
      fixture,
      fixture.syntax.envelope,
    );
    const second = await createDistributionTree(
      "second",
      fixture,
      envelopeWithSignatureByte(fixture.syntax.envelope, 7),
    );

    const report =
      await verifyCppCuteBrowserFullDistributionReproducibility({
        buildInputLock: fixture.buildInputLock,
        first,
        second,
      });

    expect(report).toMatchObject({
      schema:
        "browsergrad.compiler.cpp-cute.browser-full-distribution-reproducibility",
      version: 1,
      authority:
        "two-root-complete-distribution-output-reproducibility-only",
      totals: {
        outputCount: 25,
        deterministicSubjectCount: 24,
        detachedEvidenceCount: 1,
      },
      claims: {
        twoDistinctPrivateOutputRootsVerified: true,
        exactBuildLockOutputPlanMatched: true,
        exactOutputsRehashedInBothRoots: true,
        deterministicSubjectsByteIdentical: true,
        detachedEvidenceBuildSubjectMatched: true,
        fullDistributedOutputSetReproducible: true,
        detachedSignatureVerified: false,
        licenseReviewComplete: false,
        distributionAuthorized: false,
        producerTrusted: false,
        workerExecutionObserved: false,
        loweringAuthorityMinted: false,
        backendExecutionObserved: false,
        releaseReady: false,
      },
    });
    expect(report.deterministicOutputs).toHaveLength(24);
    expect(report.detachedEvidence.firstSha256)
      .not.toBe(report.detachedEvidence.secondSha256);
    expect(report.detachedEvidence.buildSubjectId)
      .toBe(fixture.syntax.buildSubject.buildSubjectId);
    expect(Object.isFrozen(report)).toBe(true);
    expect(
      canonicalCppCuteBrowserFullDistributionReproducibilityBytes(report),
    ).toEqual(canonicalJsonBytes(report));
    expect(() =>
      requireCppCuteBrowserFullDistributionReproducibilityAuthority({
        ...report,
      }),
    ).toThrow(CppCuteBrowserFullDistributionReproducibilityError);
  });

  it("discovers only current planned paths before independent exact-tree verification", async () => {
    const fixture = await createFixture();
    const first = await createDistributionTree(
      "observe-first",
      fixture,
      fixture.syntax.envelope,
    );
    const second = await createDistributionTree(
      "observe-second",
      fixture,
      envelopeWithSignatureByte(fixture.syntax.envelope, 11),
    );

    const report =
      await observeCppCuteBrowserFullDistributionReproducibility({
        firstOutputRoot: first.outputRoot,
        secondOutputRoot: second.outputRoot,
      });

    expect(report.claims).toMatchObject({
      exactBuildLockOutputPlanMatched: true,
      exactOutputsRehashedInBothRoots: true,
      deterministicSubjectsByteIdentical: true,
      detachedEvidenceBuildSubjectMatched: true,
      fullDistributedOutputSetReproducible: true,
    });
  });

  it("rejects deterministic byte drift between the complete trees", async () => {
    const fixture = await createFixture();
    const first = await createDistributionTree(
      "drift-first",
      fixture,
      fixture.syntax.envelope,
    );
    const driftPath = deterministicOutputPaths(fixture)[0];
    if (driftPath === undefined) {
      throw new Error("fixture deterministic output is missing");
    }
    const second = await createDistributionTree(
      "drift-second",
      fixture,
      envelopeWithSignatureByte(fixture.syntax.envelope, 3),
      { driftPath },
    );

    await expect(
      verifyCppCuteBrowserFullDistributionReproducibility({
        buildInputLock: fixture.buildInputLock,
        first,
        second,
      }),
    ).rejects.toThrow("deterministic distribution subject differs");
  });

  it("rejects detached envelopes that bind different build subjects", async () => {
    const fixture = await createFixture();
    const first = await createDistributionTree(
      "subject-first",
      fixture,
      fixture.syntax.envelope,
    );
    const second = await createDistributionTree(
      "subject-second",
      fixture,
      envelopeWithBuildSubject(fixture, "e".repeat(64)),
    );

    await expect(
      verifyCppCuteBrowserFullDistributionReproducibility({
        buildInputLock: fixture.buildInputLock,
        first,
        second,
      }),
    ).rejects.toThrow("detached provenance envelopes bind different");
  });

  it("rejects overlapping roots and forged build-lock authority", async () => {
    const fixture = await createFixture();
    const tree = await createDistributionTree(
      "authority",
      fixture,
      fixture.syntax.envelope,
    );
    await expect(
      verifyCppCuteBrowserFullDistributionReproducibility({
        buildInputLock: fixture.buildInputLock,
        first: tree,
        second: tree,
      }),
    ).rejects.toThrow("roots must be distinct and non-overlapping");

    await expect(
      verifyCppCuteBrowserFullDistributionReproducibility({
        buildInputLock: {
          ...fixture.buildInputLock,
        },
        first: tree,
        second: {
          ...tree,
          outputRoot: `${tree.outputRoot}/nested`,
        },
      }),
    ).rejects.toThrow("opaque prepared build-input lock");
  });
});

async function createDistributionTree(
  name: string,
  fixture: FullDistributionFixture,
  envelope: CppCuteBrowserBuildProvenanceEnvelopeV1,
  options: Readonly<{ driftPath?: string }> = {},
): Promise<DistributionTree> {
  const outputRoot = await realpath(
    await mkdtemp(join(tmpdir(), `browsergrad-full-${name}-`)),
  );
  ROOTS.push(outputRoot);
  const outputs =
    unwrapPreparedCppCuteBrowserBuildInputLock(
      fixture.buildInputLock,
    ).lock.body.recipe.distributedOutputPlan.outputs;
  const expectedOutputs = [];
  for (const output of outputs) {
    const bytes = output.path === DETACHED_OUTPUT
      ? canonicalJsonBytes(envelope)
      : new TextEncoder().encode(
        options.driftPath === output.path
          ? `drift:${output.path}\n`
          : `deterministic:${output.path}\n`,
      );
    await mkdir(join(outputRoot, dirname(output.path)), {
      recursive: true,
      mode: 0o700,
    });
    await writeFile(join(outputRoot, output.path), bytes, { mode: 0o400 });
    expectedOutputs.push(Object.freeze({
      outputPath: output.path,
      sha256: sha256(bytes),
      byteLength: String(bytes.byteLength),
    }));
  }
  return Object.freeze({
    outputRoot,
    expectedOutputs: Object.freeze(expectedOutputs),
  });
}

function deterministicOutputPaths(
  fixture: FullDistributionFixture,
): readonly string[] {
  return unwrapPreparedCppCuteBrowserBuildInputLock(
    fixture.buildInputLock,
  ).lock.body.recipe.distributedOutputPlan.outputs
    .filter((output) => output.reproducibilityClass === "deterministic-subject")
    .map((output) => output.path);
}

function envelopeWithSignatureByte(
  envelope: CppCuteBrowserBuildProvenanceEnvelopeV1,
  value: number,
): CppCuteBrowserBuildProvenanceEnvelopeV1 {
  const signature = envelope.signatures[0];
  if (signature === undefined) throw new Error("fixture signature is missing");
  return {
    ...envelope,
    signatures: [{
      ...signature,
      sig: encodeBase64(new Uint8Array(64).fill(value)),
    }],
  };
}

function envelopeWithBuildSubject(
  fixture: FullDistributionFixture,
  digest: string,
): CppCuteBrowserBuildProvenanceEnvelopeV1 {
  const buildSubjectId = `bg.cpp.browser-build-subject.sha256.${digest}`;
  const statement = {
    ...fixture.syntax.statement,
    subject: [{
      name: buildSubjectId,
      digest: { sha256: digest },
    }] as const,
    predicate: {
      ...fixture.syntax.statement.predicate,
      buildSubject: {
        buildSubjectId,
        buildSubjectSha256: digest,
      },
    },
  };
  return {
    ...fixture.syntax.envelope,
    payload: cppCuteBrowserBuildProvenancePayloadBase64(statement),
  };
}

async function createFixture(): Promise<FullDistributionFixture> {
  const [syntax, buildInputLock] = await Promise.all([
    createCppCuteBrowserBuildProvenanceSyntaxFixture(),
    decodeCppCuteBrowserBuildInputLock(
      cppCuteBrowserBuildInputLockResourceBytes(),
    ),
  ]);
  return Object.freeze({ syntax, buildInputLock });
}

function encodeBase64(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes));
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}
