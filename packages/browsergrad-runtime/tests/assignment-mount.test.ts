import { describe, expect, it } from "vitest";
import { VALID_PROFILE } from "./assignment-fixtures";
import {
  createAssignmentMountPreflightReport,
  createAssignmentMountPlan,
  createAssignmentDatasetCachePlan,
  createAssignmentRunPlan,
  evaluateAssignmentMountContents,
  materializeAssignmentMountPlan,
  parseAssignmentProfile,
  verifyAssignmentMountContentHashes,
} from "../src/index";

describe("assignment mount planning", () => {
  it("creates a deterministic mount plan from a run plan", () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });

    expect(createAssignmentMountPlan(plan)).toEqual({
      root: "/assignments/cs336-assignment1",
      files: [
        {
          role: "rubric",
          path: "/assignments/cs336-assignment1/rubric.py",
          required: true,
        },
        {
          role: "starter",
          path: "/assignments/cs336-assignment1/assignment.py",
          required: false,
        },
        {
          role: "reference",
          path: "/assignments/cs336-assignment1/reference.py",
          required: false,
        },
      ],
      datasets: [
        {
          name: "tiny",
          url: "/fixtures/tiny.txt",
          hash: "sha256:abc",
          mountPath: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
        },
      ],
    });
  });

  it("creates deterministic dataset cache entries from mount plans", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      datasets: [
        {
          name: "hashed",
          url: "/fixtures/hashed.bin?download=1",
          hash: "sha256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        },
        {
          name: "source-only",
          url: "https://example.test/data/tiny.txt?version=1",
        },
        {
          name: "bad-hash",
          url: "/fixtures/bad.txt",
          hash: "sha256:replace-with-fixture-hash",
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(plan);

    expect(createAssignmentDatasetCachePlan(mountPlan)).toEqual({
      root: "/assignments/cs336-assignment1",
      datasets: [
        {
          name: "hashed",
          url: "/fixtures/hashed.bin?download=1",
          hash: "sha256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
          mountPath: "/assignments/cs336-assignment1/fixtures/datasets/hashed.bin",
          strategy: "content-addressed",
          cacheKey: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          cachePath: "datasets/sha256/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        },
        {
          name: "source-only",
          url: "https://example.test/data/tiny.txt?version=1",
          mountPath: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
          strategy: "source-addressed",
          cacheKey: "url:https://example.test/data/tiny.txt?version=1",
          cachePath: "datasets/url/https%3A%2F%2Fexample.test%2Fdata%2Ftiny.txt%3Fversion%3D1",
        },
        {
          name: "bad-hash",
          url: "/fixtures/bad.txt",
          hash: "sha256:replace-with-fixture-hash",
          mountPath: "/assignments/cs336-assignment1/fixtures/datasets/bad.txt",
          strategy: "invalid-hash",
          cacheKey: "url:/fixtures/bad.txt",
          cachePath: "datasets/url/%2Ffixtures%2Fbad.txt",
        },
      ],
    });
  });

  it("materializes provided assignment mount contents into SessionFS", async () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const runPlan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(runPlan);
    const writes: Array<{ path: string; content: string | Uint8Array }> = [];

    const materialized = await materializeAssignmentMountPlan(
      {
        async write(path, content) {
          writes.push({ path, content });
        },
        async read(path) {
          const content = writes.find((write) => write.path === path)?.content;
          return typeof content === "string" ? content : "";
        },
        async readBytes(path) {
          const content = writes.find((write) => write.path === path)?.content;
          return typeof content === "string" || content === undefined
            ? new Uint8Array()
            : content;
        },
      },
      mountPlan,
      {
        files: {
          "/assignments/cs336-assignment1/rubric.py": "print('rubric')",
          "/assignments/cs336-assignment1/assignment.py": "answer = 42",
        },
        datasets: {
          tiny: "fixture text",
        },
      },
    );

    expect(materialized).toEqual({
      writtenPaths: [
        "/assignments/cs336-assignment1/rubric.py",
        "/assignments/cs336-assignment1/assignment.py",
        "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
      ],
      skippedOptionalPaths: ["/assignments/cs336-assignment1/reference.py"],
    });
    expect(writes).toEqual([
      {
        path: "/assignments/cs336-assignment1/rubric.py",
        content: "print('rubric')",
      },
      {
        path: "/assignments/cs336-assignment1/assignment.py",
        content: "answer = 42",
      },
      {
        path: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
        content: "fixture text",
      },
    ]);
  });

  it("evaluates assignment mount contents without writing to SessionFS", () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const runPlan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(runPlan);

    expect(
      evaluateAssignmentMountContents(mountPlan, {
        files: {
          "/assignments/cs336-assignment1/assignment.py": "answer = 42",
        },
      }),
    ).toEqual({
      ok: false,
      writablePaths: ["/assignments/cs336-assignment1/assignment.py"],
      missingRequiredFiles: ["/assignments/cs336-assignment1/rubric.py"],
      missingDatasets: ["tiny"],
      skippedOptionalPaths: ["/assignments/cs336-assignment1/reference.py"],
    });

    expect(
      evaluateAssignmentMountContents(mountPlan, {
        files: {
          "/assignments/cs336-assignment1/rubric.py": "print('rubric')",
        },
        datasets: {
          tiny: "fixture text",
        },
      }).ok,
    ).toBe(true);
  });

  it("verifies mounted dataset hashes for text and binary contents", async () => {
    const textSha256 =
      "5cb72f90e968922d30557d0af8f719d21f61792becaa87eb32477767d739dc0b";
    const binarySha256 =
      "26a66b061e8f48f39927c312f25293959729eee95978e2892d49d3512a5cc092";
    const wrongSha256 =
      "a40d856f7bd138b40fc924da7f59edc8edb9e4a749994ca49a4a5e5f7a32602d";
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      datasets: [
        {
          name: "tiny",
          url: "/fixtures/tiny.txt",
          hash: `sha256:${textSha256}`,
        },
        {
          name: "tiny-bin",
          url: "/fixtures/tiny.bin",
          hash: `sha256:${binarySha256}`,
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const mountPlan = createAssignmentMountPlan(
      createAssignmentRunPlan(result.profile, { capabilities: ["pyodide"] }),
    );

    await expect(
      verifyAssignmentMountContentHashes(mountPlan, {
        files: { "/assignments/cs336-assignment1/rubric.py": "print('rubric')" },
        datasets: {
          tiny: "fixture text",
          "tiny-bin": Uint8Array.of(0, 1, 255),
        },
      }),
    ).resolves.toEqual({
      ok: true,
      checks: [
        {
          name: "tiny",
          path: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
          algorithm: "sha256",
          expected: `sha256:${textSha256}`,
          actual: `sha256:${textSha256}`,
          ok: true,
          status: "match",
        },
        {
          name: "tiny-bin",
          path: "/assignments/cs336-assignment1/fixtures/datasets/tiny.bin",
          algorithm: "sha256",
          expected: `sha256:${binarySha256}`,
          actual: `sha256:${binarySha256}`,
          ok: true,
          status: "match",
        },
      ],
    });

    await expect(
      verifyAssignmentMountContentHashes(mountPlan, {
        files: { "/assignments/cs336-assignment1/rubric.py": "print('rubric')" },
        datasets: {
          tiny: "wrong fixture",
          "tiny-bin": Uint8Array.of(0, 1, 255),
        },
      }),
    ).resolves.toMatchObject({
      ok: false,
      checks: [
        {
          name: "tiny",
          expected: `sha256:${textSha256}`,
          actual: `sha256:${wrongSha256}`,
          ok: false,
          status: "mismatch",
        },
        {
          name: "tiny-bin",
          ok: true,
          status: "match",
        },
      ],
    });
  });

  it("bundles mount content readiness and hash verification", async () => {
    const textSha256 =
      "5cb72f90e968922d30557d0af8f719d21f61792becaa87eb32477767d739dc0b";
    const wrongSha256 =
      "a40d856f7bd138b40fc924da7f59edc8edb9e4a749994ca49a4a5e5f7a32602d";
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      datasets: [
        {
          name: "tiny",
          url: "/fixtures/tiny.txt",
          hash: `sha256:${textSha256}`,
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const mountPlan = createAssignmentMountPlan(
      createAssignmentRunPlan(result.profile, { capabilities: ["pyodide"] }),
    );

    await expect(
      createAssignmentMountPreflightReport(mountPlan, {
        files: { "/assignments/cs336-assignment1/rubric.py": "print('rubric')" },
        datasets: { tiny: "wrong fixture" },
      }),
    ).resolves.toMatchObject({
      ok: false,
      content: {
        ok: true,
        missingRequiredFiles: [],
        missingDatasets: [],
        skippedOptionalPaths: ["/assignments/cs336-assignment1/assignment.py",
          "/assignments/cs336-assignment1/reference.py"],
      },
      hashes: {
        ok: false,
        checks: [
          {
            name: "tiny",
            expected: `sha256:${textSha256}`,
            actual: `sha256:${wrongSha256}`,
            ok: false,
            status: "mismatch",
          },
        ],
      },
    });
  });

  it("materializes binary assignment dataset contents", async () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const runPlan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(runPlan);
    const writes: Array<{ path: string; content: string | Uint8Array }> = [];
    const binaryFixture = Uint8Array.of(0x50, 0x54, 0x00, 0xff);

    const materialized = await materializeAssignmentMountPlan(
      {
        async write(path, content) {
          writes.push({ path, content });
        },
        async read() {
          return "";
        },
        async readBytes() {
          return new Uint8Array();
        },
      },
      mountPlan,
      {
        files: {
          "/assignments/cs336-assignment1/rubric.py": "print('rubric')",
        },
        datasets: {
          tiny: binaryFixture,
        },
      },
    );

    expect(materialized.writtenPaths).toEqual([
      "/assignments/cs336-assignment1/rubric.py",
      "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
    ]);
    expect(writes.at(-1)).toEqual({
      path: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
      content: binaryFixture,
    });
  });

  it("rejects missing required mount content before writing optional files", async () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const runPlan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(runPlan);
    const writes: Array<{ path: string; content: string | Uint8Array }> = [];

    await expect(
      materializeAssignmentMountPlan(
        {
          async write(path, content) {
            writes.push({ path, content });
          },
          async read() {
            return "";
          },
          async readBytes() {
            return new Uint8Array();
          },
        },
        mountPlan,
        {
          files: {
            "/assignments/cs336-assignment1/assignment.py": "answer = 42",
          },
          datasets: {
            tiny: "fixture text",
          },
        },
      ),
    ).rejects.toThrow(
      "missing required assignment file content: /assignments/cs336-assignment1/rubric.py",
    );
    expect(writes).toEqual([]);
  });

  it("rejects missing dataset mount content before writing files", async () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const runPlan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(runPlan);
    const writes: Array<{ path: string; content: string | Uint8Array }> = [];

    await expect(
      materializeAssignmentMountPlan(
        {
          async write(path, content) {
            writes.push({ path, content });
          },
          async read() {
            return "";
          },
          async readBytes() {
            return new Uint8Array();
          },
        },
        mountPlan,
        {
          files: {
            "/assignments/cs336-assignment1/rubric.py": "print('rubric')",
          },
        },
      ),
    ).rejects.toThrow("missing assignment dataset content: tiny");
    expect(writes).toEqual([]);
  });
});
