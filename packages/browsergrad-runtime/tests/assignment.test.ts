import { describe, expect, it } from "vitest";
import {
  parseAssignmentProfile,
  profileOracleJsModules,
} from "../src/assignment";

const VALID_PROFILE = {
  id: "cs336-assignment1",
  version: "1.0.0",
  requires_browsergrad: "^0.1.0",
  runtime_packages: ["numpy", "regex", "pytest"],
  files: {
    root: "/assignments/cs336-assignment1",
    rubric_path: "rubric.py",
    starter_path: "assignment.py",
    reference_path: "reference.py",
    fixtures_path: "fixtures",
  },
  timeouts: {
    setup_ms: 10_000,
    test_ms: 30_000,
    worker_ms: 60_000,
  },
  allowed_tests: ["test_train_bpe_tiny", "test_encode_iterable_streams"],
  oracles: [
    {
      name: "_bg_tokenizers",
      js_module: "/assets/tokenizer-oracle.js",
      export_name: "oracle",
    },
  ],
  gates: [
    {
      name: "encode_iterable_streaming",
      kind: "streaming",
      options: { max_chunks_before_first_yield: 2 },
    },
  ],
  datasets: [{ name: "tiny", url: "/fixtures/tiny.txt", hash: "sha256:abc" }],
};

describe("parseAssignmentProfile", () => {
  it("accepts a complete assignment profile", () => {
    const result = parseAssignmentProfile(VALID_PROFILE);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.profile.id).toBe("cs336-assignment1");
      expect(result.profile.runtime_packages).toEqual(["numpy", "regex", "pytest"]);
      expect(result.profile.gates[0]?.kind).toBe("streaming");
    }
  });

  it("rejects malformed profile shape", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      id: "Bad_ID",
      files: { root: "/x" },
      timeouts: { test_ms: -1 },
      gates: [{ name: "bad", kind: "memory-rss" }],
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.some((error) => error.includes("id"))).toBe(true);
      expect(result.errors.some((error) => error.includes("rubric_path"))).toBe(true);
      expect(result.errors.some((error) => error.includes("test_ms"))).toBe(true);
      expect(result.errors.some((error) => error.includes("kind"))).toBe(true);
    }
  });

  it("maps oracle specs to runtime JS module registrations", () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(profileOracleJsModules(result.profile)).toEqual([
      {
        name: "_bg_tokenizers",
        importURL: "/assets/tokenizer-oracle.js",
        exportName: "oracle",
      },
    ]);
  });
});
