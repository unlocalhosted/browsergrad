import { describe, expect, it } from "vitest";
import { VALID_PROFILE } from "./assignment-fixtures";
import {
  parseAssignmentProfile,
  profileOracleJsModules,
} from "../src/index";

describe("assignment profile parsing", () => {
  it("accepts a complete assignment profile", () => {
    const result = parseAssignmentProfile(VALID_PROFILE);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.profile.id).toBe("cs336-assignment1");
      expect(result.profile.metadata?.course).toBe("Stanford CS336");
      expect(result.profile.metadata?.lecture_urls).toEqual([
        "https://www.youtube.com/watch?v=example",
      ]);
      expect(result.profile.runtime_packages).toEqual(["numpy", "regex", "pytest"]);
      expect(result.profile.gates[0]?.kind).toBe("capability");
      expect(result.profile.gates[1]?.kind).toBe("streaming");
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

  it("rejects malformed capability gate options", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "bad_capability_options",
          kind: "capability",
          options: {
            requires: ["pyodide", 123],
            any_of: [["webgpu"], "native-cuda"],
            message: false,
          },
        },
      ],
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors).toEqual([
        "gates[0].options.requires[1]: must be a string",
        "gates[0].options.any_of[1]: must be a string array",
        "gates[0].options.message: must be a string when present",
      ]);
    }
  });

  it("rejects malformed browser-safe behavioral gate options", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "bad_streaming",
          kind: "streaming",
          options: {
            max_chunks_before_first_yield: -1,
            chunk_count: "many",
          },
        },
        {
          name: "bad_forbidden_read",
          kind: "forbidden-read",
          options: {
            methods: ["read", 42],
          },
        },
        {
          name: "bad_timeout",
          kind: "timeout",
          options: {
            timeout_ms: 1.5,
          },
        },
      ],
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors).toEqual([
        "gates[0].options.max_chunks_before_first_yield: must be a non-negative integer",
        "gates[0].options.chunk_count: must be a non-negative integer when present",
        "gates[1].options.methods[1]: must be a string",
        "gates[2].options.timeout_ms: must be a non-negative integer",
      ]);
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
