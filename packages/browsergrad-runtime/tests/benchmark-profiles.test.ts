import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  parseAssignmentProfile,
  requiredAssignmentCapabilities,
} from "../src/index";

const PROFILE_FILES = [
  "cs336-assignment1.profile.json",
  "cs336-assignment2-systems.profile.json",
  "cs336-assignment3-scaling.profile.json",
  "cs336-assignment4-data.profile.json",
  "cs336-assignment5-alignment.profile.json",
  "gpu-puzzles.profile.json",
  "cs149-assignment1.profile.json",
  "cs149-assignment2.profile.json",
  "cs149-assignment3.profile.json",
  "cs149gpt.profile.json",
];

describe("benchmark assignment profiles", () => {
  for (const file of PROFILE_FILES) {
    it(`parses ${file} and declares capability requirements`, () => {
      const profileJson = JSON.parse(
        readFileSync(new URL(`../../../docs/internal/${file}`, import.meta.url), "utf8"),
      );

      const result = parseAssignmentProfile(profileJson);
      expect(result).toMatchObject({ ok: true });
      if (!result.ok) return;

      expect(result.profile.metadata?.source_url).toMatch(/^https:\/\/github.com\//);
      expect(requiredAssignmentCapabilities(result.profile).length).toBeGreaterThan(0);
    });
  }
});
