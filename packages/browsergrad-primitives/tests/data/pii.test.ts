import { describe, expect, it } from "vitest";
import { maskPii } from "../../src/data";

describe("maskPii", () => {
  it("masks emails, phones, and IP addresses with deterministic spans", () => {
    const result = maskPii(
      "Email Jane.Doe+lab@cs.stanford.edu, call (415) 555-1212, host 192.168.0.1.",
    );

    expect(result.text).toBe("Email <EMAIL>, call <PHONE>, host <IP>.");
    expect(result.spans).toEqual([
      {
        kind: "email",
        original: "Jane.Doe+lab@cs.stanford.edu",
        replacement: "<EMAIL>",
        start: 6,
        end: 34,
      },
      {
        kind: "phone",
        original: "(415) 555-1212",
        replacement: "<PHONE>",
        start: 41,
        end: 55,
      },
      {
        kind: "ip",
        original: "192.168.0.1",
        replacement: "<IP>",
        start: 62,
        end: 73,
      },
    ]);
  });
});
