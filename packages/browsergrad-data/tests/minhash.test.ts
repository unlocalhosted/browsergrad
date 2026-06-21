import { describe, expect, it } from "vitest";
import { minhashDeduplicateDocuments } from "../src/index";

describe("minhashDeduplicateDocuments", () => {
  it("removes exact and fuzzy document duplicates while keeping first occurrences", () => {
    const result = minhashDeduplicateDocuments(
      [
        {
          id: "rails_mit_license",
          text: "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files.",
        },
        {
          id: "react_mit_license",
          text: "Permission   is hereby granted free of charge to any person obtaining a copy of this software and associated documentation files without restriction.",
        },
        {
          id: "moby",
          text: "Call me Ishmael. Some years ago never mind how long precisely having little or no money in my purse.",
        },
        {
          id: "moby-copy",
          text: "Call me Ishmael. Some years ago never mind how long precisely having little or no money in my purse.",
        },
      ],
      {
        ngrams: 3,
        numHashes: 64,
        numBands: 8,
        jaccardThreshold: 0.8,
      },
    );

    expect(result.keptDocuments.map((document) => document.id)).toEqual([
      "rails_mit_license",
      "moby",
    ]);
    expect(result.duplicates.map((duplicate) => duplicate.id)).toEqual([
      "react_mit_license",
      "moby-copy",
    ]);
    expect(result.duplicates[0]).toMatchObject({
      id: "react_mit_license",
      duplicateOf: "rails_mit_license",
      candidateReason: "lsh-band",
    });
    expect(result.duplicates[0]?.jaccard).toBeGreaterThanOrEqual(0.8);
    expect(result.duplicates[0]?.jaccard).toBeLessThan(1);
    expect(result.duplicates[1]).toEqual({
      id: "moby-copy",
      duplicateOf: "moby",
      jaccard: 1,
      candidateReason: "lsh-band",
    });
  });
});
