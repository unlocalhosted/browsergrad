#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { buildAssignmentRequirementRegistrySource } from "./assignment-requirement-registry.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const vocabularyFile = path.join(
  repoRoot,
  "architecture/platform-vocabulary.json",
);
const outputFile = path.join(
  repoRoot,
  "packages/browsergrad-runtime/src/assignment-requirement-registry.generated.ts",
);
const vocabulary = JSON.parse(fs.readFileSync(vocabularyFile, "utf8"));
const source = buildAssignmentRequirementRegistrySource(
  vocabulary.legacyAssignmentRequirements,
);
fs.writeFileSync(outputFile, source);
process.stdout.write(
  `Wrote ${path.relative(repoRoot, outputFile)} with ${vocabulary.legacyAssignmentRequirements.length} definitions.\n`,
);
