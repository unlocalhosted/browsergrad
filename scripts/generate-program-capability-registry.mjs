#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { buildProgramCapabilityRegistrySource } from "./program-capability-registry.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const vocabularyFile = path.join(
  repoRoot,
  "architecture/platform-vocabulary.json",
);
const outputFile = path.join(
  repoRoot,
  "packages/browsergrad-runtime/src/program-capability-registry.generated.ts",
);
const vocabulary = JSON.parse(fs.readFileSync(vocabularyFile, "utf8"));
const source = buildProgramCapabilityRegistrySource(vocabulary);
fs.writeFileSync(outputFile, source);
process.stdout.write(
  `Wrote ${path.relative(repoRoot, outputFile)} with ${vocabulary.semanticCapabilities.length} capabilities and ${vocabulary.backends.length} backends.\n`,
);
