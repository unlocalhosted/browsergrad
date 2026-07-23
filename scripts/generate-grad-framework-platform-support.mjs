#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { buildGradFrameworkPlatformSupportSource } from "./grad-framework-platform-support.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const inventoryFile = path.join(
  repoRoot,
  "architecture/grad-compatibility-inventory.json",
);
const outputFile = path.join(
  repoRoot,
  "packages/browsergrad-grad/src/framework-platform-support.generated.ts",
);
const inventory = JSON.parse(fs.readFileSync(inventoryFile, "utf8"));
const source = buildGradFrameworkPlatformSupportSource(inventory);
fs.writeFileSync(outputFile, source);
process.stdout.write(
  `Wrote ${path.relative(repoRoot, outputFile)} with ${inventory.behaviors.length} Grad contracts.\n`,
);
