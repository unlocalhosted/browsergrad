#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { buildAssignmentRequirementUsage } from "./semantic-architecture-check.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(repoRoot, "architecture/assignment-requirement-usage.generated.json");
const usage = buildAssignmentRequirementUsage(repoRoot);
fs.writeFileSync(output, `${JSON.stringify(usage, null, 2)}\n`);
process.stdout.write(`Wrote ${path.relative(repoRoot, output)} with ${usage.requirements.length} requirement IDs.\n`);
