#!/usr/bin/env node
import { readFileSync } from "node:fs";

const files = process.argv.slice(2);

if (files.length === 0) {
  console.error("Usage: node scripts/validate-research-gated-prd.mjs <prd.md> [...]");
  process.exit(2);
}

const requiredSections = [
  "Problem Statement",
  "Solution",
  "User Stories",
  "Research Dossier",
  "Grill Decisions",
  "Novelty Reach",
  "Implementation Decisions",
  "Testing Decisions",
  "Out of Scope",
  "Further Notes",
];

let failed = false;

for (const file of files) {
  const text = readFileSync(file, "utf8");
  const headings = parseHeadings(text);
  const errors = [];

  for (const section of requiredSections) {
    if (!headings.has(section)) {
      errors.push(`missing section: ${section}`);
      continue;
    }
    const body = sectionBody(text, section);
    if (!hasMeaningfulBody(body)) {
      errors.push(`empty section: ${section}`);
    }
  }

  const userStories = sectionBody(text, "User Stories");
  const storyCount = userStories
    .split("\n")
    .filter((line) => /^\d+\.\s+As .+, I want .+, so that .+/i.test(line.trim()))
    .length;
  if (storyCount < 3) {
    errors.push(`User Stories must include at least 3 numbered stories (found ${storyCount})`);
  }

  const grillDecisions = sectionBody(text, "Grill Decisions");
  if (!/question:/i.test(grillDecisions) || !/recommended answer:/i.test(grillDecisions)) {
    errors.push("Grill Decisions must include Question and Recommended answer entries");
  }

  const researchDossier = sectionBody(text, "Research Dossier");
  if (!/https?:\/\//.test(researchDossier) && !/repo exploration:/i.test(researchDossier)) {
    errors.push("Research Dossier must include links or explicit repo exploration notes");
  }

  if (errors.length > 0) {
    failed = true;
    console.error(`${file}:`);
    for (const error of errors) {
      console.error(`  - ${error}`);
    }
  } else {
    console.log(`${file}: ok`);
  }
}

process.exit(failed ? 1 : 0);

function parseHeadings(text) {
  const headings = new Set();
  for (const line of text.split("\n")) {
    const match = /^##\s+(.+?)\s*$/.exec(line);
    if (match?.[1]) headings.add(match[1].trim());
  }
  return headings;
}

function sectionBody(text, heading) {
  const pattern = new RegExp(`^##\\s+${escapeRegExp(heading)}\\s*$`, "m");
  const match = pattern.exec(text);
  if (!match) return "";
  const start = match.index + match[0].length;
  const next = /^##\s+.+$/m.exec(text.slice(start));
  return next ? text.slice(start, start + next.index) : text.slice(start);
}

function hasMeaningfulBody(body) {
  return body
    .split("\n")
    .map((line) => line.trim())
    .some((line) => line && !line.startsWith("<!--") && !line.endsWith("-->"));
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
