#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { findRepoRoot } from "./cuda-lite-webgpu-cli.mjs";

const scriptPath = fileURLToPath(import.meta.url);

function parseArgs(argv) {
  const options = {
    json: false,
    root: undefined,
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--json") options.json = true;
    else if (arg === "--root") options.root = argv[++index];
    else if (arg.startsWith("--root=")) options.root = arg.slice("--root=".length);
    else throw new Error(`unknown argument: ${arg}`);
  }
  return options;
}

export function gatherStatus(root) {
  const progressPath = path.join(root, "docs/internal/compiler-bugbash-progress.md");
  const failurePath = path.join(root, ".tmp/cuda-lite-last-failures.json");
  const progress = readProgress(progressPath);
  const failures = readJson(failurePath);
  const git = gitStatus(root);
  const activeFailureCases = Array.isArray(failures?.cases) ? failures.cases.filter(Boolean) : [];
  const progressSummary = summarizeProgress(progress, activeFailureCases);
  return {
    tool: "browsergrad-cuda-lite-bugbash-status",
    root,
    progressPath: fs.existsSync(progressPath) ? progressPath : undefined,
    failurePath: fs.existsSync(failurePath) ? failurePath : undefined,
    lastUpdated: progress.lastUpdated,
    dashboard: progress.dashboard,
    progress: progressSummary,
    latestGreenGates: progress.latestGreenGates,
    remainingProbes: progress.remainingProbes,
    activeFailures: {
      cases: activeFailureCases,
      failedByStage: failures?.failedByStage ?? {},
      replayCommand: activeFailureCases.length > 0
        ? "pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:last-failures"
        : undefined,
    },
    git,
    nextCommands: nextCommands(activeFailureCases),
  };
}

export function formatStatus(status) {
  const lines = [];
  lines.push("Compiler Bugbash Status");
  lines.push(`Last updated: ${status.lastUpdated ?? "unknown"}`);
  if (status.dashboard["Overall status"]) lines.push(`Overall: ${status.dashboard["Overall status"]}`);
  if (status.dashboard["Active work item"]) lines.push(`Active: ${status.dashboard["Active work item"]}`);
  if (status.dashboard["Skip policy"]) lines.push(`Skips: ${status.dashboard["Skip policy"]}`);
  lines.push(`Git: ${status.git.summary}`);
  lines.push("Progress:");
  if (status.progress.movement) lines.push(`- Movement: ${status.progress.movement}`);
  lines.push(`- Active failure cases: ${status.progress.activeFailureCount}`);
  if (status.progress.latestUnitGate) lines.push(`- Latest unit proof: ${status.progress.latestUnitGate}`);
  if (status.progress.latestSmokeGate) lines.push(`- Latest smoke proof: ${status.progress.latestSmokeGate}`);
  if (status.progress.latestVerifierGate) lines.push(`- Latest verifier proof: ${status.progress.latestVerifierGate}`);
  lines.push(`- Remaining probe entries: ${status.progress.remainingProbeCount}`);
  if (status.activeFailures.cases.length > 0) {
    lines.push(`Active failures: ${status.activeFailures.cases.length}`);
    for (const name of status.activeFailures.cases) lines.push(`- ${name}`);
    const stages = Object.entries(status.activeFailures.failedByStage)
      .map(([stage, count]) => `${stage}=${count}`)
      .join(", ");
    if (stages) lines.push(`Failure stages: ${stages}`);
  } else {
    lines.push("Active failures: 0");
  }
  if (status.latestGreenGates.length > 0) {
    lines.push("Latest green gates:");
    for (const gate of status.latestGreenGates.slice(-6)) lines.push(`- ${gate}`);
  }
  if (status.remainingProbes.length > 0) {
    lines.push("Remaining probes:");
    for (const probe of status.remainingProbes.slice(0, 8)) lines.push(`- ${probe}`);
  }
  lines.push("Next commands:");
  for (const command of status.nextCommands) lines.push(`- ${command}`);
  return `${lines.join("\n")}\n`;
}

function nextCommands(activeFailureCases) {
  const commands = [];
  if (activeFailureCases.length > 0) {
    commands.push("pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:last-failures");
  }
  commands.push("pnpm --filter @unlocalhosted/browsergrad-compiler run verify:changed:plan");
  commands.push("pnpm --filter @unlocalhosted/browsergrad-compiler run verify:changed");
  return commands;
}

function summarizeProgress(progress, activeFailureCases) {
  return {
    movement: progress.dashboard["Fixed failure movement"],
    activeFailureCount: activeFailureCases.length,
    latestUnitGate: latestGate(progress.latestGreenGates, /compiler unit suite/i),
    latestSmokeGate: latestGate(progress.latestGreenGates, /WebGPU smoke/i),
    latestVerifierGate: latestGate(progress.latestGreenGates, /verifier/i),
    remainingProbeCount: progress.remainingProbes.length,
  };
}

function latestGate(gates, pattern) {
  return gates.filter((gate) => pattern.test(gate)).at(-1);
}

function readProgress(progressPath) {
  if (!fs.existsSync(progressPath)) return { dashboard: {}, latestGreenGates: [] };
  const source = fs.readFileSync(progressPath, "utf8");
  return {
    lastUpdated: /^Last updated:\s*(.+)$/mu.exec(source)?.[1],
    dashboard: parseDashboard(source),
    latestGreenGates: parseBulletSection(source, "Latest Proven Green Gates"),
    remainingProbes: parseRemainingProbeMap(source),
  };
}

function parseDashboard(source) {
  const section = sectionText(source, "Dashboard");
  const dashboard = {};
  for (const line of section.split(/\r?\n/u)) {
    const match = /^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$/u.exec(line);
    if (!match) continue;
    const key = match[1].trim();
    const value = match[2].trim();
    if (key === "Field" || /^-+$/u.test(key)) continue;
    dashboard[key] = value;
  }
  return dashboard;
}

function parseBulletSection(source, heading) {
  return sectionText(source, heading)
    .split(/\r?\n/u)
    .map((line) => /^-\s+(.+)$/u.exec(line)?.[1]?.trim())
    .filter(Boolean);
}

function parseRemainingProbeMap(source) {
  const probes = [];
  let group;
  for (const line of sectionText(source, "Remaining Probe Map").split(/\r?\n/u)) {
    const groupMatch = /^-\s+(.+?):\s*$/u.exec(line);
    if (groupMatch) {
      group = groupMatch[1].trim();
      continue;
    }
    const itemMatch = /^\s+-\s+(.+)$/u.exec(line);
    if (itemMatch && group) probes.push(`${group}: ${itemMatch[1].trim()}`);
  }
  return probes;
}

function sectionText(source, heading) {
  const start = source.search(new RegExp(`^## ${escapeRegExp(heading)}\\s*$`, "mu"));
  if (start < 0) return "";
  const rest = source.slice(start).split(/\r?\n/u).slice(1);
  const out = [];
  for (const line of rest) {
    if (/^##\s+/u.test(line)) break;
    out.push(line);
  }
  return out.join("\n");
}

function readJson(filePath) {
  if (!fs.existsSync(filePath)) return undefined;
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function gitStatus(root) {
  const result = spawnSync("git", ["status", "--short"], { cwd: root, encoding: "utf8" });
  if (result.error || result.status !== 0) {
    return { summary: "unavailable", changedFiles: [] };
  }
  const changedFiles = result.stdout.split(/\r?\n/u).filter(Boolean);
  return {
    summary: changedFiles.length === 0 ? "clean" : `dirty, ${changedFiles.length} changed entries`,
    changedFiles,
  };
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const root = path.resolve(options.root ?? findRepoRoot(path.dirname(scriptPath)));
  const status = gatherStatus(root);
  if (options.json) console.log(JSON.stringify(status, null, 2));
  else process.stdout.write(formatStatus(status));
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
