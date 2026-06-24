#!/usr/bin/env node
import { spawn } from "node:child_process";
import { mkdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const compilerDir = path.join(root, "packages/browsergrad-compiler");
const lockDir = path.join(root, "node_modules/.cache/browsergrad/cuda-lite-tool.lock");
const staleLockMs = 30 * 60 * 1000;

const tools = {
  bench: { script: "benchmark-cuda-lite-compiler.mjs" },
  "bench:browser": { script: "benchmark-cuda-lite-webgpu.mjs" },
  "e2e:webgpu": { script: "e2e-cuda-lite-webgpu.mjs" },
  "audit:corpus": { script: "audit-cuda-lite-corpus.mjs" },
  "audit:real-world-cuda": { script: "audit-real-world-cuda-corpora.mjs" },
  "audit:cuda-120": {
    script: "audit-real-world-cuda-corpora.mjs",
    args: [
      "--only",
      "cuda-120",
    ],
  },
};

const [toolName, ...rawArgs] = process.argv.slice(2);
if (!toolName || toolName === "--help" || !(toolName in tools)) {
  console.error(`usage: node scripts/run-cuda-lite-tool.mjs <${Object.keys(tools).join("|")}> [args...]`);
  process.exit(toolName ? 2 : 0);
}

const tool = tools[toolName];
const forwardedArgs = rawArgs[0] === "--" ? rawArgs.slice(1) : rawArgs;

await withLock(lockDir, async () => {
  await run(pnpmBin(), ["run", "build"], compilerDir);
  await run(process.execPath, [
    path.join(scriptDir, tool.script),
    ...(tool.args ?? []),
    ...forwardedArgs,
  ], root);
});

function pnpmBin() {
  return process.platform === "win32" ? "pnpm.cmd" : "pnpm";
}

async function withLock(dir, fn) {
  await mkdir(path.dirname(dir), { recursive: true });
  while (true) {
    try {
      await mkdir(dir);
      await writeFile(path.join(dir, "owner.json"), JSON.stringify({
        pid: process.pid,
        createdAt: new Date().toISOString(),
        tool: toolName,
      }));
      break;
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
      await removeStaleLock(dir);
      await sleep(75);
    }
  }

  try {
    await fn();
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

async function removeStaleLock(dir) {
  try {
    const owner = await readLockOwner(dir);
    if (owner?.pid !== undefined && !isProcessAlive(owner.pid)) {
      await rm(dir, { recursive: true, force: true });
      return;
    }
    const info = await stat(dir);
    if (Date.now() - info.mtimeMs > staleLockMs) {
      await rm(dir, { recursive: true, force: true });
    }
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
}

async function readLockOwner(dir) {
  try {
    const raw = await readFile(path.join(dir, "owner.json"), "utf8");
    const owner = JSON.parse(raw);
    return Number.isInteger(owner?.pid) && owner.pid > 0 ? { pid: owner.pid } : undefined;
  } catch {
    return undefined;
  }
}

function isProcessAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error?.code === "EPERM";
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function run(command, args, cwd) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, stdio: "inherit" });
    child.on("error", reject);
    child.on("exit", (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(signal ? `${command} ${args.join(" ")} exited via ${signal}` : `${command} ${args.join(" ")} exited ${code}`));
    });
  });
}
