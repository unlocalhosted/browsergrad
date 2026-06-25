#!/usr/bin/env node
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { withDirectoryLock } from "./cuda-lite-tool-lock.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const compilerDir = path.join(root, "packages/browsergrad-compiler");
const lockDir = path.join(root, "node_modules/.cache/browsergrad/cuda-lite-tool.lock");

const tools = {
  bench: { script: "benchmark-cuda-lite-compiler.mjs" },
  "bench:browser": { script: "benchmark-cuda-lite-webgpu.mjs" },
  "e2e:webgpu": { script: "e2e-cuda-lite-webgpu.mjs" },
  "audit:corpus": { script: "audit-cuda-lite-corpus.mjs" },
  "audit:real-world-cuda": { script: "audit-real-world-cuda-corpora.mjs" },
  "verify:real-world-cuda": { script: "verify-real-world-cuda.mjs" },
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

await withDirectoryLock(lockDir, { tool: toolName }, async () => {
  await run(pnpmBin(), ["--filter", "@unlocalhosted/browsergrad-kernels", "run", "build"], root);
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
