#!/usr/bin/env node
import process from "node:process";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS,
} from "./cpp_cute_browser_build/cpp_cute_browser_build_executor_process.mjs";

const MAXIMUM_COMMAND_OUTPUT_BYTES = 64 * 1024 * 1024;
const MAXIMUM_COMMAND_DURATION_MS = 60 * 60 * 1_000;
const COMPILER_DIRECTORY = resolve(import.meta.dirname, "..");

export class VerifyCompilerCommandError extends Error {
  /**
   * @param {Readonly<{ id: string; executable: string; arguments: readonly string[] }>} command
   * @param {string} message
   * @param {ErrorOptions} [options]
   */
  constructor(command, message, options) {
    super(`verify:compiler command '${command.id}' ${message}`, options);
    this.name = "VerifyCompilerCommandError";
    this.commandId = command.id;
  }
}

export class VerifyCompilerSignalError extends Error {
  /** @param {NodeJS.Signals} signal */
  constructor(signal) {
    super(`verify:compiler received ${signal}`);
    this.name = "VerifyCompilerSignalError";
    this.signal = signal;
  }
}

export function createVerifyCompilerPlan() {
  const command = (id, ...arguments_) => Object.freeze({
    id,
    executable: process.platform === "win32" ? "pnpm.cmd" : "pnpm",
    arguments: Object.freeze(arguments_),
  });
  const lane = (id, commands) => Object.freeze({ id, commands: Object.freeze(commands) });

  return Object.freeze({
    schema: "browsergrad.compiler.verify-plan",
    version: 1,
    maximumConcurrentLanes: 4,
    prerequisites: Object.freeze([
      command("root-architecture", "--dir", "../..", "run", "architecture:check"),
      command(
        "kernels-build",
        "--dir",
        "../..",
        "--filter",
        "@unlocalhosted/browsergrad-kernels",
        "run",
        "build",
      ),
      command("compiler-build", "run", "build"),
    ]),
    lanes: Object.freeze([
      lane("static-analysis", [
        command("compiler-typecheck", "run", "typecheck"),
        command("compiler-lint", "run", "lint"),
        command("compiler-architecture", "run", "architecture:check"),
      ]),
      lane("docker-shell", [
        command("docker-shell-tests", "run", "test:aot-docker-shell:run"),
      ]),
      lane("compiler-tests", [
        command(
          "clang-wasm-build-plan-tests",
          "run",
          "test:browser-clang-wasm-build-plan:run",
        ),
        command("compiler-tests", "run", "test"),
      ]),
      lane("fixture-and-cli-tests", [
        command("synthetic-input", "run", "test:synthetic-input:run"),
        command("source-normalizer", "run", "test:source-normalizer"),
        command("webgpu-fixtures", "run", "test:webgpu-fixtures"),
        command("test-scope", "run", "test:test-scope"),
        command("bugbash-status", "run", "test:bugbash-status"),
        command("verify-real-world-cli", "run", "test:verify-real-world-cli"),
        command("tool-lock", "run", "test:tool-lock"),
        command("corpus-audit", "run", "test:audit-corpus:run"),
        command("corpus-provisioning", "run", "test:corpus-provisioning:run"),
      ]),
    ]),
  });
}

/** @param {unknown} value */
export function validateVerifyCompilerPlan(value) {
  if (value === null || typeof value !== "object") {
    throw new TypeError("verify:compiler plan must be an object");
  }
  const plan = value;
  if (plan.schema !== "browsergrad.compiler.verify-plan" || plan.version !== 1) {
    throw new TypeError("verify:compiler plan has an unsupported schema or version");
  }
  if (!Number.isSafeInteger(plan.maximumConcurrentLanes) ||
      plan.maximumConcurrentLanes < 1 ||
      plan.maximumConcurrentLanes > 4) {
    throw new TypeError("verify:compiler plan must allow between one and four concurrent lanes");
  }
  if (!Array.isArray(plan.prerequisites) || plan.prerequisites.length === 0) {
    throw new TypeError("verify:compiler plan must contain serial prerequisites");
  }
  if (!Array.isArray(plan.lanes) ||
      plan.lanes.length === 0 ||
      plan.lanes.length > plan.maximumConcurrentLanes) {
    throw new TypeError("verify:compiler plan exceeds its bounded lane concurrency");
  }

  const commandIds = new Set();
  const commandInvocations = new Set();
  const laneIds = new Set();
  const inspectCommand = (command, location) => {
    if (command === null || typeof command !== "object") {
      throw new TypeError(`${location} must be a command object`);
    }
    if (typeof command.id !== "string" || command.id.length === 0) {
      throw new TypeError(`${location}.id must be a non-empty string`);
    }
    if (commandIds.has(command.id)) {
      throw new TypeError(`verify:compiler command id '${command.id}' is duplicated`);
    }
    commandIds.add(command.id);
    if (typeof command.executable !== "string" || command.executable.length === 0 ||
        !Array.isArray(command.arguments) ||
        command.arguments.some((argument) => typeof argument !== "string")) {
      throw new TypeError(`${location} must contain one executable and string arguments`);
    }
    const invocation = JSON.stringify([command.executable, ...command.arguments]);
    if (commandInvocations.has(invocation)) {
      throw new TypeError(`verify:compiler invocation ${invocation} is duplicated`);
    }
    commandInvocations.add(invocation);
  };

  plan.prerequisites.forEach((command, index) => {
    inspectCommand(command, `prerequisites[${index}]`);
  });
  plan.lanes.forEach((lane, laneIndex) => {
    if (lane === null || typeof lane !== "object" ||
        typeof lane.id !== "string" || lane.id.length === 0) {
      throw new TypeError(`lanes[${laneIndex}] must have a non-empty id`);
    }
    if (laneIds.has(lane.id)) {
      throw new TypeError(`verify:compiler lane id '${lane.id}' is duplicated`);
    }
    laneIds.add(lane.id);
    if (!Array.isArray(lane.commands) || lane.commands.length === 0) {
      throw new TypeError(`verify:compiler lane '${lane.id}' must contain commands`);
    }
    lane.commands.forEach((command, commandIndex) => {
      inspectCommand(command, `lanes[${laneIndex}].commands[${commandIndex}]`);
    });
  });

  const dockerLane = plan.lanes.find((lane) =>
    lane.commands.some((command) => command.id === "docker-shell-tests"));
  if (dockerLane === undefined || dockerLane.commands.length !== 1) {
    throw new TypeError("Docker-shell verification must remain isolated in its own lane");
  }
  return plan;
}

/**
 * @param {ReturnType<typeof createVerifyCompilerPlan>} plan
 * @param {Readonly<{
 *   executeCommand?: typeof executeVerifyCompilerCommand;
 *   signal?: AbortSignal;
 *   onEvent?: (event: Readonly<Record<string, unknown>>) => void;
 * }>} [options]
 */
export async function executeVerifyCompilerPlan(plan, options = {}) {
  validateVerifyCompilerPlan(plan);
  const executeCommand = options.executeCommand ?? executeVerifyCompilerCommand;
  const controller = new AbortController();
  let firstFailure;

  const stop = (failure) => {
    if (firstFailure !== undefined) return;
    firstFailure = failure instanceof Error ? failure : new Error(String(failure));
    controller.abort(firstFailure);
  };
  const onExternalAbort = () => stop(
    options.signal?.reason instanceof Error
      ? options.signal.reason
      : new Error("verify:compiler was cancelled"),
  );
  if (options.signal?.aborted === true) onExternalAbort();
  else options.signal?.addEventListener("abort", onExternalAbort, { once: true });

  const runCommand = async (command, phase, laneId) => {
    if (controller.signal.aborted) throw firstFailure;
    options.onEvent?.(Object.freeze({ type: "command-start", phase, laneId, commandId: command.id }));
    try {
      await executeCommand(command, { signal: controller.signal });
      options.onEvent?.(Object.freeze({ type: "command-pass", phase, laneId, commandId: command.id }));
    } catch (cause) {
      const failure = cause instanceof VerifyCompilerCommandError ||
        cause instanceof VerifyCompilerSignalError
        ? cause
        : new VerifyCompilerCommandError(command, "failed", {
            cause: cause instanceof Error ? cause : new Error(String(cause)),
          });
      options.onEvent?.(Object.freeze({ type: "command-fail", phase, laneId, commandId: command.id }));
      stop(failure);
      // Preserve the first terminal cause. In particular, an external abort can
      // arrive while an active process is still settling its cancellation; the
      // process boundary then rejects with its own cancelled-command error.
      // That settlement error must not replace the signal that initiated it.
      throw firstFailure ?? failure;
    }
  };

  try {
    for (const command of plan.prerequisites) {
      await runCommand(command, "prerequisite", undefined);
    }

    const laneResults = await Promise.allSettled(plan.lanes.map(async (lane) => {
      options.onEvent?.(Object.freeze({ type: "lane-start", laneId: lane.id }));
      for (const command of lane.commands) {
        await runCommand(command, "lane", lane.id);
      }
      options.onEvent?.(Object.freeze({ type: "lane-pass", laneId: lane.id }));
    }));
    if (firstFailure !== undefined) throw firstFailure;
    const rejected = laneResults.find((result) => result.status === "rejected");
    if (rejected?.status === "rejected") throw rejected.reason;
  } finally {
    options.signal?.removeEventListener("abort", onExternalAbort);
  }
}

/**
 * @param {Readonly<{ id: string; executable: string; arguments: readonly string[] }>} command
 * @param {Readonly<{ signal: AbortSignal }>} options
 */
export async function executeVerifyCompilerCommand(command, options) {
  const result = await CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS.run({
    executable: command.executable,
    arguments: command.arguments,
    cwd: COMPILER_DIRECTORY,
    environment: stringEnvironment(process.env),
    maximumOutputByteLength: MAXIMUM_COMMAND_OUTPUT_BYTES,
    maximumDurationMs: MAXIMUM_COMMAND_DURATION_MS,
    signal: options.signal,
    onOutputChunk: async (stream, chunk) => {
      await writeOutput(stream === "stdout" ? process.stdout : process.stderr, chunk);
    },
  });
  if (result.exitCode !== 0 || result.terminationSignal !== null) {
    const ending = result.terminationSignal === null
      ? `exited with code ${result.exitCode}`
      : `terminated via ${result.terminationSignal}`;
    throw new VerifyCompilerCommandError(command, ending);
  }
}

function stringEnvironment(environment) {
  return Object.fromEntries(Object.entries(environment)
    .filter((entry) => typeof entry[1] === "string"));
}

async function writeOutput(output, chunk) {
  await new Promise((resolveWrite, rejectWrite) => {
    output.write(Buffer.from(chunk), (error) => {
      if (error) rejectWrite(error);
      else resolveWrite();
    });
  });
}

function reportEvent(event) {
  if (event.type === "command-start") {
    const lane = event.laneId === undefined ? "prerequisite" : `lane:${event.laneId}`;
    process.stderr.write(`[verify:compiler] ${lane} start ${event.commandId}\n`);
  } else if (event.type === "command-pass") {
    process.stderr.write(`[verify:compiler] pass ${event.commandId}\n`);
  }
}

async function main() {
  const arguments_ = process.argv.slice(2);
  const plan = createVerifyCompilerPlan();
  validateVerifyCompilerPlan(plan);
  if (arguments_.length === 1 && (arguments_[0] === "--plan" || arguments_[0] === "--dry-run")) {
    process.stdout.write(`${JSON.stringify(plan, null, 2)}\n`);
    return;
  }
  if (arguments_.length !== 0) {
    throw new Error("usage: node scripts/verify_compiler.mjs [--plan|--dry-run]");
  }

  const controller = new AbortController();
  /** @type {NodeJS.Signals | undefined} */
  let receivedSignal;
  const signalHandlers = new Map(["SIGINT", "SIGTERM"].map((signal) => [signal, () => {
    if (receivedSignal !== undefined) return;
    receivedSignal = signal;
    process.exitCode = signal === "SIGINT" ? 130 : 143;
    controller.abort(new VerifyCompilerSignalError(signal));
  }]));
  for (const [signal, handler] of signalHandlers) process.once(signal, handler);
  try {
    await executeVerifyCompilerPlan(plan, {
      signal: controller.signal,
      onEvent: reportEvent,
    });
  } finally {
    for (const [signal, handler] of signalHandlers) process.removeListener(signal, handler);
  }
}

const mainUrl = process.argv[1] === undefined
  ? undefined
  : pathToFileURL(resolve(process.argv[1])).href;
if (mainUrl === import.meta.url) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    if (process.exitCode === undefined) process.exitCode = 1;
  });
}
