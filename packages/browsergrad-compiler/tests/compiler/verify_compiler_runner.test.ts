import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  VerifyCompilerSignalError,
  createVerifyCompilerPlan,
  executeVerifyCompilerCommand,
  executeVerifyCompilerPlan,
  validateVerifyCompilerPlan,
} from "../../scripts/verify_compiler.mjs";

type Plan = ReturnType<typeof createVerifyCompilerPlan>;
type Command = Plan["prerequisites"][number];

const EXPECTED_INVOCATIONS = [
  "pnpm --dir ../.. run architecture:check",
  "pnpm --dir ../.. --filter @unlocalhosted/browsergrad-kernels run build",
  "pnpm run build",
  "pnpm run test:browser-clang-wasm-build-plan:check",
  "pnpm run test:browser-clang-wasm-build-plan:native",
  "pnpm run typecheck",
  "pnpm run lint",
  "pnpm run architecture:check",
  "pnpm run test:aot-docker-shell:run",
  "pnpm run test:browser-clang-wasm-build-plan:surface",
  "pnpm run test",
  "pnpm run test:synthetic-input:run",
  "pnpm run test:source-normalizer",
  "pnpm run test:webgpu-fixtures",
  "pnpm run test:test-scope",
  "pnpm run test:bugbash-status",
  "pnpm run test:verify-real-world-cli",
  "pnpm run test:tool-lock",
  "pnpm run test:audit-corpus:run",
  "pnpm run test:corpus-provisioning:run",
];

describe("verify:compiler bounded runner", () => {
  it("covers every verification command exactly once with native Clang isolated", () => {
    const plan = validateVerifyCompilerPlan(createVerifyCompilerPlan());
    const commands = allCommands(plan);

    expect(plan.maximumConcurrentLanes).toBe(4);
    expect(plan.prerequisites.map((command) => command.id)).toEqual([
      "root-architecture",
      "kernels-build",
      "compiler-build",
      "clang-wasm-build-plan-check",
      "clang-wasm-native-tests",
    ]);
    expect(plan.lanes.map((lane) => lane.id)).toEqual([
      "static-analysis",
      "docker-shell",
      "compiler-tests",
      "fixture-and-cli-tests",
    ]);
    expect(commands.map(invocation)).toEqual(EXPECTED_INVOCATIONS);
    expect(new Set(commands.map((command) => command.id)).size).toBe(commands.length);
    expect(plan.lanes.find((lane) => lane.id === "docker-shell")?.commands.map(
      (command) => command.id,
    )).toEqual(["docker-shell-tests"]);
    expect(plan.lanes.find((lane) => lane.id === "compiler-tests")?.commands.map(
      (command) => command.id,
    )).toEqual(["clang-wasm-surface-tests", "compiler-tests"]);
  });

  it("rejects duplicate command ids, invocations, lanes, and excess concurrency", () => {
    const duplicateId = mutablePlan();
    duplicateId.lanes[0]!.commands[1]!.id = duplicateId.lanes[0]!.commands[0]!.id;
    expect(() => validateVerifyCompilerPlan(duplicateId)).toThrow(/command id .* duplicated/u);

    const duplicateInvocation = mutablePlan();
    duplicateInvocation.lanes[0]!.commands[1]!.arguments = [
      ...duplicateInvocation.lanes[0]!.commands[0]!.arguments,
    ];
    expect(() => validateVerifyCompilerPlan(duplicateInvocation)).toThrow(/invocation .* duplicated/u);

    const duplicateLane = mutablePlan();
    duplicateLane.lanes[1]!.id = duplicateLane.lanes[0]!.id;
    expect(() => validateVerifyCompilerPlan(duplicateLane)).toThrow(/lane id .* duplicated/u);

    const excessConcurrency = mutablePlan();
    excessConcurrency.maximumConcurrentLanes = 5;
    expect(() => validateVerifyCompilerPlan(excessConcurrency)).toThrow(/between one and four/u);

    const nativeLane = mutablePlan();
    const nativeCommand = nativeLane.prerequisites.pop()!;
    nativeLane.lanes[0]!.commands.push(nativeCommand);
    expect(() => validateVerifyCompilerPlan(nativeLane))
      .toThrow(/native Clang verification must remain an isolated serial prerequisite/u);

    const misplacedSurface = mutablePlan();
    const compilerLane = misplacedSurface.lanes.find((lane) => lane.id === "compiler-tests")!;
    const surfaceCommandIndex = compilerLane.commands.findIndex(
      (command) => command.id === "clang-wasm-surface-tests",
    );
    const [surfaceCommand] = compilerLane.commands.splice(surfaceCommandIndex, 1);
    misplacedSurface.lanes[0]!.commands.push(surfaceCommand!);
    expect(() => validateVerifyCompilerPlan(misplacedSurface))
      .toThrow(/non-native Clang-Wasm verification must remain in the compiler-test lane/u);
  });

  it("keeps prerequisites serial, starts four lanes together, and serializes each lane", async () => {
    let active = 0;
    let maximumActive = 0;
    let sequence = 0;
    const intervals = new Map<string, { start: number; finish: number }>();

    await executeVerifyCompilerPlan(createVerifyCompilerPlan(), {
      executeCommand: async (command) => {
        active++;
        maximumActive = Math.max(maximumActive, active);
        const start = sequence++;
        await Promise.resolve();
        const finish = sequence++;
        intervals.set(command.id, { start, finish });
        active--;
      },
    });

    expect(maximumActive).toBe(4);
    expect(intervals.get("root-architecture")!.finish)
      .toBeLessThan(intervals.get("kernels-build")!.start);
    expect(intervals.get("kernels-build")!.finish)
      .toBeLessThan(intervals.get("compiler-build")!.start);
    expect(intervals.get("compiler-build")!.finish)
      .toBeLessThan(intervals.get("clang-wasm-build-plan-check")!.start);
    expect(intervals.get("clang-wasm-build-plan-check")!.finish)
      .toBeLessThan(intervals.get("clang-wasm-native-tests")!.start);
    for (const lane of createVerifyCompilerPlan().lanes) {
      for (let index = 1; index < lane.commands.length; index++) {
        const previous = lane.commands[index - 1]!;
        const current = lane.commands[index]!;
        expect(intervals.get(previous.id)!.finish).toBeLessThan(intervals.get(current.id)!.start);
      }
    }
  });

  it("fails fast, aborts every active sibling boundary, and starts no later commands", async () => {
    const started: string[] = [];
    const cancelled: string[] = [];
    const injected = new Error("injected Docker-shell failure");

    const execution = executeVerifyCompilerPlan(createVerifyCompilerPlan(), {
      executeCommand: async (command, { signal }) => {
        started.push(command.id);
        if ([
          "root-architecture",
          "kernels-build",
          "compiler-build",
          "clang-wasm-build-plan-check",
          "clang-wasm-native-tests",
        ].includes(command.id)) return;
        await new Promise<void>((resolve, reject) => {
          let settled = false;
          const onAbort = () => {
            if (settled) return;
            settled = true;
            cancelled.push(command.id);
            reject(signal.reason);
          };
          signal.addEventListener("abort", onAbort, { once: true });
          if (command.id === "docker-shell-tests") {
            queueMicrotask(() => {
              if (settled) return;
              settled = true;
              signal.removeEventListener("abort", onAbort);
              reject(injected);
            });
          }
          void resolve;
        });
      },
    });

    await expect(execution).rejects.toMatchObject({
      name: "VerifyCompilerCommandError",
      commandId: "docker-shell-tests",
      cause: injected,
    });
    expect(started).toEqual([
      "root-architecture",
      "kernels-build",
      "compiler-build",
      "clang-wasm-build-plan-check",
      "clang-wasm-native-tests",
      "compiler-typecheck",
      "docker-shell-tests",
      "clang-wasm-surface-tests",
      "synthetic-input",
    ]);
    expect(cancelled.sort()).toEqual([
      "clang-wasm-surface-tests",
      "compiler-typecheck",
      "synthetic-input",
    ]);
  });

  it("preserves external termination identity while a serial prerequisite settles", async () => {
    const controller = new AbortController();
    const termination = new VerifyCompilerSignalError("SIGTERM");
    const started: string[] = [];
    let cancellationSettled = false;

    const execution = executeVerifyCompilerPlan(createVerifyCompilerPlan(), {
      signal: controller.signal,
      executeCommand: async (command, { signal }) => {
        started.push(command.id);
        await new Promise<void>((_resolve, reject) => {
          signal.addEventListener("abort", () => {
            queueMicrotask(() => {
              cancellationSettled = true;
              reject(new Error("executor cancellation settled"));
            });
          }, { once: true });
          queueMicrotask(() => controller.abort(termination));
        });
      },
    });

    await expect(execution).rejects.toBe(termination);
    expect(cancellationSettled).toBe(true);
    expect(started).toEqual(["root-architecture"]);
  });

  it("waits for prerequisite process-group settlement before reporting external termination", async () => {
    const root = await mkdtemp(join(tmpdir(), "browsergrad-verify-prerequisite-"));
    const readyPath = join(root, "ready");
    const settledPath = join(root, "settled");
    const controller = new AbortController();
    const termination = new VerifyCompilerSignalError("SIGTERM");
    const plan = mutablePlan();
    plan.prerequisites[0] = {
      id: "root-architecture",
      executable: process.execPath,
      arguments: [
        "-e",
        [
          "const fs = require('node:fs')",
          "process.on('SIGTERM', () => {",
          "  fs.writeFileSync(process.argv[2], 'settled')",
          "  process.exit(0)",
          "})",
          "fs.writeFileSync(process.argv[1], 'ready')",
          "setInterval(() => {}, 1000)",
        ].join(";"),
        readyPath,
        settledPath,
      ],
    };

    const execution = executeVerifyCompilerPlan(plan as unknown as Plan, {
      signal: controller.signal,
      executeCommand: executeVerifyCompilerCommand,
    });
    try {
      await waitForFile(readyPath);
      controller.abort(termination);
      await expect(execution).rejects.toBe(termination);
      await expect(readFile(settledPath, "utf8")).resolves.toBe("settled");
    } finally {
      if (!controller.signal.aborted) controller.abort(termination);
      await execution.catch(() => undefined);
      await rm(root, { recursive: true, force: true });
    }
  });

  it("does not mask a genuine serial prerequisite failure", async () => {
    const injected = new Error("injected root architecture failure");
    const started: string[] = [];

    const execution = executeVerifyCompilerPlan(createVerifyCompilerPlan(), {
      executeCommand: async (command) => {
        started.push(command.id);
        throw injected;
      },
    });

    await expect(execution).rejects.toMatchObject({
      name: "VerifyCompilerCommandError",
      commandId: "root-architecture",
      cause: injected,
    });
    expect(started).toEqual(["root-architecture"]);
  });

  it("relays external termination to all active command boundaries", async () => {
    const controller = new AbortController();
    const started: string[] = [];
    const observedReasons: unknown[] = [];

    const execution = executeVerifyCompilerPlan(createVerifyCompilerPlan(), {
      signal: controller.signal,
      executeCommand: async (command, { signal }) => {
        started.push(command.id);
        if ([
          "root-architecture",
          "kernels-build",
          "compiler-build",
          "clang-wasm-build-plan-check",
          "clang-wasm-native-tests",
        ].includes(command.id)) return;
        await new Promise<void>((_resolve, reject) => {
          signal.addEventListener("abort", () => {
            observedReasons.push(signal.reason);
            reject(signal.reason);
          }, { once: true });
          if (started.length === 9) {
            queueMicrotask(() => controller.abort(new VerifyCompilerSignalError("SIGTERM")));
          }
        });
      },
    });

    await expect(execution).rejects.toBeInstanceOf(VerifyCompilerSignalError);
    expect(observedReasons).toHaveLength(4);
    expect(observedReasons.every((reason) =>
      reason instanceof VerifyCompilerSignalError && reason.signal === "SIGTERM")).toBe(true);
  });
});

function allCommands(plan: Plan): Command[] {
  return [
    ...plan.prerequisites,
    ...plan.lanes.flatMap((lane) => lane.commands),
  ];
}

function invocation(command: Command): string {
  return ["pnpm", ...command.arguments].join(" ");
}

async function waitForFile(path: string): Promise<void> {
  const deadline = Date.now() + 2_000;
  while (Date.now() < deadline) {
    try {
      await readFile(path);
      return;
    } catch (error) {
      if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error(`timed out waiting for ${path}`);
}

function mutablePlan(): {
  schema: string;
  version: number;
  maximumConcurrentLanes: number;
  prerequisites: Array<{ id: string; executable: string; arguments: string[] }>;
  lanes: Array<{
    id: string;
    commands: Array<{ id: string; executable: string; arguments: string[] }>;
  }>;
} {
  return JSON.parse(JSON.stringify(createVerifyCompilerPlan())) as ReturnType<typeof mutablePlan>;
}
