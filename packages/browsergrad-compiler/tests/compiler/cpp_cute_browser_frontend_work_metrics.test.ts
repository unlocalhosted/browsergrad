import { beforeAll, describe, expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_BYTE_LENGTH,
  CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_MAGIC,
  CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_VERSION,
  CppCuteBrowserFrontendWorkMetricsError,
  closeCppCuteBrowserFrontendWorkMetrics,
  completeCppCuteBrowserFrontendWorkMetrics,
  prepareCppCuteBrowserFrontendWorkMetrics,
} from "../../src/cpp_cute_browser_frontend_work_metrics.js";
import {
  prepareCppCuteFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import { createCppCuteBrowserProfileInput } from "./support/cpp_cute_frontend_fixtures.js";

const RECORD_POINTER = 64;

interface Counters {
  readonly includeDepth: bigint;
  readonly macroExpansions: bigint;
  readonly preprocessedTokens: bigint;
  readonly astNodes: bigint;
  readonly constexprSteps: bigint;
  readonly templateInstantiations: bigint;
  readonly templateDepth: bigint;
  readonly completedSemanticPasses: bigint;
}

let profile: PreparedCppCuteFrontendProfile;

beforeAll(async () => {
  profile = await prepareCppCuteFrontendProfile(createCppCuteBrowserProfileInput());
});

describe("Clang-Wasm frontend-work record", () => {
  it("admits one exact complete generation and confirms reset before disclosure", () => {
    const memory = new WebAssembly.Memory({ initial: 1, maximum: 2 });
    writeRecord(memory, 0, 1, 7n, zeroCounters());
    const prepared = prepareCppCuteBrowserFrontendWorkMetrics({
      profile,
      memory,
      recordPointer: RECORD_POINTER,
    });

    writeRecord(memory, 2, 1, 8n, {
      includeDepth: 3n,
      macroExpansions: 11n,
      preprocessedTokens: 101n,
      astNodes: 67n,
      constexprSteps: 13n,
      templateInstantiations: 17n,
      templateDepth: 4n,
      completedSemanticPasses: 2n,
    });
    completeCppCuteBrowserFrontendWorkMetrics(prepared);
    writeRecord(memory, 0, 1, 8n, zeroCounters());
    const observation = closeCppCuteBrowserFrontendWorkMetrics(prepared);

    expect(observation).toEqual({
      authority: "wasm-frontend-work-local-observation-only",
      protocol: "browsergrad.compiler.cpp-cute.frontend-work-metrics@1",
      profileHash: profile.profileHash,
      source: "wasm-memory-frontend-work-metrics-record-v1",
      confidence: "record-exact-unverified-producer",
      generation: "8",
      values: {
        includeDepth: "3",
        macroExpansions: "11",
        preprocessedTokens: "101",
        astNodes: "67",
        constexprSteps: "13",
        templateInstantiations: "17",
        templateDepth: "4",
        completedSemanticPasses: "2",
      },
      resetConfirmed: true,
      workerExecutionObserved: false,
      loweringAuthorityReady: false,
    });
    expect(Object.isFrozen(observation)).toBe(true);
    expect(Object.isFrozen(observation.values)).toBe(true);
  });

  it("rejects non-idle initial state and forged authorities", () => {
    const memory = new WebAssembly.Memory({ initial: 1, maximum: 2 });
    writeRecord(memory, 1, 1, 0n, zeroCounters());
    expectFrontendError(
      () => prepareCppCuteBrowserFrontendWorkMetrics({
        profile,
        memory,
        recordPointer: RECORD_POINTER,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-MISMATCH",
    );
    expectFrontendError(
      () => completeCppCuteBrowserFrontendWorkMetrics(Object.freeze({}) as never),
      "BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-UNVERIFIED",
    );
  });

  it("fails closed on generation, phase, pass-count, ceiling, and reset drift", () => {
    for (const mutate of [
      (values: Counters) => ({ phase: 2, flags: 1, generation: 0n, values }),
      (values: Counters) => ({ phase: 1, flags: 1, generation: 1n, values }),
      (values: Counters) => ({
        phase: 2,
        flags: 1,
        generation: 1n,
        values: { ...values, completedSemanticPasses: 1n },
      }),
      (values: Counters) => ({
        phase: 2,
        flags: 1,
        generation: 1n,
        values: { ...values, includeDepth: 1_025n },
      }),
    ]) {
      const memory = new WebAssembly.Memory({ initial: 1, maximum: 2 });
      writeRecord(memory, 0, 1, 0n, zeroCounters());
      const prepared = prepareCppCuteBrowserFrontendWorkMetrics({
        profile,
        memory,
        recordPointer: RECORD_POINTER,
      });
      const candidate = mutate(validCounters());
      writeRecord(
        memory,
        candidate.phase,
        candidate.flags,
        candidate.generation,
        candidate.values,
      );
      expectFrontendError(
        () => completeCppCuteBrowserFrontendWorkMetrics(prepared),
        candidate.values.includeDepth === 1_025n
          ? "BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-RESOURCE-LIMIT"
          : "BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-MISMATCH",
      );
    }

    const memory = new WebAssembly.Memory({ initial: 1, maximum: 2 });
    writeRecord(memory, 0, 1, 0n, zeroCounters());
    const prepared = prepareCppCuteBrowserFrontendWorkMetrics({
      profile,
      memory,
      recordPointer: RECORD_POINTER,
    });
    writeRecord(memory, 2, 1, 1n, validCounters());
    completeCppCuteBrowserFrontendWorkMetrics(prepared);
    writeRecord(memory, 0, 1, 2n, zeroCounters());
    expectFrontendError(
      () => closeCppCuteBrowserFrontendWorkMetrics(prepared),
      "BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-MISMATCH",
    );
  });
});

function validCounters(): Counters {
  return {
    includeDepth: 1n,
    macroExpansions: 2n,
    preprocessedTokens: 3n,
    astNodes: 4n,
    constexprSteps: 5n,
    templateInstantiations: 6n,
    templateDepth: 2n,
    completedSemanticPasses: 2n,
  };
}

function zeroCounters(): Counters {
  return {
    includeDepth: 0n,
    macroExpansions: 0n,
    preprocessedTokens: 0n,
    astNodes: 0n,
    constexprSteps: 0n,
    templateInstantiations: 0n,
    templateDepth: 0n,
    completedSemanticPasses: 0n,
  };
}

function writeRecord(
  memory: WebAssembly.Memory,
  phase: number,
  flags: number,
  generation: bigint,
  counters: Counters,
): void {
  const bytes = new Uint8Array(
    memory.buffer,
    RECORD_POINTER,
    CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_BYTE_LENGTH,
  );
  bytes.fill(0);
  bytes.set(new TextEncoder().encode(CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_MAGIC));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  view.setUint32(8, CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_VERSION, true);
  view.setUint32(12, CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_BYTE_LENGTH, true);
  view.setUint32(16, phase, true);
  view.setUint32(20, flags, true);
  view.setBigUint64(24, generation, true);
  view.setBigUint64(32, counters.includeDepth, true);
  view.setBigUint64(40, counters.macroExpansions, true);
  view.setBigUint64(48, counters.preprocessedTokens, true);
  view.setBigUint64(56, counters.astNodes, true);
  view.setBigUint64(64, counters.constexprSteps, true);
  view.setBigUint64(72, counters.templateInstantiations, true);
  view.setBigUint64(80, counters.templateDepth, true);
  view.setBigUint64(88, counters.completedSemanticPasses, true);
}

function expectFrontendError(
  operation: () => unknown,
  code: CppCuteBrowserFrontendWorkMetricsError["code"],
): void {
  let observed: unknown;
  try {
    operation();
  } catch (cause) {
    observed = cause;
  }
  expect(observed).toBeInstanceOf(CppCuteBrowserFrontendWorkMetricsError);
  expect(observed).toMatchObject({ code });
}
