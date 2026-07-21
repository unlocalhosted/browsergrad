import { performance } from "node:perf_hooks";

/**
 * Owns non-authoritative timing state outside canonical execution evidence.
 * Each registry has an independent WeakMap authority, so another importer
 * cannot mint an observation accepted by the executor's registry.
 *
 * @param {(path: string, message: string) => never} unverified
 */
export function createCppCuteClangWasmBuildTimingRegistry(unverified) {
  const builds = new WeakMap();
  const observations = new WeakMap();

  const storedBuild = (build) => {
    const stored = builds.get(build);
    if (stored === undefined) throw new Error("internal: expected active Clang-Wasm build timing");
    return stored;
  };

  return Object.freeze({
    beginBuild() {
      const build = Object.freeze({});
      builds.set(build, { startedAt: performance.now(), phases: [] });
      return build;
    },

    beginPhase() {
      return performance.now();
    },

    /**
     * @param {object} build
     * @param {Readonly<{ id: string; stageId: "native-tablegen" | "clang-extractor-wasm"; kind: "configure" | "build" }>} step
     * @param {number} startedAt
     */
    finishPhase(build, step, startedAt) {
      storedBuild(build).phases.push(Object.freeze({
        id: step.id,
        stageId: step.stageId,
        kind: step.kind,
        durationMs: performance.now() - startedAt,
      }));
    },

    /** @param {object} build @param {object} executed */
    attach(build, executed) {
      const stored = storedBuild(build);
      builds.delete(build);
      observations.set(executed, Object.freeze({
        clock: "monotonic-performance-now",
        unit: "milliseconds",
        phases: Object.freeze(stored.phases),
        totalDurationMs: performance.now() - stored.startedAt,
      }));
    },

    /** @param {object} executed */
    observe(executed) {
      const timing = observations.get(executed);
      if (timing === undefined) {
        unverified("$executed", "expected executor-issued successful build timing observation");
      }
      return timing;
    },
  });
}
