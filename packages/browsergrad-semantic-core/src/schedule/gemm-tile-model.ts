import type { JsonObject } from "../schema/json.js";
import type { WireU64 } from "../schema/integers.js";

export interface LogicalGemmPhysicalTile extends JsonObject {
  readonly m: WireU64;
  readonly n: WireU64;
  readonly k: WireU64;
}

/**
 * Backend-neutral physical requirements for realizing one verified logical
 * GEMM tile. Logical views, effects, numerical policy, and operation meaning
 * remain exclusively in the referenced kernel artifact.
 */
export type LogicalGemmTileSchedule = JsonObject & {
  readonly kind: "logical-gemm-tile-schedule";
  readonly version: { readonly major: 1; readonly minor: 0 };
  readonly physicalTile: LogicalGemmPhysicalTile;
  readonly workgroup: {
    readonly size: { readonly x: WireU64; readonly y: WireU64; readonly z: WireU64 };
    readonly x: "physical-tile-column";
    readonly y: "physical-tile-row";
    readonly z: "singleton";
  };
  readonly invocation: {
    readonly output: "one-element";
    readonly localX: "output-column";
    readonly localY: "output-row";
    readonly localZ: "unused";
  };
  readonly staging: {
    readonly space: "workgroup";
    readonly lhs: "cooperative";
    readonly rhs: "cooperative";
    readonly buffering: "single";
  };
  readonly participation: {
    readonly workgroup: "all-invocations";
    readonly boundaryLanes: "participate";
    readonly earlyExit: "forbidden";
  };
  readonly uniformity: {
    readonly barrierControl: "workgroup-uniform";
    readonly activeMaskScope: "memory-effects-only";
  };
  readonly vectorization: {
    readonly lhsLoad: WireU64;
    readonly rhsLoad: WireU64;
    readonly destinationStore: WireU64;
  };
  readonly barriers: {
    readonly afterCooperativeLoad: LogicalGemmWorkgroupBarrier;
    readonly beforeStagingReuse: LogicalGemmWorkgroupBarrier;
  };
  readonly masks: {
    readonly lhsLoad: "zero-fill";
    readonly rhsLoad: "zero-fill";
    readonly destinationStore: "suppress";
  };
};

export interface LogicalGemmWorkgroupBarrier extends JsonObject {
  readonly scope: "workgroup";
  readonly memory: "workgroup";
  readonly semantics: "acquire-release";
}
