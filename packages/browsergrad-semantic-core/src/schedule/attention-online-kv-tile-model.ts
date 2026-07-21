import type { JsonObject } from "../schema/json.js";
import type { WireU64 } from "../schema/integers.js";

export const INITIAL_ATTENTION_ONLINE_KV_MAX_TILE_ROWS = 256n;

export interface AttentionOnlineKvPhysicalTile extends JsonObject {
  readonly queryRows: WireU64;
  readonly keyRows: WireU64;
}

export interface AttentionOnlineKvWorkgroupBarrier extends JsonObject {
  readonly scope: "workgroup";
  readonly memory: "workgroup";
  readonly semantics: "acquire-release";
}

/**
 * Backend-neutral physical requirements for realizing one verified
 * attention-forward operation with tiled online softmax. Logical view, mask,
 * dtype, scale, numerical, effect, and autodiff meaning remain exclusively in
 * the referenced kernel artifact.
 */
export type AttentionOnlineKvTileSchedule = JsonObject & {
  readonly kind: "attention-online-kv-tile-schedule";
  readonly version: { readonly major: 1; readonly minor: 0 };
  readonly physicalTile: AttentionOnlineKvPhysicalTile;
  readonly workgroup: {
    readonly size: { readonly x: WireU64; readonly y: WireU64; readonly z: WireU64 };
    readonly dispatchX: "query-tile";
    readonly dispatchY: "head";
    readonly dispatchZ: "batch";
  };
  readonly invocation: {
    readonly localX: "query-row-within-tile";
    readonly localY: "unused";
    readonly localZ: "unused";
    readonly privateQuery: "one-logical-query-row";
    readonly privateOutput: "one-logical-output-row";
  };
  readonly traversal: {
    readonly keyTiles: "increasing-key-index";
    readonly keysWithinTile: "increasing-key-index";
    readonly coverage: "complete-logical-key-range";
    readonly tail: "masked-final-tile";
  };
  readonly staging: {
    readonly space: "workgroup";
    readonly key: "cooperative";
    readonly value: "cooperative";
    readonly layout: "key-major-contiguous-depth";
    readonly buffering: "single";
  };
  readonly onlineSoftmax: {
    readonly state: "running-maximum-denominator-and-weighted-value";
    readonly tileScores: "scaled-query-key-dot-products";
    readonly tileMaximum: "maximum-over-valid-tile-scores";
    readonly tileReductionOrder: "increasing-key-index";
    readonly update: "rescale-prior-state-then-accumulate-current-tile";
    readonly priorRescale: "exp-previous-maximum-minus-new-maximum";
    readonly currentWeight: "exp-score-minus-new-maximum";
    readonly finalize: "divide-weighted-value-by-denominator-after-all-key-tiles";
  };
  readonly participation: {
    readonly workgroup: "all-invocations";
    readonly boundaryQueryLanes: "participate";
    readonly earlyExit: "forbidden";
  };
  readonly uniformity: {
    readonly barrierControl: "workgroup-uniform";
    readonly activeMaskScope: "memory-effects-and-online-state-only";
  };
  readonly vectorization: {
    readonly keyLoad: WireU64;
    readonly valueLoad: WireU64;
    readonly destinationStore: WireU64;
  };
  readonly barriers: {
    readonly afterCooperativeLoad: AttentionOnlineKvWorkgroupBarrier;
    readonly beforeStagingReuse: AttentionOnlineKvWorkgroupBarrier;
  };
  readonly masks: {
    readonly queryLane: "suppress-logical-state-and-store";
    readonly keyLoad: "zero-fill";
    readonly valueLoad: "zero-fill";
    readonly invalidKeyScore: "exclude-before-online-state-update";
    readonly logicalMask: "exclude-before-online-state-update";
    readonly destinationStore: "suppress";
  };
};
