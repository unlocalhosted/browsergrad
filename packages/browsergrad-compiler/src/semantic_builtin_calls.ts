import { CUDA_ADDRESS_SPACE_PREDICATE_CALL_NAMES } from "./cuda_pointer_calls.js";

export const SEMANTIC_LOCAL_ARRAY_FILL_CALLS = new Set(["fill_1D_regs", "fill_2D_regs", "fill_3D_regs"]);

export const SEMANTIC_NOOP_CALLS = new Set([
  "__nanosleep",
  "__prof_trigger",
  "__trap",
]);

export const SEMANTIC_ADDRESS_PREDICATE_CALLS: ReadonlySet<string> = new Set(CUDA_ADDRESS_SPACE_PREDICATE_CALL_NAMES);

export const SEMANTIC_SUBGROUP_CALLS = new Set([
  "__activemask",
  "__any",
  "__all",
  "__ballot",
  "__any_sync",
  "__all_sync",
  "__ballot_sync",
  "__match_any_sync",
  "__reduce_add_sync",
  "__reduce_min_sync",
  "__reduce_max_sync",
  "__reduce_and_sync",
  "__reduce_or_sync",
  "__reduce_xor_sync",
  "__shfl",
  "__shfl_down",
  "__shfl_up",
  "__shfl_xor",
  "__shfl_sync",
  "__shfl_down_sync",
  "__shfl_up_sync",
  "__shfl_xor_sync",
]);
