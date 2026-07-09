import { CUDA_ADDRESS_SPACE_PREDICATE_CALL_NAMES } from "./cuda_pointer_calls.js";
import { CUDA_SUBGROUP_CALL_NAMES } from "./cuda_subgroup_calls.js";

export const SEMANTIC_LOCAL_ARRAY_FILL_CALLS = new Set(["fill_1D_regs", "fill_2D_regs", "fill_3D_regs"]);

export const SEMANTIC_NOOP_CALLS = new Set([
  "__nanosleep",
  "__prof_trigger",
  "__trap",
]);

export const SEMANTIC_ADDRESS_PREDICATE_CALLS: ReadonlySet<string> = new Set(CUDA_ADDRESS_SPACE_PREDICATE_CALL_NAMES);

export const SEMANTIC_SUBGROUP_CALLS: ReadonlySet<string> = new Set(CUDA_SUBGROUP_CALL_NAMES);
