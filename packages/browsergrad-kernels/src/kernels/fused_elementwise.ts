/**
 * Generic fused-elementwise kernel — runtime WGSL codegen.
 *
 * Input: the same `ops` list shape the Python realizer hands us:
 *   ops: Array<[opcode: string, lhs_ref: number, rhs_ref: number]>
 *   Negative ref → external input N (via `-ref - 1` for 0-indexed).
 *   Non-negative ref → output of step N.
 *
 * Output: a single WGSL compute shader that:
 *   1. Reads the i-th element from each input buffer into a local.
 *   2. Walks the ops list, emitting `let stepK: f32 = <expr>;` per op.
 *   3. Writes the last step's value to output[i].
 *
 * Pipeline cache key is the hash of the ops list. Two graphs with the
 * same ops structure compile the WGSL once and reuse it.
 *
 * Carmack-shaped design — we own the codegen. No template engine, no
 * runtime parser. WGSL is a string we assemble in TS. The "compiler"
 * is a 40-line function that walks the ops list and string-concats
 * expressions. The browser handles the actual SPIR-V emission via its
 * built-in WGSL parser.
 *
 * Why no broadcasting in v0:
 *   The Python fusion pass (_fusion.py) only builds FUSED_ELEMENTWISE
 *   from a chain whose terminals all share the same shape (broadcasting
 *   is materialized as BROADCAST_TO UOps upstream). So at the realizer
 *   layer, every input to FUSED_ELEMENTWISE has shape == output shape.
 *   This kernel relies on that invariant — it just walks N elements
 *   linearly, no per-input stride logic.
 *
 * What this is NOT:
 *   - WMMA / subgroup-accelerated. Pure scalar f32 in v0.
 *   - Multi-output. One output buffer per fused kernel; if the IR ever
 *     fuses ops with two terminal sinks (e.g. shared softmax), we'd
 *     need a different opcode (`OP_FUSED_MULTI_*`) and a different
 *     codegen.
 *   - Reductive. Sum/max inside a fused chain stay as separate REDUCE
 *     UOps; the chain is strictly elementwise.
 */

import {
  runDirect,
  type KernelDescriptor,
  type DirectDispatchResult,
} from "../runner.js";
import { KernelError, type KernelDevice } from "../types.js";

/** Op signature: opcode string + lhs/rhs refs (rhs may be unused for unary ops). */
export type FusedOp = readonly [string, number, number];

/** Stable hash of an ops list — deterministic, browser-portable.
 *  Uses an FNV-1a-shaped fold so we don't drag in SubtleCrypto for a 32-bit
 *  hash. Collision probability is negligible across the op-sequences we
 *  realistically see (max ~10 ops/chain × ~20 opcodes = small space). */
function hashOps(ops: readonly FusedOp[], numInputs: number): string {
  let h = 0x811c9dc5;
  const mix = (v: number): void => {
    h = (h ^ v) >>> 0;
    h = Math.imul(h, 0x01000193) >>> 0;
  };
  mix(numInputs);
  for (const [opcode, lhs, rhs] of ops) {
    for (let i = 0; i < opcode.length; i++) mix(opcode.charCodeAt(i));
    mix(lhs + 0x10000); // shift so negatives don't collide with zero
    mix(rhs + 0x10000);
  }
  return h.toString(16).padStart(8, "0");
}

/** Resolve a ref into a WGSL identifier:
 *    -1 → in0[i], -2 → in1[i], 0 → step0, 1 → step1, ... */
function refExpr(ref: number): string {
  if (ref < 0) {
    const idx = -ref - 1;
    return `in${idx}[i]`;
  }
  return `step${ref}`;
}

/** Map opcode → WGSL expression template. The expression uses `LHS` and
 *  `RHS` placeholders that get string-replaced with the resolved refs. */
const OPCODE_TO_WGSL: Record<string, (lhs: string, rhs: string) => string> = {
  ADD: (lhs, rhs) => `${lhs} + ${rhs}`,
  MUL: (lhs, rhs) => `${lhs} * ${rhs}`,
  DIV: (lhs, rhs) => `${lhs} / ${rhs}`,
  NEG: (lhs) => `-${lhs}`,
  EXP: (lhs) => `exp(${lhs})`,
  LOG: (lhs) => `log(${lhs})`,
};

/** Generate the WGSL source for a fused-elementwise chain. */
export function generateFusedWgsl(
  ops: readonly FusedOp[],
  numInputs: number,
): string {
  if (numInputs < 1 || numInputs > 8) {
    throw new KernelError(
      `fused_elementwise: numInputs=${numInputs} outside supported range [1, 8]. ` +
        `WGSL bind group has 9 slots reserved for inputs + 1 output + 1 uniform; ` +
        `extending requires a different shader template.`,
    );
  }
  if (ops.length < 1 || ops.length > 32) {
    throw new KernelError(
      `fused_elementwise: ops.length=${ops.length} outside supported range [1, 32]. ` +
        `Long chains should be split — single-shader register pressure becomes ` +
        `the limiting factor past ~32 ops.`,
    );
  }

  // Storage bindings: one per input + one output.
  const bindings: string[] = [];
  for (let i = 0; i < numInputs; i++) {
    bindings.push(`@group(0) @binding(${i}) var<storage, read> in${i}: array<f32>;`);
  }
  bindings.push(`@group(0) @binding(${numInputs}) var<storage, read_write> out: array<f32>;`);
  bindings.push(`
struct Params { N: u32, _pad0: u32, _pad1: u32, _pad2: u32, };
@group(0) @binding(${numInputs + 1}) var<uniform> params: Params;`);

  // Step expressions.
  const steps: string[] = [];
  for (let i = 0; i < ops.length; i++) {
    const [opcode, lhsRef, rhsRef] = ops[i]!;
    const builder = OPCODE_TO_WGSL[opcode];
    if (builder === undefined) {
      throw new KernelError(
        `fused_elementwise: opcode ${JSON.stringify(opcode)} has no WGSL builder. ` +
          `Supported: ${Object.keys(OPCODE_TO_WGSL).join(", ")}.`,
      );
    }
    const lhs = refExpr(lhsRef);
    const rhs = refExpr(rhsRef);
    steps.push(`  let step${i}: f32 = ${builder(lhs, rhs)};`);
  }

  // Terminal output: last step.
  const lastStep = `step${ops.length - 1}`;

  return `
${bindings.join("\n")}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.N) { return; }
${steps.join("\n")}
  out[i] = ${lastStep};
}
`;
}

/** Direct-dispatch fused elementwise kernel. Inputs and output are GPUBuffers.
 *  Caller owns the output buffer's lifetime. */
export function fusedElementwiseDirect(
  device: KernelDevice,
  inputs: readonly GPUBuffer[],
  ops: readonly FusedOp[],
  outputLength: number,
): DirectDispatchResult {
  const wgsl = generateFusedWgsl(ops, inputs.length);
  const hash = hashOps(ops, inputs.length);
  const desc: KernelDescriptor = {
    name: `fused_elementwise_${hash}`,
    wgsl,
    workgroupSize: [64, 1, 1],
  };
  const params = new Uint32Array([outputLength, 0, 0, 0]);
  return runDirect(device, desc, {
    inputBuffers: inputs,
    outputLength,
    params,
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: hash,
  });
}
