import type {
  HostGraphCollectiveDType,
  HostGraphCollectiveReduction,
} from "@unlocalhosted/browsergrad-semantic-core/graph";
import {
  hashNamedComponents,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  defineWgslKernelProgram,
  type WgslKernelProgram,
} from "./wgsl_program.js";

export interface PreparedHostGraphCollectiveWgsl {
  readonly program: WgslKernelProgram;
  readonly moduleHash: string;
  readonly usesNumericalStatus: boolean;
}

export interface PreparedHostGraphReplicationWgsl {
  readonly program: WgslKernelProgram;
  readonly moduleHash: string;
}

export async function prepareHostGraphCollectiveWgsl(
  dtype: HostGraphCollectiveDType,
  reduction: HostGraphCollectiveReduction,
  workgroupSize: number,
): Promise<PreparedHostGraphCollectiveWgsl> {
  const source = collectiveSource(dtype, reduction, workgroupSize);
  const moduleHash = await hashNamedComponents({
    profile: "browsergrad.host-graph.webgpu-collective@1",
    dtype,
    reduction,
    workgroupSize,
    source,
  });
  const usesNumericalStatus = dtype === "f32";
  const program = freezeProgram(defineWgslKernelProgram({
    name: `bg_host_graph_reduce_${dtype}_${reduction}_${moduleHash}`,
    wgsl: source,
    bindings: [
      {
        kind: "storage",
        name: "accumulator",
        valueType: dtype,
        access: "read_write",
        binding: 0,
      },
      {
        kind: "storage",
        name: "operand",
        valueType: dtype,
        access: "read",
        binding: 1,
      },
      ...(usesNumericalStatus
        ? [{
            kind: "storage" as const,
            name: "numerical_status",
            valueType: "u32" as const,
            access: "read_write" as const,
            binding: 2,
          }]
        : []),
    ],
    workgroupSize: [workgroupSize, 1, 1],
  }));
  return Object.freeze({ program, moduleHash, usesNumericalStatus });
}

export async function prepareHostGraphReplicationWgsl(
  workgroupSize: number,
): Promise<PreparedHostGraphReplicationWgsl> {
  const source = replicationSource(workgroupSize);
  const moduleHash = await hashNamedComponents({
    profile: "browsergrad.host-graph.webgpu-replication@1",
    workgroupSize,
    source,
  });
  return Object.freeze({
    program: freezeProgram(defineWgslKernelProgram({
      name: `bg_host_graph_replicate_${moduleHash}`,
      wgsl: source,
      bindings: [
        {
          kind: "storage",
          name: "source_words",
          valueType: "u32",
          access: "read",
          binding: 0,
        },
        {
          kind: "storage",
          name: "destination_words",
          valueType: "u32",
          access: "read_write",
          binding: 1,
        },
      ],
      workgroupSize: [workgroupSize, 1, 1],
    })),
    moduleHash,
  });
}

function collectiveSource(
  dtype: HostGraphCollectiveDType,
  reduction: HostGraphCollectiveReduction,
  workgroupSize: number,
): string {
  const expression = reductionExpression(dtype, reduction);
  const status = dtype === "f32"
    ? `
struct NumericalStatus {
  value: atomic<u32>,
}

@group(0) @binding(2)
var<storage, read_write> numerical_status: NumericalStatus;

fn bg_is_finite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
${f32ReductionHelpers(reduction)}
`
    : "";
  const finiteGuard = dtype === "f32"
    ? `
  if (!bg_is_finite(left) || !bg_is_finite(right)) {
    atomicStore(&numerical_status.value, 1u);
    return;
  }
`
    : "";
  const resultGuard = dtype === "f32"
    ? `
  if (!bg_is_finite(result)) {
    atomicStore(&numerical_status.value, 1u);
    return;
  }
`
    : "";
  return `@group(0) @binding(0)
var<storage, read_write> accumulator: array<${dtype}>;

@group(0) @binding(1)
var<storage, read> operand: array<${dtype}>;
${status}
@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index = global_id.x;
  if (index >= arrayLength(&accumulator)) {
    return;
  }
  let left = accumulator[index];
  let right = operand[index];${finiteGuard}
  let result = ${expression};${resultGuard}
  accumulator[index] = result;
}
`;
}

function reductionExpression(
  dtype: HostGraphCollectiveDType,
  reduction: HostGraphCollectiveReduction,
): string {
  if (dtype === "f32" && reduction === "min") {
    return "bg_f32_min(left, right)";
  }
  if (dtype === "f32" && reduction === "max") {
    return "bg_f32_max(left, right)";
  }
  if (reduction === "min") return "min(left, right)";
  if (reduction === "max") return "max(left, right)";
  if (dtype === "i32") {
    return "bitcast<i32>(bitcast<u32>(left) + bitcast<u32>(right))";
  }
  return "left + right";
}

function f32ReductionHelpers(
  reduction: HostGraphCollectiveReduction,
): string {
  if (reduction === "min") {
    return `
fn bg_f32_min(left: f32, right: f32) -> f32 {
  if (left < right) {
    return left;
  }
  if (right < left) {
    return right;
  }
  return bitcast<f32>(bitcast<u32>(left) | bitcast<u32>(right));
}
`;
  }
  if (reduction === "max") {
    return `
fn bg_f32_max(left: f32, right: f32) -> f32 {
  if (left > right) {
    return left;
  }
  if (right > left) {
    return right;
  }
  return bitcast<f32>(bitcast<u32>(left) & bitcast<u32>(right));
}
`;
  }
  return "";
}

function replicationSource(workgroupSize: number): string {
  return `@group(0) @binding(0)
var<storage, read> source_words: array<u32>;

@group(0) @binding(1)
var<storage, read_write> destination_words: array<u32>;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index = global_id.x;
  if (index >= arrayLength(&source_words)) {
    return;
  }
  destination_words[index] = source_words[index];
}
`;
}

function freezeProgram(program: WgslKernelProgram): WgslKernelProgram {
  return Object.freeze({
    ...program,
    bindings: Object.freeze(
      program.bindings.map((binding) => Object.freeze({ ...binding })),
    ),
    workgroupSize: Object.freeze([...program.workgroupSize]) as unknown as
      readonly [number, number, number],
  });
}
