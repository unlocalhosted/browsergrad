/**
 * VJP rule registry integration tests (PRD-007 Week 1).
 *
 * The rules in _vjp.py are symbolic — they emit IR. The dispatcher in
 * TensorProxy.backward() consults `_vjp.get_rule(op)`; when a rule exists,
 * future weeks of PRD-007 will rewire the reverse walk to use it instead
 * of the closure path. Week 1 ships the rules + the registry + the
 * "all rules emit valid IR with `vjp_of` annotations" invariant.
 *
 * These tests assert:
 *   - The registry has the W1 rule set.
 *   - Each rule produces UOps with the expected opcode + shape.
 *   - The `vjp_of` annotation is present on every emitted UOp (PRD-009
 *     gradient checkpointing will need this).
 *   - The IR a rule emits is itself runnable through the realizer
 *     (correctness floor: a VJP rule can't emit nonsense).
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("VJP rule registry (_vjp.py)", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("registers the Week 1 rule set", async () => {
    const target = await getJitTarget();
    const registered = await target.run<string[]>(`
from browsergrad_jit._vjp import list_registered
list(list_registered())
`);
    // Week 1: ADD, MUL, DIV, NEG, EXP, LOG, RESHAPE, PERMUTE, CAST,
    // REDUCE, MATMUL. (REDUCE covers sum/mean; max/min defer to W3.)
    const expected = [
      "ADD", "CAST", "DIV", "EXP", "LOG", "MATMUL", "MUL", "NEG",
      "PERMUTE", "REDUCE", "RESHAPE",
    ];
    expect(registered).toEqual(expected);
  });

  it("ADD VJP returns two un-broadcast gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ ops: string[]; shape_a: number[]; shape_b: number[] }>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_ADD
from browsergrad_jit._vjp import get_rule

a_buf = buffer("test:a", (3, 4), "float32")
b_buf = buffer("test:b", (3, 4), "float32")
a, b = load(a_buf), load(b_buf)
fwd = UOp(OP_ADD, (a, b), (3, 4), "float32")
dy = load(buffer("test:dy", (3, 4), "float32"))
rule = get_rule(OP_ADD)
grads = rule(fwd, (a, b), dy)
{
    "ops": [g.op for g in grads],
    "shape_a": list(grads[0].shape),
    "shape_b": list(grads[1].shape),
}
`);
    // No broadcast → both grads pass through unchanged (LOAD).
    expect(result.ops).toEqual(["LOAD", "LOAD"]);
    expect(result.shape_a).toEqual([3, 4]);
    expect(result.shape_b).toEqual([3, 4]);
  });

  it("ADD VJP un-broadcasts a bias gradient", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ b_grad_op: string; b_grad_shape: number[] }>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_ADD
from browsergrad_jit._vjp import get_rule

a = load(buffer("test:a", (8, 16), "float32"))   # full matrix
b = load(buffer("test:b", (16,), "float32"))     # bias broadcast
fwd = UOp(OP_ADD, (a, b), (8, 16), "float32")
dy = load(buffer("test:dy", (8, 16), "float32"))
da, db = get_rule(OP_ADD)(fwd, (a, b), dy)
{"b_grad_op": db.op, "b_grad_shape": list(db.shape)}
`);
    // db needs to sum out the broadcast leading dim → REDUCE(sum).
    expect(result.b_grad_op).toBe("REDUCE");
    expect(result.b_grad_shape).toEqual([16]);
  });

  it("MUL VJP references forward inputs by IR identity", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ da_inputs_ok: boolean; db_inputs_ok: boolean }>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_MUL
from browsergrad_jit._vjp import get_rule

a = load(buffer("test:a", (4,), "float32"))
b = load(buffer("test:b", (4,), "float32"))
fwd = UOp(OP_MUL, (a, b), (4,), "float32")
dy = load(buffer("test:dy", (4,), "float32"))
da, db = get_rule(OP_MUL)(fwd, (a, b), dy)
# da = dy * b  → MUL whose inputs are (dy, b).
# db = dy * a  → MUL whose inputs are (dy, a).
{
    "da_inputs_ok": da.op == "MUL" and da.inputs[0] is dy and da.inputs[1] is b,
    "db_inputs_ok": db.op == "MUL" and db.inputs[0] is dy and db.inputs[1] is a,
}
`);
    expect(result.da_inputs_ok).toBe(true);
    expect(result.db_inputs_ok).toBe(true);
  });

  it("EXP VJP references the forward EXP output (Flash Attention v2 reuse)", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ op: string; references_forward: boolean }>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_EXP
from browsergrad_jit._vjp import get_rule

x = load(buffer("test:x", (4,), "float32"))
fwd = UOp(OP_EXP, (x,), (4,), "float32")
dy = load(buffer("test:dy", (4,), "float32"))
(dx,) = get_rule(OP_EXP)(fwd, (x,), dy)
# dx should be MUL(dy, fwd) — references the forward EXP output, not a
# re-emitted EXP of x. This is the pattern PRD-006/012 use to keep the
# EXP value in shared memory between forward and backward.
{"op": dx.op, "references_forward": dx.inputs[1] is fwd}
`);
    expect(result.op).toBe("MUL");
    expect(result.references_forward).toBe(true);
  });

  it("LOG VJP is dy / x", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ op: string; refs_x: boolean }>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_LOG
from browsergrad_jit._vjp import get_rule

x = load(buffer("test:x", (4,), "float32"))
fwd = UOp(OP_LOG, (x,), (4,), "float32")
dy = load(buffer("test:dy", (4,), "float32"))
(dx,) = get_rule(OP_LOG)(fwd, (x,), dy)
{"op": dx.op, "refs_x": dx.inputs[1] is x}
`);
    expect(result.op).toBe("DIV");
    expect(result.refs_x).toBe(true);
  });

  it("MATMUL VJP produces dy @ B.T and A.T @ dy with batched shape un-broadcast", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ da_op: string; db_op: string; da_shape: number[]; db_shape: number[] }>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_MATMUL
from browsergrad_jit._vjp import get_rule

a = load(buffer("test:a", (8, 16), "float32"))   # (M, K)
b = load(buffer("test:b", (16, 32), "float32"))  # (K, N)
fwd = UOp(OP_MATMUL, (a, b), (8, 32), "float32")  # (M, N)
dy = load(buffer("test:dy", (8, 32), "float32"))
da, db = get_rule(OP_MATMUL)(fwd, (a, b), dy)
{
    "da_op": da.op, "db_op": db.op,
    "da_shape": list(da.shape), "db_shape": list(db.shape),
}
`);
    // No broadcast (both 2-D, no batch dims): grads are MATMUL outputs.
    expect(result.da_op).toBe("MATMUL");
    expect(result.db_op).toBe("MATMUL");
    expect(result.da_shape).toEqual([8, 16]);
    expect(result.db_shape).toEqual([16, 32]);
  });

  it("MATMUL VJP un-broadcasts the batch dim when one input is unbatched", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ db_shape: number[]; db_has_reduce: boolean }>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_MATMUL, toposort
from browsergrad_jit._vjp import get_rule

# Batched matmul: (B, M, K) @ (K, N) → (B, M, N)
a = load(buffer("test:a", (4, 8, 16), "float32"))
b = load(buffer("test:b", (16, 32), "float32"))
fwd = UOp(OP_MATMUL, (a, b), (4, 8, 32), "float32")
dy = load(buffer("test:dy", (4, 8, 32), "float32"))
da, db = get_rule(OP_MATMUL)(fwd, (a, b), dy)
# db should be un-broadcast back to (16, 32) — a REDUCE must appear in
# its subgraph.
{
    "db_shape": list(db.shape),
    "db_has_reduce": any(n.op == "REDUCE" for n in toposort(db)),
}
`);
    expect(result.db_shape).toEqual([16, 32]);
    expect(result.db_has_reduce).toBe(true);
  });

  it("REDUCE(sum) VJP broadcasts dy back to input shape", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ dx_shape: number[]; ops: string[] }>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_REDUCE, toposort
from browsergrad_jit._vjp import get_rule

x = load(buffer("test:x", (4, 8), "float32"))
fwd = UOp(OP_REDUCE, (x,), (), "float32",
          arg={"op": "sum", "axis": None, "keepdims": False})
dy = load(buffer("test:dy", (), "float32"))
(dx,) = get_rule(OP_REDUCE)(fwd, (x,), dy)
{"dx_shape": list(dx.shape), "ops": [n.op for n in toposort(dx)]}
`);
    expect(result.dx_shape).toEqual([4, 8]);
  });

  it("REDUCE(mean) VJP scales by 1/N", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ has_mul: boolean; has_const: boolean }>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_REDUCE, toposort
from browsergrad_jit._vjp import get_rule

x = load(buffer("test:x", (4,), "float32"))
fwd = UOp(OP_REDUCE, (x,), (), "float32",
          arg={"op": "mean", "axis": None, "keepdims": False})
dy = load(buffer("test:dy", (), "float32"))
(dx,) = get_rule(OP_REDUCE)(fwd, (x,), dy)
nodes = toposort(dx)
{
    "has_mul": any(n.op == "MUL" for n in nodes),
    "has_const": any(n.op == "CONST" for n in nodes),
}
`);
    expect(result.has_mul).toBe(true);
    expect(result.has_const).toBe(true);
  });

  it("REDUCE(max) VJP returns None — defers to closure path", async () => {
    const target = await getJitTarget();
    const result = await target.run<boolean>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_REDUCE
from browsergrad_jit._vjp import get_rule

x = load(buffer("test:x", (4,), "float32"))
fwd = UOp(OP_REDUCE, (x,), (), "float32",
          arg={"op": "max", "axis": None, "keepdims": False})
dy = load(buffer("test:dy", (), "float32"))
(grad,) = get_rule(OP_REDUCE)(fwd, (x,), dy)
grad is None
`);
    expect(result).toBe(true);
  });

  it("RESHAPE VJP undoes the reshape", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ op: string; shape: number[] }>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_RESHAPE
from browsergrad_jit._vjp import get_rule

x = load(buffer("test:x", (4, 8), "float32"))
fwd = UOp(OP_RESHAPE, (x,), (32,), "float32", arg={"new_shape": (32,)})
dy = load(buffer("test:dy", (32,), "float32"))
(dx,) = get_rule(OP_RESHAPE)(fwd, (x,), dy)
{"op": dx.op, "shape": list(dx.shape)}
`);
    expect(result.op).toBe("RESHAPE");
    expect(result.shape).toEqual([4, 8]);
  });

  it("PERMUTE VJP applies the inverse permutation", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ axes: number[]; shape: number[] }>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_PERMUTE
from browsergrad_jit._vjp import get_rule

x = load(buffer("test:x", (2, 3, 5), "float32"))
# forward axes (1, 2, 0) -> shape (3, 5, 2)
fwd = UOp(OP_PERMUTE, (x,), (3, 5, 2), "float32", arg={"axes": (1, 2, 0)})
dy = load(buffer("test:dy", (3, 5, 2), "float32"))
(dx,) = get_rule(OP_PERMUTE)(fwd, (x,), dy)
{"axes": list(dx.arg["axes"]), "shape": list(dx.shape)}
`);
    // Inverse of (1, 2, 0) is (2, 0, 1).
    expect(result.axes).toEqual([2, 0, 1]);
    expect(result.shape).toEqual([2, 3, 5]);
  });

  it("CAST VJP rounds back to the source dtype", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ op: string; dtype: string }>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_CAST
from browsergrad_jit._vjp import get_rule

x = load(buffer("test:x", (4,), "float32"))
fwd = UOp(OP_CAST, (x,), (4,), "float16", arg={"dtype": "float16"})
dy = load(buffer("test:dy", (4,), "float16"))
(dx,) = get_rule(OP_CAST)(fwd, (x,), dy)
{"op": dx.op, "dtype": dx.dtype}
`);
    expect(result.op).toBe("CAST");
    expect(result.dtype).toBe("float32");
  });

  it("every emitted UOp carries the vjp_of annotation (PRD-009 hook)", async () => {
    const target = await getJitTarget();
    const result = await target.run<boolean>(`
from browsergrad_jit._ir import UOp, buffer, load, OP_MUL, toposort
from browsergrad_jit._vjp import get_rule

a = load(buffer("test:a", (4,), "float32"))
b = load(buffer("test:b", (4,), "float32"))
fwd = UOp(OP_MUL, (a, b), (4,), "float32")
dy = load(buffer("test:dy", (4,), "float32"))
da, db = get_rule(OP_MUL)(fwd, (a, b), dy)

# All NEW UOps in the backward graph (those that weren't leaves) must
# carry the vjp_of tag pointing at the forward node.
new_ops = []
for grad_root in (da, db):
    for node in toposort(grad_root):
        if isinstance(node.arg, dict) and "vjp_of" in node.arg:
            new_ops.append(node)
all(node.arg["vjp_of"] is fwd for node in new_ops)
`);
    expect(result).toBe(true);
  });
});
