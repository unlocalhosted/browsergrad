import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_EINSUM_CONFORMANCE } from "../../../test-support/framework-einsum-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed torch.einsum contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("matches shared general contractions, gradients, dtypes, aliases, and refusals", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_EINSUM_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})

def tensor(shape, values, dtype="float32", requires_grad=False):
    return bg.from_numpy(
        np.asarray(values, dtype=np.dtype(dtype)).reshape(tuple(shape)),
        requires_grad=requires_grad,
    )

matmul = fixture["matmul"]
left = tensor(matmul["leftShape"], matmul["leftValues"], requires_grad=True)
right = tensor(matmul["rightShape"], matmul["rightValues"], requires_grad=True)

einsum_calls = 0
original_einsum = np.einsum
def forbidden_einsum(*args, **kwargs):
    global einsum_calls
    einsum_calls += 1
    raise RuntimeError("construction executed NumPy einsum")
np.einsum = forbidden_einsum
output = bg.einsum(matmul["equation"], left, right)
np.einsum = original_einsum
output_values = output.numpy()
output.sum().backward()

implicit = bg.einsum(
    matmul["implicitEquation"],
    tensor(matmul["leftShape"], matmul["leftValues"]),
    tensor(matmul["rightShape"], matmul["rightValues"]),
)

three = fixture["threeOperand"]
three_inputs = tuple(
    tensor(shape, values, requires_grad=True)
    for shape, values in zip(three["shapes"], three["values"])
)
three_output = bg.einsum(three["equation"], *three_inputs)
three_output.sum().backward()

broadcast = fixture["broadcast"]
broadcast_inputs = tuple(
    tensor(shape, values, requires_grad=True)
    for shape, values in zip(broadcast["shapes"], broadcast["values"])
)
broadcast_output = bg.einsum(broadcast["equation"], *broadcast_inputs)
broadcast_output.sum().backward()

ellipsis = fixture["ellipsis"]
ellipsis_output = bg.einsum(
    ellipsis["equation"],
    *(tensor(shape, values) for shape, values in zip(ellipsis["shapes"], ellipsis["values"])),
)
reduction_input = tensor(
    ellipsis["reductionInputShape"],
    ellipsis["reductionInputValues"],
    requires_grad=True,
)
reduction_output = bg.einsum(ellipsis["reductionEquation"], reduction_input)
reduction_output.sum().backward()

diagonal = fixture["diagonal"]
diagonal_input = tensor(
    diagonal["inputShape"], diagonal["inputValues"], requires_grad=True
)
diagonal_output = bg.einsum(diagonal["equation"], diagonal_input)
diagonal_output.sum().backward()
trace = bg.einsum(
    diagonal["traceEquation"],
    tensor(diagonal["inputShape"], diagonal["inputValues"]),
)

scalar = fixture["scalar"]
scalar_inputs = tuple(
    tensor([], [value], requires_grad=True) for value in scalar["values"]
)
scalar_output = bg.einsum(scalar["equation"], *scalar_inputs)
scalar_output.backward()

uppercase = fixture["uppercaseImplicit"]
uppercase_output = bg.einsum(
    uppercase["equation"],
    tensor(uppercase["inputShape"], uppercase["inputValues"]),
)

empty = fixture["emptyBroadcast"]
empty_output = bg.einsum(
    empty["equation"],
    *(tensor(shape, []) if shape == [0] else tensor(shape, [1]) for shape in empty["shapes"]),
)

dtypes = []
half_gradient_dtypes = []
for case in fixture["dtypeCases"]:
    left_dtype = case["leftDtype"]
    right_dtype = case["rightDtype"]
    differentiable = left_dtype == "float16" and right_dtype == "float16"
    typed_left = tensor([2], [1, 2], left_dtype, differentiable)
    typed_right = tensor([2], [3, 4], right_dtype, differentiable)
    typed_output = bg.einsum("i,i->", typed_left, typed_right)
    dtypes.append(typed_output.dtype)
    if differentiable:
        typed_output.backward()
        half_gradient_dtypes = [typed_left.grad.dtype, typed_right.grad.dtype]

with bg.no_grad():
    detached = bg.einsum(
        "i,i->",
        tensor([2], [1, 2], requires_grad=True),
        tensor([2], [3, 4], requires_grad=True),
    )

bg.install_torch_alias()
import torch
alias = torch.einsum(
    matmul["canonicalEquation"],
    torch.tensor(np.asarray(matmul["leftValues"], dtype=np.float32).reshape(matmul["leftShape"])),
    torch.tensor(np.asarray(matmul["rightValues"], dtype=np.float32).reshape(matmul["rightShape"])),
)

{
    "schema": fixture["schema"],
    "op": output._uop.op,
    "equation": output._uop.arg["equation"],
    "batchRank": output._uop.arg["batch_rank"],
    "constructionEinsumCalls": einsum_calls,
    "matmul": output_values.reshape(-1).tolist(),
    "matmulShape": list(output.shape),
    "matmulGradients": [left.grad.numpy().reshape(-1).tolist(), right.grad.numpy().reshape(-1).tolist()],
    "implicit": implicit.numpy().reshape(-1).tolist(),
    "implicitEquation": implicit._uop.arg["equation"],
    "three": three_output.numpy().reshape(-1).tolist(),
    "threeGradients": [value.grad.numpy().reshape(-1).tolist() for value in three_inputs],
    "broadcast": broadcast_output.numpy().reshape(-1).tolist(),
    "broadcastGradients": [value.grad.numpy().reshape(-1).tolist() for value in broadcast_inputs],
    "ellipsis": ellipsis_output.numpy().reshape(-1).tolist(),
    "ellipsisShape": list(ellipsis_output.shape),
    "reduction": reduction_output.numpy().reshape(-1).tolist(),
    "reductionGradient": reduction_input.grad.numpy().reshape(-1).tolist(),
    "diagonal": diagonal_output.numpy().reshape(-1).tolist(),
    "diagonalGradient": diagonal_input.grad.numpy().reshape(-1).tolist(),
    "trace": float(trace.item()),
    "scalar": float(scalar_output.item()),
    "scalarGradients": [float(value.grad.item()) for value in scalar_inputs],
    "uppercase": uppercase_output.numpy().reshape(-1).tolist(),
    "uppercaseShape": list(uppercase_output.shape),
    "uppercaseEquation": uppercase_output._uop.arg["equation"],
    "emptyShape": list(empty_output.shape),
    "empty": empty_output.numpy().reshape(-1).tolist(),
    "dtypes": dtypes,
    "halfGradientDtypes": half_gradient_dtypes,
    "ownsData": bool(output_values.flags["OWNDATA"]),
    "detached": detached.requires_grad,
    "alias": alias.numpy().reshape(-1).tolist(),
}
`);

    const fixture = FRAMEWORK_EINSUM_CONFORMANCE;
    expect(result.schema).toBe(fixture.schema);
    expect(result.op).toBe("EINSUM");
    expect(result.equation).toBe(fixture.matmul.canonicalEquation);
    expect(result.batchRank).toBe(0);
    expect(result.constructionEinsumCalls).toBe(0);
    expect(result.matmulShape).toEqual(fixture.matmul.outputShape);
    expect(result.matmul).toEqual(fixture.matmul.outputValues);
    expect(result.matmulGradients).toEqual([
      fixture.matmul.leftGradient,
      fixture.matmul.rightGradient,
    ]);
    expect(result.implicit).toEqual(fixture.matmul.outputValues);
    expect(result.implicitEquation).toBe(fixture.matmul.canonicalEquation);
    expect(result.three).toEqual(fixture.threeOperand.outputValues);
    expect(result.threeGradients).toEqual(fixture.threeOperand.gradients);
    expect(result.broadcast).toEqual(fixture.broadcast.outputValues);
    expect(result.broadcastGradients).toEqual(fixture.broadcast.gradients);
    expect(result.ellipsisShape).toEqual(fixture.ellipsis.outputShape);
    expect(result.ellipsis).toEqual(fixture.ellipsis.outputValues);
    expect(result.reduction).toEqual(fixture.ellipsis.reductionOutputValues);
    expect(result.reductionGradient).toEqual(fixture.ellipsis.reductionGradient);
    expect(result.diagonal).toEqual(fixture.diagonal.outputValues);
    expect(result.diagonalGradient).toEqual(fixture.diagonal.gradient);
    expect(result.trace).toBe(fixture.diagonal.traceValue);
    expect(result.scalar).toBe(fixture.scalar.outputValue);
    expect(result.scalarGradients).toEqual(fixture.scalar.gradients);
    expect(result.uppercaseShape).toEqual(fixture.uppercaseImplicit.outputShape);
    expect(result.uppercase).toEqual(fixture.uppercaseImplicit.outputValues);
    expect(result.uppercaseEquation).toBe(fixture.uppercaseImplicit.canonicalEquation);
    expect(result.emptyShape).toEqual(fixture.emptyBroadcast.outputShape);
    expect(result.empty).toEqual(fixture.emptyBroadcast.outputValues);
    expect(result.dtypes).toEqual(fixture.dtypeCases.map(({ expectedDtype }) => expectedDtype));
    expect(result.halfGradientDtypes).toEqual(["float16", "float16"]);
    expect(result.ownsData).toBe(true);
    expect(result.detached).toBe(false);
    expect(result.alias).toEqual(fixture.matmul.outputValues);
  });

  it("provides symbolic VJP, captured-safe vmap, resolved ONNX export, and device refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import OP_EINSUM, OP_EINSUM_VJP, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

left = bg.from_numpy(np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
right = bg.from_numpy(np.asarray([[4, 5, 6], [7, 8, 9]], dtype=np.float32))
captured = bg.from_numpy(np.asarray([1, 1, 1], dtype=np.float32))

symbolic_left = bg.func.grad(
    lambda value: bg.einsum("bi,bi->b", value, right).sum()
)(bg.from_numpy(np.asarray([[1, 2, 3]], dtype=np.float32)))
diagonal_source = bg.from_numpy(np.arange(1, 10, dtype=np.float32).reshape(3, 3))
symbolic_diagonal = bg.func.grad(
    lambda value: bg.einsum("ii->", value)
)(diagonal_source)

mapped = bg.func.vmap(lambda a, b: bg.einsum("i,i->", a, b))(left, right)
captured_mapped = bg.func.vmap(lambda row: bg.einsum("i,i->", row, captured))(left)
ellipsis_input = bg.from_numpy(np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3))
ellipsis_mapped = bg.func.vmap(
    lambda block: bg.einsum("...i->i", block)
)(ellipsis_input)

onnx_left = bg.from_numpy(np.ones((2, 3), dtype=np.float32))
onnx_right = bg.from_numpy(np.ones((3, 2), dtype=np.float32))
onnx_output = bg.einsum("ij,jk->ik", onnx_left, onnx_right)
onnx = bg.onnx.export_inference(onnx_output, input_buffers=(onnx_left, onnx_right))
bool_left = bg.from_numpy(np.asarray([True, False], dtype=np.bool_))
bool_right = bg.from_numpy(np.asarray([True, True], dtype=np.bool_))
bool_output = bg.einsum("i,i->", bool_left, bool_right)
many_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZa"
many = bg.from_numpy(np.ones((1,) * len(many_labels), dtype=np.float32))
many_output = bg.einsum(many_labels + "->", many)

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "registered": get_rule(OP_EINSUM) is not None,
    "symbolicLeft": symbolic_left.numpy().reshape(-1).tolist(),
    "symbolicLeftOps": [node.op for node in toposort(symbolic_left._uop)],
    "symbolicDiagonal": symbolic_diagonal.numpy().reshape(-1).tolist(),
    "symbolicDiagonalOps": [node.op for node in toposort(symbolic_diagonal._uop)],
    "mapped": mapped.numpy().reshape(-1).tolist(),
    "capturedMapped": captured_mapped.numpy().reshape(-1).tolist(),
    "ellipsisMapped": ellipsis_mapped.numpy().tolist(),
    "ellipsisBatchRank": ellipsis_mapped._uop.arg["batch_rank"],
    "onnxEinsum": b"Einsum" in onnx,
    "onnxEquation": b"ab,bc->ac" in onnx,
    "boolOnnxError": error(lambda: bg.onnx.export_inference(bool_output, input_buffers=(bool_left, bool_right))),
    "labelOnnxError": error(lambda: bg.onnx.export_inference(many_output, input_buffers=(many,))),
    "planError": error(lambda: bg.gpu_plan_summary(onnx_output)),
    "webgpuSupported": OP_EINSUM in supported_opcodes() or OP_EINSUM_VJP in supported_opcodes(),
}
`);

    expect(result.registered).toBe(true);
    expect(result.symbolicLeft).toEqual([11, 13, 15]);
    expect(result.symbolicLeftOps).toContain("EINSUM_VJP");
    expect(result.symbolicDiagonal).toEqual(FRAMEWORK_EINSUM_CONFORMANCE.diagonal.gradient);
    expect(result.symbolicDiagonalOps).toContain("EINSUM_VJP");
    expect(result.mapped).toEqual([32, 122]);
    expect(result.capturedMapped).toEqual([6, 15]);
    expect(result.ellipsisMapped).toEqual([[5, 7, 9], [17, 19, 21]]);
    expect(result.ellipsisBatchRank).toBe(1);
    expect(result.onnxEinsum).toBe(true);
    expect(result.onnxEquation).toBe(true);
    expect(result.boolOnnxError).toMatch(/^OnnxUnmappableOp: .*bool/u);
    expect(result.labelOnnxError).toMatch(/^ShapeError: .*at most 26 resolved labels/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*EINSUM/u);
    expect(result.webgpuSupported).toBe(false);
  });

  it("rejects hostile equations, malformed IR, and bounded resource amplification", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_EINSUM_CONFORMANCE);
    const result = await target.run<{ errors: Record<string, string>; hostileCalls: number }>(`
import browsergrad_jit as bg
import json
import numpy as np
from browsergrad_jit._framework_contracts import validate_einsum_contract, validate_einsum_vjp_contract
from browsergrad_jit._ir import UOp, OP_BUFFER, OP_EINSUM, OP_EINSUM_VJP
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

def inputs(shapes, dtype="float32"):
    return tuple(bg.from_numpy(np.zeros(tuple(shape), dtype=np.dtype(dtype))) for shape in shapes)

class HostileEquation:
    calls = 0
    def __str__(self):
        HostileEquation.calls += 1
        raise RuntimeError("equation coercion executed")

errors = {
    case["id"]: error(lambda case=case: bg.einsum(case["equation"], *inputs(case["shapes"])))
    for case in fixture["invalid"]
}
errors["non-string"] = error(lambda: bg.einsum(HostileEquation(), bg.from_numpy(np.ones((1,), dtype=np.float32))))
errors["no-operands"] = error(lambda: bg.einsum("->"))
errors["equation-bytes"] = error(lambda: bg.einsum("i" * 4097, bg.from_numpy(np.ones((1,), dtype=np.float32))))
scalar = UOp(OP_BUFFER, (), (), "float32", {"buffer_id": "scalar"})
errors["operand-count"] = error(lambda: UOp(
    OP_EINSUM,
    (scalar,) * 65,
    (),
    "float32",
    {"equation": ",".join([""] * 65) + "->", "batch_rank": 0},
))

left = bg.from_numpy(np.ones((2, 3), dtype=np.float32))
right = bg.from_numpy(np.ones((3, 2), dtype=np.float32))
valid = bg.einsum("ij,jk->ik", left, right)
bad_arg = bg.einsum("ij,jk->ik", left, right)
bad_arg._uop.arg["batch_rank"] = True
open_arg = bg.einsum("ij,jk->ik", left, right)
open_arg._uop.arg["extra"] = 1
dy = bg.from_numpy(np.ones((2, 2), dtype=np.float32))._uop

errors["bad-arg-cpu"] = error(lambda: bad_arg.numpy())
errors["bad-arg-vjp"] = error(lambda: get_rule(OP_EINSUM)(bad_arg._uop, bad_arg._uop.inputs, dy))
errors["bad-arg-vmap"] = error(lambda: get_vmap_rule(OP_EINSUM)(bad_arg._uop, {id(node): node for node in bad_arg._uop.inputs}, 2))
errors["bad-arg-onnx"] = error(lambda: bg.onnx.export_inference(bad_arg, input_buffers=(left, right)))
errors["bad-arg-plan"] = error(lambda: bg.gpu_plan_summary(bad_arg))
errors["open-arg"] = error(lambda: validate_einsum_contract(open_arg._uop))
errors["noncanonical"] = error(lambda: UOp(
    OP_EINSUM, valid._uop.inputs, (2, 2), "float32",
    {"equation": "ij,jk", "batch_rank": 0},
))
errors["wrong-output-shape"] = error(lambda: UOp(
    OP_EINSUM, valid._uop.inputs, (4,), "float32",
    {"equation": "ij,jk->ik", "batch_rank": 0},
))
errors["wrong-output-dtype"] = error(lambda: UOp(
    OP_EINSUM, valid._uop.inputs, (2, 2), "float64",
    {"equation": "ij,jk->ik", "batch_rank": 0},
))

huge_output = UOp(OP_BUFFER, (), (1 << 28,), "float32", {"buffer_id": "huge-output"})
errors["output-ceiling"] = error(lambda: UOp(
    OP_EINSUM, (huge_output,), (1 << 28,), "float32",
    {"equation": "i->i", "batch_rank": 0},
))
huge_work_left = UOp(OP_BUFFER, (), (16384, 16384), "float32", {"buffer_id": "work-left"})
huge_work_right = UOp(OP_BUFFER, (), (16384, 16384), "float32", {"buffer_id": "work-right"})
errors["work-ceiling"] = error(lambda: UOp(
    OP_EINSUM, (huge_work_left, huge_work_right), (), "float32",
    {"equation": "ij,ij->", "batch_rank": 0},
))
huge_workspace = UOp(OP_BUFFER, (), (50000000,), "float16", {"buffer_id": "workspace"})
errors["workspace-ceiling"] = error(lambda: UOp(
    OP_EINSUM, (huge_workspace,), (), "float16",
    {"equation": "i->", "batch_rank": 0},
))
contraction_workspace = UOp(OP_BUFFER, (), (40000000,), "int64", {"buffer_id": "contraction-workspace"})
errors["contraction-workspace-ceiling"] = error(lambda: UOp(
    OP_EINSUM, (contraction_workspace,), (), "int64",
    {"equation": "i->", "batch_rank": 0},
))
upper = UOp(OP_BUFFER, (), (1,) * 26, "float32", {"buffer_id": "upper"})
lower = UOp(OP_BUFFER, (), (1,) * 26, "float32", {"buffer_id": "lower"})
ellipsis = UOp(OP_BUFFER, (), (1,), "float32", {"buffer_id": "ellipsis"})
errors["label-ceiling"] = error(lambda: UOp(
    OP_EINSUM, (upper, lower, ellipsis), (), "float32",
    {"equation": "ABCDEFGHIJKLMNOPQRSTUVWXYZ,abcdefghijklmnopqrstuvwxyz,...->", "batch_rank": 0},
))

vjp = get_rule(OP_EINSUM)(valid._uop, valid._uop.inputs, dy)[0]
vjp.arg["operand"] = len(valid._uop.inputs)
errors["vjp-operand"] = error(lambda: validate_einsum_vjp_contract(vjp))
errors["valid"] = error(lambda: validate_einsum_contract(valid._uop))

{"errors": errors, "hostileCalls": HostileEquation.calls}
`);

    expect(result.hostileCalls).toBe(0);
    for (const invalid of FRAMEWORK_EINSUM_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id], invalid.id).toContain(invalid.message);
    }
    expect(result.errors["non-string"]).toContain("equation must be an exact string");
    expect(result.errors["no-operands"]).toContain("need at least one operand");
    expect(result.errors["equation-bytes"]).toContain("4096-byte ceiling");
    expect(result.errors["operand-count"]).toContain("64-operand ceiling");
    expect(result.errors["bad-arg-cpu"]).toContain("batch_rank must be");
    expect(result.errors["bad-arg-vjp"]).toContain("batch_rank must be");
    expect(result.errors["bad-arg-vmap"]).toContain("batch_rank must be");
    expect(result.errors["bad-arg-onnx"]).toContain("batch_rank must be");
    expect(result.errors["bad-arg-plan"]).toContain("batch_rank must be");
    expect(result.errors["open-arg"]).toContain("arg fields must be exactly");
    expect(result.errors.noncanonical).toContain("equation must be canonical");
    expect(result.errors["wrong-output-shape"]).toContain("does not match derived shape");
    expect(result.errors["wrong-output-dtype"]).toContain("does not match promoted dtype");
    expect(result.errors["output-ceiling"]).toContain("output requires");
    expect(result.errors["work-ceiling"]).toContain("projected contraction work");
    expect(result.errors["workspace-ceiling"]).toContain("projected output/cast/contraction/gradient workspace");
    expect(result.errors["contraction-workspace-ceiling"]).toContain("projected output/cast/contraction/gradient workspace");
    expect(result.errors["label-ceiling"]).toContain("52-label NumPy execution ceiling");
    expect(result.errors["vjp-operand"]).toContain("normalized input index");
    expect(result.errors.valid).toBe("no_error");
  });
});
