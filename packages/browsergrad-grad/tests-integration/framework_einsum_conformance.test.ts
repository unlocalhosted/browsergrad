import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_EINSUM_CONFORMANCE } from "../../../test-support/framework-einsum-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager torch.einsum conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches general contractions, PyTorch ellipses, diagonals, dtypes, gradients, and aliases", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_EINSUM_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})

def tensor(shape, values, dtype="float32", requires_grad=False):
    return grad.Tensor(
        np.asarray(values, dtype=np.dtype(dtype)).reshape(tuple(shape)),
        dtype=dtype,
        requires_grad=requires_grad,
    )

matmul = fixture["matmul"]
left = tensor(matmul["leftShape"], matmul["leftValues"], requires_grad=True)
right = tensor(matmul["rightShape"], matmul["rightValues"], requires_grad=True)
output = grad.einsum(matmul["equation"], left, right)
output.sum().backward()
implicit = grad.einsum(
    matmul["implicitEquation"],
    tensor(matmul["leftShape"], matmul["leftValues"]),
    tensor(matmul["rightShape"], matmul["rightValues"]),
)

three = fixture["threeOperand"]
three_inputs = tuple(
    tensor(shape, values, requires_grad=True)
    for shape, values in zip(three["shapes"], three["values"])
)
three_output = grad.einsum(three["equation"], *three_inputs)
three_output.sum().backward()

broadcast = fixture["broadcast"]
broadcast_inputs = tuple(
    tensor(shape, values, requires_grad=True)
    for shape, values in zip(broadcast["shapes"], broadcast["values"])
)
broadcast_output = grad.einsum(broadcast["equation"], *broadcast_inputs)
broadcast_output.sum().backward()

ellipsis = fixture["ellipsis"]
ellipsis_output = grad.einsum(
    ellipsis["equation"],
    *(tensor(shape, values) for shape, values in zip(ellipsis["shapes"], ellipsis["values"])),
)
reduction_input = tensor(
    ellipsis["reductionInputShape"],
    ellipsis["reductionInputValues"],
    requires_grad=True,
)
reduction_output = grad.einsum(ellipsis["reductionEquation"], reduction_input)
reduction_output.sum().backward()

diagonal = fixture["diagonal"]
diagonal_input = tensor(
    diagonal["inputShape"], diagonal["inputValues"], requires_grad=True
)
diagonal_output = grad.einsum(diagonal["equation"], diagonal_input)
diagonal_output.sum().backward()
trace = grad.einsum(
    diagonal["traceEquation"],
    tensor(diagonal["inputShape"], diagonal["inputValues"]),
)

scalar = fixture["scalar"]
scalar_inputs = tuple(
    tensor([], [value], requires_grad=True) for value in scalar["values"]
)
scalar_output = grad.einsum(scalar["equation"], *scalar_inputs)
scalar_output.backward()

uppercase = fixture["uppercaseImplicit"]
uppercase_output = grad.einsum(
    uppercase["equation"],
    tensor(uppercase["inputShape"], uppercase["inputValues"]),
)

empty = fixture["emptyBroadcast"]
empty_output = grad.einsum(
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
    typed_output = grad.einsum("i,i->", typed_left, typed_right)
    dtypes.append(typed_output.dtype)
    if differentiable:
        typed_output.backward()
        half_gradient_dtypes = [typed_left.grad.dtype, typed_right.grad.dtype]

snapshot_left = tensor(matmul["leftShape"], matmul["leftValues"], requires_grad=True)
snapshot_right = tensor(matmul["rightShape"], matmul["rightValues"], requires_grad=True)
snapshot_output = grad.einsum(matmul["canonicalEquation"], snapshot_left, snapshot_right)
snapshot_left.data[:] = -1000.0
snapshot_right.data[:] = 1000.0
snapshot_output.sum().backward()

with grad.no_grad():
    detached = grad.einsum(
        "i,i->",
        tensor([2], [1, 2], requires_grad=True),
        tensor([2], [3, 4], requires_grad=True),
    )

grad.install_torch_alias()
import torch
alias = torch.einsum(
    matmul["canonicalEquation"],
    torch.tensor(np.asarray(matmul["leftValues"], dtype=np.float32).reshape(matmul["leftShape"])),
    torch.tensor(np.asarray(matmul["rightValues"], dtype=np.float32).reshape(matmul["rightShape"])),
)

{
    "schema": fixture["schema"],
    "matmul": output.data.reshape(-1).tolist(),
    "matmulShape": list(output.shape),
    "matmulGradients": [left.grad.data.reshape(-1).tolist(), right.grad.data.reshape(-1).tolist()],
    "implicit": implicit.data.reshape(-1).tolist(),
    "three": three_output.data.reshape(-1).tolist(),
    "threeGradients": [value.grad.data.reshape(-1).tolist() for value in three_inputs],
    "broadcast": broadcast_output.data.reshape(-1).tolist(),
    "broadcastGradients": [value.grad.data.reshape(-1).tolist() for value in broadcast_inputs],
    "ellipsis": ellipsis_output.data.reshape(-1).tolist(),
    "ellipsisShape": list(ellipsis_output.shape),
    "reduction": reduction_output.data.reshape(-1).tolist(),
    "reductionGradient": reduction_input.grad.data.reshape(-1).tolist(),
    "diagonal": diagonal_output.data.reshape(-1).tolist(),
    "diagonalGradient": diagonal_input.grad.data.reshape(-1).tolist(),
    "trace": float(trace.item()),
    "scalar": float(scalar_output.item()),
    "scalarGradients": [float(value.grad.item()) for value in scalar_inputs],
    "uppercase": uppercase_output.data.reshape(-1).tolist(),
    "uppercaseShape": list(uppercase_output.shape),
    "emptyShape": list(empty_output.shape),
    "empty": empty_output.data.reshape(-1).tolist(),
    "dtypes": dtypes,
    "halfGradientDtypes": half_gradient_dtypes,
    "ownsData": bool(output.data.flags["OWNDATA"]),
    "snapshotGradients": [
        snapshot_left.grad.data.reshape(-1).tolist(),
        snapshot_right.grad.data.reshape(-1).tolist(),
    ],
    "detached": detached.requires_grad,
    "alias": alias.data.reshape(-1).tolist(),
}
`);

    const fixture = FRAMEWORK_EINSUM_CONFORMANCE;
    expect(result.schema).toBe(fixture.schema);
    expect(result.matmulShape).toEqual(fixture.matmul.outputShape);
    expect(result.matmul).toEqual(fixture.matmul.outputValues);
    expect(result.matmulGradients).toEqual([
      fixture.matmul.leftGradient,
      fixture.matmul.rightGradient,
    ]);
    expect(result.implicit).toEqual(fixture.matmul.outputValues);
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
    expect(result.emptyShape).toEqual(fixture.emptyBroadcast.outputShape);
    expect(result.empty).toEqual(fixture.emptyBroadcast.outputValues);
    expect(result.dtypes).toEqual(fixture.dtypeCases.map(({ expectedDtype }) => expectedDtype));
    expect(result.halfGradientDtypes).toEqual(["float16", "float16"]);
    expect(result.ownsData).toBe(true);
    expect(result.snapshotGradients).toEqual([
      fixture.matmul.leftGradient,
      fixture.matmul.rightGradient,
    ]);
    expect(result.detached).toBe(false);
    expect(result.alias).toEqual(fixture.matmul.outputValues);
  });

  it("rejects malformed equations and bounded hostile inputs before execution", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_EINSUM_CONFORMANCE);
    const errors = await target.run<Record<string, string>>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

def operands(shapes):
    return tuple(grad.Tensor(np.zeros(tuple(shape), dtype=np.float32)) for shape in shapes)

class HostileEquation:
    def __str__(self):
        raise RuntimeError("equation coercion executed")

errors = {
    case["id"]: error(lambda case=case: grad.einsum(case["equation"], *operands(case["shapes"])))
    for case in fixture["invalid"]
}
errors["non-string"] = error(lambda: grad.einsum(HostileEquation(), grad.Tensor([1.0])))
errors["no-operands"] = error(lambda: grad.einsum("->"))
errors["equation-bytes"] = error(lambda: grad.einsum("i" * 4097, grad.Tensor([1.0])))
errors["operand-count"] = error(lambda: grad.einsum(
    ",".join([""] * 65) + "->",
    *(grad.Tensor(np.asarray(1.0, dtype=np.float32)) for _ in range(65)),
))
errors
`);

    for (const invalid of FRAMEWORK_EINSUM_CONFORMANCE.invalid) {
      expect(errors[invalid.id], invalid.id).toContain(invalid.message);
    }
    expect(errors["non-string"]).toContain("equation must be an exact string");
    expect(errors["non-string"]).not.toContain("equation coercion executed");
    expect(errors["no-operands"]).toContain("operands must be a non-empty plain tuple");
    expect(errors["equation-bytes"]).toContain("4096-byte ceiling");
    expect(errors["operand-count"]).toContain("64-operand ceiling");
  });
});
