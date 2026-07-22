import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_STACK_CONFORMANCE } from "../../../test-support/framework-stack-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager torch.stack conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("shares exact shape, promotion, scalar/empty, gradient, ownership, and refusal semantics", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_STACK_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      shape: number[];
      values: number[][];
      dtype: string;
      gradients: number[][];
      dtypeCases: Array<{ dtype: string; values: unknown[] }>;
      scalar: { shape: number[]; values: number[] };
      empty: { shape: number[]; values: unknown[] };
      mixedGradient: { floating: number[]; integralPresent: boolean };
      ownsData: boolean;
      rerunValues: number[][];
      errors: Record<string, string>;
      hostileCalls: { axis: number; sequence: number };
    }>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
inputs = tuple(
    grad.Tensor(np.asarray(values, dtype=np.float32), requires_grad=True, dtype="float32")
    for values in valid["inputValues"]
)
output = grad.stack(list(inputs), dim=np.int32(valid["axis"]))
values = output.data.tolist()
output.backward(grad.Tensor(np.asarray(valid["cotangent"], dtype=np.float32), dtype="float32"))

dtype_cases = []
for case in fixture["dtypeCases"]:
    typed_inputs = [
        grad.Tensor(np.asarray(values, dtype=np.dtype(dtype)), dtype=dtype)
        for values, dtype in zip(case["values"], case["dtypes"])
    ]
    observed = grad.stack(typed_inputs, dim=0)
    dtype_cases.append({"dtype": observed.dtype, "values": observed.data.tolist()})

scalar_spec = fixture["scalar"]
scalar = grad.stack(tuple(grad.Tensor(value) for value in scalar_spec["values"]), dim=scalar_spec["axis"])

empty_spec = fixture["empty"]
empty = grad.stack(tuple(
    grad.Tensor(
        np.empty(tuple(empty_spec["inputShape"]), dtype=np.float32),
        dtype="float32",
    )
    for _ in range(2)
), dim=empty_spec["axis"])

mixed_spec = fixture["mixedGradient"]
mixed_float = grad.Tensor(
    np.asarray(mixed_spec["floatingValues"], dtype=np.float32),
    requires_grad=True,
    dtype="float32",
)
mixed_int = grad.Tensor(
    np.asarray(mixed_spec["integralValues"], dtype=np.int64),
    requires_grad=True,
    dtype="int64",
)
grad.stack((mixed_float, mixed_int), dim=0).sum().backward()

class HostileAxis:
    calls = 0
    def __index__(self):
        HostileAxis.calls += 1
        return 0
    def __int__(self):
        HostileAxis.calls += 1
        return 0

class HostileSequence:
    calls = 0
    def __iter__(self):
        HostileSequence.calls += 1
        return iter(inputs)

errors = {}
attempts = {
    "non-sequence": lambda: grad.stack((tensor for tensor in inputs), 1),
    "empty-sequence": lambda: grad.stack([], 0),
    "non-tensor": lambda: grad.stack([inputs[0], [1.0, 2.0]], 0),
    "bool-axis": lambda: grad.stack(inputs, True),
    "float-axis": lambda: grad.stack(inputs, 1.0),
    "low-axis": lambda: grad.stack(inputs, -3),
    "high-axis": lambda: grad.stack(inputs, 2),
    "hostile-axis": lambda: grad.stack(inputs, HostileAxis()),
    "hostile-sequence": lambda: grad.stack(HostileSequence(), 1),
    "shape-mismatch": lambda: grad.stack((inputs[0], grad.Tensor([1.0, 2.0, 3.0])), 0),
    "unsupported-dtype": lambda: grad.stack((
        grad.Tensor(np.ones((1,), dtype=np.uint16), dtype="uint16"),
        grad.Tensor(np.ones((1,), dtype=np.uint16), dtype="uint16"),
    ), 0),
    "out-mutation": lambda: grad.stack(inputs, 1, out=inputs[0]),
}
for case in fixture["invalid"]:
    try:
        attempts[case["id"]]()
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = type(exc).__name__ + ": " + str(exc)

owns_data = bool(output.data.flags["OWNDATA"])
output.data[0, 0] = 999
rerun = grad.stack(inputs, dim=-1)

{
    "schema": fixture["schema"],
    "shape": list(output.shape),
    "values": values,
    "dtype": output.dtype,
    "gradients": [tensor.grad.data.tolist() for tensor in inputs],
    "dtypeCases": dtype_cases,
    "scalar": {"shape": list(scalar.shape), "values": scalar.data.tolist()},
    "empty": {"shape": list(empty.shape), "values": empty.data.tolist()},
    "mixedGradient": {
        "floating": mixed_float.grad.data.tolist(),
        "integralPresent": mixed_int.grad is not None,
    },
    "ownsData": owns_data,
    "rerunValues": rerun.data.tolist(),
    "errors": errors,
    "hostileCalls": {"axis": HostileAxis.calls, "sequence": HostileSequence.calls},
}
`);

    expect(result.schema).toBe(FRAMEWORK_STACK_CONFORMANCE.schema);
    expect(result.shape).toEqual(FRAMEWORK_STACK_CONFORMANCE.valid.expectedShape);
    expect(result.values).toEqual(FRAMEWORK_STACK_CONFORMANCE.valid.expectedValues);
    expect(result.dtype).toBe("float32");
    expect(result.gradients).toEqual(FRAMEWORK_STACK_CONFORMANCE.valid.expectedGradients);
    expect(result.dtypeCases).toEqual(
      FRAMEWORK_STACK_CONFORMANCE.dtypeCases.map((testCase) => ({
        dtype: testCase.expectedDtype,
        values: testCase.expectedStackValues,
      })),
    );
    expect(result.scalar).toEqual({
      shape: [...FRAMEWORK_STACK_CONFORMANCE.scalar.expectedShape],
      values: FRAMEWORK_STACK_CONFORMANCE.scalar.expectedValues,
    });
    expect(result.empty).toEqual({
      shape: [...FRAMEWORK_STACK_CONFORMANCE.empty.expectedShape],
      values: FRAMEWORK_STACK_CONFORMANCE.empty.expectedValues,
    });
    expect(result.mixedGradient).toEqual({
      floating: FRAMEWORK_STACK_CONFORMANCE.mixedGradient.expectedFloatingGradient,
      integralPresent: FRAMEWORK_STACK_CONFORMANCE.mixedGradient.expectedIntegralGradientPresent,
    });
    expect(result.ownsData).toBe(true);
    expect(result.rerunValues).toEqual(FRAMEWORK_STACK_CONFORMANCE.valid.expectedValues);
    expect(result.hostileCalls).toEqual({ axis: 0, sequence: 0 });
    for (const invalid of FRAMEWORK_STACK_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).not.toBe("no_error");
    }
  });
});
