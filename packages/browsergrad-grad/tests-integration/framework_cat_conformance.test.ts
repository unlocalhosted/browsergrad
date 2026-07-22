import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_CAT_CONFORMANCE } from "../../../test-support/framework-cat-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager torch.cat conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("shares exact shape, promotion, empty, gradient, ownership, and refusal semantics", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_CAT_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      shape: number[];
      values: number[][];
      dtype: string;
      gradients: unknown[];
      dtypeCases: Array<{ dtype: string; values: unknown[] }>;
      legacy: {
        shape: number[];
        dtype: string;
        values: number[][];
        emptyGradient: unknown[];
        emptyGradientDtype: string;
        matrixGradient: number[][];
        matrixGradientDtype: string;
      };
      allEmpty: { shape: number[]; dtype: string; values: unknown[] };
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
    grad.Tensor(np.asarray(values, dtype=np.float32).reshape(shape), requires_grad=True, dtype="float32")
    for values, shape in zip(valid["inputValues"], valid["inputShapes"])
)
output = grad.cat(list(inputs), dim=np.int32(valid["axis"]))
values = output.data.tolist()
output.backward(grad.Tensor(np.asarray(valid["cotangent"], dtype=np.float32), dtype="float32"))

dtype_cases = []
for case in fixture["dtypeCases"]:
    typed_inputs = [
        grad.Tensor(np.asarray(values, dtype=np.dtype(dtype)), dtype=dtype)
        for values, dtype in zip(case["values"], case["dtypes"])
    ]
    observed = grad.cat(typed_inputs, dim=0)
    dtype_cases.append({"dtype": observed.dtype, "values": observed.data.tolist()})

legacy_spec = fixture["legacyEmpty"]
legacy_empty = grad.Tensor(
    np.empty((0,), dtype=np.dtype(legacy_spec["emptyDtype"])),
    requires_grad=True,
    dtype=legacy_spec["emptyDtype"],
)
legacy_matrix = grad.Tensor(
    np.asarray(legacy_spec["matrixValues"], dtype=np.dtype(legacy_spec["matrixDtype"])),
    requires_grad=True,
    dtype=legacy_spec["matrixDtype"],
)
legacy_output = grad.cat((legacy_empty, legacy_matrix), dim=legacy_spec["axis"])
legacy_output.sum().backward()

all_empty_spec = fixture["allEmpty"]
all_empty = grad.cat(tuple(
    grad.Tensor(np.empty((0,), dtype=np.dtype(dtype)), dtype=dtype)
    for dtype in all_empty_spec["dtypes"]
), dim=all_empty_spec["axis"])

mixed_gradient_spec = fixture["mixedGradient"]
mixed_float = grad.Tensor(
    np.asarray(mixed_gradient_spec["floatingValues"], dtype=np.float32),
    requires_grad=True,
    dtype="float32",
)
mixed_int = grad.Tensor(
    np.asarray(mixed_gradient_spec["integralValues"], dtype=np.int64),
    requires_grad=True,
    dtype="int64",
)
grad.cat((mixed_float, mixed_int), dim=0).sum().backward()

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
    "non-sequence": lambda: grad.cat((tensor for tensor in inputs), 1),
    "empty-sequence": lambda: grad.cat([], 0),
    "non-tensor": lambda: grad.cat([inputs[0], [[1.0, 2.0]]], 0),
    "scalar-input": lambda: grad.cat([grad.Tensor(1.0), grad.Tensor(2.0)], 0),
    "bool-axis": lambda: grad.cat(inputs, True),
    "float-axis": lambda: grad.cat(inputs, 1.0),
    "low-axis": lambda: grad.cat(inputs, -3),
    "high-axis": lambda: grad.cat(inputs, 2),
    "hostile-axis": lambda: grad.cat(inputs, HostileAxis()),
    "hostile-sequence": lambda: grad.cat(HostileSequence(), 1),
    "shape-mismatch": lambda: grad.cat((
        grad.Tensor(np.ones((2, 2), dtype=np.float32)),
        grad.Tensor(np.ones((1, 1), dtype=np.float32)),
    ), 0),
    "rank-mismatch": lambda: grad.cat((
        grad.Tensor(np.ones((2, 2), dtype=np.float32)),
        grad.Tensor(np.ones((2,), dtype=np.float32)),
    ), 0),
    "unsupported-dtype": lambda: grad.cat((
        grad.Tensor(np.ones((1,), dtype=np.uint16), dtype="uint16"),
        grad.Tensor(np.ones((1,), dtype=np.uint16), dtype="uint16"),
    ), 0),
    "out-mutation": lambda: grad.cat(inputs, 1, out=inputs[0]),
}
for case in fixture["invalid"]:
    try:
        attempts[case["id"]]()
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = type(exc).__name__ + ": " + str(exc)

owns_data = bool(output.data.flags["OWNDATA"])
output.data[0, 0] = 999
rerun = grad.cat(inputs, dim=-1)

{
    "schema": fixture["schema"],
    "shape": list(output.shape),
    "values": values,
    "dtype": output.dtype,
    "gradients": [tensor.grad.data.tolist() for tensor in inputs],
    "dtypeCases": dtype_cases,
    "legacy": {
        "shape": list(legacy_output.shape),
        "dtype": legacy_output.dtype,
        "values": legacy_output.data.tolist(),
        "emptyGradient": legacy_empty.grad.data.tolist(),
        "emptyGradientDtype": legacy_empty.grad.dtype,
        "matrixGradient": legacy_matrix.grad.data.tolist(),
        "matrixGradientDtype": legacy_matrix.grad.dtype,
    },
    "allEmpty": {
        "shape": list(all_empty.shape),
        "dtype": all_empty.dtype,
        "values": all_empty.data.tolist(),
    },
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

    expect(result.schema).toBe(FRAMEWORK_CAT_CONFORMANCE.schema);
    expect(result.shape).toEqual(FRAMEWORK_CAT_CONFORMANCE.valid.expectedShape);
    expect(result.values).toEqual(FRAMEWORK_CAT_CONFORMANCE.valid.expectedValues);
    expect(result.dtype).toBe("float32");
    expect(result.gradients).toEqual(FRAMEWORK_CAT_CONFORMANCE.valid.expectedGradients);
    expect(result.dtypeCases).toEqual(
      FRAMEWORK_CAT_CONFORMANCE.dtypeCases.map((testCase) => ({
        dtype: testCase.expectedDtype,
        values: testCase.expectedValues,
      })),
    );
    expect(result.legacy).toEqual({
      shape: [...FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.expectedShape],
      dtype: FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.expectedDtype,
      values: FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.expectedValues,
      emptyGradient: FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.expectedEmptyGradient,
      emptyGradientDtype: FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.emptyDtype,
      matrixGradient: FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.expectedMatrixGradient,
      matrixGradientDtype: FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.matrixDtype,
    });
    expect(result.allEmpty).toEqual({
      shape: [...FRAMEWORK_CAT_CONFORMANCE.allEmpty.expectedShape],
      dtype: FRAMEWORK_CAT_CONFORMANCE.allEmpty.expectedDtype,
      values: FRAMEWORK_CAT_CONFORMANCE.allEmpty.expectedValues,
    });
    expect(result.mixedGradient).toEqual({
      floating: FRAMEWORK_CAT_CONFORMANCE.mixedGradient.expectedFloatingGradient,
      integralPresent: FRAMEWORK_CAT_CONFORMANCE.mixedGradient.expectedIntegralGradientPresent,
    });
    expect(result.ownsData).toBe(true);
    expect(result.rerunValues).toEqual(FRAMEWORK_CAT_CONFORMANCE.valid.expectedValues);
    expect(result.hostileCalls).toEqual({ axis: 0, sequence: 0 });
    for (const invalid of FRAMEWORK_CAT_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).not.toBe("no_error");
    }
  });
});
