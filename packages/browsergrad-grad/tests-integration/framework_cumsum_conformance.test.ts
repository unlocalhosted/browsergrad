import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_CUMSUM_CONFORMANCE } from "../../../test-support/framework-cumsum-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager Tensor.cumsum conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches inclusive scan values, promotion, gradients, ownership, and refusals", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_CUMSUM_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      values: number[][];
      alternateAxisValues: number[][];
      gradient: number[][];
      typedGradient: number[];
      typedGradientDtype: string;
      autogradRequiresGrad: boolean[];
      dtypeCases: Array<{ dtype: string; values: unknown[] }>;
      emptyShape: number[];
      emptyValues: unknown[];
      ownsData: boolean;
      rerunValues: number[][];
      errors: Record<string, string>;
      hostileCalls: { axis: number; dtype: number };
    }>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
base = np.asarray(valid["inputValues"], dtype=np.float32).reshape(valid["inputShape"])
source = grad.Tensor(base, requires_grad=True, dtype="float32")
output = source.cumsum(np.int32(valid["axis"]))
output.sum().backward()
alternate = grad.cumsum(source, dim=-2)

typed_spec = fixture["typedGradient"]
typed_source = grad.Tensor(
    np.asarray(typed_spec["values"], dtype=np.dtype(typed_spec["sourceDtype"])),
    requires_grad=True,
    dtype=typed_spec["sourceDtype"],
)
typed_cotangent = grad.Tensor(
    np.asarray(typed_spec["cotangent"], dtype=np.dtype(typed_spec["outputDtype"])),
    dtype=typed_spec["outputDtype"],
)
(typed_source.cumsum(0, dtype=typed_spec["outputDtype"]) * typed_cotangent).sum().backward()
float_to_integral = source.cumsum(1, dtype="int64")
integral_to_float = grad.Tensor(
    np.asarray([1, 2], dtype=np.int32), requires_grad=True, dtype="int32"
).cumsum(0, dtype="float32")

dtype_cases = []
for case in fixture["dtypeCases"]:
    typed = grad.Tensor(
        np.asarray(case["values"], dtype=np.dtype(case["sourceDtype"])),
        dtype=case["sourceDtype"],
    )
    kwargs = {} if case["dtype"] is None else {"dtype": case["dtype"]}
    observed = typed.cumsum(0, **kwargs)
    dtype_cases.append({"dtype": observed.dtype, "values": observed.data.tolist()})

empty_spec = fixture["empty"]
empty = grad.cumsum(
    grad.Tensor(np.empty(empty_spec["shape"], dtype=np.float32), dtype="float32"),
    dim=empty_spec["axis"],
)

class HostileAxis:
    calls = 0
    def __int__(self):
        HostileAxis.calls += 1
        return 0

class HostileDtype:
    calls = 0
    def __str__(self):
        HostileDtype.calls += 1
        return "float32"

errors = {}
attempts = {
    "scalar-input": lambda: grad.cumsum(grad.Tensor(np.asarray(1.0, dtype=np.float32), dtype="float32"), 0),
    "bool-axis": lambda: source.cumsum(True),
    "float-axis": lambda: source.cumsum(1.0),
    "low-axis": lambda: source.cumsum(-3),
    "high-axis": lambda: source.cumsum(2),
    "hostile-axis": lambda: source.cumsum(HostileAxis()),
    "list-dtype": lambda: source.cumsum(1, dtype=["float32"]),
    "hostile-dtype": lambda: source.cumsum(1, dtype=HostileDtype()),
    "unknown-dtype": lambda: source.cumsum(1, dtype="complex64"),
    "unsupported-source-dtype": lambda: grad.Tensor(np.ones((2, 3), dtype=np.uint16), dtype="uint16").cumsum(1),
    "out-mutation": lambda: grad.cumsum(source, 1, out=source),
}
for case in fixture["invalid"]:
    try:
        attempts[case["id"]]()
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = type(exc).__name__ + ": " + str(exc)

owns_data = bool(output.data.flags["OWNDATA"])
output_values = output.data.tolist()
output.data[0, 0] = 999
rerun = source.cumsum(valid["axis"])

{
    "schema": fixture["schema"],
    "values": output_values,
    "alternateAxisValues": alternate.data.tolist(),
    "gradient": source.grad.data.tolist(),
    "typedGradient": typed_source.grad.data.tolist(),
    "typedGradientDtype": typed_source.grad.dtype,
    "autogradRequiresGrad": [float_to_integral.requires_grad, integral_to_float.requires_grad],
    "dtypeCases": dtype_cases,
    "emptyShape": list(empty.shape),
    "emptyValues": empty.data.tolist(),
    "ownsData": owns_data,
    "rerunValues": rerun.data.tolist(),
    "errors": errors,
    "hostileCalls": {"axis": HostileAxis.calls, "dtype": HostileDtype.calls},
}
`);

    expect(result.schema).toBe(FRAMEWORK_CUMSUM_CONFORMANCE.schema);
    expect(result.values).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.valid.values);
    expect(result.alternateAxisValues).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.valid.alternateAxisValues);
    expect(result.gradient).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.valid.gradient);
    expect(result.typedGradient).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.typedGradient.expected);
    expect(result.typedGradientDtype).toBe(FRAMEWORK_CUMSUM_CONFORMANCE.typedGradient.expectedDtype);
    expect(result.autogradRequiresGrad).toEqual([
      FRAMEWORK_CUMSUM_CONFORMANCE.autograd.floatToIntegralRequiresGrad,
      FRAMEWORK_CUMSUM_CONFORMANCE.autograd.integralToFloatRequiresGrad,
    ]);
    expect(result.dtypeCases).toEqual(
      FRAMEWORK_CUMSUM_CONFORMANCE.dtypeCases.map((testCase) => ({
        dtype: testCase.expectedDtype,
        values: testCase.expectedValues,
      })),
    );
    expect(result.emptyShape).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.empty.shape);
    expect(result.emptyValues).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.empty.values);
    expect(result.ownsData).toBe(true);
    expect(result.rerunValues).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.valid.values);
    expect(result.hostileCalls).toEqual({ axis: 0, dtype: 0 });
    for (const invalid of FRAMEWORK_CUMSUM_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).not.toBe("no_error");
    }
  });
});
