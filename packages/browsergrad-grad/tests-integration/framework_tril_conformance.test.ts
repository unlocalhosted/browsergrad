import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_TRIL_CONFORMANCE } from "../../../test-support/framework-tril-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager Tensor.tril conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("preserves batched matrix shape/dtype with canonical diagonals and gradients", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_TRIL_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      values: number[];
      gradient: number[];
      upperValues: number[];
      positiveValues: number[];
      negativeValues: number[];
      ownsData: boolean;
      dtypes: string[];
      emptyShapes: number[][];
      errors: Record<string, string>;
      hostileCalls: number;
    }>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
shape = tuple(valid["inputShape"])
base = np.asarray(valid["inputValues"], dtype=np.float32).reshape(shape)
source = grad.Tensor(base, requires_grad=True)
weights = grad.Tensor(np.arange(1, 13, dtype=np.float32).reshape(shape))
output = source.tril(np.int32(valid["diagonal"]))
(output * weights).sum().backward()
upper = grad.tril(source, np.uint64(1))
positive = source.tril(10 ** 200)
negative = source.tril(-(10 ** 200))

dtypes = []
for case in fixture["dtypeCases"]:
    dtype = case["dtype"]
    typed_values = np.asarray(valid["inputValues"], dtype=np.dtype(dtype)).reshape(shape)
    typed = grad.Tensor(typed_values, dtype=dtype)
    dtypes.append(typed.tril(valid["diagonal"]).dtype)
empty_shapes = [
    list(grad.Tensor(np.empty((0, 3), dtype=np.float32)).tril(-1).shape),
    list(grad.Tensor(np.empty((2, 0), dtype=np.float32)).tril(1).shape),
]

class HostileIndex:
    calls = 0
    def __index__(self):
        HostileIndex.calls += 1
        return 0

errors = {}
for case in fixture["invalid"]:
    try:
        if case["kind"] == "rank":
            grad.Tensor(np.ones(case["value"], dtype=np.float32)).tril()
        elif case["kind"] == "diagonal":
            source.tril(case["value"])
        elif case["kind"] == "dtype":
            bad = np.asarray(valid["inputValues"], dtype=np.dtype(case["value"])).reshape(shape)
            grad.Tensor(bad, dtype=case["value"]).tril()
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

try:
    source.tril(HostileIndex())
    errors["hostile-diagonal"] = "no_error"
except Exception as exc:
    errors["hostile-diagonal"] = str(exc)

{
    "schema": fixture["schema"],
    "values": output.data.reshape(-1).tolist(),
    "gradient": source.grad.data.reshape(-1).tolist(),
    "upperValues": upper.data.reshape(-1).tolist(),
    "positiveValues": positive.data.reshape(-1).tolist(),
    "negativeValues": negative.data.reshape(-1).tolist(),
    "ownsData": bool(output.data.flags["OWNDATA"]),
    "dtypes": dtypes,
    "emptyShapes": empty_shapes,
    "errors": errors,
    "hostileCalls": HostileIndex.calls,
}
`);

    expect(result.schema).toBe(FRAMEWORK_TRIL_CONFORMANCE.schema);
    expect(result.values).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.outputValues);
    expect(result.gradient).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.sourceGradient);
    expect(result.upperValues).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.upperDiagonalValues);
    expect(result.positiveValues).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.inputValues);
    expect(result.negativeValues).toEqual(new Array(12).fill(0));
    expect(result.ownsData).toBe(true);
    expect(result.dtypes).toEqual(
      FRAMEWORK_TRIL_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    expect(result.emptyShapes).toEqual([[0, 3], [2, 0]]);
    expect(result.hostileCalls).toBe(0);
    expect(result.errors["hostile-diagonal"]).toContain("integer scalar");
    for (const invalid of FRAMEWORK_TRIL_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).toContain(invalid.message);
    }
  });
});
