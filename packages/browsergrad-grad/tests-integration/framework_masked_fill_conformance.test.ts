import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_MASKED_FILL_CONFORMANCE } from "../../../test-support/framework-masked-fill-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager Tensor.masked_fill conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("preserves source shape/dtype with strict bool-mask broadcasting and gradients", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_MASKED_FILL_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      values: number[];
      gradient: number[];
      ownsData: boolean;
      infinityCount: number;
      dtypes: string[];
      inplaceDtype: string;
      inplaceValues: number[];
      errors: Record<string, string>;
      hostileCalls: number;
    }>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
base = np.asarray(valid["inputValues"], dtype=np.float32).reshape(valid["inputShape"])
source = grad.Tensor(base, requires_grad=True)
mask = grad.Tensor(np.asarray(valid["maskValues"], dtype=np.bool_), dtype="bool")
output = source.masked_fill(mask, valid["fillValue"])
output.sum().backward()
infinite = source.masked_fill(mask, float("-inf"))

dtypes = []
for case in fixture["dtypeCases"]:
    dtype = case["dtype"]
    if dtype == "bool":
        values = np.asarray(valid["inputValues"], dtype=np.bool_).reshape(valid["inputShape"])
        fill = case["fillValue"]
    else:
        values = np.asarray(valid["inputValues"], dtype=np.dtype(dtype)).reshape(valid["inputShape"])
        fill = valid["fillValue"]
    typed = grad.Tensor(values, dtype=dtype)
    dtypes.append(typed.masked_fill(mask, fill).dtype)

inplace = grad.Tensor(np.asarray(valid["inputValues"], dtype=np.float16).reshape(valid["inputShape"]), dtype="float16")
inplace.masked_fill_(mask, valid["fillValue"])

class HostileValue:
    calls = 0
    def __float__(self):
        HostileValue.calls += 1
        return -1.0
    def __int__(self):
        HostileValue.calls += 1
        return -1

errors = {}
for case in fixture["invalid"]:
    try:
        if case["kind"] == "mask-value":
            source.masked_fill(case["value"], -1)
        elif case["kind"] == "mask-dtype":
            bad = grad.Tensor(np.ones(valid["maskShape"], dtype=np.int32), dtype="int32")
            source.masked_fill(bad, -1)
        elif case["kind"] == "mask-shape":
            bad = grad.Tensor(np.zeros(case["value"], dtype=np.bool_), dtype="bool")
            source.masked_fill(bad, -1)
        elif case["kind"] == "fill":
            source.masked_fill(mask, case["value"])
        elif case["kind"] == "integer-fill":
            integer_source = grad.Tensor(np.asarray(valid["inputValues"], dtype=np.int32).reshape(valid["inputShape"]), dtype="int32")
            integer_source.masked_fill(mask, case["value"])
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

try:
    source.masked_fill(mask, HostileValue())
    errors["hostile-fill"] = "no_error"
except Exception as exc:
    errors["hostile-fill"] = str(exc)

{
    "schema": fixture["schema"],
    "values": output.data.reshape(-1).tolist(),
    "gradient": source.grad.data.reshape(-1).tolist(),
    "ownsData": bool(output.data.flags["OWNDATA"]),
    "infinityCount": int(np.isneginf(infinite.data).sum()),
    "dtypes": dtypes,
    "inplaceDtype": inplace.dtype,
    "inplaceValues": inplace.data.reshape(-1).tolist(),
    "errors": errors,
    "hostileCalls": HostileValue.calls,
}
`);

    expect(result.schema).toBe(FRAMEWORK_MASKED_FILL_CONFORMANCE.schema);
    expect(result.values).toEqual(FRAMEWORK_MASKED_FILL_CONFORMANCE.valid.outputValues);
    expect(result.gradient).toEqual(FRAMEWORK_MASKED_FILL_CONFORMANCE.valid.sourceGradient);
    expect(result.ownsData).toBe(true);
    expect(result.infinityCount).toBe(4);
    expect(result.dtypes).toEqual(
      FRAMEWORK_MASKED_FILL_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    expect(result.inplaceDtype).toBe("float16");
    expect(result.inplaceValues).toEqual(FRAMEWORK_MASKED_FILL_CONFORMANCE.valid.outputValues);
    expect(result.hostileCalls).toBe(0);
    expect(result.errors["hostile-fill"]).toContain("real scalar");
    for (const invalid of FRAMEWORK_MASKED_FILL_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).toContain(invalid.message);
    }
  });
});
