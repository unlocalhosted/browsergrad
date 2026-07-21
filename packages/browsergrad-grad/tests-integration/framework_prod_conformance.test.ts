import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_PROD_CONFORMANCE } from "../../../test-support/framework-prod-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager Tensor.prod conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches shared product values, dtype, ownership, and refusals", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_PROD_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      outputShape: number[];
      outputValues: number[];
      keepdimsShape: number[];
      keepdimsValues: number[][];
      fullValue: number;
      ownsData: boolean;
      dtypes: string[];
      errors: Record<string, string>;
    }>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
base = np.asarray(valid["inputValues"], dtype=np.float32).reshape(valid["inputShape"])
source = grad.Tensor(base)
output = source.prod(dim=valid["axis"])
keepdims = source.prod(dim=valid["axis"], keepdim=True)
full = source.prod()

dtypes = []
for case in fixture["dtypeCases"]:
    typed = np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(valid["inputShape"])
    dtypes.append(grad.Tensor(typed, dtype=case["dtype"]).prod(dim=valid["axis"]).dtype)

errors = {}
for case in fixture["invalid"]:
    try:
        if case["kind"] == "both":
            source.prod(dim=0, axis=1)
        elif case["kind"] == "keepdims":
            source.prod(dim=0, keepdim=case["value"])
        elif case["kind"] == "scalar":
            grad.Tensor(np.asarray(2.0, dtype=np.float32)).prod(dim=case["value"])
        else:
            source.prod(dim=case["value"])
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

{
    "schema": fixture["schema"],
    "outputShape": list(output.shape),
    "outputValues": output.tolist(),
    "keepdimsShape": list(keepdims.shape),
    "keepdimsValues": keepdims.tolist(),
    "fullValue": float(full.item()),
    "ownsData": bool(output.data.flags["OWNDATA"] and full.data.flags["OWNDATA"]),
    "dtypes": dtypes,
    "errors": errors,
}
`);

    expect(result.schema).toBe(FRAMEWORK_PROD_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_PROD_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_PROD_CONFORMANCE.valid.outputValues);
    expect(result.keepdimsShape).toEqual(FRAMEWORK_PROD_CONFORMANCE.valid.keepdimsShape);
    expect(result.keepdimsValues).toEqual(FRAMEWORK_PROD_CONFORMANCE.valid.keepdimsValues);
    expect(result.fullValue).toBe(FRAMEWORK_PROD_CONFORMANCE.valid.fullValue);
    expect(result.ownsData).toBe(true);
    expect(result.dtypes).toEqual(
      FRAMEWORK_PROD_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    for (const fixture of FRAMEWORK_PROD_CONFORMANCE.invalid) {
      expect(result.errors[fixture.id]).toContain(fixture.message);
    }
  });
});
