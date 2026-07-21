import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_EXPAND_CONFORMANCE } from "../../../test-support/framework-expand-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager Tensor.expand conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches the shared eager/lazy expand values, dtype, and refusal fixture", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_EXPAND_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      outputShape: number[];
      outputValues: number[][];
      dtypes: string[];
      errors: Record<string, string>;
    }>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
base = np.asarray(valid["inputValues"], dtype=np.float32).reshape(valid["inputShape"])
output = grad.Tensor(base).expand(*valid["requestedShape"])

dtypes = []
for case in fixture["dtypeCases"]:
    source = np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(valid["inputShape"])
    dtypes.append(grad.Tensor(source, dtype=case["dtype"]).expand(*valid["requestedShape"]).dtype)

errors = {}
for case in fixture["invalid"]:
    source = np.arange(np.prod(case["inputShape"]), dtype=np.float32).reshape(case["inputShape"])
    try:
        grad.Tensor(source).expand(*case["requestedShape"])
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

{
    "schema": fixture["schema"],
    "outputShape": list(output.shape),
    "outputValues": output.tolist(),
    "dtypes": dtypes,
    "errors": errors,
}
`);

    expect(result.schema).toBe(FRAMEWORK_EXPAND_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_EXPAND_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_EXPAND_CONFORMANCE.valid.outputValues);
    expect(result.dtypes).toEqual(
      FRAMEWORK_EXPAND_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    for (const fixture of FRAMEWORK_EXPAND_CONFORMANCE.invalid) {
      expect(result.errors[fixture.id]).toContain(fixture.message);
    }
  });
});
