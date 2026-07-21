import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_REPEAT_CONFORMANCE } from "../../../test-support/framework-repeat-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager Tensor.repeat conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches the shared eager/lazy repeat values, dtype, and refusal fixture", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_REPEAT_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      outputShape: number[];
      outputValues: number[][];
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
output = grad.Tensor(base).repeat(*valid["requestedRepeats"])

dtypes = []
for case in fixture["dtypeCases"]:
    source = np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(valid["inputShape"])
    dtypes.append(grad.Tensor(source, dtype=case["dtype"]).repeat(*valid["requestedRepeats"]).dtype)

errors = {}
for case in fixture["invalid"]:
    source = np.arange(np.prod(case["inputShape"]), dtype=np.float32).reshape(case["inputShape"])
    try:
        grad.Tensor(source).repeat(*case["requestedRepeats"])
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

{
    "schema": fixture["schema"],
    "outputShape": list(output.shape),
    "outputValues": output.tolist(),
    "ownsData": bool(output.data.flags["OWNDATA"]),
    "dtypes": dtypes,
    "errors": errors,
}
`);

    expect(result.schema).toBe(FRAMEWORK_REPEAT_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_REPEAT_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_REPEAT_CONFORMANCE.valid.outputValues);
    expect(result.ownsData).toBe(true);
    expect(result.dtypes).toEqual(
      FRAMEWORK_REPEAT_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    for (const fixture of FRAMEWORK_REPEAT_CONFORMANCE.invalid) {
      expect(result.errors[fixture.id]).toContain(fixture.message);
    }
  });
});
