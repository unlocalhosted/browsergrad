import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE } from "../../../test-support/framework-repeat-interleave-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager Tensor.repeat_interleave conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches shared values, dtype preservation, ownership, and refusals", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE);
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
output = grad.Tensor(base).repeat_interleave(valid["repeats"], dim=valid["dim"])

dtypes = []
for case in fixture["dtypeCases"]:
    source = np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(valid["inputShape"])
    dtypes.append(
        grad.Tensor(source, dtype=case["dtype"])
        .repeat_interleave(valid["repeats"], dim=valid["dim"])
        .dtype
    )

errors = {}
for case in fixture["invalid"]:
    source = np.arange(np.prod(case["inputShape"]), dtype=np.float32).reshape(case["inputShape"])
    try:
        grad.Tensor(source).repeat_interleave(case["repeats"], dim=case["dim"])
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

    expect(result.schema).toBe(FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE.valid.outputValues);
    expect(result.ownsData).toBe(true);
    expect(result.dtypes).toEqual(
      FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    for (const fixture of FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE.invalid) {
      expect(result.errors[fixture.id]).toContain(fixture.message);
    }
  });
});
