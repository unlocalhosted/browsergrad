import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_VAR_CONFORMANCE } from "../../../test-support/framework-var-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager Tensor.var conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches canonical variance values, correction, dtype, gradient, and refusals", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_VAR_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      outputShape: number[];
      outputValues: number[];
      negativeAxisValues: number[];
      keepdimsShape: number[];
      keepdimsValues: number[][];
      fullValue: number;
      populationValues: number[];
      legacyPopulationValues: number[];
      negativeCorrectionValues: number[];
      multiAxisPopulation: number;
      sourceGradient: number[];
      ownsData: boolean;
      dtypes: string[];
      halfGradientDtype: string;
      zeroDenominatorInfinite: boolean;
      scalarDefaultNan: boolean;
      scalarPopulation: number;
      errors: Record<string, string>;
    }>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
base = np.asarray(valid["inputValues"], dtype=np.float32).reshape(valid["inputShape"])
source = grad.Tensor(base, requires_grad=True)
output = source.var(dim=valid["axis"], correction=valid["correction"])
negative_axis = source.var(dim=valid["negativeAxis"])
keepdims = source.var(dim=valid["axis"], keepdim=True)
full = source.var()
population = source.var(dim=valid["axis"], correction=0)
legacy_population = source.var(dim=valid["axis"], unbiased=False)
negative_correction = source.var(dim=valid["axis"], correction=-1)
multi_axis_population = source.var(dim=(0, 1), correction=0)
output.sum().backward()

dtypes = []
half_gradient_dtype = None
for case in fixture["dtypeCases"]:
    typed = grad.Tensor(
        np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(valid["inputShape"]),
        dtype=case["dtype"],
        requires_grad=case["dtype"] == "float16",
    )
    variance = typed.var(dim=valid["axis"])
    dtypes.append(variance.dtype)
    if case["dtype"] == "float16":
        variance.sum().backward()
        half_gradient_dtype = typed.grad.dtype

zero_denominator = source.var(dim=1, correction=3)
scalar = grad.Tensor(np.asarray(2.0, dtype=np.float32))

errors = {}
for case in fixture["invalid"]:
    try:
        if case["kind"] == "axis":
            source.var(dim=case["value"])
        elif case["kind"] == "both-axis-aliases":
            source.var(dim=0, axis=1)
        elif case["kind"] == "keepdims":
            source.var(dim=0, keepdim=case["value"])
        elif case["kind"] == "correction":
            source.var(dim=0, correction=case["value"])
        elif case["kind"] == "unbiased":
            source.var(dim=0, unbiased=case["value"])
        elif case["kind"] == "both-correction-aliases":
            source.var(dim=0, correction=0, unbiased=False)
        elif case["kind"] == "dtype":
            typed = grad.Tensor(
                np.asarray(valid["inputValues"], dtype=np.dtype(case["value"])).reshape(valid["inputShape"]),
                dtype=case["value"],
            )
            typed.var(dim=1)
        elif case["kind"] == "scalar":
            scalar.var(dim=case["value"])
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

{
    "schema": fixture["schema"],
    "outputShape": list(output.shape),
    "outputValues": output.tolist(),
    "negativeAxisValues": negative_axis.tolist(),
    "keepdimsShape": list(keepdims.shape),
    "keepdimsValues": keepdims.tolist(),
    "fullValue": float(full.item()),
    "populationValues": population.tolist(),
    "legacyPopulationValues": legacy_population.tolist(),
    "negativeCorrectionValues": negative_correction.tolist(),
    "multiAxisPopulation": float(multi_axis_population.item()),
    "sourceGradient": source.grad.data.reshape(-1).tolist(),
    "ownsData": bool(output.data.flags["OWNDATA"] and full.data.flags["OWNDATA"]),
    "dtypes": dtypes,
    "halfGradientDtype": half_gradient_dtype,
    "zeroDenominatorInfinite": bool(np.isinf(zero_denominator.data).all()),
    "scalarDefaultNan": bool(np.isnan(scalar.var().item())),
    "scalarPopulation": float(scalar.var(correction=0).item()),
    "errors": errors,
}
`);

    expect(result.schema).toBe(FRAMEWORK_VAR_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_VAR_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_VAR_CONFORMANCE.valid.outputValues);
    expect(result.negativeAxisValues).toEqual(FRAMEWORK_VAR_CONFORMANCE.valid.outputValues);
    expect(result.keepdimsShape).toEqual(FRAMEWORK_VAR_CONFORMANCE.valid.keepdimsShape);
    expect(result.keepdimsValues).toEqual(FRAMEWORK_VAR_CONFORMANCE.valid.keepdimsValues);
    expect(result.fullValue).toBe(FRAMEWORK_VAR_CONFORMANCE.valid.fullValue);
    for (const value of result.populationValues) expect(value).toBeCloseTo(2 / 3, 6);
    for (const value of result.legacyPopulationValues) expect(value).toBeCloseTo(2 / 3, 6);
    expect(result.negativeCorrectionValues).toEqual([0.5, 0.5]);
    expect(result.multiAxisPopulation).toBeCloseTo(35 / 12, 6);
    expect(result.sourceGradient).toEqual(FRAMEWORK_VAR_CONFORMANCE.valid.sourceGradient);
    expect(result.ownsData).toBe(true);
    expect(result.dtypes).toEqual(
      FRAMEWORK_VAR_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    expect(result.halfGradientDtype).toBe("float16");
    expect(result.zeroDenominatorInfinite).toBe(true);
    expect(result.scalarDefaultNan).toBe(true);
    expect(result.scalarPopulation).toBe(0);
    for (const invalid of FRAMEWORK_VAR_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).toContain(invalid.message);
    }
  });
});
