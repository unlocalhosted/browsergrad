import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import {
  FRAMEWORK_TRIL_CONFORMANCE,
  FRAMEWORK_TRIU_CONFORMANCE,
} from "../../../test-support/framework-triangular-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

const TRIANGULAR_FIXTURES = Object.freeze({
  tril: FRAMEWORK_TRIL_CONFORMANCE,
  triu: FRAMEWORK_TRIU_CONFORMANCE,
});

describe("Gate 6 eager triangular-selection conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("preserves batched matrix shape/dtype with canonical diagonals and gradients", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(TRIANGULAR_FIXTURES);
    const result = await target.run<Record<string, {
      schema: string;
      values: number[];
      gradient: number[];
      alternateValues: number[];
      positiveValues: number[];
      negativeValues: number[];
      ownsData: boolean;
      dtypes: string[];
      emptyShapes: number[][];
      errors: Record<string, string>;
    }> & { hostileCalls: number }>(`
import browsergrad_grad as grad
import json
import numpy as np

fixtures = json.loads(${JSON.stringify(fixtureJson)})

class HostileIndex:
    calls = 0
    def __index__(self):
        HostileIndex.calls += 1
        return 0

results = {}
for operation, fixture in fixtures.items():
    valid = fixture["valid"]
    shape = tuple(valid["inputShape"])
    base = np.asarray(valid["inputValues"], dtype=np.float32).reshape(shape)
    source = grad.Tensor(base, requires_grad=True)
    weights = grad.Tensor(np.arange(1, 13, dtype=np.float32).reshape(shape))
    output = getattr(source, operation)(np.int32(valid["diagonal"]))
    (output * weights).sum().backward()
    alternate_diagonal = 1 if operation == "tril" else 0
    alternate = getattr(grad, operation)(source, np.uint64(alternate_diagonal))
    positive = getattr(source, operation)(10 ** 200)
    negative = getattr(source, operation)(-(10 ** 200))

    dtypes = []
    for case in fixture["dtypeCases"]:
        typed_values = np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(shape)
        typed = grad.Tensor(typed_values, dtype=case["dtype"])
        dtypes.append(getattr(typed, operation)(valid["diagonal"]).dtype)
    empty_shapes = [
        list(getattr(grad.Tensor(np.empty((0, 3), dtype=np.float32)), operation)(-1).shape),
        list(getattr(grad.Tensor(np.empty((2, 0), dtype=np.float32)), operation)(1).shape),
    ]

    errors = {}
    for case in fixture["invalid"]:
        try:
            if case["kind"] == "rank":
                getattr(grad.Tensor(np.ones(case["value"], dtype=np.float32)), operation)()
            elif case["kind"] == "diagonal":
                getattr(source, operation)(case["value"])
            elif case["kind"] == "dtype":
                bad = np.asarray(valid["inputValues"], dtype=np.dtype(case["value"])).reshape(shape)
                getattr(grad.Tensor(bad, dtype=case["value"]), operation)()
            errors[case["id"]] = "no_error"
        except Exception as exc:
            errors[case["id"]] = str(exc)
    try:
        getattr(source, operation)(HostileIndex())
        errors["hostile-diagonal"] = "no_error"
    except Exception as exc:
        errors["hostile-diagonal"] = str(exc)

    results[operation] = {
        "schema": fixture["schema"],
        "values": output.data.reshape(-1).tolist(),
        "gradient": source.grad.data.reshape(-1).tolist(),
        "alternateValues": alternate.data.reshape(-1).tolist(),
        "positiveValues": positive.data.reshape(-1).tolist(),
        "negativeValues": negative.data.reshape(-1).tolist(),
        "ownsData": bool(output.data.flags["OWNDATA"]),
        "dtypes": dtypes,
        "emptyShapes": empty_shapes,
        "errors": errors,
    }
results["hostileCalls"] = HostileIndex.calls
results
`);

    expect(result.hostileCalls).toBe(0);
    for (const [operation, fixture] of Object.entries(TRIANGULAR_FIXTURES)) {
      const observed = result[operation];
      expect(observed?.schema).toBe(fixture.schema);
      expect(observed?.values).toEqual(fixture.valid.outputValues);
      expect(observed?.gradient).toEqual(fixture.valid.sourceGradient);
      expect(observed?.ownsData).toBe(true);
      expect(observed?.dtypes).toEqual(
        fixture.dtypeCases.map(({ expectedDtype }) => expectedDtype),
      );
      expect(observed?.emptyShapes).toEqual([[0, 3], [2, 0]]);
      expect(observed?.errors["hostile-diagonal"]).toContain("integer scalar");
      for (const invalid of fixture.invalid) {
        expect(observed?.errors[invalid.id]).toContain(invalid.message);
      }
    }
    expect(result.tril?.alternateValues).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.upperDiagonalValues);
    expect(result.tril?.positiveValues).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.inputValues);
    expect(result.tril?.negativeValues).toEqual(new Array(12).fill(0));
    expect(result.triu?.alternateValues).toEqual(FRAMEWORK_TRIU_CONFORMANCE.valid.mainDiagonalValues);
    expect(result.triu?.positiveValues).toEqual(new Array(12).fill(0));
    expect(result.triu?.negativeValues).toEqual(FRAMEWORK_TRIU_CONFORMANCE.valid.inputValues);
  });
});
