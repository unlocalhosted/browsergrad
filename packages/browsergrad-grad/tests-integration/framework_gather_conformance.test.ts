import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_GATHER_CONFORMANCE } from "../../../test-support/framework-gather-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager Tensor.gather conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches strict gather values, dtype, gradient, ownership, and refusals", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_GATHER_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      outputShape: number[];
      outputValues: number[];
      negativeAxisValues: number[];
      sourceGradient: number[];
      ownsData: boolean;
      dtypes: string[];
      halfGradientDtype: string;
      partialValues: number[][];
      indexViewDtypes: string[];
      indexViewGatherValues: number[];
      emptyShape: number[];
      emptyGradient: number[][];
      errors: Record<string, string>;
    }>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
base = np.asarray(valid["inputValues"], dtype=np.float32).reshape(valid["inputShape"])
indices = grad.Tensor(
    np.asarray(valid["indexValues"], dtype=np.int64).reshape(valid["indexShape"]),
    dtype="int64",
)
source = grad.Tensor(base, requires_grad=True)
output = source.gather(valid["axis"], indices)
negative_axis = grad.gather(source, valid["negativeAxis"], indices)
output.sum().backward()

dtypes = []
half_gradient_dtype = None
for case in fixture["dtypeCases"]:
    typed = grad.Tensor(
        np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(valid["inputShape"]),
        dtype=case["dtype"],
        requires_grad=case["dtype"] == "float16",
    )
    gathered = typed.gather(valid["axis"], indices)
    dtypes.append(gathered.dtype)
    if case["dtype"] == "float16":
        gathered.sum().backward()
        half_gradient_dtype = typed.grad.dtype

partial_source = grad.Tensor(np.arange(9, dtype=np.float32).reshape(3, 3))
partial_index = grad.Tensor(np.array([[2, 0], [1, 1]], dtype=np.int64), dtype="int64")
partial = partial_source.gather(1, partial_index)

view_index_source = grad.Tensor(np.array([0, 2, 1, 0], dtype=np.int64), dtype="int64")
view_index_slice = view_index_source[1:].unsqueeze(-1)
view_index_reshape = view_index_source.reshape(2, 2)
view_index_transpose = view_index_reshape.transpose(0, 1)
view_index_permute = view_index_reshape.permute(1, 0)
view_source = grad.Tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
view_gather = view_source.gather(1, view_index_slice)

empty_source = grad.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
empty_index = grad.Tensor(np.empty((2, 0), dtype=np.int64), dtype="int64")
empty = empty_source.gather(1, empty_index)
empty.sum().backward()

errors = {}
for case in fixture["invalid"]:
    try:
        if case["kind"] == "axis":
            source.gather(case["value"], indices)
        elif case["kind"] == "index-dtype":
            bad = grad.Tensor(
                np.asarray(valid["indexValues"], dtype=np.dtype(case["value"])).reshape(valid["indexShape"]),
                dtype=case["value"],
            )
            source.gather(valid["axis"], bad)
        elif case["kind"] == "index-rank":
            bad = grad.Tensor(np.asarray(case["value"], dtype=np.int64), dtype="int64")
            source.gather(valid["axis"], bad)
        elif case["kind"] == "index-shape":
            bad = grad.Tensor(np.zeros(case["value"], dtype=np.int64), dtype="int64")
            source.gather(valid["axis"], bad)
        elif case["kind"] == "non-tensor":
            source.gather(valid["axis"], case["value"])
        elif case["kind"] == "index-values":
            bad = grad.Tensor(np.asarray(case["value"], dtype=np.int64).reshape(valid["indexShape"]), dtype="int64")
            source.gather(valid["axis"], bad)
        elif case["kind"] == "scalar-source":
            scalar_index = grad.Tensor(np.asarray(0, dtype=np.int64), dtype="int64")
            grad.Tensor(np.asarray(2.0, dtype=np.float32)).gather(case["value"], scalar_index)
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

{
    "schema": fixture["schema"],
    "outputShape": list(output.shape),
    "outputValues": output.data.reshape(-1).tolist(),
    "negativeAxisValues": negative_axis.data.reshape(-1).tolist(),
    "sourceGradient": source.grad.data.reshape(-1).tolist(),
    "ownsData": bool(output.data.flags["OWNDATA"]),
    "dtypes": dtypes,
    "halfGradientDtype": half_gradient_dtype,
    "partialValues": partial.tolist(),
    "indexViewDtypes": [
        view_index_source[1:].dtype,
        view_index_slice.dtype,
        view_index_reshape.dtype,
        view_index_transpose.dtype,
        view_index_permute.dtype,
    ],
    "indexViewGatherValues": view_gather.data.reshape(-1).tolist(),
    "emptyShape": list(empty.shape),
    "emptyGradient": empty_source.grad.tolist(),
    "errors": errors,
}
`);

    expect(result.schema).toBe(FRAMEWORK_GATHER_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_GATHER_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_GATHER_CONFORMANCE.valid.outputValues);
    expect(result.negativeAxisValues).toEqual(FRAMEWORK_GATHER_CONFORMANCE.valid.outputValues);
    expect(result.sourceGradient).toEqual(FRAMEWORK_GATHER_CONFORMANCE.valid.sourceGradient);
    expect(result.ownsData).toBe(true);
    expect(result.dtypes).toEqual(
      FRAMEWORK_GATHER_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    expect(result.halfGradientDtype).toBe("float16");
    expect(result.partialValues).toEqual([[2, 0], [4, 4]]);
    expect(result.indexViewDtypes).toEqual(["int64", "int64", "int64", "int64", "int64"]);
    expect(result.indexViewGatherValues).toEqual([2, 5, 8]);
    expect(result.emptyShape).toEqual([2, 0]);
    expect(result.emptyGradient).toEqual([[0, 0, 0], [0, 0, 0]]);
    for (const invalid of FRAMEWORK_GATHER_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).toContain(invalid.message);
    }
  });
});
