import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_SCATTER_CONFORMANCE } from "../../../test-support/framework-scatter-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager torch.scatter conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches overwrite values, dtypes, gradients, ownership, aliases, and refusals", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_SCATTER_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      outputShape: number[];
      outputValues: number[];
      negativeAxisValues: number[];
      scalarOutputValues: number[];
      targetGradient: number[];
      sourceGradient: number[];
      ownsData: boolean;
      dtypes: string[];
      halfGradientDtypes: string[];
      emptyValues: number[];
      aliasValues: number[];
      errors: Record<string, string>;
    }>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
target_values = np.asarray(valid["targetValues"], dtype=np.float32).reshape(valid["targetShape"])
index_values = np.asarray(valid["indexValues"], dtype=np.int64).reshape(valid["indexShape"])
source_values = np.asarray(valid["sourceValues"], dtype=np.float32).reshape(valid["indexShape"])

target = grad.Tensor(target_values, requires_grad=True)
index = grad.Tensor(index_values, dtype="int64")
source = grad.Tensor(source_values, requires_grad=True)
output = target.scatter(valid["axis"], index, source)
negative_axis = grad.scatter(
    grad.Tensor(target_values),
    valid["negativeAxis"],
    index,
    grad.Tensor(source_values),
)
scalar_output = grad.scatter(
    grad.Tensor(target_values),
    valid["axis"],
    index,
    valid["scalarSource"],
)
output.sum().backward()

dtypes = []
half_gradient_dtypes = []
for case in fixture["dtypeCases"]:
    dtype = case["dtype"]
    if dtype == "bool":
        typed_target_values = (target_values != 0)
        typed_source_values = np.asarray([[False, True], [True, False]], dtype=np.bool_)
    else:
        typed_target_values = target_values.astype(np.dtype(dtype))
        typed_source_values = source_values.astype(np.dtype(dtype))
    typed_target = grad.Tensor(
        typed_target_values,
        dtype=dtype,
        requires_grad=dtype == "float16",
    )
    typed_source = grad.Tensor(
        typed_source_values,
        dtype=dtype,
        requires_grad=dtype == "float16",
    )
    typed_output = typed_target.scatter(valid["axis"], index, typed_source)
    dtypes.append(typed_output.dtype)
    if dtype == "float16":
        typed_output.sum().backward()
        half_gradient_dtypes = [typed_target.grad.dtype, typed_source.grad.dtype]

empty_index = grad.Tensor(np.empty((2, 0), dtype=np.int64), dtype="int64")
empty_source = grad.Tensor(np.empty((2, 0), dtype=np.float32))
empty_values = grad.Tensor(target_values).scatter(1, empty_index, empty_source).data.reshape(-1).tolist()

grad.install_torch_alias()
import torch
alias_values = torch.scatter(
    torch.tensor(target_values),
    1,
    torch.tensor(index_values, dtype=torch.int64),
    torch.tensor(source_values),
).data.reshape(-1).tolist()

errors = {}
for case in fixture["invalid"]:
    try:
        kind = case["kind"]
        if kind == "axis":
            grad.scatter(grad.Tensor(target_values), case["value"], index, grad.Tensor(source_values))
        elif kind == "index-dtype":
            bad = grad.Tensor(index_values.astype(np.dtype(case["value"])), dtype=case["value"])
            grad.scatter(grad.Tensor(target_values), 1, bad, grad.Tensor(source_values))
        elif kind == "index-shape":
            shape = tuple(case["value"])
            bad = grad.Tensor(np.zeros(shape, dtype=np.int64), dtype="int64")
            bad_source = grad.Tensor(np.zeros(shape, dtype=np.float32))
            grad.scatter(grad.Tensor(target_values), 1, bad, bad_source)
        elif kind == "non-tensor-index":
            grad.scatter(grad.Tensor(target_values), 1, case["value"], grad.Tensor(source_values))
        elif kind == "source-shape":
            bad_source = grad.Tensor(np.zeros(tuple(case["value"]), dtype=np.float32))
            grad.scatter(grad.Tensor(target_values), 1, index, bad_source)
        elif kind == "source-dtype":
            bad_source = grad.Tensor(source_values.astype(np.float64), dtype="float64")
            grad.scatter(grad.Tensor(target_values), 1, index, bad_source)
        elif kind == "index-values":
            bad = grad.Tensor(
                np.asarray(case["value"], dtype=np.int64).reshape(valid["indexShape"]),
                dtype="int64",
            )
            grad.scatter(grad.Tensor(target_values), 1, bad, grad.Tensor(source_values))
        elif kind == "reduce":
            grad.scatter(
                grad.Tensor(target_values),
                1,
                index,
                grad.Tensor(source_values),
                reduce=case["value"],
            )
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

{
    "schema": fixture["schema"],
    "outputShape": list(output.shape),
    "outputValues": output.data.reshape(-1).tolist(),
    "negativeAxisValues": negative_axis.data.reshape(-1).tolist(),
    "scalarOutputValues": scalar_output.data.reshape(-1).tolist(),
    "targetGradient": target.grad.data.reshape(-1).tolist(),
    "sourceGradient": source.grad.data.reshape(-1).tolist(),
    "ownsData": bool(output.data.flags["OWNDATA"]),
    "dtypes": dtypes,
    "halfGradientDtypes": half_gradient_dtypes,
    "emptyValues": empty_values,
    "aliasValues": alias_values,
    "errors": errors,
}
`);

    expect(result.schema).toBe(FRAMEWORK_SCATTER_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.outputValues);
    expect(result.negativeAxisValues).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.outputValues);
    expect(result.scalarOutputValues).toEqual(
      FRAMEWORK_SCATTER_CONFORMANCE.valid.scalarOutputValues,
    );
    expect(result.targetGradient).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.targetGradient);
    expect(result.sourceGradient).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.sourceGradient);
    expect(result.ownsData).toBe(true);
    expect(result.dtypes).toEqual(
      FRAMEWORK_SCATTER_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    expect(result.halfGradientDtypes).toEqual(["float16", "float16"]);
    expect(result.emptyValues).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.targetValues);
    expect(result.aliasValues).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.outputValues);
    for (const invalid of FRAMEWORK_SCATTER_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id], invalid.id).toContain(invalid.message);
    }
  });
});
