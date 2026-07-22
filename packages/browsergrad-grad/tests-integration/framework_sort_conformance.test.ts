import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_SORT_CONFORMANCE } from "../../../test-support/framework-sort-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager torch.sort conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("shares ordering, tie, dtype, ownership, gradient, and refusal semantics", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_SORT_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
source = grad.Tensor(
    np.asarray(valid["inputValues"], dtype=np.float32),
    requires_grad=True,
    dtype="float32",
)
ascending_values, ascending_indices = grad.sort(
    source, dim=np.int32(valid["dim"]), stable=True
)
descending_values, descending_indices = source.sort(
    dim=1, descending=True, stable=True
)
ascending_values.backward(
    grad.Tensor(np.asarray(valid["cotangent"], dtype=np.float32), dtype="float32")
)

dtype_cases = []
for case in fixture["dtypeCases"]:
    typed = grad.Tensor(np.asarray(case["input"], dtype=np.dtype(case["dtype"])), dtype=case["dtype"])
    values, indices = grad.sort(typed, stable=True)
    dtype_cases.append({
        "valueDtype": values.dtype,
        "indexDtype": indices.dtype,
        "values": values.data.tolist(),
        "indices": indices.data.tolist(),
    })

scalar_values, scalar_indices = grad.sort(
    grad.Tensor(np.asarray(fixture["scalar"]["value"], dtype=np.float32), dtype="float32"),
    dim=0,
    stable=True,
)
empty_values, empty_indices = grad.sort(
    grad.Tensor(np.empty(tuple(fixture["empty"]["inputShape"]), dtype=np.float32), dtype="float32"),
    stable=True,
)
uint_values, uint_indices = grad.sort(
    grad.Tensor(np.asarray([0, 255, 1, 255], dtype=np.uint8), dtype="uint8"),
    descending=True,
    stable=True,
)
nan_source = grad.Tensor(np.asarray([np.nan, 1.0, np.nan], dtype=np.float32), dtype="float32")
_, nan_ascending_indices = grad.sort(nan_source, stable=True)
_, nan_descending_indices = grad.sort(nan_source, descending=True, stable=True)
signed_min_source = grad.Tensor(
    np.asarray([np.iinfo(np.int64).min, 1, 0], dtype=np.int64),
    dtype="int64",
)
signed_min_values, signed_min_indices = grad.sort(
    signed_min_source,
    descending=True,
    stable=True,
)

mutable_source = grad.Tensor(np.asarray(valid["inputValues"], dtype=np.float32), requires_grad=True, dtype="float32")
mutable_values, mutable_indices = grad.sort(mutable_source, stable=True)
mutable_indices.data[...] = 0
mutable_values.backward(grad.Tensor(np.asarray(valid["cotangent"], dtype=np.float32)))

class HostileDim:
    calls = 0
    def __index__(self):
        HostileDim.calls += 1
        return 0

class HostileBool:
    calls = 0
    def __bool__(self):
        HostileBool.calls += 1
        return False

errors = {}
attempts = {
    "non-tensor": lambda: grad.sort([[1.0]]),
    "bool-dim": lambda: grad.sort(source, dim=True),
    "float-dim": lambda: grad.sort(source, dim=1.0),
    "out-of-range-dim": lambda: grad.sort(source, dim=2),
    "descending-type": lambda: grad.sort(source, descending=1),
    "stable-type": lambda: grad.sort(source, stable=1),
    "out-mutation": lambda: grad.sort(source, out=(source, source)),
    "unsupported-dtype": lambda: grad.sort(grad.Tensor(np.ones((1,), dtype=np.uint16), dtype="uint16")),
    "hostile-dim": lambda: grad.sort(source, dim=HostileDim()),
    "hostile-descending": lambda: grad.sort(source, descending=HostileBool()),
    "hostile-stable": lambda: grad.sort(source, stable=HostileBool()),
    "zero-size-oversized-axis": lambda: grad.sort(
        grad.Tensor(np.empty((0, (1 << 28) + 1), dtype=np.float32), dtype="float32"),
        dim=0,
    ),
}
for case in fixture["invalid"]:
    try:
        attempts[case["id"]]()
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = type(exc).__name__ + ": " + str(exc)

values_array = ascending_values.data
indices_array = ascending_indices.data
values_own = bool(values_array.flags["OWNDATA"])
indices_own = bool(indices_array.flags["OWNDATA"])
values_array[0, 0] = 999
indices_array[0, 0] = 999
rerun_values, rerun_indices = grad.sort(source, stable=True)

{
    "ascendingValues": rerun_values.data.tolist(),
    "ascendingIndices": rerun_indices.data.tolist(),
    "descendingValues": descending_values.data.tolist(),
    "descendingIndices": descending_indices.data.tolist(),
    "gradient": source.grad.data.tolist(),
    "mutationSafeGradient": mutable_source.grad.data.tolist(),
    "dtypeCases": dtype_cases,
    "scalar": {"value": scalar_values.item(), "index": scalar_indices.item()},
    "empty": {"valueShape": list(empty_values.shape), "indexShape": list(empty_indices.shape)},
    "edgeOrdering": {
        "uintValues": uint_values.data.tolist(),
        "uintIndices": uint_indices.data.tolist(),
        "nanAscendingIndices": nan_ascending_indices.data.tolist(),
        "nanDescendingIndices": nan_descending_indices.data.tolist(),
        "signedMinValuesCorrect": bool(np.array_equal(
            signed_min_values.data,
            np.asarray([1, 0, np.iinfo(np.int64).min], dtype=np.int64),
        )),
        "signedMinIndices": signed_min_indices.data.tolist(),
    },
    "ownsData": [values_own, indices_own],
    "errors": errors,
    "hostileCalls": {"dim": HostileDim.calls, "bool": HostileBool.calls},
}
`);

    expect(result).toMatchObject({
      ascendingValues: FRAMEWORK_SORT_CONFORMANCE.valid.ascendingValues,
      ascendingIndices: FRAMEWORK_SORT_CONFORMANCE.valid.ascendingIndices,
      descendingValues: FRAMEWORK_SORT_CONFORMANCE.valid.descendingValues,
      descendingIndices: FRAMEWORK_SORT_CONFORMANCE.valid.descendingIndices,
      gradient: FRAMEWORK_SORT_CONFORMANCE.valid.expectedGradient,
      mutationSafeGradient: FRAMEWORK_SORT_CONFORMANCE.valid.expectedGradient,
      scalar: {
        value: FRAMEWORK_SORT_CONFORMANCE.scalar.value,
        index: FRAMEWORK_SORT_CONFORMANCE.scalar.expectedIndex,
      },
      empty: {
        valueShape: FRAMEWORK_SORT_CONFORMANCE.empty.expectedShape,
        indexShape: FRAMEWORK_SORT_CONFORMANCE.empty.expectedShape,
      },
      edgeOrdering: {
        uintValues: [255, 255, 1, 0],
        uintIndices: [1, 3, 2, 0],
        nanAscendingIndices: [1, 0, 2],
        nanDescendingIndices: [0, 2, 1],
        signedMinValuesCorrect: true,
        signedMinIndices: [1, 2, 0],
      },
      ownsData: [true, true],
      hostileCalls: { dim: 0, bool: 0 },
    });
    expect(result.dtypeCases).toEqual(
      FRAMEWORK_SORT_CONFORMANCE.dtypeCases.map((testCase) => ({
        valueDtype: testCase.dtype,
        indexDtype: "int64",
        values: testCase.values,
        indices: testCase.indices,
      })),
    );
    const errors = result.errors as Record<string, string>;
    for (const invalid of FRAMEWORK_SORT_CONFORMANCE.invalid) {
      expect(errors[invalid.id]).not.toBe("no_error");
    }
  });
});
