import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_PAD_CONFORMANCE } from "../../../test-support/framework-pad-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager torch.nn.functional.pad conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("shares exact padding, fill, dtype, ownership, gradient, and refusal semantics", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_PAD_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      shape: number[];
      values: number[][];
      dtype: string;
      gradient: number[][];
      dtypeCases: Array<{ dtype: string; value: number | boolean }>;
      zeroPad: { shape: number[]; values: number[][]; ownsData: boolean };
      empty: { shape: number[]; values: number[][] };
      integralGradientPresent: boolean;
      ownsData: boolean;
      rerunValues: number[][];
      errors: Record<string, string>;
      hostileCalls: { padding: number; mode: number; value: number };
    }>(`
import browsergrad_grad as grad
import browsergrad_grad.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
source = grad.Tensor(
    np.asarray(valid["inputValues"], dtype=np.float32),
    requires_grad=True,
    dtype="float32",
)
output = F.pad(
    source,
    tuple(np.int32(value) for value in valid["pad"]),
    value=np.float32(valid["value"]),
)
values = output.data.tolist()
output.backward(grad.Tensor(np.asarray(valid["cotangent"], dtype=np.float32)))

dtype_cases = []
for case in fixture["dtypeCases"]:
    typed = grad.Tensor(
        np.asarray([[1]], dtype=np.dtype(case["dtype"])),
        dtype=case["dtype"],
    )
    padded = F.pad(typed, (1, 0), value=case["value"])
    dtype_cases.append({"dtype": padded.dtype, "value": padded.data.tolist()[0][0]})

zero_pad = F.pad(
    grad.Tensor(np.asarray(valid["inputValues"], dtype=np.float32), dtype="float32"),
    tuple(),
)

empty_spec = fixture["empty"]
empty = F.pad(
    grad.Tensor(
        np.empty(tuple(empty_spec["inputShape"]), dtype=np.float32),
        dtype="float32",
    ),
    tuple(empty_spec["pad"]),
    value=empty_spec["value"],
)

integer_source = grad.Tensor(
    np.asarray([1, 2], dtype=np.int64),
    requires_grad=True,
    dtype="int64",
)
F.pad(integer_source, (1, 1)).sum().backward()

class HostilePadding:
    calls = 0
    def __len__(self):
        HostilePadding.calls += 1
        return 2
    def __iter__(self):
        HostilePadding.calls += 1
        return iter((1, 1))

class HostileMode:
    calls = 0
    def __eq__(self, other):
        HostileMode.calls += 1
        return True

class HostileValue:
    calls = 0
    def __float__(self):
        HostileValue.calls += 1
        return 0.0
    def __int__(self):
        HostileValue.calls += 1
        return 0

errors = {}
attempts = {
    "non-tensor": lambda: F.pad([[1.0]], (1, 1)),
    "non-sequence": lambda: F.pad(source, (value for value in (1, 1))),
    "odd-length": lambda: F.pad(source, (1,)),
    "too-many-dimensions": lambda: F.pad(source, (1, 1, 1, 1, 1, 1)),
    "bool-padding": lambda: F.pad(source, (True, 1)),
    "float-padding": lambda: F.pad(source, (1.0, 1)),
    "negative-padding": lambda: F.pad(source, (-1, 0)),
    "hostile-padding": lambda: F.pad(source, HostilePadding()),
    "mode-type": lambda: F.pad(source, (1, 1), mode=7),
    "unsupported-mode": lambda: F.pad(source, (1, 1), mode="reflect"),
    "hostile-mode": lambda: F.pad(source, (1, 1), mode=HostileMode()),
    "hostile-value": lambda: F.pad(source, (1, 1), value=HostileValue()),
    "unsupported-dtype": lambda: F.pad(
        grad.Tensor(np.ones((1,), dtype=np.uint16), dtype="uint16"), (1, 1)
    ),
    "fractional-integer-fill": lambda: F.pad(
        grad.Tensor(np.ones((1,), dtype=np.int32), dtype="int32"),
        (1, 1),
        value=1.5,
    ),
    "overflowing-fill": lambda: F.pad(
        grad.Tensor(np.ones((1,), dtype=np.uint8), dtype="uint8"),
        (1, 1),
        value=256,
    ),
    "nonfinite-floating-fill": lambda: F.pad(source, (1, 1), value=float("inf")),
    "zero-size-oversized-axis": lambda: F.pad(
        grad.Tensor(np.empty((0, 1), dtype=np.float32), dtype="float32"),
        (0, (1 << 28)),
    ),
}
for case in fixture["invalid"]:
    try:
        attempts[case["id"]]()
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = type(exc).__name__ + ": " + str(exc)

owns_data = bool(output.data.flags["OWNDATA"])
output.data[0, 0] = 999
rerun = F.pad(source, tuple(valid["pad"]), value=valid["value"])

{
    "schema": fixture["schema"],
    "shape": list(output.shape),
    "values": values,
    "dtype": output.dtype,
    "gradient": source.grad.data.tolist(),
    "dtypeCases": dtype_cases,
    "zeroPad": {
        "shape": list(zero_pad.shape),
        "values": zero_pad.data.tolist(),
        "ownsData": bool(zero_pad.data.flags["OWNDATA"]),
    },
    "empty": {"shape": list(empty.shape), "values": empty.data.tolist()},
    "integralGradientPresent": integer_source.grad is not None,
    "ownsData": owns_data,
    "rerunValues": rerun.data.tolist(),
    "errors": errors,
    "hostileCalls": {
        "padding": HostilePadding.calls,
        "mode": HostileMode.calls,
        "value": HostileValue.calls,
    },
}
`);

    expect(result.schema).toBe(FRAMEWORK_PAD_CONFORMANCE.schema);
    expect(result.shape).toEqual(FRAMEWORK_PAD_CONFORMANCE.valid.expectedShape);
    expect(result.values).toEqual(FRAMEWORK_PAD_CONFORMANCE.valid.expectedValues);
    expect(result.dtype).toBe("float32");
    expect(result.gradient).toEqual(FRAMEWORK_PAD_CONFORMANCE.valid.expectedGradient);
    expect(result.dtypeCases).toEqual(
      FRAMEWORK_PAD_CONFORMANCE.dtypeCases.map((testCase) => ({
        dtype: testCase.dtype,
        value: testCase.expectedValue,
      })),
    );
    expect(result.zeroPad).toEqual({
      shape: FRAMEWORK_PAD_CONFORMANCE.zeroPad.expectedShape,
      values: FRAMEWORK_PAD_CONFORMANCE.zeroPad.expectedValues,
      ownsData: true,
    });
    expect(result.empty).toEqual({
      shape: FRAMEWORK_PAD_CONFORMANCE.empty.expectedShape,
      values: FRAMEWORK_PAD_CONFORMANCE.empty.expectedValues,
    });
    expect(result.integralGradientPresent).toBe(false);
    expect(result.ownsData).toBe(true);
    expect(result.rerunValues).toEqual(FRAMEWORK_PAD_CONFORMANCE.valid.expectedValues);
    expect(result.hostileCalls).toEqual({ padding: 0, mode: 0, value: 0 });
    for (const invalid of FRAMEWORK_PAD_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).not.toBe("no_error");
    }
  });
});
