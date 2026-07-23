import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_DROPOUT_CONFORMANCE } from "../../../test-support/framework-dropout-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager torch.nn.functional.dropout conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("shares the typed branch, validation, dtype, ownership, and module contract", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_DROPOUT_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_grad as bg
import browsergrad_grad.functional as F
import browsergrad_grad.nn as nn
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

x = bg.Tensor(np.ones(tuple(fixture["inputShape"]), dtype=np.float32))
empty = bg.Tensor(np.empty((0, 3), dtype=np.float32))
drop_all = {}
for dtype in fixture["deterministicDropAllDtypes"]:
    values = np.ones((2, 3), dtype=np.dtype(dtype))
    source = bg.Tensor(values, dtype=dtype)
    output = F.dropout(source, p=1.0, training=True)
    drop_all[dtype] = {
        "dtype": output.data.dtype.name,
        "allZero": bool(np.all(output.data == 0)),
        "owning": bool(not np.shares_memory(output.data, values)),
    }

module = nn.Dropout(1.0)
module_values = module(x).data

{
    "identityP0": F.dropout(x, p=0.0, training=True) is x,
    "identityEval": F.dropout(x, p=0.5, training=False) is x,
    "identityEmpty": F.dropout(empty, p=0.5, training=True) is empty,
    "dropAll": drop_all,
    "module": {
        "p": module.p,
        "inplace": module.inplace,
        "allZero": bool(np.all(module_values == 0)),
        "repr": repr(module),
    },
    "errors": {
        "invalidEvalProbability": error(lambda: F.dropout(x, p=1.1, training=False)),
        "nan": error(lambda: F.dropout(x, p=float("nan"))),
        "inf": error(lambda: F.dropout(x, p=float("inf"))),
        "boolProbability": error(lambda: F.dropout(x, p=True)),
        "training": error(lambda: F.dropout(x, training=1)),
        "inplaceType": error(lambda: F.dropout(x, inplace=1)),
        "inplace": error(lambda: F.dropout(x, inplace=True)),
        "input": error(lambda: F.dropout([1.0])),
        "integerStochastic": error(lambda: F.dropout(
            bg.Tensor(np.ones((2,), dtype=np.int32), dtype="int32"),
            p=0.5,
        )),
        "moduleProbability": error(lambda: nn.Dropout(float("nan"))),
        "moduleInplace": error(lambda: nn.Dropout(0.5, inplace=True)),
    },
}
`);

    expect(result.identityP0).toBe(true);
    expect(result.identityEval).toBe(true);
    expect(result.identityEmpty).toBe(true);
    for (const dtype of FRAMEWORK_DROPOUT_CONFORMANCE.deterministicDropAllDtypes) {
      expect((result.dropAll as Record<string, unknown>)[dtype]).toEqual({
        dtype,
        allZero: true,
        owning: true,
      });
    }
    expect(result.module).toEqual({
      p: 1,
      inplace: false,
      allZero: true,
      repr: "Dropout(p=1.0, inplace=False)",
    });
    for (const [name, value] of Object.entries(result.errors as Record<string, string>)) {
      expect(value, name).not.toBe("no_error");
    }
    expect((result.errors as Record<string, string>).inplace).toMatch(/NotImplementedError/u);
  });

  it("preserves floating dtype and snapshots the exact forward mask for backward", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_DROPOUT_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_grad as bg
import browsergrad_grad.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
observed = {}
for index, dtype in enumerate(fixture["floatingDtypes"]):
    np.random.seed(200 + index)
    source = bg.Tensor(
        np.ones(tuple(fixture["inputShape"]), dtype=np.dtype(dtype)),
        requires_grad=True,
        dtype=dtype,
    )
    output = F.dropout(source, p=fixture["probability"], training=True)
    forward = output.data.copy()
    source.data[...] = 9
    output.sum().backward()
    observed[dtype] = {
        "outputDtype": output.data.dtype.name,
        "gradientDtype": source.grad.data.dtype.name,
        "maskMatches": bool(np.array_equal(output.data, source.grad.data)),
        "sourceMutationDidNotChangeOutput": bool(np.array_equal(output.data, forward)),
    }
observed
`);

    for (const dtype of FRAMEWORK_DROPOUT_CONFORMANCE.floatingDtypes) {
      expect(result[dtype]).toEqual({
        outputDtype: dtype,
        gradientDtype: dtype,
        maskMatches: true,
        sourceMutationDidNotChangeOutput: true,
      });
    }
  });
});
