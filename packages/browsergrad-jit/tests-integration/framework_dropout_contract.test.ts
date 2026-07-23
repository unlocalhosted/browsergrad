import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_DROPOUT_CONFORMANCE } from "../../../test-support/framework-dropout-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed torch.nn.functional.dropout contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("validates before identity branches and preserves exact branch/dtype behavior", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_DROPOUT_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

x = bg.from_numpy(np.ones(tuple(fixture["inputShape"]), dtype=np.float32))
np.random.seed(91)
identity_p0 = F.dropout(x, p=0.0, training=True)
after_p0 = int(np.random.randint(0, np.iinfo(np.int64).max, dtype=np.int64))
np.random.seed(91)
expected_after_p0 = int(np.random.randint(0, np.iinfo(np.int64).max, dtype=np.int64))

empty = bg.from_numpy(np.empty((0, 3), dtype=np.float32))
drop_all = {}
for dtype in fixture["deterministicDropAllDtypes"]:
    values = np.ones((2, 3), dtype=np.dtype(dtype))
    source = bg.from_numpy(values)
    output = F.dropout(source, p=1.0, training=True)
    realized = output.numpy()
    drop_all[dtype] = {
        "opcode": output._uop.op,
        "dtype": realized.dtype.name,
        "allZero": bool(np.all(realized == 0)),
        "owning": bool(not np.shares_memory(realized, values)),
    }

module = bg.nn.Dropout(1.0)
module_value = module(x).numpy()

{
    "identityP0": identity_p0 is x,
    "identityEval": F.dropout(x, p=0.5, training=False) is x,
    "identityEmpty": F.dropout(empty, p=0.5, training=True) is empty,
    "identityConsumesNoSeed": after_p0 == expected_after_p0,
    "dropAll": drop_all,
    "module": {
        "p": module.p,
        "inplace": module.inplace,
        "allZero": bool(np.all(module_value == 0)),
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
            bg.from_numpy(np.ones((2,), dtype=np.int32)),
            p=0.5,
        )),
        "moduleProbability": error(lambda: bg.nn.Dropout(float("nan"))),
        "moduleInplace": error(lambda: bg.nn.Dropout(0.5, inplace=True)),
    },
}
`);

    expect(result.identityP0).toBe(true);
    expect(result.identityEval).toBe(true);
    expect(result.identityEmpty).toBe(true);
    expect(result.identityConsumesNoSeed).toBe(true);
    for (const dtype of FRAMEWORK_DROPOUT_CONFORMANCE.deterministicDropAllDtypes) {
      expect((result.dropAll as Record<string, unknown>)[dtype]).toEqual({
        opcode: "DROPOUT",
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
    for (const value of Object.values(result.errors as Record<string, string>)) {
      expect(value).not.toBe("no_error");
    }
    expect((result.errors as Record<string, string>).inplace).toMatch(/JitNotImplementedError/u);
  });

  it("owns an immutable seed and replays the exact mask for closure and symbolic gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._framework_contracts import validate_dropout_contract
from browsergrad_jit._vjp import get_rule

bg.manual_seed(1729)
x = bg.from_numpy(np.ones((4, 8), dtype=np.float32), requires_grad=True)
y = F.dropout(x, p=0.25, training=True)
first = y.numpy()
second = y.numpy()
y.sum().backward()
closure_grad = x.grad.numpy()
contract = validate_dropout_contract(y._uop)

bg.manual_seed(1729)
x_functional = bg.from_numpy(np.ones((4, 8), dtype=np.float32))
functional_grad = bg.func.grad(
    lambda value: F.dropout(value, p=0.25, training=True).sum()
)(x_functional)
functional_values = functional_grad.numpy()

rule = get_rule("DROPOUT")
dy = bg.from_numpy(np.ones((4, 8), dtype=np.float32))._uop
symbolic = rule(y._uop, y._uop.inputs, dy)[0]

original_p = y._uop.arg["p"]
y._uop.arg["p"] = 2.0
try:
    y.numpy()
    mutation_error = "no_error"
except Exception as exc:
    mutation_error = type(exc).__name__ + ": " + str(exc)
y._uop.arg["p"] = original_p

{
    "opcode": y._uop.op,
    "argFields": sorted(y._uop.arg.keys()),
    "mode": contract.mode,
    "sameReplay": bool(np.array_equal(first, second)),
    "closureMaskMatches": bool(np.array_equal(closure_grad, first)),
    "functionalMatches": bool(np.array_equal(functional_values, first)),
    "symbolicOpcode": symbolic.op,
    "symbolicSeed": symbolic.arg["seed_key"],
    "forwardSeed": y._uop.arg["seed_key"],
    "mutationError": mutation_error,
}
`);

    expect(result).toMatchObject({
      opcode: "DROPOUT",
      argFields: ["inplace", "p", "seed_key", "training"],
      mode: "stochastic",
      sameReplay: true,
      closureMaskMatches: true,
      functionalMatches: true,
      symbolicOpcode: "DROPOUT_VJP",
    });
    expect(result.symbolicSeed).toBe(result.forwardSeed);
    expect(result.mutationError).toMatch(/^RealizationError:/u);
  });

  it("preserves keyed dropout through checkpoints and refuses unevidenced portable/randomness routes", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit.utils.checkpoint import checkpoint

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

bg.manual_seed(404)
reference_input = bg.from_numpy(np.ones((3, 6), dtype=np.float32), requires_grad=True)
reference = F.dropout(reference_input, p=0.4, training=True)
reference_values = reference.numpy()
reference.sum().backward()
reference_gradient = reference_input.grad.numpy()

bg.manual_seed(404)
checkpoint_input = bg.from_numpy(np.ones((3, 6), dtype=np.float32), requires_grad=True)
checkpointed = checkpoint(
    lambda value: F.dropout(value, p=0.4, training=True),
    checkpoint_input,
)
checkpoint_values = checkpointed.numpy()
checkpointed.sum().backward()
checkpoint_gradient = checkpoint_input.grad.numpy()
checkpoint_false_shape = checkpoint(
    lambda value: value + 1,
    checkpoint_input,
    preserve_rng_state=False,
).shape

mapped_input = bg.from_numpy(np.ones((3, 6), dtype=np.float32))
mapped_drop_all = bg.func.vmap(
    lambda row: F.dropout(row, p=1.0, training=True)
)(mapped_input).numpy()
stochastic_input = bg.from_numpy(np.ones((6,), dtype=np.float32))
stochastic = F.dropout(
    stochastic_input,
    p=0.4,
    training=True,
)

{
    "checkpointForward": bool(np.array_equal(reference_values, checkpoint_values)),
    "checkpointGradient": bool(np.array_equal(reference_gradient, checkpoint_gradient)),
    "checkpointFalseShape": checkpoint_false_shape,
    "mappedDropAll": bool(np.all(mapped_drop_all == 0)),
    "errors": {
        "vmap": error(lambda: bg.func.vmap(
            lambda row: F.dropout(row, p=0.4, training=True)
        )(mapped_input)),
        "onnx": error(lambda: bg.onnx.export_inference(
            stochastic,
            input_buffers=(stochastic_input,),
        )),
        "plan": error(lambda: bg.gpu_plan_summary(stochastic)),
        "checkpointFlagType": error(lambda: checkpoint(
            lambda value: value + 1,
            mapped_input,
            preserve_rng_state=1,
        )),
    },
}
`);

    expect(result.checkpointForward).toBe(true);
    expect(result.checkpointGradient).toBe(true);
    expect(result.checkpointFalseShape).toEqual([3, 6]);
    expect(result.mappedDropAll).toBe(true);
    expect((result.errors as Record<string, string>).vmap).toMatch(/randomness policy/u);
    expect((result.errors as Record<string, string>).onnx).toMatch(/OnnxUnmappableOp/u);
    expect((result.errors as Record<string, string>).plan).toMatch(/GpuPlanUnsupported/u);
    expect((result.errors as Record<string, string>).checkpointFlagType).toMatch(/CheckpointError/u);
  });

  it("does not trace-cache stochastic keys while retaining identity-mode cacheability", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import numpy as np

bg.jit.clear_trace_cache()
bg.manual_seed(55)
training = bg.nn.Dropout(0.5)
x = bg.from_numpy(np.ones((8,), dtype=np.float32))
first = training(x)
second = training(x)
training_stats = bg.jit.trace_cache_stats()

bg.jit.clear_trace_cache()
identity = bg.nn.Dropout(0.5)
identity.eval()
identity(x)
identity(x)
identity_stats = bg.jit.trace_cache_stats()

{
    "distinctSeeds": first._uop.arg["seed_key"] != second._uop.arg["seed_key"],
    "trainingStats": training_stats,
    "identityStats": identity_stats,
}
`);

    expect(result.distinctSeeds).toBe(true);
    expect(result.trainingStats).toMatchObject({ entries: 0, hits: 0 });
    expect(result.identityStats).toMatchObject({ entries: 1, hits: 1 });
  });

  it("bounds metadata before allocation and revalidates runtime arrays", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, string>>(`
import numpy as np
from browsergrad_jit._framework_contracts import (
    execute_dropout_array,
    execute_dropout_vjp_array,
    infer_dropout_contract,
)

class Source:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

valid = infer_dropout_contract(Source((2, 3), "float32"), 0.25, True, False, 7)
source = np.ones((2, 3), dtype=np.float32)

{
    "rank": error(lambda: infer_dropout_contract(
        Source((1,) * 33, "float32"), 0.25, True, False, 7
    )),
    "extent": error(lambda: infer_dropout_contract(
        Source(((1 << 28) + 1,), "float32"), 0.25, True, False, 7
    )),
    "output": error(lambda: infer_dropout_contract(
        Source(((1 << 25) + 1,), "float64"), 1.0, True, False, 7
    )),
    "work": error(lambda: infer_dropout_contract(
        Source(((1 << 25) + 1,), "float32"), 0.25, True, False, 7
    )),
    "workspace": error(lambda: infer_dropout_contract(
        Source((25_000_000,), "float16"), 0.25, True, False, 7
    )),
    "seedType": error(lambda: infer_dropout_contract(
        Source((2, 3), "float32"), 0.25, True, False, True
    )),
    "seedRange": error(lambda: infer_dropout_contract(
        Source((2, 3), "float32"), 0.25, True, False, 1 << 63
    )),
    "runtimeShape": error(lambda: execute_dropout_array(
        valid, np.ones((3, 2), dtype=np.float32)
    )),
    "runtimeDtype": error(lambda: execute_dropout_array(
        valid, np.ones((2, 3), dtype=np.float64)
    )),
    "runtimeGradient": error(lambda: execute_dropout_vjp_array(
        valid, np.ones((2, 3), dtype=np.float64), source
    )),
}
`);

    for (const value of Object.values(result)) {
      expect(value).toMatch(/^ShapeError:/u);
    }
    expect(result.rank).toMatch(/rank ceiling/u);
    expect(result.extent).toMatch(/per-axis ceiling/u);
    expect(result.output).toMatch(/output requires/u);
    expect(result.work).toMatch(/projected work/u);
    expect(result.workspace).toMatch(/projected workspace/u);
  });
});
