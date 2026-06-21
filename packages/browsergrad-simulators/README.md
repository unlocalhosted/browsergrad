# @unlocalhosted/browsergrad-simulators

Deterministic browser-safe simulators for BrowserGrad parallel and distributed
execution.

This package is intentionally small and dependency-free. It models event traces
and collective behavior for browser workloads that need distributed or parallel
systems concepts without depending on native threads, CUDA, MPI, or Linux
process behavior.

## Public Surface

```ts
import {
  createDeterministicMesh,
  createTaskGraphSimulator,
  partitionStaticWork,
  simulateVectorizedArraySum,
  simulateVectorizedClampedExp,
  simulateDdpGradientSynchronization,
  simulateFsdpGradientReduceScatter,
  simulateFsdpParameterSharding,
  simulateShardedAdamWStep,
} from "@unlocalhosted/browsergrad-simulators";

const mesh = createDeterministicMesh({ ranks: 4 });
const reduced = mesh.allReduce({
  tag: "grad-bucket-0",
  values: [1, 2, 3, 4],
  op: "sum",
});

console.log(reduced); // [10, 10, 10, 10]
console.log(mesh.trace());

const tasks = createTaskGraphSimulator({ workers: 2 });
tasks.addTask({ id: "load", duration: 2 });
tasks.addTask({ id: "decode", duration: 3 });
tasks.addTask({ id: "train", duration: 4, dependsOn: ["load", "decode"] });
console.log(tasks.run().events);

const ddp = simulateDdpGradientSynchronization({
  parameters: [{ name: "fc.weight" }],
  rankGradients: [{ "fc.weight": [1, 3] }, { "fc.weight": [3, 5] }],
});
console.log(ddp.synchronizedGradients);

const fsdp = simulateFsdpParameterSharding({
  ranks: 2,
  parameters: [{ name: "embedding.weight", values: [0, 1, 2, 3] }],
});
console.log(simulateFsdpGradientReduceScatter({
  shardPlan: fsdp.shardPlan,
  rankGradients: [
    { "embedding.weight": [1, 3, 5, 7] },
    { "embedding.weight": [3, 5, 7, 9] },
  ],
}).rankGradientShards);

console.log(simulateShardedAdamWStep({
  ranks: 2,
  parameters: [{ name: "fc.weight", values: [1], gradients: [1] }],
  optimizer: { lr: 0.1 },
}).updatedParameters);

const clamped = simulateVectorizedClampedExp({
  values: [2, 3, -2, 4, 2],
  exponents: [0, 2, 3, 4, 5],
  vectorWidth: 4,
});
console.log(clamped.output, clamped.stats.utilization);

console.log(simulateVectorizedArraySum({
  values: [1, 2, 3, 4, 5, 6, 7, 8],
  vectorWidth: 4,
}).sum);

console.log(partitionStaticWork({ items: 10, workers: 3, chunkSize: 2 }));
```

Use it as a rubric oracle for DDP/FSDP/sharded-optimizer/task-system labs when
the learning goal is ordering, participation, reduction, sharding, and update
semantics rather than native performance. For CS149 A1-style CPU labs, use the
SIMD helpers to verify clamped exponentiation, vector-reduction behavior,
active-lane utilization, tail masking, and static row/task decomposition before
an external C++/ISPC runner is attached.
