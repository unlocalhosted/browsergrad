# @unlocalhosted/browsergrad-scaling

Browser-safe scaling-law and hosted API oracles for BrowserGrad assignment
rubrics.

This package exists for CS336 Assignment 3-style labs where the upstream path
uses a hosted training API, budget accounting, scheduler fairness, and scaling
law projections. It does not run JAX, Modal, Postgres, or W&B in the browser.
It gives platforms a deterministic fixture-scale truth source first.

## Install

```sh
npm install @unlocalhosted/browsergrad-scaling
```

## API

```ts
import {
  createHostedScalingApiMock,
  fitPowerLawScalingLaw,
  selectExperimentsForDispatch,
} from "@unlocalhosted/browsergrad-scaling";
```

### Hosted API Mock

```ts
const api = createHostedScalingApiMock({ totalBudgetSeconds: 60 });
api.registerUser({ sunetId: "alice", apiKey: "test-api-key" });

api.submitExperiment("test-api-key", {
  architecture_config: { hidden_size: 256 },
  optimizer_config: { learning_rate: 0.001 },
  max_runtime_seconds: 15,
});

api.getBudget("test-api-key");
api.listExperiments("test-api-key");
api.saveFinalSubmission("test-api-key", config, 2.75);
```

The response shapes intentionally mirror the portable CS336 A3 API tests:
`/budget`, `/submit`, `/experiments`, `/experiment/{id}`,
`/final_submission`, and the dashboard HTML smoke.

### Scheduler Selector

```ts
const selection = selectExperimentsForDispatch(experiments, {
  maxConcurrentWorkers: 9,
});
```

The selector matches the upstream fairness contract: queued experiments are
ranked by effective per-user running count, then queue time, then ID.

### Scaling-Law Fit

```ts
const fit = fitPowerLawScalingLaw(
  [
    { compute: 1, loss: 4 },
    { compute: 4, loss: 2 },
    { compute: 16, loss: 1 },
  ],
  { x: "compute", y: "loss" },
);

fit.predict(64); // 0.5
```

Fits `y = multiplier * x^exponent` with ordinary least squares in log space.
Non-positive samples are rejected loudly because log-space fits cannot represent
them.

## BrowserGrad Profile Use

Use with profile capabilities:

- `hosted-api-mock`
- `server-fixture`
- `scheduler-simulator`
- `scaling-law-oracle`

Keep full hosted/server/JAX paths as explicit `external` capability routes.
