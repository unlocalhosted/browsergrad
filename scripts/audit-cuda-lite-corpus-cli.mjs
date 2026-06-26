export function parseArgs(args) {
  let corpusPathArg;
  let details = false;
  let includeSources = false;
  let emitKernelFile;
  let emitKernelName;
  const expectations = {};
  let firstFailureLimit = 80;
  let help = false;
  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === "--") continue;
    if (arg === "--details" || arg === "--json") {
      details = true;
      continue;
    }
    if (arg === "--sources") {
      details = true;
      includeSources = true;
      continue;
    }
    if (arg === "--emit-kernel-source") {
      emitKernelFile = args[++index];
      if (!emitKernelFile) {
        console.error("--emit-kernel-source expects a relative file path");
        process.exit(2);
      }
      continue;
    }
    if (arg === "--kernel-name") {
      emitKernelName = args[++index];
      if (!emitKernelName) {
        console.error("--kernel-name expects a CUDA kernel name");
        process.exit(2);
      }
      continue;
    }
    if (arg === "--limit") {
      const value = Number(args[++index]);
      if (!Number.isInteger(value) || value < 0) {
        console.error("--limit expects a non-negative integer");
        process.exit(2);
      }
      firstFailureLimit = value;
      continue;
    }
    if (arg?.startsWith("--limit=")) {
      const value = Number(arg.slice("--limit=".length));
      if (!Number.isInteger(value) || value < 0) {
        console.error("--limit expects a non-negative integer");
        process.exit(2);
      }
      firstFailureLimit = value;
      continue;
    }
    const expectation = parseExpectationArg(arg, args, index);
    if (expectation) {
      expectations[expectation.key] = expectation.value;
      index = expectation.nextIndex;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      help = true;
      continue;
    }
    if (!corpusPathArg) {
      corpusPathArg = arg;
      continue;
    }
    console.error(`unexpected argument: ${arg}`);
    process.exit(2);
  }
  const emitKernelSource = emitKernelFile !== undefined || emitKernelName !== undefined
    ? { file: emitKernelFile, kernelName: emitKernelName }
    : undefined;
  if (emitKernelSource !== undefined && (!emitKernelSource.file || !emitKernelSource.kernelName)) {
    console.error("--emit-kernel-source requires --kernel-name");
    process.exit(2);
  }
  return { corpusPathArg, details, emitKernelSource, expectations, firstFailureLimit, help, includeSources };
}

export function parseExpectationArg(arg, args, index) {
  const specs = {
    "--expect-total": "totalKernelDefinitions",
    "--expect-ok-min": "okMin",
    "--expect-single-dispatch-min": "singleDispatchPlanCompiledMin",
    "--expect-webgpu-lifted-min": "hostOrchestratedPlanCompiledMin",
    "--expect-host-orchestrated-plan-compiled-min": "hostOrchestratedPlanCompiledMin",
    "--expect-plan-compiled-min": "planCompiledMin",
    "--expect-compile-codegen-min": "compileCodegenMin",
    "--expect-webgpu-min": "planCompiledMin",
    "--expect-reference-fallback-min": "referenceFallbackMin",
    "--expect-reference-only-max": "referenceOnlyMax",
    "--expect-hard-fail-max": "hardFailMax",
  };
  for (const [flag, key] of Object.entries(specs)) {
    if (arg === flag) {
      return { key, value: parseExpectationValue(flag, args[index + 1]), nextIndex: index + 1 };
    }
    if (arg?.startsWith(`${flag}=`)) {
      return { key, value: parseExpectationValue(flag, arg.slice(flag.length + 1)), nextIndex: index };
    }
  }
  return undefined;
}

export function parseExpectationValue(flag, rawValue) {
  const value = Number(rawValue);
  if (!Number.isInteger(value) || value < 0) {
    console.error(`${flag} expects a non-negative integer`);
    process.exit(2);
  }
  return value;
}

export function checkExpectations(summary, expectations) {
  const failures = [];
  if (expectations.totalKernelDefinitions !== undefined && summary.totalKernelDefinitions !== expectations.totalKernelDefinitions) {
    failures.push(`totalKernelDefinitions expected ${expectations.totalKernelDefinitions}, got ${summary.totalKernelDefinitions}`);
  }
  minExpectation(failures, summary.ok, expectations.okMin, "ok");
  minExpectation(failures, summary.singleDispatchPlanCompiledOk, expectations.singleDispatchPlanCompiledMin, "singleDispatchPlanCompiledOk");
  minExpectation(failures, summary.hostOrchestratedPlanCompiledOk, expectations.hostOrchestratedPlanCompiledMin, "hostOrchestratedPlanCompiledOk");
  minExpectation(failures, summary.planCompiledOk, expectations.planCompiledMin, "planCompiledOk");
  minExpectation(failures, summary.compileCodegenOk, expectations.compileCodegenMin, "compileCodegenOk");
  minExpectation(failures, summary.referenceFallbackOk, expectations.referenceFallbackMin, "referenceFallbackOk");
  maxExpectation(failures, summary.referenceOnlyOk, expectations.referenceOnlyMax, "referenceOnlyOk");
  maxExpectation(failures, summary.hardFail, expectations.hardFailMax, "hardFail");
  return failures;
}

export function minExpectation(failures, actual, expected, label) {
  if (expected !== undefined && actual < expected) failures.push(`${label} expected >= ${expected}, got ${actual}`);
}

export function maxExpectation(failures, actual, expected, label) {
  if (expected !== undefined && actual > expected) failures.push(`${label} expected <= ${expected}, got ${actual}`);
}
