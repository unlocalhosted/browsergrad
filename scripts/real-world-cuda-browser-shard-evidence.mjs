import { readFileSync } from "node:fs";

export function aggregateBrowserShardReports(steps, loadReport = loadBrowserShardReport) {
  const browserSteps = steps.filter((step) => step.kind === "browser-e2e");
  const grouped = new Map();
  for (const step of browserSteps) {
    const group = grouped.get(step.bundle) ?? [];
    group.push(step);
    grouped.set(step.bundle, group);
  }
  const bundles = [...grouped.entries()].map(([bundle, shardSteps]) =>
    aggregateBrowserBundleReports(bundle, shardSteps, loadReport));
  return Object.freeze({
    kind: "browsergrad-real-world-cuda-browser-shard-evidence",
    version: 1,
    ok: bundles.every((entry) => entry.ok),
    bundles,
  });
}

function aggregateBrowserBundleReports(bundle, shardSteps, loadReport) {
  const failures = [];
  const sortedSteps = [...shardSteps].sort((left, right) => left.shardIndex - right.shardIndex);
  const declaredShardCount = sortedSteps[0]?.shardCount ?? 0;
  if (sortedSteps.length !== declaredShardCount) {
    failures.push(`expected ${declaredShardCount} shard report(s), planned ${sortedSteps.length}`);
  }
  const expectedIndexes = Array.from({ length: declaredShardCount }, (_, index) => index + 1);
  const actualIndexes = sortedSteps.map((step) => step.shardIndex);
  if (!sameValues(actualIndexes, expectedIndexes)) {
    failures.push(`planned shard indexes ${actualIndexes.join(",")} do not match ${expectedIndexes.join(",")}`);
  }
  if (!sortedSteps.every((step) => step.forbidSkips)) {
    failures.push("every browser shard must forbid skips");
  }
  if (new Set(sortedSteps.map((step) => step.caseTimeoutMs)).size > 1) {
    failures.push("browser shards must use one case timeout");
  }

  const expectedFixtureNames = sortedSteps.flatMap((step) => step.expectedFixtureCases);
  const expectedOutputFixtureNames = sortedSteps.flatMap((step) => step.expectedOutputFixtureCases);
  const duplicatePlannedFixtures = duplicateValues(expectedFixtureNames);
  if (duplicatePlannedFixtures.length > 0) {
    failures.push(`fixture partition duplicated: ${duplicatePlannedFixtures.join(", ")}`);
  }

  const loaded = [];
  for (const step of sortedSteps) {
    try {
      loaded.push({ step, report: loadReport(step.reportPath) });
    } catch (error) {
      failures.push(
        `shard ${step.shardIndex}/${step.shardCount} report unavailable: ` +
        String(error?.message ?? error),
      );
    }
  }

  const unavailableReports = loaded.filter(({ report }) => report?.available !== true);
  const requireWebGpu = sortedSteps.some((step) => step.requireWebGpu);
  if (requireWebGpu && unavailableReports.length > 0) {
    failures.push(
      `WebGPU unavailable in shard(s): ` +
      unavailableReports.map(({ step }) => step.shardIndex).join(","),
    );
  }
  const evaluateCoverage = unavailableReports.length === 0 && loaded.length === sortedSteps.length;
  const fixtureCaseNames = [];
  const outputPinnedFixtureNames = [];
  const autoCorpusCaseNames = [];
  let expectedAutoCorpusCases = 0;
  let passed = 0;
  let failed = 0;
  let skipped = 0;
  let warmupFailed = 0;
  const shards = [];

  for (const { step, report } of loaded) {
    const cases = Array.isArray(report?.cases) ? report.cases : [];
    const shardFixtureNames = cases
      .filter((item) => item?.name?.startsWith("corpus:"))
      .map((item) => item.name);
    const shardOutputPinnedNames = cases
      .filter((item) => item?.name?.startsWith("corpus:") && item.expectedOutputPinned === true)
      .map((item) => item.name);
    const shardAutoCorpusNames = cases
      .filter((item) => item?.name?.startsWith("auto-corpus:"))
      .map((item) => item.name);
    fixtureCaseNames.push(...shardFixtureNames);
    outputPinnedFixtureNames.push(...shardOutputPinnedNames);
    autoCorpusCaseNames.push(...shardAutoCorpusNames);
    expectedAutoCorpusCases += report?.autoCorpusSmokeExpectedCovered ?? 0;
    passed += report?.passed ?? 0;
    failed += report?.failed ?? 0;
    skipped += report?.skipped ?? 0;
    warmupFailed += report?.warmupFailed ?? 0;

    if (report?.bundle !== bundle) {
      failures.push(`shard ${step.shardIndex} reported bundle ${String(report?.bundle)}`);
    }
    if (report?.available === true) {
      if (
        report?.autoCorpusSmokeShard?.index !== step.shardIndex ||
        report?.autoCorpusSmokeShard?.count !== step.shardCount
      ) {
        failures.push(
          `shard ${step.shardIndex} reported auto-corpus shard ` +
          JSON.stringify(report?.autoCorpusSmokeShard),
        );
      }
      if ((report.failed ?? 0) !== 0) {
        failures.push(`shard ${step.shardIndex} reported ${report.failed} failed case(s)`);
      }
      if ((report.skipped ?? 0) !== 0) {
        failures.push(`shard ${step.shardIndex} reported ${report.skipped} skipped case(s)`);
      }
      if ((report.warmupFailed ?? 0) !== 0) {
        failures.push(`shard ${step.shardIndex} reported ${report.warmupFailed} warmup failure(s)`);
      }
      if ((report.autoCorpusSmokeCovered ?? 0) !== (report.autoCorpusSmokeExpectedCovered ?? 0)) {
        failures.push(
          `shard ${step.shardIndex} auto-corpus coverage ` +
          `${report.autoCorpusSmokeCovered ?? 0}/${report.autoCorpusSmokeExpectedCovered ?? 0}`,
        );
      }
    }
    shards.push({
      index: step.shardIndex,
      count: step.shardCount,
      available: report?.available === true,
      passed: report?.passed ?? 0,
      failed: report?.failed ?? 0,
      skipped: report?.skipped ?? 0,
      warmupFailed: report?.warmupFailed ?? 0,
      fixtureCases: shardFixtureNames.length,
      outputPinnedFixtureCases: shardOutputPinnedNames.length,
      autoCorpusSmokeCovered: report?.autoCorpusSmokeCovered ?? 0,
      autoCorpusSmokeExpectedCovered: report?.autoCorpusSmokeExpectedCovered ?? 0,
    });
  }

  const missingFixtureCases = difference(expectedFixtureNames, fixtureCaseNames);
  const unexpectedFixtureCases = difference(fixtureCaseNames, expectedFixtureNames);
  const duplicateFixtureCases = duplicateValues(fixtureCaseNames);
  const missingOutputPinnedFixtureCases = difference(expectedOutputFixtureNames, outputPinnedFixtureNames);
  const unexpectedOutputPinnedFixtureCases = difference(
    outputPinnedFixtureNames,
    expectedOutputFixtureNames,
  );
  const duplicateAutoCorpusCases = duplicateValues(autoCorpusCaseNames);
  if (evaluateCoverage) {
    pushNamedFailure(failures, "missing fixture case(s)", missingFixtureCases);
    pushNamedFailure(failures, "unexpected fixture case(s)", unexpectedFixtureCases);
    pushNamedFailure(failures, "duplicate fixture case(s)", duplicateFixtureCases);
    pushNamedFailure(
      failures,
      "fixture output threshold missing case(s)",
      missingOutputPinnedFixtureCases,
    );
    pushNamedFailure(
      failures,
      "unexpected fixture output threshold case(s)",
      unexpectedOutputPinnedFixtureCases,
    );
    pushNamedFailure(failures, "duplicate auto-corpus case(s)", duplicateAutoCorpusCases);
    if (autoCorpusCaseNames.length !== expectedAutoCorpusCases) {
      failures.push(`aggregate auto-corpus coverage ${autoCorpusCaseNames.length}/${expectedAutoCorpusCases}`);
    }
    if (passed !== fixtureCaseNames.length + autoCorpusCaseNames.length) {
      failures.push(
        `aggregate passed count ${passed} does not match ` +
        `${fixtureCaseNames.length + autoCorpusCaseNames.length} covered case(s)`,
      );
    }
  }

  return Object.freeze({
    bundle,
    shardCount: declaredShardCount,
    ok: failures.length === 0,
    requireWebGpu,
    forbidSkips: sortedSteps.every((step) => step.forbidSkips),
    caseTimeoutMs: sortedSteps[0]?.caseTimeoutMs ?? 0,
    coverageEvaluated: evaluateCoverage,
    thresholds: {
      fixtureCasesExact: expectedFixtureNames.length,
      outputPinnedFixtureCasesExact: expectedOutputFixtureNames.length,
      autoCorpusSmokeCasesExact: expectedAutoCorpusCases,
      failedCasesMax: 0,
      skippedCasesMax: 0,
      warmupFailedMax: 0,
    },
    expectedFixtureCases: expectedFixtureNames.length,
    observedFixtureCases: fixtureCaseNames.length,
    expectedOutputPinnedFixtureCases: expectedOutputFixtureNames.length,
    observedOutputPinnedFixtureCases: outputPinnedFixtureNames.length,
    expectedAutoCorpusSmokeCases: expectedAutoCorpusCases,
    observedAutoCorpusSmokeCases: autoCorpusCaseNames.length,
    passed,
    failed,
    skipped,
    warmupFailed,
    missingFixtureCases,
    unexpectedFixtureCases,
    duplicateFixtureCases,
    missingOutputPinnedFixtureCases,
    unexpectedOutputPinnedFixtureCases,
    duplicateAutoCorpusCases,
    shards,
    failures,
  });
}

function loadBrowserShardReport(reportPath) {
  return JSON.parse(readFileSync(reportPath, "utf8"));
}

function sameValues(actual, expected) {
  return actual.length === expected.length && actual.every((value, index) => value === expected[index]);
}

function difference(left, right) {
  const rightSet = new Set(right);
  return [...new Set(left)].filter((value) => !rightSet.has(value));
}

function duplicateValues(values) {
  const seen = new Set();
  const duplicates = new Set();
  for (const value of values) {
    if (seen.has(value)) duplicates.add(value);
    seen.add(value);
  }
  return [...duplicates];
}

function pushNamedFailure(failures, label, names) {
  if (names.length > 0) failures.push(`${label}: ${names.join(", ")}`);
}
