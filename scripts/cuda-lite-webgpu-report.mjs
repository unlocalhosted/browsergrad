import { cudaLiteCorpusExecutionFixtureBaseline } from "./cuda-lite-corpus-registry.mjs";

export function summarizeReport(data) {
  const cases = data.cases ?? [];
  const repeatStats = repeatCaseStats(cases);
  return {
    tool: data.tool,
    bundle: data.bundle ?? "src",
    available: data.available,
    caseFilters: data.caseFilters ?? [],
    reason: data.available ? undefined : data.reason ?? "unknown",
    passed: data.passed ?? 0,
    failed: data.failed ?? 0,
    skipped: data.skipped ?? 0,
    corpusFixturePassed: data.corpusFixturePassed ?? 0,
    corpusFixtureCases: data.corpusFixtureCases ?? 0,
    corpusFixtureExpectedOutputCases: data.corpusFixtureExpectedOutputCases ?? 0,
    corpusFixturePassedByCorpus: data.corpusFixturePassedByCorpus ?? {},
    failedByStage: data.failedByStage ?? {},
    autoCorpusSmokePassed: data.autoCorpusSmokePassed ?? 0,
    autoCorpusSmokeSkipped: data.autoCorpusSmokeSkipped ?? 0,
    autoCorpusSmokeCovered: data.autoCorpusSmokeCovered ?? data.autoCorpusSmokePassed ?? 0,
    autoCorpusSmokeCases: data.autoCorpusSmokeCases ?? 0,
    autoCorpusSmokePassedByCorpus: data.autoCorpusSmokePassedByCorpus ?? {},
    autoCorpusSmokeSkippedByCorpus: data.autoCorpusSmokeSkippedByCorpus ?? {},
    autoCorpusSmokeMode: data.autoCorpusSmokeMode,
    autoCorpusSmokeProfile: data.autoCorpusSmokeProfile,
    compileOnly: data.compileOnly,
    warmup: data.warmup ?? 0,
    warmupCases: data.warmupCases ?? 0,
    warmupFailed: data.warmupFailed ?? 0,
    warmupFailedByStage: data.warmupFailedByStage ?? {},
    repeat: data.repeat ?? 1,
    autoCorpusSmokeOnly: data.autoCorpusSmokeOnly,
    autoCorpusSmokeLimit: data.autoCorpusSmokeLimit,
    autoCorpusSmokeExpectedCovered: data.autoCorpusSmokeExpectedCovered,
    autoCorpusSmokeShard: data.autoCorpusSmokeShard,
    repeatStats,
    failedCases: cases.filter((item) => !item.ok).map((item) => ({
      name: item.name,
      repeat: item.repeat,
      stage: item.stage,
      plan: item.plan,
      output: item.output,
      maxAbsDiff: item.maxAbsDiff,
      tolerance: item.tolerance,
      firstDiff: item.firstDiff,
      error: item.error,
      corpusId: item.corpusId,
    })),
    skippedCases: cases.filter((item) => item.skipped).map((item) => ({
      name: item.name,
      repeat: item.repeat,
      stage: item.stage,
      plan: item.plan,
      corpusId: item.corpusId,
    })),
    slowestCases: [...cases]
      .filter((item) => Number.isFinite(item.ms))
      .sort((left, right) => right.ms - left.ms)
      .slice(0, 12)
      .map((item) => ({
        name: item.name,
        repeat: item.repeat,
        stage: item.stage,
        plan: item.plan,
        ms: item.ms,
        corpusId: item.corpusId,
        ...(item.profile === undefined ? {} : { profile: item.profile }),
      })),
  };
}

export function repeatCaseStats(cases) {
  const byName = new Map();
  for (const item of cases) {
    if (!Number.isFinite(item.ms) || !Number.isInteger(item.repeat)) continue;
    const entries = byName.get(item.name) ?? [];
    entries.push(item);
    byName.set(item.name, entries);
  }
  return [...byName.entries()]
    .map(([name, entries]) => {
      const sorted = entries.sort((left, right) => left.repeat - right.repeat);
      const cold = sorted.find((item) => item.repeat === 1);
      const warm = sorted.filter((item) => item.repeat > 1);
      if (!cold || warm.length === 0) return undefined;
      const bestWarm = warm.reduce((best, item) => item.ms < best.ms ? item : best, warm[0]);
      return {
        name,
        coldMs: cold.ms,
        bestWarmMs: bestWarm.ms,
        bestWarmRepeat: bestWarm.repeat,
        speedup: bestWarm.ms > 0 ? Math.round((cold.ms / bestWarm.ms) * 100) / 100 : Number.POSITIVE_INFINITY,
      };
    })
    .filter(Boolean)
    .sort((left, right) => right.speedup - left.speedup);
}

export function validateWarmSpeedup(report, minSpeedup) {
  if (minSpeedup === undefined) return;
  const stats = repeatCaseStats(report.cases ?? []);
  if (stats.length === 0) {
    throw new Error("Warm speedup gate needs repeat >= 2; got no repeat stats");
  }
  const slow = stats.filter((item) => item.speedup < minSpeedup);
  if (slow.length > 0) {
    throw new Error(`Warm speedup gate failed: ${slow.map((item) => `${item.name} ${item.speedup}x < ${minSpeedup}x`).join(", ")}`);
  }
}

export function validateWarmMsMax(report, maxMs) {
  if (maxMs === undefined) return;
  const stats = repeatCaseStats(report.cases ?? []);
  if (stats.length === 0) {
    throw new Error("Warm ms gate needs repeat >= 2; got no repeat stats");
  }
  const slow = stats.filter((item) => item.bestWarmMs > maxMs);
  if (slow.length > 0) {
    throw new Error(`Warm ms gate failed: ${slow.map((item) => `${item.name} ${item.bestWarmMs}ms > ${maxMs}ms`).join(", ")}`);
  }
}

export function failureReplayCases(report) {
  const cases = report.cases ?? [];
  return [...new Set(cases
    .filter((item) => !item.ok && item.name)
    .map((item) => item.name))];
}

export function markdownReport(data) {
  const lines = [
    "# BrowserGrad CUDA-lite WebGPU e2e",
    "",
    `Bundle: \`${data.bundle ?? "src"}\``,
    `User agent: \`${data.userAgent ?? "unknown"}\``,
    `Available: \`${data.available}\``,
    "",
  ];
  if (!data.available) {
    lines.push(`Reason: ${data.reason ?? "unknown"}`, "");
    return `${lines.join("\n")}\n`;
  }
  lines.push("| Case | Stage | Plan | Output | Max abs diff | OK | ms |");
  lines.push("| --- | --- | --- | --- | ---: | --- | ---: |");
  for (const item of data.cases ?? []) {
    lines.push(`| \`${item.name}\` | \`${item.stage ?? "unknown"}\` | \`${item.plan}\` | \`${item.output}\` | ${item.maxAbsDiff} | \`${item.ok}\` | ${item.ms} |`);
  }
  lines.push("", `Passed: \`${data.passed}\`, failed: \`${data.failed}\`, skipped: \`${data.skipped ?? 0}\``, "");
  if ((data.warmupCases ?? 0) > 0) {
    lines.push(`Warmup: \`${data.warmupCases - (data.warmupFailed ?? 0)}/${data.warmupCases}\`, failed: \`${data.warmupFailed ?? 0}\``, "");
  }
  if (data.failedByStage && Object.keys(data.failedByStage).length > 0) {
    lines.push(`Failed by stage: \`${JSON.stringify(data.failedByStage)}\``, "");
  }
  lines.push(`Corpus fixtures: \`${data.corpusFixturePassed ?? 0}/${data.corpusFixtureCases ?? 0}\``, "");
  lines.push(`Expected-output fixtures: \`${data.corpusFixtureExpectedOutputCases ?? 0}/${data.corpusFixtureBaseline?.expectedOutputMin ?? 0}\``, "");
  if ((data.autoCorpusSmokeCases ?? 0) > 0) {
    lines.push(`Auto corpus smoke: \`${data.autoCorpusSmokePassed ?? 0}/${data.autoCorpusSmokeCases ?? 0}\`, skipped: \`${data.autoCorpusSmokeSkipped ?? 0}\``, "");
  }
  const repeatStats = repeatCaseStats(data.cases ?? []);
  if (repeatStats.length > 0) {
    lines.push("| Repeat case | cold ms | best warm ms | best repeat | speedup |");
    lines.push("| --- | ---: | ---: | ---: | ---: |");
    for (const item of repeatStats) {
      lines.push(`| \`${item.name}\` | ${item.coldMs} | ${item.bestWarmMs} | ${item.bestWarmRepeat} | ${item.speedup}x |`);
    }
    lines.push("");
  }
  if (data.corpusFixturePassedByCorpus) {
    lines.push("| Corpus | passed | cases | baseline min |");
    lines.push("| --- | ---: | ---: | ---: |");
    const baseline = data.corpusFixtureBaseline?.byCorpusMin ?? {};
    const corpusIds = Object.keys({ ...(data.corpusFixtureCasesByCorpus ?? {}), ...baseline }).sort();
    for (const corpusId of corpusIds) {
      lines.push(
        `| \`${corpusId}\` | ${data.corpusFixturePassedByCorpus[corpusId] ?? 0} | ${data.corpusFixtureCasesByCorpus?.[corpusId] ?? 0} | ${baseline[corpusId] ?? 0} |`,
      );
    }
    lines.push("");
  }
  return `${lines.join("\n")}\n`;
}

export function validateCorpusFixtureBaseline(report) {
  const baseline = report.corpusFixtureBaseline ?? cudaLiteCorpusExecutionFixtureBaseline;
  if ((report.corpusFixturePassed ?? 0) < baseline.totalMin) {
    throw new Error(`Corpus fixture baseline failed: ${report.corpusFixturePassed ?? 0}/${baseline.totalMin} passed`);
  }
  if ((report.corpusFixtureExpectedOutputCases ?? 0) < (baseline.expectedOutputMin ?? 0)) {
    throw new Error(`Corpus fixture expected-output baseline failed: ${report.corpusFixtureExpectedOutputCases ?? 0}/${baseline.expectedOutputMin ?? 0} pinned`);
  }
  for (const [corpusId, min] of Object.entries(baseline.byCorpusMin ?? {})) {
    const passed = report.corpusFixturePassedByCorpus?.[corpusId] ?? 0;
    if (passed < min) {
      throw new Error(`Corpus fixture baseline failed for ${corpusId}: ${passed}/${min} passed`);
    }
  }
}
