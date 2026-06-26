import { cudaLiteCorpusExecutionFixtureBaseline } from "./cuda-lite-corpus-registry.mjs";

export function summarizeReport(data) {
  const cases = data.cases ?? [];
  return {
    tool: data.tool,
    bundle: data.bundle ?? "src",
    available: data.available,
    reason: data.available ? undefined : data.reason ?? "unknown",
    passed: data.passed ?? 0,
    failed: data.failed ?? 0,
    corpusFixturePassed: data.corpusFixturePassed ?? 0,
    corpusFixtureCases: data.corpusFixtureCases ?? 0,
    corpusFixtureExpectedOutputCases: data.corpusFixtureExpectedOutputCases ?? 0,
    corpusFixturePassedByCorpus: data.corpusFixturePassedByCorpus ?? {},
    autoCorpusSmokePassed: data.autoCorpusSmokePassed ?? 0,
    autoCorpusSmokeCases: data.autoCorpusSmokeCases ?? 0,
    autoCorpusSmokePassedByCorpus: data.autoCorpusSmokePassedByCorpus ?? {},
    autoCorpusSmokeMode: data.autoCorpusSmokeMode,
    autoCorpusSmokeLimit: data.autoCorpusSmokeLimit,
    failedCases: cases.filter((item) => !item.ok).map((item) => ({
      name: item.name,
      plan: item.plan,
      error: item.error,
    })),
    slowestCases: [...cases]
      .filter((item) => Number.isFinite(item.ms))
      .sort((left, right) => right.ms - left.ms)
      .slice(0, 12)
      .map((item) => ({
        name: item.name,
        plan: item.plan,
        ms: item.ms,
        corpusId: item.corpusId,
      })),
  };
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
  lines.push("| Case | Plan | Output | Max abs diff | OK | ms |");
  lines.push("| --- | --- | --- | ---: | --- | ---: |");
  for (const item of data.cases ?? []) {
    lines.push(`| \`${item.name}\` | \`${item.plan}\` | \`${item.output}\` | ${item.maxAbsDiff} | \`${item.ok}\` | ${item.ms} |`);
  }
  lines.push("", `Passed: \`${data.passed}\`, failed: \`${data.failed}\``, "");
  lines.push(`Corpus fixtures: \`${data.corpusFixturePassed ?? 0}/${data.corpusFixtureCases ?? 0}\``, "");
  lines.push(`Expected-output fixtures: \`${data.corpusFixtureExpectedOutputCases ?? 0}/${data.corpusFixtureBaseline?.expectedOutputMin ?? 0}\``, "");
  if ((data.autoCorpusSmokeCases ?? 0) > 0) {
    lines.push(`Auto corpus smoke: \`${data.autoCorpusSmokePassed ?? 0}/${data.autoCorpusSmokeCases ?? 0}\``, "");
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
