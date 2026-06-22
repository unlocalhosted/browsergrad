import type { PiiKind, PiiMaskResult, PiiSpan } from "./data-types.js";

const EMAIL_RE = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/giu;
const IP_RE =
  /\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b/gu;
const PHONE_RE =
  /(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)/gu;

export function maskPii(input: string): PiiMaskResult {
  const matches = [
    ...findPiiMatches(input, "email", EMAIL_RE, "<EMAIL>"),
    ...findPiiMatches(input, "ip", IP_RE, "<IP>"),
    ...findPiiMatches(input, "phone", PHONE_RE, "<PHONE>"),
  ].sort((a, b) => a.start - b.start || b.end - a.end);
  const selected: PiiSpan[] = [];
  let lastEnd = -1;
  for (const match of matches) {
    if (match.start < lastEnd) continue;
    selected.push(match);
    lastEnd = match.end;
  }

  let cursor = 0;
  let text = "";
  for (const span of selected) {
    text += input.slice(cursor, span.start);
    text += span.replacement;
    cursor = span.end;
  }
  text += input.slice(cursor);

  return { text, spans: selected };
}

function findPiiMatches(
  input: string,
  kind: PiiKind,
  pattern: RegExp,
  replacement: string,
): PiiSpan[] {
  pattern.lastIndex = 0;
  const spans: PiiSpan[] = [];
  for (const match of input.matchAll(pattern)) {
    const original = match[0];
    const start = match.index;
    if (start === undefined) continue;
    spans.push({
      kind,
      original,
      replacement,
      start,
      end: start + original.length,
    });
  }
  return spans;
}
