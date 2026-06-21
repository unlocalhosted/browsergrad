const EMAIL_RE = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/giu;
const IP_RE = /\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b/gu;
const PHONE_RE = /(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)/gu;
export function maskPii(input) {
    const matches = [
        ...findPiiMatches(input, "email", EMAIL_RE, "<EMAIL>"),
        ...findPiiMatches(input, "ip", IP_RE, "<IP>"),
        ...findPiiMatches(input, "phone", PHONE_RE, "<PHONE>"),
    ].sort((a, b) => a.start - b.start || b.end - a.end);
    const selected = [];
    let lastEnd = -1;
    for (const match of matches) {
        if (match.start < lastEnd)
            continue;
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
export function exactLineDeduplicate(lines, options = {}) {
    const seen = new Map();
    const keptLines = [];
    const duplicates = [];
    for (let index = 0; index < lines.length; index++) {
        const line = lines[index] ?? "";
        const key = normalizeLineKey(line, options);
        const firstIndex = seen.get(key);
        if (firstIndex !== undefined) {
            duplicates.push({ line, index, firstIndex, key });
            continue;
        }
        seen.set(key, index);
        keptLines.push(line);
    }
    return { keptLines, duplicates };
}
export function extractVisibleTextFromHtml(html) {
    const withoutBlocks = html
        .replace(/<!--[\s\S]*?-->/g, " ")
        .replace(/<script\b[\s\S]*?<\/script>/gi, " ")
        .replace(/<style\b[\s\S]*?<\/style>/gi, " ");
    const withBreaks = withoutBlocks
        .replace(/<\s*br\s*\/?>/gi, " ")
        .replace(/<\/(?:p|div|section|article|h[1-6]|li|tr|table|body|html)>/gi, " ");
    const withoutTags = withBreaks.replace(/<[^>]+>/g, " ");
    return decodeBasicHtmlEntities(withoutTags)
        .replace(/\s+/g, " ")
        .trim();
}
function findPiiMatches(input, kind, pattern, replacement) {
    pattern.lastIndex = 0;
    const spans = [];
    for (const match of input.matchAll(pattern)) {
        const original = match[0];
        const start = match.index;
        if (start === undefined)
            continue;
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
function normalizeLineKey(line, options) {
    let key = options.trim ? line.trim() : line;
    if (options.collapseWhitespace) {
        key = key.replace(/\s+/g, " ");
    }
    if (options.caseSensitive === false) {
        key = key.toLowerCase();
    }
    return key;
}
function decodeBasicHtmlEntities(text) {
    return text.replace(/&(#x[0-9a-f]+|#\d+|amp|lt|gt|quot|apos|nbsp);/gi, (entity, body) => {
        const normalized = body.toLowerCase();
        if (normalized === "amp")
            return "&";
        if (normalized === "lt")
            return "<";
        if (normalized === "gt")
            return ">";
        if (normalized === "quot")
            return '"';
        if (normalized === "apos")
            return "'";
        if (normalized === "nbsp")
            return " ";
        if (normalized.startsWith("#x")) {
            return codePointToString(Number.parseInt(normalized.slice(2), 16), entity);
        }
        if (normalized.startsWith("#")) {
            return codePointToString(Number.parseInt(normalized.slice(1), 10), entity);
        }
        return entity;
    });
}
function codePointToString(codePoint, fallback) {
    if (!Number.isInteger(codePoint) || codePoint < 0)
        return fallback;
    try {
        return String.fromCodePoint(codePoint);
    }
    catch {
        return fallback;
    }
}
//# sourceMappingURL=index.js.map