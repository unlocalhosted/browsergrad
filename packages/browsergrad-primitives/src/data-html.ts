export function extractVisibleTextFromHtml(html: string): string {
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

function decodeBasicHtmlEntities(text: string): string {
  return text.replace(
    /&(#x[0-9a-f]+|#\d+|amp|lt|gt|quot|apos|nbsp);/gi,
    (entity, body: string) => {
      const normalized = body.toLowerCase();
      if (normalized === "amp") return "&";
      if (normalized === "lt") return "<";
      if (normalized === "gt") return ">";
      if (normalized === "quot") return '"';
      if (normalized === "apos") return "'";
      if (normalized === "nbsp") return " ";
      if (normalized.startsWith("#x")) {
        return codePointToString(Number.parseInt(normalized.slice(2), 16), entity);
      }
      if (normalized.startsWith("#")) {
        return codePointToString(Number.parseInt(normalized.slice(1), 10), entity);
      }
      return entity;
    },
  );
}

function codePointToString(codePoint: number, fallback: string): string {
  if (!Number.isInteger(codePoint) || codePoint < 0) return fallback;
  try {
    return String.fromCodePoint(codePoint);
  } catch {
    return fallback;
  }
}
