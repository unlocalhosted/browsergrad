import { describe, expect, it } from "vitest";
import { extractVisibleTextFromHtml } from "../src/index";

describe("extractVisibleTextFromHtml", () => {
  it("extracts visible text while removing scripts, styles, comments, and entities", () => {
    const html = `
      <html>
        <head><style>.x { color: red; }</style><script>bad()</script></head>
        <body><!-- skip --><h1>Moby&nbsp;Dick</h1><p>Call &amp; response<br>again.</p></body>
      </html>
    `;

    expect(extractVisibleTextFromHtml(html)).toBe("Moby Dick Call & response again.");
  });
});
