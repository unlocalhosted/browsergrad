export const CPP_CUTE_VIRTUAL_PATH_MAXIMUM_UTF8_BYTE_LENGTH = 4_096;

/**
 * Returns one stable reason when a path cannot cross the C++/CuTe VFS seam.
 *
 * Keep this allocation-free before the byte ceiling is known: caller input is
 * untrusted, and TextEncoder would allocate proportionally to an oversized
 * string. JavaScript lone surrogates are rejected instead of being silently
 * replaced while computing their UTF-8 length.
 */
export function findCppCuteVirtualPathError(value: string): string | null {
  if (value.length === 0 || value.charCodeAt(0) !== 0x2f) {
    return "virtual path must be an absolute POSIX path";
  }

  let utf8ByteLength = 0;
  for (let index = 0; index < value.length; index += 1) {
    const codeUnit = value.charCodeAt(index);
    if (codeUnit === 0x5c || codeUnit <= 0x1f || codeUnit === 0x7f) {
      return "virtual path must not contain backslashes, C0 controls, or DEL";
    }
    if (codeUnit <= 0x7f) {
      utf8ByteLength += 1;
    } else if (codeUnit <= 0x7ff) {
      utf8ByteLength += 2;
    } else if (codeUnit >= 0xd800 && codeUnit <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (next < 0xdc00 || next > 0xdfff) {
        return "virtual path must contain only valid Unicode scalar values";
      }
      utf8ByteLength += 4;
      index += 1;
    } else if (codeUnit >= 0xdc00 && codeUnit <= 0xdfff) {
      return "virtual path must contain only valid Unicode scalar values";
    } else {
      utf8ByteLength += 3;
    }
    if (utf8ByteLength > CPP_CUTE_VIRTUAL_PATH_MAXIMUM_UTF8_BYTE_LENGTH) {
      return `virtual path must be at most ${CPP_CUTE_VIRTUAL_PATH_MAXIMUM_UTF8_BYTE_LENGTH} UTF-8 bytes`;
    }
  }

  if (value === "/") return null;
  let segmentBegin = 1;
  for (let index = 1; index <= value.length; index += 1) {
    if (index < value.length && value.charCodeAt(index) !== 0x2f) continue;
    const segmentLength = index - segmentBegin;
    const isDot = segmentLength === 1 && value.charCodeAt(segmentBegin) === 0x2e;
    const isDotDot = segmentLength === 2 && value.charCodeAt(segmentBegin) === 0x2e &&
      value.charCodeAt(segmentBegin + 1) === 0x2e;
    if (segmentLength === 0 || isDot || isDotDot) {
      return "virtual path must not contain empty, dot, or parent segments";
    }
    segmentBegin = index + 1;
  }
  return null;
}
