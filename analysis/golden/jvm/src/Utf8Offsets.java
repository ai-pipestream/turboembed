import java.nio.charset.StandardCharsets;

/**
 * Converts UTF-16 offsets of one text into UTF-8 byte offsets.
 *
 * <p>The table is built once per text. An offset that falls between the two halves of a surrogate
 * pair has no UTF-8 offset; {@link #isBoundary(int)} reports it and {@link #utf8(int)} refuses it.
 */
final class Utf8Offsets {

  private final int[] utf8;
  private final boolean[] boundary;

  Utf8Offsets(String text) {
    int n = text.length();
    utf8 = new int[n + 1];
    boundary = new boolean[n + 1];
    int bytes = 0;
    boundary[0] = true;
    for (int i = 0; i < n; i++) {
      utf8[i] = bytes;
      char c = text.charAt(i);
      if (Character.isHighSurrogate(c) && i + 1 < n && Character.isLowSurrogate(text.charAt(i + 1))) {
        // The pair is one character of four bytes; the offset between the halves is not a boundary.
        utf8[i + 1] = bytes;
        bytes += 4;
        i++;
        utf8[i + 1] = bytes;
        boundary[i + 1] = true;
        continue;
      }
      boundary[i + 1] = true;
      bytes += text.substring(i, i + 1).getBytes(StandardCharsets.UTF_8).length;
    }
    utf8[n] = bytes;
    boundary[n] = true;
  }

  /** Whether {@code utf16} is a character boundary (not between the halves of a surrogate pair). */
  boolean isBoundary(int utf16) {
    return boundary[utf16];
  }

  /** The UTF-8 byte offset of a UTF-16 offset that {@link #isBoundary(int)} accepts. */
  int utf8(int utf16) {
    if (!boundary[utf16]) {
      throw new IllegalArgumentException("UTF-16 offset " + utf16 + " is inside a surrogate pair");
    }
    return utf8[utf16];
  }
}
