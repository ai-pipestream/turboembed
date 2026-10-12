package ai.pipestream.turbo.goldens;

/**
 * Converts UTF-16 indices into one input text to UTF-8 byte offsets into the same text.
 *
 * <p>OpenNLP reports spans as UTF-16 indices into a Java string; the golden files hold UTF-8
 * byte offsets. The table is a prefix sum over the text's code points: the byte offset of every
 * UTF-16 index that starts a code point, and a marker for the index between the two halves of a
 * surrogate pair. An index on that marker is not a position in the UTF-8 text, so converting it
 * fails with the input id and the span instead of rounding to a neighbour.</p>
 */
final class Utf8Offsets {

  private static final int INSIDE_PAIR = -1;

  private final String inputId;
  private final int[] byteAt;

  /**
   * Builds the table for {@code text}, which must be well formed: a lone surrogate has no UTF-8
   * form and is refused.
   */
  Utf8Offsets(String inputId, String text) {
    this.inputId = inputId;
    this.byteAt = new int[text.length() + 1];
    int bytes = 0;
    int i = 0;
    while (i < text.length()) {
      int cp = text.codePointAt(i);
      if (Character.isSurrogate(text.charAt(i)) && Character.charCount(cp) == 1) {
        throw new IllegalArgumentException("input " + inputId + ": lone surrogate at UTF-16 index " + i);
      }
      byteAt[i] = bytes;
      if (Character.charCount(cp) == 2) {
        byteAt[i + 1] = INSIDE_PAIR;
      }
      bytes += utf8Length(cp);
      i += Character.charCount(cp);
    }
    byteAt[text.length()] = bytes;
  }

  /** The number of UTF-8 bytes a code point takes. */
  static int utf8Length(int cp) {
    if (cp < 0x80) {
      return 1;
    } else if (cp < 0x800) {
      return 2;
    } else if (cp < 0x10000) {
      return 3;
    }
    return 4;
  }

  /** The text's length in UTF-8 bytes. */
  int utf8Length() {
    return byteAt[byteAt.length - 1];
  }

  /**
   * Converts the half open UTF-16 span {@code [start, end)} to a UTF-8 byte span. Fails when
   * either end is outside the text, is between the halves of a surrogate pair, or the span is
   * reversed.
   */
  ByteSpan span(int start, int end) {
    if (start < 0 || end > byteAt.length - 1 || start > end) {
      throw new IllegalArgumentException(
          "input " + inputId + ": span [" + start + ", " + end + ") is not inside the text of "
              + (byteAt.length - 1) + " UTF-16 units");
    }
    int s = byteAt[start];
    int e = byteAt[end];
    if (s == INSIDE_PAIR || e == INSIDE_PAIR) {
      throw new IllegalArgumentException(
          "input " + inputId + ": span [" + start + ", " + end + ") has an end inside a surrogate pair");
    }
    return new ByteSpan(s, e);
  }

  /**
   * Converts one UTF-16 index, a position OpenNLP decided at rather than a span end, to a byte
   * offset. An index between the two halves of a surrogate pair is a real decision point for
   * an annotator that walks UTF-16 units, so it is not refused: it gives the byte offset of the
   * pair's code point with unit 1, where an index that starts a code point gives unit 0.
   */
  Position position(int index) {
    if (index < 0 || index > byteAt.length - 1) {
      throw new IllegalArgumentException("input " + inputId + ": index " + index
          + " is not inside the text of " + (byteAt.length - 1) + " UTF-16 units");
    }
    if (byteAt[index] == INSIDE_PAIR) {
      return new Position(byteAt[index - 1], 1);
    }
    return new Position(byteAt[index], 0);
  }

  /** A half open span of UTF-8 byte offsets. */
  record ByteSpan(int start, int end) {}

  /**
   * A UTF-8 byte offset and the UTF-16 unit within the code point there: 0 at its start, 1
   * between the halves of a surrogate pair.
   */
  record Position(int offset, int unit) {}
}
