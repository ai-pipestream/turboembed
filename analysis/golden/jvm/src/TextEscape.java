/**
 * Escapes text so that it fits on one line of a golden file.
 *
 * <p>A backslash, line feed, carriage return and tab become {@code \\}, {@code \n}, {@code \r}
 * and {@code \t}. Every other UTF-16 code unit outside U+0020..U+007E becomes {@code \}{@code uXXXX}
 * with four upper case hex digits. A character outside the BMP is written as its two surrogate
 * escapes, as Java source would write it.
 */
final class TextEscape {

  private TextEscape() {
  }

  static String escape(CharSequence text) {
    StringBuilder out = new StringBuilder(text.length() + 16);
    for (int i = 0; i < text.length(); i++) {
      char c = text.charAt(i);
      switch (c) {
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        default -> {
          if (c >= 0x20 && c <= 0x7E) {
            out.append(c);
          } else {
            out.append("\\u").append(String.format("%04X", (int) c));
          }
        }
      }
    }
    return out.toString();
  }
}
