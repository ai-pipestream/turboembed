package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.List;

/**
 * The inputs that hold strings a UTF-8 file cannot hold as they are, such as half a surrogate pair
 * or a tab inside a token: one string per line, with a backslash written as two, a tab, line feed
 * and carriage return as backslash and {@code t}, {@code n} or {@code r}, and any UTF-16 unit as a
 * backslash, {@code u} and four hex digits.
 */
final class EscapedLines {

  private EscapedLines() {}

  /** Every line of the input, unescaped. */
  static List<String> read(Input input) {
    List<String> out = new ArrayList<>();
    for (String line : input.lines()) {
      out.add(unescape(line));
    }
    return out;
  }

  /** The string a line stands for. */
  static String unescape(String s) {
    StringBuilder b = new StringBuilder(s.length());
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c != '\\') {
        b.append(c);
        continue;
      }
      if (i + 1 >= s.length()) {
        throw new IllegalArgumentException("a backslash at the end of " + s);
      }
      char n = s.charAt(++i);
      switch (n) {
        case '\\' -> b.append('\\');
        case 't' -> b.append('\t');
        case 'n' -> b.append('\n');
        case 'r' -> b.append('\r');
        case 'u' -> {
          b.append((char) Integer.parseInt(s.substring(i + 1, i + 5), 16));
          i += 4;
        }
        default -> throw new IllegalArgumentException("an unknown escape in " + s);
      }
    }
    return b.toString();
  }
}
