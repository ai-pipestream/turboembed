import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;

/** Escaping, headers and inputs shared by the golden writers. */
final class Common {

  private Common() {
  }

  /**
   * Writes every UTF-16 unit outside U+0020..U+007E, and the backslash, as backslash-u and four
   * upper case hex digits, except tab, line feed and carriage return, which are backslash-t,
   * backslash-n and backslash-r.
   */
  static String escape(String s) {
    StringBuilder b = new StringBuilder(s.length());
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      switch (c) {
        case '\\' -> b.append("\\\\");
        case '\t' -> b.append("\\t");
        case '\n' -> b.append("\\n");
        case '\r' -> b.append("\\r");
        default -> {
          if (c >= 0x20 && c <= 0x7E) {
            b.append(c);
          } else {
            b.append(String.format("\\u%04X", (int) c));
          }
        }
      }
    }
    return b.toString();
  }

  /** Reverses {@link #escape}. */
  static String unescape(String s) {
    StringBuilder b = new StringBuilder(s.length());
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c != '\\') {
        b.append(c);
        continue;
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
        default -> throw new IllegalArgumentException("bad escape in " + s);
      }
    }
    return b.toString();
  }

  /** The strings escaped and joined by tabs. */
  static String join(String[] values) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < values.length; i++) {
      if (i > 0) {
        b.append('\t');
      }
      b.append(escape(values[i]));
    }
    return b.toString();
  }

  static String join(List<String> values) {
    return join(values.toArray(new String[0]));
  }

  /** The raw bits of a double, sixteen lower case hex digits. */
  static String bits(double d) {
    return String.format("%016x", Double.doubleToRawLongBits(d));
  }

  /** A UTF-8 file written with line feeds. */
  static Writer writer(Path path) throws IOException {
    Files.createDirectories(path.getParent());
    return Files.newBufferedWriter(path, StandardCharsets.UTF_8);
  }

  static void line(Writer w, String line) throws IOException {
    w.write(line);
    w.write('\n');
  }

  /** The header every golden file starts with. */
  static void header(Writer w, String component, String commit) throws IOException {
    line(w, "component: " + component);
    line(w, "opennlp: " + commit);
    line(w, "jdk: " + System.getProperty("java.version"));
  }

  /** The sentences of a file, one per line, tokens separated by single spaces. */
  static List<String[]> sentences(Path file) throws IOException {
    List<String[]> out = new ArrayList<>();
    for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
      out.add(line.split(" "));
    }
    return out;
  }

  static String sha256(Path file) throws IOException {
    try {
      MessageDigest md = MessageDigest.getInstance("SHA-256");
      return HexFormat.of().formatHex(md.digest(Files.readAllBytes(file)));
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException(e);
    }
  }

  /** The outcomes the feature dumps pass as the previous outcomes of a sentence. */
  static String[] syntheticOutcomes(int sentence, int length) {
    String[] choices = {"other", "person-start", "person-cont", "NN", "té"};
    String[] out = new String[length];
    for (int i = 0; i < length; i++) {
      out[i] = choices[(i + sentence) % choices.length];
    }
    return out;
  }
}
