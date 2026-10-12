package ai.pipestream.turbo.goldens;

import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Locale;

/**
 * Writes a golden file byte for byte as the format says: UTF-8, LF line ends, one tab between
 * fields, the header records in a fixed order, then each layer block with its rows and its
 * {@code end} line. Nothing depends on the platform or the locale, so writing the same values
 * twice gives the same bytes.
 */
final class GoldenWriter {

  /** The format version on the first line. */
  static final int FORMAT_VERSION = 1;

  /** The generator's name and version on the {@code generator} line. */
  static final String GENERATOR = "turbo-goldens-java";
  static final int GENERATOR_VERSION = 1;

  private GoldenWriter() {}

  /** Returns the whole file for one header and its layers, in UTF-8. */
  static byte[] write(Header h, List<Layer> layers) {
    if (layers.isEmpty()) {
      throw new IllegalArgumentException("a golden needs at least one layer");
    }
    StringBuilder out = new StringBuilder();
    out.append("golden ").append(FORMAT_VERSION).append('\n');
    line(out, "generator", GENERATOR, Integer.toString(GENERATOR_VERSION));
    line(out, "opennlp", h.opennlpCommit(), escape(h.opennlpVersion()));
    line(out, "jdk", escape(h.jdk().javaVersion()), escape(h.jdk().vmVersion()), escape(h.jdk().vendor()));
    line(out, "unicode", escape(h.jdk().unicodeVersion()));
    line(out, "input", escape(h.inputId()), u32(h.utf8Length()), u32(h.utf16Length()), h.inputSha256());
    for (ModelRef m : h.models()) {
      line(out, "model", escape(m.fileName()), m.sha512());
    }
    line(out, "annotator", escape(h.annotatorClass()), escape(h.method()));
    for (Layer layer : layers) {
      layer(out, layer);
    }
    return utf8(out);
  }

  private static void layer(StringBuilder out, Layer layer) {
    out.append("layer\t").append(escape(layer.id())).append('\t').append(escape(layer.anchor()))
        .append('\t').append(layer.rows());
    for (Column c : layer.columns()) {
      out.append('\t').append(escape(c.name())).append(':').append(c.type());
    }
    out.append('\n');
    for (int row = 0; row < layer.rows(); row++) {
      for (int col = 0; col < layer.columns().size(); col++) {
        if (col > 0) {
          out.append('\t');
        }
        out.append(cell(layer.columns().get(col), row));
      }
      out.append('\n');
    }
    line(out, "end", escape(layer.id()));
  }

  private static String cell(Column c, int row) {
    return switch (c) {
      case Column.U32 u -> u32(u.values()[row]);
      case Column.F64 f -> f64(f.values()[row]);
      case Column.Str s -> escape(s.values()[row]);
    };
  }

  private static void line(StringBuilder out, String... fields) {
    out.append(String.join("\t", fields)).append('\n');
  }

  /**
   * Encodes the text as UTF-8, failing on anything without a UTF-8 form instead of writing a
   * replacement character. After {@link #escape} nothing such is left, so a failure is a bug.
   */
  static byte[] utf8(CharSequence text) {
    try {
      var bytes = StandardCharsets.UTF_8.newEncoder().onMalformedInput(CodingErrorAction.REPORT)
          .onUnmappableCharacter(CodingErrorAction.REPORT).encode(CharBuffer.wrap(text));
      byte[] out = new byte[bytes.remaining()];
      bytes.get(out);
      return out;
    } catch (CharacterCodingException e) {
      throw new IllegalStateException("text without a UTF-8 form", e);
    }
  }

  /** A {@code u32} cell: decimal, no sign, no leading zeros. */
  static String u32(long v) {
    if (v < 0 || v > 0xffff_ffffL) {
      throw new IllegalArgumentException(v + " is not a u32");
    }
    return Long.toString(v);
  }

  /**
   * An {@code f64} cell: the 16 lowercase hex digits of the raw IEEE 754 bits, a space, and
   * {@link Double#toString(double)} for a person reading the file.
   */
  static String f64(double v) {
    String hex = Long.toHexString(Double.doubleToRawLongBits(v));
    return "0".repeat(16 - hex.length()) + hex + " " + Double.toString(v);
  }

  /**
   * A {@code str} cell: {@code \t}, {@code \n}, {@code \r} and {@code \\} as those two
   * characters, any other character below U+0020 as {@code \}{@code uXXXX} with four lowercase
   * hex digits, and everything else as it is. A surrogate that is not half of a pair has no UTF-8
   * form, so it is written as {@code \}{@code uXXXX} too rather than being replaced on encoding;
   * the n-grams of the language detector and the contexts of the tokenizer are cut at UTF-16
   * units and hold such halves.
   */
  static String escape(String s) {
    StringBuilder b = null;
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      String rep = switch (c) {
        case '\t' -> "\\t";
        case '\n' -> "\\n";
        case '\r' -> "\\r";
        case '\\' -> "\\\\";
        default -> c < 0x20 || isLoneSurrogate(s, i) ? String.format(Locale.ROOT, "\\u%04x", (int) c) : null;
      };
      if (rep != null && b == null) {
        b = new StringBuilder(s.length() + 8).append(s, 0, i);
      }
      if (b != null) {
        if (rep != null) {
          b.append(rep);
        } else {
          b.append(c);
        }
      }
    }
    return b == null ? s : b.toString();
  }

  /** Whether the unit at {@code i} is a surrogate that does not form a pair with its neighbour. */
  private static boolean isLoneSurrogate(String s, int i) {
    char c = s.charAt(i);
    if (Character.isHighSurrogate(c)) {
      return i + 1 >= s.length() || !Character.isLowSurrogate(s.charAt(i + 1));
    }
    if (Character.isLowSurrogate(c)) {
      return i == 0 || !Character.isHighSurrogate(s.charAt(i - 1));
    }
    return false;
  }
}
