import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CoderResult;
import java.nio.charset.CodingErrorAction;
import java.util.TreeSet;

/**
 * Writes {@code src/charset/tables.rs}: for each single byte charset named on the command line,
 * its canonical name, its aliases and the character the JDK's decoder gives for each byte from
 * 0x80 to 0xFF, or 0xFFFF where a strict decoder reports the byte. Run it with the JDK the goldens
 * are made with (the base image of analysis/goldens-tool/container/Dockerfile):
 *
 * <pre>
 * java CharsetTables.java ISO-8859-1 ISO-8859-2 ... &gt; ../src/charset/tables.rs
 * </pre>
 *
 * <p>The names the crate's tables were written from are listed at the top of the file it wrote. The
 * file also lists every charset name and alias the JDK knows, so a name the port has no table for
 * is told apart from one Java does not know either.</p>
 */
public class CharsetTables {
  public static void main(String[] args) {
    StringBuilder out = new StringBuilder();
    out.append("//! The single byte charsets `SET` may name, as the JDK decodes them. Written by\n")
        .append("//! `tools/CharsetTables.java` on ").append(System.getProperty("java.vendor")).append(' ')
        .append(System.getProperty("java.runtime.version")).append(" from these names:\n//!\n//!");
    for (String name : args) {
      out.append(' ').append(name);
    }
    out.append("\n//!\n//! Do not edit; run the program again.\n\n");
    out.append("use super::SingleByte;\n\n");
    out.append("/// The charsets, each with every name `Charset.forName` knows it by.\n");
    out.append("pub(super) static SINGLE_BYTE: [SingleByte; ").append(args.length).append("] = [\n");
    for (String name : args) {
      Charset cs = Charset.forName(name);
      for (int b = 0; b < 0x80; b++) {
        CharBuffer c = decode(cs, b);
        if (c == null || c.remaining() != 1 || c.get(0) != b) {
          throw new IllegalStateException(name + " is not ASCII at " + b);
        }
      }
      TreeSet<String> names = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
      names.add(cs.name());
      names.addAll(cs.aliases());
      out.append("    SingleByte {\n        name: \"").append(cs.name()).append("\",\n        aliases: &[");
      boolean first = true;
      for (String a : names) {
        out.append(first ? "" : ", ").append('"').append(a).append('"');
        first = false;
      }
      out.append("],\n        high: [");
      for (int b = 0x80; b < 0x100; b++) {
        CharBuffer c = decode(cs, b);
        int v;
        if (c == null) {
          v = 0xFFFF;
        } else if (c.remaining() != 1) {
          throw new IllegalStateException(name + " gives " + c.remaining() + " chars for " + b);
        } else {
          v = c.get(0);
        }
        out.append((b - 0x80) % 8 == 0 ? "\n            " : " ").append(String.format("0x%04X,", v));
      }
      out.append("\n        ],\n    },\n");
    }
    out.append("];\n\n");
    TreeSet<String> known = new TreeSet<>();
    for (Charset cs : Charset.availableCharsets().values()) {
      known.add(cs.name().toLowerCase(java.util.Locale.ROOT));
      for (String a : cs.aliases()) {
        known.add(a.toLowerCase(java.util.Locale.ROOT));
      }
    }
    out.append("/// Every name and alias of every charset the JDK has, lower case and sorted: the names\n")
        .append("/// `Charset.forName` accepts.\n");
    out.append("pub(super) static JAVA_NAMES: [&str; ").append(known.size()).append("] = [");
    int n = 0;
    for (String k : known) {
      out.append(n++ % 6 == 0 ? "\n    " : " ").append('"').append(k).append("\",");
    }
    out.append("\n];\n");
    System.out.print(out);
  }

  /** The chars a strict decoder gives for the one byte, or null when it reports it. */
  private static CharBuffer decode(Charset cs, int b) {
    CharsetDecoder d = cs.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT);
    CharBuffer c = CharBuffer.allocate(4);
    CoderResult r = d.decode(ByteBuffer.wrap(new byte[] {(byte) b}), c, true);
    if (r.isError()) {
      return null;
    }
    r = d.flush(c);
    if (r.isError()) {
      return null;
    }
    c.flip();
    return c;
  }
}
