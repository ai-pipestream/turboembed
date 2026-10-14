import java.text.BreakIterator;
import java.util.Locale;

/**
 * Prints what the root locale's word BreakIterator makes of every assigned
 * code point: one line per code point, {@code cp signature}, the code point in
 * hexadecimal and sixteen boundary answers in ten short contexts (a letter or a
 * digit on either side, the character doubled, at the start or the end, after a
 * dot, before a dot or a comma). The jdk test of turbo-unicode maps each
 * signature to a word class and compares it with the crate's table.
 */
public final class WordClasses {

  private static final BreakIterator WORDS = BreakIterator.getWordInstance(Locale.ROOT);

  private WordClasses() {
  }

  public static void main(String[] args) {
    StringBuilder out = new StringBuilder(1 << 23);
    out.append(Runtime.version().feature()).append('\n');
    for (int cp = 0; cp <= Character.MAX_CODE_POINT; cp++) {
      if (cp >= Character.MIN_SURROGATE && cp <= Character.MAX_SURROGATE) {
        continue;
      }
      if (Character.getType(cp) == Character.UNASSIGNED) {
        continue;
      }
      String c = new String(Character.toChars(cp));
      int w = c.length();
      out.append(Integer.toHexString(cp)).append(' ');
      bits(out, "a" + c + "a", 1, 1 + w);
      bits(out, "1" + c + "1", 1, 1 + w);
      bits(out, "a" + c + "1", 1, 1 + w);
      bits(out, "1" + c + "a", 1, 1 + w);
      bits(out, "a" + c + c + "a", 1, 1 + w, 1 + 2 * w);
      bits(out, c + "a", w);
      bits(out, "a" + c, 1);
      bits(out, "a." + c, 2);
      bits(out, "a" + c + ".a", 1 + w);
      bits(out, "1" + c + ",1", 1 + w);
      out.append('\n');
    }
    System.out.print(out);
    System.out.flush();
  }

  /** Appends one digit per offset: 1 where the iterator puts a boundary. */
  private static void bits(StringBuilder out, String text, int... offsets) {
    WORDS.setText(text);
    for (int at : offsets) {
      out.append(WORDS.isBoundary(at) ? '1' : '0');
    }
  }
}
