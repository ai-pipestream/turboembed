import java.text.BreakIterator;
import java.util.Locale;

/**
 * Prints what the root locale's word BreakIterator makes of every assigned
 * code point: one line per code point, {@code cp signature}, the code point in
 * hexadecimal and its boundary answers in short contexts. A character of the
 * Basic Multilingual Plane gets seventeen answers in eleven contexts (a letter
 * or a digit on either side, the character doubled, at the start or the end,
 * after a dot, before a dot or a comma, before a mark). A supplementary
 * character gets fourteen: the
 * iterator answers true right after one that does not start the text, as it
 * resynchronizes from inside the surrogate pair, so the character is probed at
 * the start of the text (before a dash and a letter, a comma and a digit, a mark
 * and a letter, a danda, a percent sign, a letter, a digit) and only the answer
 * before it is read with a letter or a digit on either side. The jdk test of
 * turbo-unicode maps each signature to a word class and compares it with the
 * crate's table.
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
      if (w == 2) {
        bits(out, c + "-a", w, w + 1);
        bits(out, c + ",1", w, w + 1);
        bits(out, c + "\u0345a", w, w + 1);
        bits(out, c + "\u0964", w);
        bits(out, c + "%", w);
        bits(out, c + "a", w);
        bits(out, c + "1", w);
        bits(out, "a" + c + "a", 1);
        bits(out, "1" + c + "1", 1);
        bits(out, "1" + c + "a", 1);
        bits(out, "a" + c + "1", 1);
        out.append('\n');
        continue;
      }
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
      bits(out, "a" + c + "\u0345a", 1 + w);
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
