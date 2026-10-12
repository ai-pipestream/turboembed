/**
 * Prints the java.lang.Character answers that turbo-unicode reproduces, for
 * every code point of the Basic Multilingual Plane and every 257th code point
 * above it. The jdk test in turbo-unicode compiles and runs this program and
 * compares its output with the tables.
 *
 * <p>One line per code point: {@code cp type flags lower script}, with the
 * code point and {@code toLowerCase(int)} in hexadecimal, {@code getType(int)}
 * in decimal, and the flags as a hexadecimal bit mask: 0x01 isWhitespace, 0x02
 * isAlphabetic, 0x04 isUpperCase, 0x08 isLetter, 0x10 isDigit, 0x20
 * isSpaceChar. The script is the {@code Character.UnicodeScript} name for the
 * scripts turbo-unicode distinguishes and {@code OTHER} for the rest.
 */
public final class JavaChars {

  private JavaChars() {
  }

  public static void main(String[] args) {
    StringBuilder out = new StringBuilder(1 << 22);
    for (int cp = 0; cp <= 0xFFFF; cp++) {
      describe(out, cp);
    }
    for (int cp = 0x10000; cp <= Character.MAX_CODE_POINT; cp += 257) {
      describe(out, cp);
    }
    System.out.print(out);
    System.out.flush();
  }

  private static void describe(StringBuilder out, int cp) {
    int flags = 0;
    if (Character.isWhitespace(cp)) {
      flags |= 0x01;
    }
    if (Character.isAlphabetic(cp)) {
      flags |= 0x02;
    }
    if (Character.isUpperCase(cp)) {
      flags |= 0x04;
    }
    if (Character.isLetter(cp)) {
      flags |= 0x08;
    }
    if (Character.isDigit(cp)) {
      flags |= 0x10;
    }
    if (Character.isSpaceChar(cp)) {
      flags |= 0x20;
    }
    out.append(Integer.toHexString(cp)).append(' ')
        .append(Character.getType(cp)).append(' ')
        .append(Integer.toHexString(flags)).append(' ')
        .append(Integer.toHexString(Character.toLowerCase(cp))).append(' ')
        .append(script(cp)).append('\n');
  }

  private static String script(int cp) {
    Character.UnicodeScript script = Character.UnicodeScript.of(cp);
    return switch (script) {
      case HAN, HIRAGANA, KATAKANA, HANGUL, THAI, LAO, MYANMAR, KHMER, TAI_LE, NEW_TAI_LUE,
          TAI_VIET -> script.name();
      default -> "OTHER";
    };
  }
}
