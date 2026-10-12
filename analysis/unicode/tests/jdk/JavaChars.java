/**
 * Prints the java.lang.Character answers that turbo-unicode reproduces, for
 * every code point from U+0000 to U+10FFFF. The jdk test in turbo-unicode
 * compiles and runs this program and compares its output with the tables.
 *
 * <p>The first line is the JDK feature version, {@code
 * Runtime.version().feature()}. Then one line per code point: {@code cp type
 * flags lower script}, with the code point and {@code toLowerCase(int)} in
 * hexadecimal, {@code getType(int)} in decimal, and the flags as a
 * hexadecimal bit mask: 0x01 isWhitespace, 0x02 isAlphabetic, 0x04
 * isUpperCase, 0x08 isLetter, 0x10 isDigit, 0x20 isSpaceChar. The script is
 * the {@code Character.UnicodeScript} name.
 */
public final class JavaChars {

  private JavaChars() {
  }

  public static void main(String[] args) {
    StringBuilder out = new StringBuilder(1 << 25);
    out.append(Runtime.version().feature()).append('\n');
    for (int cp = 0; cp <= Character.MAX_CODE_POINT; cp++) {
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
        .append(Character.UnicodeScript.of(cp).name()).append('\n');
  }
}
