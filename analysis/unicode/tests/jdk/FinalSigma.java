import java.util.Locale;
import java.util.Random;

/**
 * Prints what {@code String.toLowerCase(Locale.ROOT)} makes of the capital
 * sigmas in random strings: {@code FinalSigma <seed> <count> <bmp|all|mixed>}
 * writes {@code count} lines, each the string's code points in hexadecimal
 * separated by hyphens, a space, and one letter per capital sigma, {@code F}
 * where the JDK wrote final sigma and {@code S} where it wrote sigma. Each
 * string holds at least one U+03A3. With {@code bmp} and {@code all} it has
 * one to seven code points, drawn from a list of characters the JDK's word
 * break rules care about (letters and digits of several scripts, marks,
 * format characters, joiners, currency and percent signs, dandas, kana,
 * kanji, supplementary characters) and from the assigned code points at
 * large: the Basic Multilingual Plane with {@code bmp}, every plane with
 * {@code all}. With {@code mixed} it has one to twenty code points from a
 * shorter list that puts dandas, marks, joiners, controls and supplementary
 * letters, digits, marks and format characters next to each other. Code
 * points the JDK reports as unassigned are not drawn.
 */
public final class FinalSigma {

  private static final int[] INTERESTING = {
      0x3A3, 0x3A3, 0x3A3, 0x391, 0x3B1, 0x3C2, 0x41, 0x61, 0x1E9E, 0xDF, 0x130, 0x1C5, 0x2B0, 0x2B9, 0x2C0, 0x37A,
      0x1D2C, 0x1D62, 0x1D78, 0xAA, 0x295, 0x2160, 0x217F, 0x2180, 0x24B6, 0xA770,
      0x31, 0x39, 0x660, 0x966, 0xB2, 0x2460, 0x2070,
      0x300, 0x301, 0x323, 0x345, 0x3099, 0x309A, 0xF7E, 0x903, 0xE48, 0x20DD, 0x1D167,
      0xAD, 0x200B, 0x200C, 0x200D, 0x2060, 0x180E, 0xFEFF, 0x202E, 0x2067, 0xE0001, 0x110BD,
      0x2E, 0x27, 0x22, 0x2C, 0x2D, 0x5F, 0x2013, 0x2027, 0x203F, 0x66B, 0x58A, 0x10EAD,
      0x964, 0x965, 0x23, 0x24, 0x20AC, 0xA2, 0x25, 0x26, 0x66A, 0x2030, 0x11FDD,
      0x20, 0xA0, 0x9, 0xA, 0x2028, 0x3000, 0x29, 0x5B, 0x21, 0x3F, 0xB7, 0x387, 0x2019,
      0x3042, 0x30A2, 0x30FC, 0x3005, 0x4E00, 0x9FA5, 0x9FA6, 0x3400, 0xFA2D, 0xFA2E, 0xAC00, 0xE02,
      0x10400, 0x10428, 0x1D400, 0x1D7CE, 0x20000, 0x1F600, 0x109B2, 0x10FFFD, 0x1E900, 0x16E40,
  };

  /** Characters whose neighbours decide the word boundaries, for {@code mixed}. */
  private static final int[] MIXED = {
      0x3A3, 0x3A3, 0x3A3, 0x3A3, 0x391, 0x3B1, 0x41, 0x61, 0x345, 0x37A, 0x2B0, 0x295, 0x2160, 0x4E00, 0x3042, 0xE02,
      0x301, 0x3099, 0xF7E, 0x903, 0x20DD, 0x1D167, 0xE01EF, 0x101FD,
      0x964, 0x965,
      0x31, 0x2460, 0x660, 0x1D7CE, 0x10107, 0x1F100,
      0x2D, 0x5F, 0xAD, 0x2027, 0x203F, 0xFE4F, 0x2E, 0x27, 0x22, 0x2C, 0x66B,
      0x24, 0x23, 0x25, 0x26, 0xA2, 0x1E2FF, 0x1ECB0,
      0x200B, 0x2060, 0x180E, 0xFEFF, 0xE0020, 0xE0001, 0x110BD, 0x1D17A,
      0x20, 0x9, 0xA, 0x85, 0x2028, 0x29, 0x3F, 0x1F600, 0x10EAD,
      0x10100, 0xF0000, 0x20000, 0x10400, 0x10428, 0x1D400, 0x109B2, 0x16E40, 0x11000,
  };

  private FinalSigma() {
  }

  public static void main(String[] args) {
    long seed = Long.parseLong(args[0]);
    int count = Integer.parseInt(args[1]);
    boolean all = args[2].equals("all");
    boolean mixed = args[2].equals("mixed");
    Random random = new Random(seed);
    StringBuilder out = new StringBuilder(count * 48);
    for (int n = 0; n < count; n++) {
      int length = 1 + random.nextInt(mixed ? 20 : 7);
      int[] cps = new int[length];
      for (int i = 0; i < length; i++) {
        cps[i] = mixed ? MIXED[random.nextInt(MIXED.length)] : draw(random, all);
      }
      cps[random.nextInt(length)] = 0x3A3;
      StringBuilder s = new StringBuilder();
      for (int cp : cps) {
        s.appendCodePoint(cp);
      }
      String text = s.toString();
      String lower = text.toLowerCase(Locale.ROOT);
      for (int i = 0; i < cps.length; i++) {
        if (i > 0) {
          out.append('-');
        }
        out.append(Integer.toHexString(cps[i]));
      }
      out.append(' ');
      // The lowercase keeps every code point's width but for U+0130, which is
      // never drawn, so the sigmas are at the same indexes.
      for (int i = 0, j = 0; i < text.length(); i += Character.charCount(text.codePointAt(i)), j++) {
        if (text.codePointAt(i) == 0x3A3) {
          out.append(lower.codePointAt(i) == 0x3C2 ? 'F' : 'S');
        }
      }
      out.append('\n');
    }
    System.out.print(out);
    System.out.flush();
  }

  /** A code point: half the time from the list, otherwise any assigned one. */
  private static int draw(Random random, boolean all) {
    if (random.nextBoolean()) {
      return INTERESTING[random.nextInt(INTERESTING.length)];
    }
    while (true) {
      int cp = random.nextInt(all ? Character.MAX_CODE_POINT + 1 : 0x10000);
      if (cp >= Character.MIN_SURROGATE && cp <= Character.MAX_SURROGATE) {
        continue;
      }
      if (Character.getType(cp) == Character.UNASSIGNED || cp == 0x130) {
        continue;
      }
      return cp;
    }
  }
}
