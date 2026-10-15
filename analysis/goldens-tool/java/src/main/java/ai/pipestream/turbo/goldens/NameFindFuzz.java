package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Random sentences built from pieces that the name finders' patterns and dictionaries look for:
 * e-mail addresses, URLs, phone numbers, MGRS coordinates, degrees, punctuation and letters.
 */
final class NameFindFuzz {

  private static final String LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

  /** Characters inserted at random: ASCII punctuation the patterns read, and non-ASCII ones. */
  private static final String[] NOISE = {".", "@", "/", ":", "-", "_", "~", "%", "#", "?", "&", "=",
      "+", "!", "$", "|", ",", "*", "'", "\"", "(", ")", "[", "]", "\\", "<", ">", ";", "`", "{", "}",
      "^", "\u00b0", "\u2019", "\u201d", "\u00e9", "e\u0301", "\u0301", "\u00f6", "\u017f", "\u212a",
      "\u0130", "\u00a0", "\uD83D\uDE00", "\uD835\uDC00", "\uD834\uDD67", "\u0000", "\t", "\n",
      "\u2028", "\u0085", "\r", "\u0663", "\u00df", "\uD805\uDF1E", "d", "D",
      "N", "s", "W", "E", "1", "2", "9", "0"};

  private static final String[] TLDS = {"com", "org", "net", "gov", "mil", "biz", "info", "mobi",
      "name", "aero", "jobs", "museum", "travel", "uk", "de", "c", "x1", "COM", "Org", "io"};

  private static final String[] PREFIXES = {"http://", "https://", "ftp://", "ftps://", "www.",
      "/", "~/", "HTTP://WWW.", "Http://", "", "x/", "a~/"};

  private NameFindFuzz() {}

  static String word(Random r, int n) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < n; i++) {
      b.append(LETTERS.charAt(r.nextInt(LETTERS.length())));
    }
    return b.toString();
  }

  static String digits(Random r, int n) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < n; i++) {
      b.append((char) ('0' + r.nextInt(10)));
    }
    return b.toString();
  }

  private static String domain(Random r) {
    StringBuilder b = new StringBuilder();
    int labels = 1 + r.nextInt(3);
    for (int i = 0; i < labels; i++) {
      b.append(r.nextInt(5) == 0 ? word(r, 1) + "-" + word(r, 2) : word(r, 1 + r.nextInt(8)));
      b.append('.');
    }
    return b.append(TLDS[r.nextInt(TLDS.length)]).toString();
  }

  private static String email(Random r) {
    return switch (r.nextInt(6)) {
      case 0 -> "\"" + word(r, 2) + (r.nextBoolean() ? "\\ " : " ") + word(r, 2) + "\"@" + domain(r);
      case 1 -> word(r, 3) + "@[" + r.nextInt(300) + "." + r.nextInt(300) + "." + r.nextInt(300) + "."
          + r.nextInt(300) + "]";
      case 2 -> word(r, 1 + r.nextInt(70)) + "@" + domain(r);
      default -> word(r, 1 + r.nextInt(5)) + (r.nextBoolean() ? "." + word(r, 3) : "") + "@" + domain(r);
    };
  }

  private static String url(Random r) {
    StringBuilder b = new StringBuilder(PREFIXES[r.nextInt(PREFIXES.length)]);
    if (r.nextInt(6) == 0) {
      b.append(word(r, 2)).append(':').append(word(r, 2)).append('@');
    }
    b.append(domain(r));
    if (r.nextInt(4) == 0) {
      b.append(':').append(digits(r, 1 + r.nextInt(6)));
    }
    int segments = r.nextInt(4);
    for (int i = 0; i < segments; i++) {
      b.append('/').append(word(r, r.nextInt(6)));
      if (r.nextInt(4) == 0) {
        b.append('%').append(Integer.toHexString(r.nextInt(256)));
      }
    }
    if (r.nextInt(3) == 0) {
      b.append('?').append(word(r, 2)).append('=').append(digits(r, 2));
      if (r.nextBoolean()) {
        b.append('&').append(word(r, 1)).append("=*:");
      }
    }
    if (r.nextInt(4) == 0) {
      b.append('#').append(word(r, 3));
    }
    return b.toString();
  }

  private static String phone(Random r) {
    return switch (r.nextInt(4)) {
      case 0 -> "(" + digits(r, 3) + ")" + (r.nextBoolean() ? " " : "") + digits(r, 3) + "-" + digits(r, 4);
      case 1 -> digits(r, 3) + "-" + digits(r, 3) + "-" + digits(r, 4);
      case 2 -> digits(r, 3) + "-" + digits(r, 4);
      default -> digits(r, 1 + r.nextInt(5)) + "-" + digits(r, 1 + r.nextInt(5));
    };
  }

  private static String space(Random r) {
    return switch (r.nextInt(5)) {
      case 0 -> " ";
      case 1 -> "\t";
      default -> "";
    };
  }

  private static String mgrs(Random r) {
    return digits(r, 1 + r.nextInt(2)) + word(r, 1) + space(r) + word(r, 2) + space(r)
        + digits(r, 1 + r.nextInt(6)) + space(r) + digits(r, 1 + r.nextInt(6));
  }

  private static String degrees(Random r) {
    String[] deg = {"d", "D", "\u00b0", " ", "|"};
    String[] min = {"'", "\u2019", " ", "|"};
    String[] sec = {"\"", "\u201d", "|"};
    String[] ns = {"N", "n", "S", "s", "|", ""};
    String[] ew = {"E", "e", "W", "w", "|", ""};
    String[] sep = {" ", ",", ", ", "", "  "};
    StringBuilder b = new StringBuilder();
    for (int half = 0; half < 2; half++) {
      if (r.nextInt(4) == 0) {
        b.append(r.nextBoolean() ? "-" : "+");
      }
      b.append(digits(r, 1 + r.nextInt(3))).append(deg[r.nextInt(deg.length)]);
      if (r.nextBoolean()) {
        b.append(space(r)).append(digits(r, 1 + r.nextInt(2))).append(min[r.nextInt(min.length)]);
      }
      if (r.nextBoolean()) {
        b.append(space(r)).append(digits(r, 1 + r.nextInt(2))).append(sec[r.nextInt(sec.length)]);
      }
      b.append(space(r)).append((half == 0 ? ns : ew)[r.nextInt(6)]);
      if (half == 0) {
        b.append(sep[r.nextInt(sep.length)]);
      }
    }
    return b.toString();
  }

  private static String piece(Random r) {
    return switch (r.nextInt(8)) {
      case 0 -> email(r);
      case 1 -> url(r);
      case 2 -> phone(r);
      case 3 -> mgrs(r);
      case 4 -> degrees(r);
      case 5 -> NOISE[r.nextInt(NOISE.length)];
      default -> word(r, 1 + r.nextInt(6));
    };
  }

  /** Pieces joined by spaces, a few characters inserted at random, split at spaces. */
  static String[] regexSentence(Random r) {
    StringBuilder b = new StringBuilder();
    int pieces = 1 + r.nextInt(8);
    for (int i = 0; i < pieces; i++) {
      if (i > 0) {
        b.append(' ');
      }
      b.append(piece(r));
    }
    int noise = r.nextInt(4);
    for (int i = 0; i < noise; i++) {
      // Insert at a character boundary, never inside a surrogate pair.
      int at = r.nextInt(b.length() + 1);
      if (at > 0 && at < b.length() && Character.isLowSurrogate(b.charAt(at))) {
        at--;
      }
      b.insert(at, NOISE[r.nextInt(NOISE.length)]);
    }
    List<String> tokens = new ArrayList<>();
    for (String t : b.toString().split(" ")) {
      if (!t.isEmpty()) {
        tokens.add(t);
      }
    }
    if (tokens.isEmpty()) {
      tokens.add("x");
    }
    return tokens.toArray(new String[0]);
  }

  private static final String[] DICT_WORDS = {"New", "new", "NEW", "York", "york", "City", "city",
      "Hall", "San", "Francisco", "S\u00e3o", "s\u00c3o", "Paulo", "\u0130stanbul", "istanbul",
      "ISTANBUL", "strasse", "STRASSE", "stra\u00dfe", "\u212a", "k", "K", "\uD83D\uDE00", "face", "FACE",
      "John", "john", "JOHN", "Smith", "smith", "Dr.", "Who", "a", "b", "c", "d", "e", "f", "A", "E",
      "the", "of", ","};

  static String[] dictionarySentence(Random r) {
    String[] tokens = new String[1 + r.nextInt(12)];
    for (int i = 0; i < tokens.length; i++) {
      tokens[i] = DICT_WORDS[r.nextInt(DICT_WORDS.length)];
    }
    return tokens;
  }
}
