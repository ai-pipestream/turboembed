import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.namefind.DictionaryNameFinder;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.RegexNameFinder;
import opennlp.tools.namefind.RegexNameFinderFactory;
import opennlp.tools.namefind.RegexNameFinderFactory.DEFAULT_REGEX_NAME_FINDER;
import opennlp.tools.namefind.TokenNameFinder;
import opennlp.tools.util.Span;

/**
 * Writes the golden files of turbo-namefind: what DictionaryNameFinder, RegexNameFinder,
 * RegexNameFinderFactory and NameFinderME.dropOverlappingSpans return.
 *
 * <p>Usage: NameFindGolden golden-directory opennlp-commit
 */
public final class NameFindGolden {

  private static final String[] DICTIONARIES = {"places", "people", "overlap", "empty"};

  /** The hand-made sentences, keyed h0, h1, ... */
  private static final Map<String, String[]> HAND = new LinkedHashMap<>();
  /** Generated sentences for the regex finders, keyed f0, f1, ... */
  private static final Map<String, String[]> FUZZ = new LinkedHashMap<>();
  /** Generated sentences for the dictionary finders, keyed d0, d1, ... */
  private static final Map<String, String[]> DICT_FUZZ = new LinkedHashMap<>();

  private NameFindGolden() {
  }

  public static void main(String[] args) throws Exception {
    Path golden = Path.of(args[0]);
    String commit = args[1];
    Path expected = golden.resolve("expected");
    int i = 0;
    for (String line : Files.readAllLines(golden.resolve("inputs/sentences.txt"), StandardCharsets.UTF_8)) {
      HAND.put("h" + i++, line.split(" "));
    }
    Random random = new Random(247);
    for (i = 0; i < 3000; i++) {
      FUZZ.put("f" + i, Fuzz.regexSentence(random));
    }
    for (i = 0; i < 1000; i++) {
      DICT_FUZZ.put("d" + i, Fuzz.dictionarySentence(random));
    }
    try (Writer w = writer(expected.resolve("sentences.txt"))) {
      header(w, "sentences", commit);
      for (Map<String, String[]> set : List.of(FUZZ, DICT_FUZZ)) {
        for (Map.Entry<String, String[]> e : set.entrySet()) {
          line(w, e.getKey() + "\t" + join(e.getValue()));
        }
      }
    }
    dictionaries(golden, expected.resolve("dictionary.txt"), commit);
    builtins(expected.resolve("regex-builtin.txt"), commit);
    factory(expected.resolve("regex-factory.txt"), commit, random);
    user(golden, expected.resolve("regex-user.txt"), commit);
    merge(golden, expected.resolve("merge.txt"), commit, random);
  }

  private static Dictionary dictionary(Path golden, String name) throws IOException {
    try (InputStream in = Files.newInputStream(golden.resolve("inputs/dictionaries/" + name + ".xml"))) {
      return new Dictionary(in);
    }
  }

  private static void dictionaries(Path golden, Path out, String commit) throws IOException {
    try (Writer w = writer(out)) {
      header(w, "dictionary", commit);
      for (String name : DICTIONARIES) {
        line(w, "finder\t" + name);
        TokenNameFinder finder = new DictionaryNameFinder(dictionary(golden, name), name);
        run(w, finder, HAND);
        run(w, finder, DICT_FUZZ);
      }
    }
  }

  private static void builtins(Path out, String commit) throws Exception {
    try (Writer w = writer(out)) {
      header(w, "regex-builtin", commit);
      for (DEFAULT_REGEX_NAME_FINDER d : DEFAULT_REGEX_NAME_FINDER.values()) {
        line(w, "finder\t" + d.name());
        RegexNameFinder finder = RegexNameFinderFactory.getDefaultRegexNameFinders(d);
        run(w, finder, HAND);
        run(w, finder, FUZZ);
      }
      line(w, "finder\tALL");
      RegexNameFinder all = RegexNameFinderFactory.getDefaultRegexNameFinders(
          DEFAULT_REGEX_NAME_FINDER.values());
      line(w, "types\t" + join(types(all)));
      run(w, all, HAND);
      run(w, all, FUZZ);

      // A config over the defaults: EMAIL replaced, a new type added.
      Map<String, Pattern[]> config = new LinkedHashMap<>();
      config.put("EMAIL", new Pattern[] {Pattern.compile("[a-z]+@[a-z]+")});
      config.put("WORD", new Pattern[] {Pattern.compile("[A-Z]\\w+"), Pattern.compile("\\d+")});
      line(w, "finder\tCONFIG");
      RegexNameFinder configured = RegexNameFinderFactory.getDefaultRegexNameFinders(
          config, DEFAULT_REGEX_NAME_FINDER.values());
      line(w, "types\t" + join(types(configured)));
      run(w, configured, HAND);
    }
  }

  /** The type order of the factory's map, for combinations of defaults and configs. */
  private static void factory(Path out, String commit, Random random) throws Exception {
    DEFAULT_REGEX_NAME_FINDER[] all = DEFAULT_REGEX_NAME_FINDER.values();
    List<DEFAULT_REGEX_NAME_FINDER[]> defaults = new ArrayList<>();
    for (int mask = 0; mask < 32; mask++) {
      List<DEFAULT_REGEX_NAME_FINDER> chosen = new ArrayList<>();
      for (int b = 0; b < 5; b++) {
        if ((mask & (1 << b)) != 0) {
          chosen.add(all[b]);
        }
      }
      defaults.add(chosen.toArray(new DEFAULT_REGEX_NAME_FINDER[0]));
    }
    defaults.add(new DEFAULT_REGEX_NAME_FINDER[] {all[4], all[3], all[2], all[1], all[0]});
    defaults.add(new DEFAULT_REGEX_NAME_FINDER[] {all[1], all[1], all[2], all[1]});
    try (Writer w = writer(out)) {
      header(w, "regex-factory", commit);
      for (DEFAULT_REGEX_NAME_FINDER[] d : defaults) {
        line(w, "defaults\t" + names(d) + "\tconfig\t\t=>\t"
            + join(types(RegexNameFinderFactory.getDefaultRegexNameFinders(d))));
      }
      String[] pool = {"PERSON", "EMAIL", "URL", "MGRS", "PHONE_NUM", "DATE", "a", "b", "Aa", "BB",
          "\u00e9t\u00e9", "\uD83D\uDE00", "x1", "x2", "x3", "ORG", "LOC"};
      for (int n = 0; n < 400; n++) {
        Map<String, Pattern[]> config = new LinkedHashMap<>();
        int size = random.nextInt(n < 200 ? 6 : 60);
        for (int k = 0; k < size; k++) {
          String key = random.nextInt(3) == 0 ? pool[random.nextInt(pool.length)]
              : Fuzz.word(random, 1 + random.nextInt(6));
          config.put(key, new Pattern[] {Pattern.compile("x")});
        }
        DEFAULT_REGEX_NAME_FINDER[] d = switch (n % 4) {
          case 0 -> null;
          case 1 -> new DEFAULT_REGEX_NAME_FINDER[0];
          case 2 -> all;
          default -> defaults.get(random.nextInt(defaults.size()));
        };
        RegexNameFinder finder = RegexNameFinderFactory.getDefaultRegexNameFinders(config, d);
        line(w, "defaults\t" + (d == null ? "null" : names(d)) + "\tconfig\t"
            + join(new ArrayList<>(config.keySet())) + "\t=>\t" + join(types(finder)));
      }
    }
  }

  private static void user(Path golden, Path out, String commit) throws IOException {
    List<String> rows = Files.readAllLines(golden.resolve("inputs/patterns.txt"), StandardCharsets.UTF_8);
    Map<String, String[]> sentences = new LinkedHashMap<>(HAND);
    int k = 0;
    for (Map.Entry<String, String[]> e : FUZZ.entrySet()) {
      if (k++ == 400) {
        break;
      }
      sentences.put(e.getKey(), e.getValue());
    }
    try (Writer w = writer(out)) {
      header(w, "regex-user", commit);
      for (String row : rows) {
        String[] f = row.split("\t", 3);
        int flags = 0;
        if (f[1].contains("i")) {
          flags |= Pattern.CASE_INSENSITIVE;
        }
        if (f[1].contains("s")) {
          flags |= Pattern.DOTALL;
        }
        Pattern p;
        try {
          p = Pattern.compile(f[2], flags);
        } catch (PatternSyntaxException e) {
          line(w, "pattern\t" + f[1] + "\t" + escape(f[2]) + "\t=>\terror");
          continue;
        }
        line(w, "pattern\t" + f[1] + "\t" + escape(f[2]) + "\t=>\tok");
        run(w, new RegexNameFinder(new Pattern[] {p}, "T"), sentences);
      }
    }
  }

  private static void merge(Path golden, Path out, String commit, Random random) throws IOException {
    String[] types = {"A", "B", "a", "\u00e9", "\uD83D\uDE00", "\uff21", ""};
    try (Writer w = writer(out)) {
      header(w, "merge", commit);
      for (int n = 0; n < 3000; n++) {
        Span[] spans = new Span[random.nextInt(13)];
        for (int k = 0; k < spans.length; k++) {
          int start = random.nextInt(10);
          spans[k] = new Span(start, start + random.nextInt(5), types[random.nextInt(types.length)]);
        }
        line(w, spans(spans) + "\t=>\t" + spans(NameFinderME.dropOverlappingSpans(spans)));
      }
      // Real finders, in this order: places, people, overlap, then every default pattern.
      List<TokenNameFinder> finders = new ArrayList<>();
      for (String name : DICTIONARIES) {
        finders.add(new DictionaryNameFinder(dictionary(golden, name), name));
      }
      finders.add(RegexNameFinderFactory.getDefaultRegexNameFinders(DEFAULT_REGEX_NAME_FINDER.values()));
      for (Map<String, String[]> set : List.of(HAND, FUZZ)) {
        for (Map.Entry<String, String[]> e : set.entrySet()) {
          List<Span> names = new ArrayList<>();
          for (TokenNameFinder finder : finders) {
            names.addAll(Arrays.asList(finder.find(e.getValue())));
          }
          Span[] kept = NameFinderME.dropOverlappingSpans(names.toArray(new Span[0]));
          if (kept.length > 0) {
            line(w, "finders\t" + e.getKey() + "\t=>\t" + spans(kept));
          }
        }
      }
    }
  }

  /** One line per sentence with names: the key and the names, in the finder's order. */
  private static void run(Writer w, TokenNameFinder finder, Map<String, String[]> sentences)
      throws IOException {
    for (Map.Entry<String, String[]> e : sentences.entrySet()) {
      Span[] found = finder.find(e.getValue());
      if (found.length > 0) {
        line(w, e.getKey() + "\t" + spans(found));
      }
    }
  }

  private static String spans(Span[] spans) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < spans.length; i++) {
      if (i > 0) {
        b.append('\t');
      }
      b.append(spans[i].getStart()).append(' ').append(spans[i].getEnd()).append(' ')
          .append(escape(spans[i].getType()));
    }
    return b.toString();
  }

  @SuppressWarnings("unchecked")
  private static List<String> types(RegexNameFinder finder) throws IOException {
    try {
      Field f = RegexNameFinder.class.getDeclaredField("regexMap");
      f.setAccessible(true);
      return new ArrayList<>(((Map<String, Pattern[]>) f.get(finder)).keySet());
    } catch (ReflectiveOperationException e) {
      throw new IOException(e);
    }
  }

  private static String names(DEFAULT_REGEX_NAME_FINDER[] d) {
    List<String> out = new ArrayList<>();
    for (DEFAULT_REGEX_NAME_FINDER x : d) {
      out.add(x.name());
    }
    return String.join(",", out);
  }

  /** Random sentences built from pieces that the patterns and dictionaries look for. */
  static final class Fuzz {

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

    private Fuzz() {
    }

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

  static String join(String[] values) {
    return join(Arrays.asList(values));
  }

  static String join(List<String> values) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < values.size(); i++) {
      if (i > 0) {
        b.append('\t');
      }
      b.append(escape(values.get(i)));
    }
    return b.toString();
  }

  static Writer writer(Path path) throws IOException {
    Files.createDirectories(path.getParent());
    return Files.newBufferedWriter(path, StandardCharsets.UTF_8);
  }

  static void line(Writer w, String line) throws IOException {
    w.write(line);
    w.write('\n');
  }

  static void header(Writer w, String component, String commit) throws IOException {
    line(w, "component: " + component);
    line(w, "opennlp: " + commit);
    line(w, "jdk: " + System.getProperty("java.version"));
  }
}
