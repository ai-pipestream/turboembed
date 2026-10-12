import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import opennlp.tools.stopword.StopwordFilter;
import opennlp.tools.stopword.StopwordFilterStream;
import opennlp.tools.stopword.StopwordFilteringTokenizer;
import opennlp.tools.stopword.StopwordLists;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Span;

/**
 * Writes the golden files of turbo-stopword: the bundled lists as StopwordLists loads them, how
 * forLanguage resolves language codes, isStopword for a set of words in every list, and what
 * the removing filter, tokenizer and stream keep.
 *
 * <p>Usage: StopwordGolden crate-directory opennlp-commit
 */
public final class StopwordGolden {

  /** Appended to words: a character outside the BMP (U+1D41A). */
  private static final String SUPPLEMENTARY = "𝐚";

  private static final String[] EXTRAS = {
    "", " ", "\t", "x", "I", "i", "İ", "ı", "ſ", "K", "Å", "ẞ",
    "SS", "ss", "123", "😀", "𝐀", "𐐀𐐨",
    "ΣΑΣ", "σας", "Ꙋ", "the\u0000", "﻿the", "the ",
    "一", "ال", "é", "é", "É"
  };

  private StopwordGolden() {
  }

  public static void main(String[] args) throws Exception {
    Path crate = Path.of(args[0]);
    String commit = args[1];
    Path expected = crate.resolve("golden/expected");
    List<String> codes = new ArrayList<>(StopwordLists.supportedLanguages());
    checkResources(crate, codes);

    Map<String, List<String>> lists = new TreeMap<>();
    try (Writer w = writer(expected.resolve("lists.txt"))) {
      header(w, "lists", commit);
      for (String code : codes) {
        StopwordFilter filter = StopwordLists.forLanguage(code);
        List<String> words = new ArrayList<>(new TreeSet<>(filter.stopwords()));
        lists.put(code, words);
        line(w, "list\t" + code + "\t" + words.size() + "\t" + filter.isCaseSensitive());
        for (String word : words) {
          line(w, escape(word));
        }
      }
    }

    languageCodes(expected.resolve("codes.txt"), commit, codes);

    Map<Integer, List<Integer>> irregular = new HashMap<>();
    Map<Integer, List<Integer>> otherUpper = new HashMap<>();
    for (int cp = 0; cp <= Character.MAX_CODE_POINT; cp++) {
      int lower = Character.toLowerCase(cp);
      int fold = Character.toLowerCase(Character.toUpperCase(cp));
      if (fold != lower) {
        irregular.computeIfAbsent(fold, k -> new ArrayList<>()).add(cp);
      }
      if (lower != cp && cp != Character.toUpperCase(lower)) {
        otherUpper.computeIfAbsent(lower, k -> new ArrayList<>()).add(cp);
      }
    }

    LinkedHashSet<String> set = new LinkedHashSet<>(Arrays.asList(EXTRAS));
    Map<String, List<String>> own = new TreeMap<>();
    for (String code : codes) {
      List<String> mine = new ArrayList<>();
      for (String word : lists.get(code)) {
        for (String v : variants(word, irregular, otherUpper)) {
          set.add(v);
          mine.add(v);
        }
      }
      own.put(code, mine);
    }
    List<String> words = new ArrayList<>(set);
    Map<String, Integer> index = new HashMap<>();
    for (int i = 0; i < words.size(); i++) {
      index.put(words.get(i), i);
    }

    try (Writer w = writer(expected.resolve("words.txt"))) {
      header(w, "words", commit);
      for (String word : words) {
        line(w, escape(word));
      }
    }

    try (Writer w = writer(expected.resolve("flags.txt"))) {
      header(w, "flags", commit);
      for (String code : codes) {
        StopwordFilter filter = StopwordLists.forLanguage(code);
        StringBuilder b = new StringBuilder(code).append('\t');
        String sep = "";
        for (int i = 0; i < words.size(); i++) {
          String word = words.get(i);
          boolean stop = filter.isStopword(word);
          if (stop != filter.isStopword(new String[] {word})) {
            throw new IllegalStateException("isStopword disagrees for " + escape(word));
          }
          if (stop) {
            b.append(sep).append(i);
            sep = " ";
          }
        }
        line(w, b.toString());
      }
    }

    removal(expected.resolve("filter.txt"), commit, codes, words, own, index);
    letters(expected.resolve("letters.txt"), commit, lists);
  }

  /** The crate's data files must be the jar's resources byte for byte. */
  private static void checkResources(Path crate, List<String> codes) throws IOException {
    for (String code : codes) {
      byte[] mine = Files.readAllBytes(crate.resolve("data/" + code + ".txt"));
      try (InputStream in = StopwordLists.class.getResourceAsStream("/opennlp/tools/stopword/" + code + ".txt")) {
        if (!Arrays.equals(mine, in.readAllBytes())) {
          throw new IllegalStateException("data/" + code + ".txt differs from the jar's resource");
        }
      }
    }
  }

  /**
   * Every code LanguageCodeValidator accepts, rebuilt as it builds its set, and case variants
   * and malformed forms of the supported ones, with the list forLanguage returns or "-".
   */
  private static void languageCodes(Path path, String commit, List<String> supported) throws IOException {
    TreeSet<String> candidates = new TreeSet<>();
    for (Locale locale : Locale.getAvailableLocales()) {
      if (!locale.getLanguage().isEmpty()) {
        candidates.add(locale.getLanguage());
      }
      try {
        if (!locale.getISO3Language().isEmpty()) {
          candidates.add(locale.getISO3Language());
        }
      } catch (Exception ignored) {
        // no three-letter code
      }
    }
    candidates.addAll(List.of("dut", "fre", "ger", "und", "x-unspecified", "", "e", "en-US", "en_US", " en", "en ",
        "engl", "İt"));
    for (String code : supported) {
      String three = Locale.forLanguageTag(code).getISO3Language();
      for (String c : List.of(code, three)) {
        candidates.add(c.toUpperCase(Locale.ROOT));
        candidates.add(c.substring(0, 1).toUpperCase(Locale.ROOT) + c.substring(1));
      }
    }
    try (Writer w = writer(path)) {
      header(w, "codes", commit);
      for (String code : candidates) {
        String result;
        try {
          StopwordFilter filter = StopwordLists.forLanguage(code);
          result = "-";
          for (String s : supported) {
            if (StopwordLists.forLanguage(s) == filter) {
              result = s;
            }
          }
          if (result.equals("-")) {
            throw new IllegalStateException("an uncached filter for " + code);
          }
        } catch (IllegalArgumentException e) {
          result = "-";
        }
        line(w, escape(code) + "\t" + result);
      }
    }
  }

  /** The word, its case variants and its spellings with other characters, in a fixed order. */
  private static List<String> variants(String word, Map<Integer, List<Integer>> irregular,
                                       Map<Integer, List<Integer>> otherUpper) {
    LinkedHashSet<String> out = new LinkedHashSet<>();
    out.add(word);
    out.add(word.toUpperCase(Locale.ROOT));
    out.add(new String(Character.toChars(Character.toTitleCase(word.codePointAt(0))))
        + word.substring(Character.charCount(word.codePointAt(0))));
    out.add(new String(word.codePoints().map(Character::toUpperCase).toArray(), 0, word.codePointCount(0,
        word.length())));
    out.add(word.toUpperCase(Locale.ROOT).toLowerCase(Locale.ROOT));
    String nfd = Normalizer.normalize(word, Normalizer.Form.NFD);
    out.add(nfd);
    out.add(Normalizer.normalize(nfd.replaceAll("\\p{M}", ""), Normalizer.Form.NFC));
    for (Map<Integer, List<Integer>> table : List.of(irregular, otherUpper)) {
      for (int cp : word.codePoints().distinct().toArray()) {
        for (int other : table.getOrDefault(cp, List.of())) {
          String from = new String(Character.toChars(cp));
          String to = new String(Character.toChars(other));
          out.add(word.replaceFirst(java.util.regex.Pattern.quote(from), to));
          out.add(word.replace(from, to));
        }
      }
    }
    out.add(word + SUPPLEMENTARY);
    out.add(SUPPLEMENTARY + word);
    out.add(word + "'");
    out.add(word + " ");
    out.add(word + "‍");
    out.add(word.substring(0, word.length() - 1));
    out.add(word + word.substring(word.length() - 1));
    return new ArrayList<>(out);
  }

  /**
   * Random token sequences per language, and the positions StopwordFilter.filter,
   * StopwordFilterStream and StopwordFilteringTokenizer.tokenizePos (over WhitespaceTokenizer)
   * keep.
   */
  private static void removal(Path path, String commit, List<String> codes, List<String> words,
                              Map<String, List<String>> own, Map<String, Integer> index) throws Exception {
    List<String> plain = new ArrayList<>();
    Set<String> isPlain = new java.util.HashSet<>();
    for (String word : words) {
      String[] t = WhitespaceTokenizer.INSTANCE.tokenize(word);
      if (t.length == 1 && t[0].equals(word)) {
        plain.add(word);
        isPlain.add(word);
      }
    }
    Random random = new Random(200);
    try (Writer w = writer(path)) {
      header(w, "filter", commit);
      for (String code : codes) {
        StopwordFilter filter = StopwordLists.forLanguage(code);
        StopwordFilteringTokenizer tokenizer = new StopwordFilteringTokenizer(WhitespaceTokenizer.INSTANCE, filter);
        List<String> mine = new ArrayList<>();
        for (String word : own.get(code)) {
          if (isPlain.contains(word)) {
            mine.add(word);
          }
        }
        for (int s = 0; s < 150; s++) {
          int n = 1 + random.nextInt(25);
          String[] tokens = new String[n];
          for (int i = 0; i < n; i++) {
            List<String> from = random.nextInt(10) < 6 ? mine : plain;
            tokens[i] = from.get(random.nextInt(from.size()));
          }
          String[] kept = filter.filter(tokens);
          String[] streamed;
          try (StopwordFilterStream stream = new StopwordFilterStream(
              ObjectStreamUtils.createObjectStream(List.<String[]>of(tokens)), filter)) {
            streamed = stream.read();
          }
          if (!Arrays.equals(kept, streamed)) {
            throw new IllegalStateException("the stream and the filter disagree");
          }
          String text = String.join(" ", tokens);
          Span[] spans = tokenizer.tokenizePos(text);
          int[] starts = new int[n];
          for (int i = 1; i < n; i++) {
            starts[i] = starts[i - 1] + tokens[i - 1].length() + 1;
          }
          StringBuilder ids = new StringBuilder();
          for (int i = 0; i < n; i++) {
            ids.append(i == 0 ? "" : " ").append(index.get(tokens[i]));
          }
          line(w, code + "\t" + ids + "\t" + positions(tokens, kept) + "\t" + spanPositions(starts, spans));
        }
      }
    }
  }

  /** The positions of the kept tokens: the filter only removes, so they are a subsequence. */
  private static String positions(String[] tokens, String[] kept) {
    StringBuilder b = new StringBuilder();
    int j = 0;
    for (int i = 0; i < tokens.length && j < kept.length; i++) {
      if (tokens[i].equals(kept[j])) {
        b.append(j == 0 ? "" : " ").append(i);
        j++;
      }
    }
    if (j != kept.length) {
      throw new IllegalStateException("kept tokens are not a subsequence");
    }
    return b.toString();
  }

  private static String spanPositions(int[] starts, Span[] spans) {
    StringBuilder b = new StringBuilder();
    for (int k = 0; k < spans.length; k++) {
      int i = Arrays.binarySearch(starts, spans[k].getStart());
      if (i < 0) {
        throw new IllegalStateException("a span that starts at no token");
      }
      b.append(k == 0 ? "" : " ").append(i);
    }
    return b.toString();
  }

  /**
   * Every code point whose Character.toLowerCase is a character of some bundled word, with that
   * lowercase: the whole of the case mapping the lookup can depend on.
   */
  private static void letters(Path path, String commit, Map<String, List<String>> lists) throws IOException {
    Set<Integer> letters = new TreeSet<>();
    for (List<String> words : lists.values()) {
      for (String word : words) {
        word.codePoints().forEach(letters::add);
      }
    }
    try (Writer w = writer(path)) {
      header(w, "letters", commit);
      for (int cp = 0; cp <= Character.MAX_CODE_POINT; cp++) {
        int lower = Character.toLowerCase(cp);
        if (letters.contains(lower)) {
          line(w, String.format("%04X %04X", cp, lower));
        }
      }
    }
  }

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
