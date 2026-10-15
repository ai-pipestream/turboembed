package ai.pipestream.turbo.goldens;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
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
 * OpenNLP's bundled stopword lists ({@code StopwordLists}), over the {@code stopword} input set.
 *
 * <p>The runs and their layers, anchored to the document:</p>
 * <ul>
 *   <li>{@code lists}, on {@code stopword/words}: {@code turbo:lists} (each language's code, number
 *       of words, whether its filter is case sensitive and the SHA-256 of the list's resource file
 *       in the OpenNLP jar, which the crate bundles) and {@code turbo:list-words} (its words,
 *       sorted);</li>
 *   <li>{@code codes}, on {@code stopword/codes}: the list {@code forLanguage} returns, or none,
 *       for every language code {@code LanguageCodeValidator} accepts (rebuilt as it builds its set,
 *       from the JDK's locales), the upper and title case forms of the supported codes and their
 *       three letter forms, and the codes of the input, one per line: {@code turbo:codes};</li>
 *   <li>{@code words}, on {@code stopword/words}: {@code turbo:words}, the words the lookup is
 *       asked about: the input's words (one per line, escaped as {@link EscapedLines} reads them),
 *       then every list word with its case variants, its decomposed and stripped forms, its
 *       spellings with the other characters whose case mapping meets its own, and with a character
 *       outside the Basic Multilingual Plane, an apostrophe, a space or a zero width joiner added,
 *       its last unit dropped or doubled; and {@code turbo:stopwords}, the words each list
 *       flags;</li>
 *   <li>{@code filter}, on {@code stopword/words}: 150 random token sequences per language from a
 *       seeded {@code Random}, and the tokens {@code StopwordFilter.filter},
 *       {@code StopwordFilterStream} and {@code StopwordFilteringTokenizer.tokenizePos} over
 *       {@code WhitespaceTokenizer} keep: {@code turbo:filter-cases} and
 *       {@code turbo:filter-tokens} (each token as a row of {@code turbo:words} and whether the
 *       filter and the tokenizer kept it);</li>
 *   <li>{@code letters}, on {@code stopword/words}: {@code turbo:letters}, every code point whose
 *       {@code Character.toLowerCase} is a character of some list word, with that lower case.</li>
 * </ul>
 */
final class StopwordGolden implements AnnotatorGolden {

  static final String SET = Runs.STOPWORD;

  /** Appended to and put before words: a character outside the BMP (U+1D41A). */
  private static final String SUPPLEMENTARY = "𝐚";

  enum Kind {
    LISTS("lists", "words"),
    CODES("codes", "codes"),
    WORDS("words", "words"),
    FILTER("filter", "words"),
    LETTERS("letters", "words");

    final String dir;
    final String input;

    Kind(String dir, String input) {
      this.dir = dir;
      this.input = input;
    }
  }

  private final Kind kind;

  private StopwordGolden(Kind kind) {
    this.kind = kind;
  }

  static List<StopwordGolden> all() {
    List<StopwordGolden> out = new ArrayList<>();
    for (Kind k : Kind.values()) {
      out.add(new StopwordGolden(k));
    }
    return out;
  }

  @Override
  public String annotator() {
    return "stopword";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return kind.dir;
  }

  @Override
  public String annotatorClass() {
    return switch (kind) {
      case LISTS, CODES -> StopwordLists.class.getName();
      case WORDS -> StopwordFilter.class.getName();
      case FILTER -> StopwordFilteringTokenizer.class.getName();
      case LETTERS -> Character.class.getName();
    };
  }

  @Override
  public String method() {
    return switch (kind) {
      case LISTS, CODES -> "forLanguage";
      case WORDS -> "isStopword";
      case FILTER -> "tokenizePos";
      case LETTERS -> "toLowerCase";
    };
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(SET) && input.id().equals(kind.input);
  }

  @Override
  public List<Layer> layers(Input input) {
    List<String> codes = new ArrayList<>(StopwordLists.supportedLanguages());
    Map<String, List<String>> lists = new TreeMap<>();
    for (String code : codes) {
      lists.put(code, new ArrayList<>(new TreeSet<>(StopwordLists.forLanguage(code).stopwords())));
    }
    return switch (kind) {
      case LISTS -> lists(codes, lists);
      case CODES -> codes(codes, input.lines());
      case WORDS -> words(codes, lists, input);
      case FILTER -> filter(codes, lists, input);
      case LETTERS -> letters(lists);
    };
  }

  private static List<Layer> lists(List<String> codes, Map<String, List<String>> lists) {
    Table summary = new Table("turbo:lists", "language:str", "words:u32", "case-sensitive:u32", "resource-sha256:str");
    Table words = new Table("turbo:list-words", "language:str", "word:str");
    for (String code : codes) {
      List<String> list = lists.get(code);
      byte[] resource;
      try (java.io.InputStream in = StopwordLists.class.getResourceAsStream("/opennlp/tools/stopword/" + code + ".txt")) {
        resource = in.readAllBytes();
      } catch (java.io.IOException e) {
        throw new IllegalStateException("the resource of " + code, e);
      }
      summary.row(code, list.size(), Table.flag(StopwordLists.forLanguage(code).isCaseSensitive()),
          Sha.sha256(resource));
      for (String w : list) {
        words.row(code, w);
      }
    }
    return List.of(summary.layer(), words.layer());
  }

  private static List<Layer> codes(List<String> supported, List<String> extra) {
    Set<String> candidates = new TreeSet<>();
    for (Locale locale : Locale.getAvailableLocales()) {
      if (!locale.getLanguage().isEmpty()) {
        candidates.add(locale.getLanguage());
      }
      try {
        if (!locale.getISO3Language().isEmpty()) {
          candidates.add(locale.getISO3Language());
        }
      } catch (java.util.MissingResourceException ignored) {
        // No three letter code.
      }
    }
    candidates.addAll(extra);
    for (String code : supported) {
      String three = Locale.forLanguageTag(code).getISO3Language();
      for (String c : List.of(code, three)) {
        candidates.add(c.toUpperCase(Locale.ROOT));
        candidates.add(c.substring(0, 1).toUpperCase(Locale.ROOT) + c.substring(1));
      }
    }
    Table t = new Table("turbo:codes", "code:str", "list:str");
    for (String code : candidates) {
      String result = "";
      try {
        StopwordFilter filter = StopwordLists.forLanguage(code);
        for (String s : supported) {
          if (StopwordLists.forLanguage(s) == filter) {
            result = s;
          }
        }
        if (result.isEmpty()) {
          throw new IllegalStateException("an uncached filter for " + code);
        }
      } catch (IllegalArgumentException e) {
        result = "";
      }
      t.row(code, result);
    }
    return List.of(t.layer());
  }

  /** The words the lookup is asked about, in order: the input's, then each list's variants. */
  private static List<String> words(List<String> codes, Map<String, List<String>> lists, Input input,
      Map<String, List<String>> own) {
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
    LinkedHashSet<String> set = new LinkedHashSet<>(EscapedLines.read(input));
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
    return new ArrayList<>(set);
  }

  private static List<Layer> words(List<String> codes, Map<String, List<String>> lists, Input input) {
    List<String> words = words(codes, lists, input, new HashMap<>());
    Table w = new Table("turbo:words", "word:str");
    for (String word : words) {
      w.row(word);
    }
    Table stops = new Table("turbo:stopwords", "language:str", "word:u32");
    for (String code : codes) {
      StopwordFilter filter = StopwordLists.forLanguage(code);
      for (int i = 0; i < words.size(); i++) {
        String word = words.get(i);
        boolean stop = filter.isStopword(word);
        if (stop != filter.isStopword(new String[] {word})) {
          throw new IllegalStateException("isStopword disagrees for " + TrainGolden.lossless(word));
        }
        if (stop) {
          stops.row(code, i);
        }
      }
    }
    return List.of(w.layer(), stops.layer());
  }

  /** The word, its case variants and its spellings with other characters, in a fixed order. */
  private static List<String> variants(String word, Map<Integer, List<Integer>> irregular,
      Map<Integer, List<Integer>> otherUpper) {
    LinkedHashSet<String> out = new LinkedHashSet<>();
    out.add(word);
    out.add(word.toUpperCase(Locale.ROOT));
    out.add(new String(Character.toChars(Character.toTitleCase(word.codePointAt(0))))
        + word.substring(Character.charCount(word.codePointAt(0))));
    out.add(new String(word.codePoints().map(Character::toUpperCase).toArray(), 0,
        word.codePointCount(0, word.length())));
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

  private static List<Layer> filter(List<String> codes, Map<String, List<String>> lists, Input input) {
    Map<String, List<String>> own = new HashMap<>();
    List<String> words = words(codes, lists, input, own);
    Map<String, Integer> index = new HashMap<>();
    for (int i = 0; i < words.size(); i++) {
      index.put(words.get(i), i);
    }
    List<String> plain = new ArrayList<>();
    Set<String> isPlain = new HashSet<>();
    for (String word : words) {
      String[] t = WhitespaceTokenizer.INSTANCE.tokenize(word);
      if (t.length == 1 && t[0].equals(word)) {
        plain.add(word);
        isPlain.add(word);
      }
    }
    Table cases = new Table("turbo:filter-cases", "case:u32", "language:str");
    Table tokensTable = new Table("turbo:filter-tokens", "case:u32", "word:u32", "filter-kept:u32",
        "tokenizer-kept:u32");
    Random random = new Random(200);
    int c = 0;
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
        try (StopwordFilterStream stream =
            new StopwordFilterStream(ObjectStreamUtils.createObjectStream(List.<String[]>of(tokens)), filter)) {
          streamed = stream.read();
        } catch (java.io.IOException e) {
          throw new IllegalStateException("a stream over one array failed", e);
        }
        if (!Arrays.equals(kept, streamed)) {
          throw new IllegalStateException("the stream and the filter disagree");
        }
        boolean[] filterKept = positions(tokens, kept);
        boolean[] tokenizerKept = spanPositions(tokens, tokenizer.tokenizePos(String.join(" ", tokens)));
        cases.row(c, code);
        for (int i = 0; i < n; i++) {
          tokensTable.row(c, index.get(tokens[i]), Table.flag(filterKept[i]), Table.flag(tokenizerKept[i]));
        }
        c++;
      }
    }
    return List.of(cases.layer(), tokensTable.layer());
  }

  /** Which tokens the filter kept: it only removes, so the kept ones are a subsequence. */
  private static boolean[] positions(String[] tokens, String[] kept) {
    boolean[] out = new boolean[tokens.length];
    int j = 0;
    for (int i = 0; i < tokens.length && j < kept.length; i++) {
      if (tokens[i].equals(kept[j])) {
        out[i] = true;
        j++;
      }
    }
    if (j != kept.length) {
      throw new IllegalStateException("kept tokens are not a subsequence");
    }
    return out;
  }

  /** Which tokens the spans start at, the tokens joined by single spaces. */
  private static boolean[] spanPositions(String[] tokens, Span[] spans) {
    int[] starts = new int[tokens.length];
    for (int i = 1; i < tokens.length; i++) {
      starts[i] = starts[i - 1] + tokens[i - 1].length() + 1;
    }
    boolean[] out = new boolean[tokens.length];
    for (Span s : spans) {
      int i = Arrays.binarySearch(starts, s.getStart());
      if (i < 0) {
        throw new IllegalStateException("a span that starts at no token");
      }
      out[i] = true;
    }
    return out;
  }

  private static List<Layer> letters(Map<String, List<String>> lists) {
    Set<Integer> letters = new TreeSet<>();
    for (List<String> words : lists.values()) {
      for (String word : words) {
        word.codePoints().forEach(letters::add);
      }
    }
    Table t = new Table("turbo:letters", "code-point:u32", "lower:u32");
    for (int cp = 0; cp <= Character.MAX_CODE_POINT; cp++) {
      int lower = Character.toLowerCase(cp);
      if (letters.contains(lower)) {
        t.row(cp, lower);
      }
    }
    return List.of(t.layer());
  }
}
