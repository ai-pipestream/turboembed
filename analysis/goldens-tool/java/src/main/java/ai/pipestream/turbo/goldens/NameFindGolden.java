package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
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
 * The name finders that need no model, run on the {@code namefind/sentences} input (hand-made
 * tokenized sentences, one per line, tokens separated by single spaces, keyed {@code h0},
 * {@code h1}, ...) and on sentences generated from a seeded {@code Random} ({@link NameFindFuzz}:
 * 3,000 for the regular expression finders, keyed {@code f0}, ..., and 1,000 for the dictionary
 * finders, keyed {@code d0}, ...). Every run draws the generated sentences first, then the
 * {@code regex-factory} cases, then the {@code merge} cases, from one {@code Random(247)}, so each
 * run's cases are the same whichever runs are selected.
 *
 * <p>The runs and their layers, anchored to the document; a name is a row with its finder or
 * sentence, its token span and its type:</p>
 * <ul>
 *   <li>{@code generated}: {@code turbo:sentences}, the tokens of the generated sentences with
 *       their keys, in order;</li>
 *   <li>{@code dictionary}: {@code DictionaryNameFinder} over each fixture dictionary (the type is
 *       the dictionary's name) on the hand-made and the dictionary sentences, {@code turbo:names};</li>
 *   <li>{@code regex-builtin}: {@code RegexNameFinderFactory}'s default finders one at a time and
 *       all together ({@code ALL}) on the hand-made and the regular expression sentences, and a
 *       configuration over the defaults ({@code CONFIG}: {@code EMAIL} replaced, a type
 *       {@code WORD} added) on the hand-made ones, {@code turbo:names}, with
 *       {@code turbo:finder-types} the type order of the combined finders;</li>
 *   <li>{@code regex-factory}: the type order of the factory's finder for every subset of the
 *       defaults, two orders with repeats, and 400 random configurations with no, all or some
 *       defaults: {@code turbo:factory-cases} (whether defaults were given), and
 *       {@code turbo:factory-defaults}, {@code turbo:factory-keys} and {@code turbo:factory-types}
 *       per case;</li>
 *   <li>{@code regex-user}: each pattern of the fixture {@code patterns.txt} (an expectation for
 *       the Rust port, flags and the pattern, separated by tabs) compiled by
 *       {@code java.util.regex} and, when it compiles, run as a {@code RegexNameFinder} of type
 *       {@code T} on the hand-made and the first 400 regular expression sentences:
 *       {@code turbo:patterns} and {@code turbo:pattern-names};</li>
 *   <li>{@code merge}: {@code NameFinderME.dropOverlappingSpans} on 3,000 random span lists
 *       ({@code turbo:merge-cases}, {@code turbo:merge-spans} and {@code turbo:merge-kept}), and
 *       on the names of the four dictionary finders and all the default finders together, in that
 *       order, for every hand-made and regular expression sentence that has any
 *       ({@code turbo:merged-names}).</li>
 * </ul>
 */
final class NameFindGolden implements AnnotatorGolden {

  static final String SET = Runs.NAMEFIND;

  /** The dictionaries, in the order the runs use them. */
  static final List<String> DICTIONARIES = List.of("places", "people", "overlap", "empty");

  enum Kind {
    GENERATED("generated"),
    DICTIONARY("dictionary"),
    REGEX_BUILTIN("regex-builtin"),
    REGEX_FACTORY("regex-factory"),
    REGEX_USER("regex-user"),
    MERGE("merge");

    final String dir;

    Kind(String dir) {
      this.dir = dir;
    }
  }

  private final Kind kind;
  private final List<ModelFile> files;

  private NameFindGolden(Kind kind, List<ModelFile> files) {
    this.kind = kind;
    this.files = files;
  }

  static List<NameFindGolden> all(Path fixtures) throws IOException {
    Path dir = fixtures.resolve(Runs.NAMEFIND_FIXTURES);
    List<ModelFile> dictionaries = new ArrayList<>();
    for (String name : DICTIONARIES) {
      dictionaries.add(ModelFile.read(dir.resolve(name + ".xml")));
    }
    ModelFile patterns = ModelFile.read(dir.resolve(Runs.NAMEFIND_PATTERNS));
    return List.of(new NameFindGolden(Kind.GENERATED, List.of()),
        new NameFindGolden(Kind.DICTIONARY, dictionaries),
        new NameFindGolden(Kind.REGEX_BUILTIN, List.of()),
        new NameFindGolden(Kind.REGEX_FACTORY, List.of()),
        new NameFindGolden(Kind.REGEX_USER, List.of(patterns, ModelFile.read(dir.resolve("regex-boundaries.txt")))),
        new NameFindGolden(Kind.MERGE, dictionaries));
  }

  /** Regenerate regex-user alone, without loading unrelated statistical models. */
  public static void main(String[] args) throws IOException {
    if (args.length != 4) {
      throw new IllegalArgumentException("fixtures input.txt output.golden opennlp-commit");
    }
    NameFindGolden run = all(Path.of(args[0])).stream().filter(a -> a.kind == Kind.REGEX_USER).findFirst().orElseThrow();
    Input input = Input.read(Path.of(args[1]), SET, "sentences");
    Header header = new Header(args[3], OpennlpVersion.read(), Jdk.current(), input.inputId(),
        input.bytes().length, input.text().length(), Sha.sha256(input.bytes()), run.models(),
        run.annotatorClass(), run.method(), run.config());
    java.nio.file.Files.write(Path.of(args[2]), Main.golden(header, run, input));
  }

  @Override
  public String annotator() {
    return "namefind";
  }

  @Override
  public List<ModelRef> models() {
    return files.stream().map(ModelFile::ref).toList();
  }

  @Override
  public String modelDir() {
    return kind.dir;
  }

  @Override
  public String annotatorClass() {
    return switch (kind) {
      case GENERATED, REGEX_BUILTIN, REGEX_USER -> RegexNameFinder.class.getName();
      case DICTIONARY -> DictionaryNameFinder.class.getName();
      case REGEX_FACTORY -> RegexNameFinderFactory.class.getName();
      case MERGE -> NameFinderME.class.getName();
    };
  }

  @Override
  public String method() {
    return switch (kind) {
      case REGEX_FACTORY -> "getDefaultRegexNameFinders";
      case MERGE -> "dropOverlappingSpans";
      default -> "find";
    };
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public List<Layer> layers(Input input) {
    Map<String, String[]> hand = new LinkedHashMap<>();
    int i = 0;
    for (String[] s : Sentences.of(input)) {
      hand.put("h" + i++, s);
    }
    Random random = new Random(247);
    Map<String, String[]> fuzz = new LinkedHashMap<>();
    for (i = 0; i < 3000; i++) {
      fuzz.put("f" + i, NameFindFuzz.regexSentence(random));
    }
    Map<String, String[]> dictFuzz = new LinkedHashMap<>();
    for (i = 0; i < 1000; i++) {
      dictFuzz.put("d" + i, NameFindFuzz.dictionarySentence(random));
    }
    try {
      return switch (kind) {
        case GENERATED -> generated(fuzz, dictFuzz);
        case DICTIONARY -> dictionaries(hand, dictFuzz);
        case REGEX_BUILTIN -> builtins(hand, fuzz);
        case REGEX_FACTORY -> factory(random);
        case REGEX_USER -> user(hand, fuzz);
        case MERGE -> {
          factory(random);
          yield merge(hand, fuzz, random);
        }
      };
    } catch (IOException e) {
      throw new IllegalStateException(kind.dir, e);
    }
  }

  private static List<Layer> generated(Map<String, String[]> fuzz, Map<String, String[]> dictFuzz) {
    Table t = new Table("turbo:sentences", "sentence:str", "token:str");
    for (Map<String, String[]> set : List.of(fuzz, dictFuzz)) {
      for (Map.Entry<String, String[]> e : set.entrySet()) {
        for (String token : e.getValue()) {
          t.row(e.getKey(), token);
        }
      }
    }
    return List.of(t.layer());
  }

  private List<Layer> dictionaries(Map<String, String[]> hand, Map<String, String[]> dictFuzz) throws IOException {
    Table names = names();
    for (int d = 0; d < DICTIONARIES.size(); d++) {
      String name = DICTIONARIES.get(d);
      TokenNameFinder finder = new DictionaryNameFinder(dictionary(files.get(d)), name);
      run(names, name, finder, hand);
      run(names, name, finder, dictFuzz);
    }
    return List.of(names.layer());
  }

  private static List<Layer> builtins(Map<String, String[]> hand, Map<String, String[]> fuzz) throws IOException {
    Table names = names();
    Table types = new Table("turbo:finder-types", "finder:str", "type:str");
    for (DEFAULT_REGEX_NAME_FINDER d : DEFAULT_REGEX_NAME_FINDER.values()) {
      RegexNameFinder finder = RegexNameFinderFactory.getDefaultRegexNameFinders(d);
      run(names, d.name(), finder, hand);
      run(names, d.name(), finder, fuzz);
    }
    RegexNameFinder all = RegexNameFinderFactory.getDefaultRegexNameFinders(DEFAULT_REGEX_NAME_FINDER.values());
    for (String t : types(all)) {
      types.row("ALL", t);
    }
    run(names, "ALL", all, hand);
    run(names, "ALL", all, fuzz);
    Map<String, Pattern[]> config = new LinkedHashMap<>();
    config.put("EMAIL", new Pattern[] {Pattern.compile("[a-z]+@[a-z]+")});
    config.put("WORD", new Pattern[] {Pattern.compile("[A-Z]\\w+"), Pattern.compile("\\d+")});
    RegexNameFinder configured = RegexNameFinderFactory.getDefaultRegexNameFinders(config,
        DEFAULT_REGEX_NAME_FINDER.values());
    for (String t : types(configured)) {
      types.row("CONFIG", t);
    }
    run(names, "CONFIG", configured, hand);
    return List.of(names.layer(), types.layer());
  }

  /** The type order of the factory's map, for combinations of defaults and configurations. */
  private static List<Layer> factory(Random random) throws IOException {
    Table cases = new Table("turbo:factory-cases", "case:u32", "has-defaults:u32");
    Table defaultsTable = new Table("turbo:factory-defaults", "case:u32", "default:str");
    Table keys = new Table("turbo:factory-keys", "case:u32", "key:str");
    Table types = new Table("turbo:factory-types", "case:u32", "type:str");
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
    int c = 0;
    for (DEFAULT_REGEX_NAME_FINDER[] d : defaults) {
      factoryCase(c++, d, Map.of(), RegexNameFinderFactory.getDefaultRegexNameFinders(d), cases, defaultsTable, keys,
          types);
    }
    String[] pool = {"PERSON", "EMAIL", "URL", "MGRS", "PHONE_NUM", "DATE", "a", "b", "Aa", "BB",
        "été", "😀", "x1", "x2", "x3", "ORG", "LOC"};
    for (int n = 0; n < 400; n++) {
      Map<String, Pattern[]> config = new LinkedHashMap<>();
      int size = random.nextInt(n < 200 ? 6 : 60);
      for (int k = 0; k < size; k++) {
        String key = random.nextInt(3) == 0 ? pool[random.nextInt(pool.length)]
            : NameFindFuzz.word(random, 1 + random.nextInt(6));
        config.put(key, new Pattern[] {Pattern.compile("x")});
      }
      DEFAULT_REGEX_NAME_FINDER[] d = switch (n % 4) {
        case 0 -> null;
        case 1 -> new DEFAULT_REGEX_NAME_FINDER[0];
        case 2 -> all;
        default -> defaults.get(random.nextInt(defaults.size()));
      };
      factoryCase(c++, d, config, RegexNameFinderFactory.getDefaultRegexNameFinders(config, d), cases, defaultsTable,
          keys, types);
    }
    return List.of(cases.layer(), defaultsTable.layer(), keys.layer(), types.layer());
  }

  private static void factoryCase(int c, DEFAULT_REGEX_NAME_FINDER[] defaults, Map<String, Pattern[]> config,
      RegexNameFinder finder, Table cases, Table defaultsTable, Table keys, Table types) throws IOException {
    cases.row(c, Table.flag(defaults != null));
    if (defaults != null) {
      for (DEFAULT_REGEX_NAME_FINDER d : defaults) {
        defaultsTable.row(c, d.name());
      }
    }
    for (String k : config.keySet()) {
      keys.row(c, k);
    }
    for (String t : types(finder)) {
      types.row(c, t);
    }
  }

  private List<Layer> user(Map<String, String[]> hand, Map<String, String[]> fuzz) throws IOException {
    Map<String, String[]> sentences = new LinkedHashMap<>(hand);
    int k = 0;
    for (Map.Entry<String, String[]> e : fuzz.entrySet()) {
      if (k++ == 400) {
        break;
      }
      sentences.put(e.getKey(), e.getValue());
    }
    // Dedicated tokens can contain line terminators, unlike the shared
    // one-sentence-per-line input. Only \n and \r escapes are decoded.
    String boundaries = new String(files.get(1).bytes(), java.nio.charset.StandardCharsets.UTF_8);
    int boundary = 0;
    for (String row : boundaries.split("\n")) {
      String[] tokens = row.split("\t", -1);
      for (int i = 0; i < tokens.length; i++) {
        tokens[i] = tokens[i].replace("\\n", "\n").replace("\\r", "\r");
      }
      sentences.put("b" + boundary++, tokens);
    }
    Table patterns = new Table("turbo:patterns", "pattern:u32", "flags:str", "regex:str", "compiled:u32");
    Table names = new Table("turbo:pattern-names", "pattern:u32", "sentence:str", "start:u32", "end:u32", "type:str");
    String text = new String(files.get(0).bytes(), java.nio.charset.StandardCharsets.UTF_8);
    List<String> rows = new ArrayList<>(Arrays.asList(text.split("\n")));
    for (int p = 0; p < rows.size(); p++) {
      String[] f = rows.get(p).split("\t", 3);
      int flags = 0;
      if (f[1].contains("i")) {
        flags |= Pattern.CASE_INSENSITIVE;
      }
      if (f[1].contains("s")) {
        flags |= Pattern.DOTALL;
      }
      Pattern compiled;
      try {
        compiled = Pattern.compile(f[2], flags);
      } catch (PatternSyntaxException e) {
        patterns.row(p, f[1], f[2], 0);
        continue;
      }
      patterns.row(p, f[1], f[2], 1);
      RegexNameFinder finder = new RegexNameFinder(new Pattern[] {compiled}, "T");
      for (Map.Entry<String, String[]> e : sentences.entrySet()) {
        for (Span s : finder.find(e.getValue())) {
          names.row(p, e.getKey(), s.getStart(), s.getEnd(), s.getType());
        }
      }
    }
    return List.of(patterns.layer(), names.layer());
  }

  private List<Layer> merge(Map<String, String[]> hand, Map<String, String[]> fuzz, Random random) throws IOException {
    String[] types = {"A", "B", "a", "é", "😀", "Ａ", ""};
    Table cases = new Table("turbo:merge-cases", "case:u32", "spans:u32");
    Table spans = new Table("turbo:merge-spans", "case:u32", "start:u32", "end:u32", "type:str");
    Table kept = new Table("turbo:merge-kept", "case:u32", "start:u32", "end:u32", "type:str");
    for (int n = 0; n < 3000; n++) {
      Span[] list = new Span[random.nextInt(13)];
      for (int k = 0; k < list.length; k++) {
        int start = random.nextInt(10);
        list[k] = new Span(start, start + random.nextInt(5), types[random.nextInt(types.length)]);
        spans.row(n, list[k].getStart(), list[k].getEnd(), list[k].getType());
      }
      cases.row(n, list.length);
      for (Span s : NameFinderME.dropOverlappingSpans(list)) {
        kept.row(n, s.getStart(), s.getEnd(), s.getType());
      }
    }
    List<TokenNameFinder> finders = new ArrayList<>();
    for (int d = 0; d < DICTIONARIES.size(); d++) {
      finders.add(new DictionaryNameFinder(dictionary(files.get(d)), DICTIONARIES.get(d)));
    }
    finders.add(RegexNameFinderFactory.getDefaultRegexNameFinders(DEFAULT_REGEX_NAME_FINDER.values()));
    Table merged = new Table("turbo:merged-names", "sentence:str", "start:u32", "end:u32", "type:str");
    for (Map<String, String[]> set : List.of(hand, fuzz)) {
      for (Map.Entry<String, String[]> e : set.entrySet()) {
        List<Span> names = new ArrayList<>();
        for (TokenNameFinder finder : finders) {
          names.addAll(Arrays.asList(finder.find(e.getValue())));
        }
        for (Span s : NameFinderME.dropOverlappingSpans(names.toArray(new Span[0]))) {
          merged.row(e.getKey(), s.getStart(), s.getEnd(), s.getType());
        }
      }
    }
    return List.of(cases.layer(), spans.layer(), kept.layer(), merged.layer());
  }

  private static Table names() {
    return new Table("turbo:names", "finder:str", "sentence:str", "start:u32", "end:u32", "type:str");
  }

  private static void run(Table names, String finderName, TokenNameFinder finder, Map<String, String[]> sentences) {
    for (Map.Entry<String, String[]> e : sentences.entrySet()) {
      for (Span s : finder.find(e.getValue())) {
        names.row(finderName, e.getKey(), s.getStart(), s.getEnd(), s.getType());
      }
    }
  }

  private static Dictionary dictionary(ModelFile file) throws IOException {
    try (InputStream in = file.stream()) {
      return new Dictionary(in);
    }
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
}
