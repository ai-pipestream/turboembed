package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.jar.JarFile;

/**
 * What {@code HunspellStemmer} returns for words, over the three published dictionaries the
 * recipe downloads into {@code <models>/hunspell} (LibreOffice's {@code en_US}, {@code
 * de_DE_frami} and {@code hu_HU}) and over the small hand-written dictionaries of {@code
 * <fixtures>/hunspell}.
 *
 * <p>The Hunspell classes are those of the fork's {@code OPENNLP-1927-hunspell-compatibility}
 * branch, which is later than the commit the image is built from. The container compiles them,
 * with {@code HunspellBridge} (which calls the package private {@code analyze}), into a jar of
 * their own; they are loaded here in a class loader that looks in that jar first for every class
 * of their package, so the commit's own Hunspell classes are never used.</p>
 *
 * <p>A run is one dictionary, loaded once from its bytes with {@code HunspellDictionary.load} on
 * two streams, strictly or, for a {@code +partial} run, with {@code LoadMode.ALLOW_PARTIAL}. It
 * reads {@code hunspell/edge} and {@code hunspell/<run>}, and the published dictionaries also
 * the sample of the Snowball vocabulary of their language in the {@code stem} set. An input holds
 * one word per line, escaped as {@link EscapedLines} reads it; a line starting with {@code #} is a
 * comment.</p>
 *
 * <p>Layers, all anchored to the document: {@code turbo:hunspell-load}, one row, the class and
 * message of the exception loading threw, empty when it loaded; {@code
 * turbo:hunspell-unsupported}, the directives a partial load skipped and their lines; {@code
 * turbo:hunspell-words}, one row per word (none when loading failed): the word, {@code stem} and
 * the class of a runtime exception stemming threw; {@code turbo:hunspell-stems}, every string
 * {@code stemAll} returned, by the row of its word; and {@code turbo:hunspell-analyses}, every
 * string {@code analyze} returned, by the row of its word.</p>
 */
final class HunspellGolden implements AnnotatorGolden {

  /** The input set of the hand-written words. */
  static final String SET = Runs.HUNSPELL;

  /** The input every run reads. */
  static final String EDGE = "edge";

  /** The system property naming the jar of the branch's classes, which the image's entrypoint sets. */
  static final String JAR_PROPERTY = "turbo.goldens.hunspellJar";

  /** The branch commit the jar must be built from, which the image writes beside it. */
  static final String BRANCH_COMMIT = "fb299cf32ccf6f5dbc81b39448d2d7bea1cf66dc";

  /** The package whose classes the branch loader takes from the jar. */
  private static final String PACKAGE = "opennlp.tools.stemmer.hunspell.";

  /** The bridge class, which the jar must hold. */
  private static final String BRIDGE = PACKAGE + "HunspellBridge";

  /** The directory below the models and the fixtures that holds the dictionaries. */
  static final String DIRECTORY = "hunspell";

  /** The published dictionaries and the Snowball sample of their language in the stem set. */
  private static final Map<String, String> PUBLISHED = Map.of("en_US", "english", "de_DE_frami", "german",
      "hu_HU", "hungarian");

  /** The prefix of a fixture whose run is also made with a partial load. */
  private static final String UNSUPPORTED_PREFIX = "unsupported";

  private final String run;
  private final String stemSample;
  private final List<ModelRef> models;
  private final Bridge bridge;
  private final Object dictionary;
  private final String loadThrown;
  private final String loadMessage;

  private HunspellGolden(String run, String stemSample, ModelFile affix, ModelFile words, boolean partial,
      Bridge bridge) {
    this.run = run;
    this.stemSample = stemSample;
    this.models = List.of(affix.ref(), words.ref());
    this.bridge = bridge;
    Object loaded = null;
    String thrown = "";
    String message = "";
    try {
      loaded = bridge.load(affix.bytes(), words.bytes(), partial);
    } catch (IOException e) {
      thrown = e.getClass().getName();
      message = e.getMessage();
    }
    this.dictionary = loaded;
    this.loadThrown = thrown;
    this.loadMessage = message;
  }

  /**
   * Every run: the published dictionaries in {@code models}, and each pair of {@code .aff} and
   * {@code .dic} files in {@code fixtures/hunspell}, a fixture whose name starts with {@code
   * unsupported} once strictly and once partially.
   */
  static List<HunspellGolden> all(Path models, Path fixtures) throws IOException {
    Bridge bridge = Bridge.open();
    List<HunspellGolden> out = new ArrayList<>();
    for (Map.Entry<String, String> e : new TreeMap<>(PUBLISHED).entrySet()) {
      Path dir = models.resolve(DIRECTORY);
      out.add(new HunspellGolden(e.getKey(), e.getValue(), ModelFile.read(dir.resolve(e.getKey() + ".aff")),
          ModelFile.read(dir.resolve(e.getKey() + ".dic")), false, bridge));
    }
    List<Path> affixes = new ArrayList<>();
    try (DirectoryStream<Path> stream = Files.newDirectoryStream(fixtures.resolve(DIRECTORY), "*.aff")) {
      for (Path p : stream) {
        affixes.add(p);
      }
    }
    affixes.sort(null);
    for (Path affix : affixes) {
      String file = affix.getFileName().toString();
      String name = "fixture-" + file.substring(0, file.length() - ".aff".length());
      ModelFile aff = ModelFile.read(affix);
      ModelFile dic = ModelFile.read(affix.resolveSibling(file.replace(".aff", ".dic")));
      out.add(new HunspellGolden(name, null, aff, dic, false, bridge));
      if (file.startsWith(UNSUPPORTED_PREFIX)) {
        out.add(new HunspellGolden(name + "+partial", null, aff, dic, true, bridge));
      }
    }
    return out;
  }

  /** The bridge's methods, from the branch's jar. */
  private record Bridge(Method load, Method unsupported, Method stem, Method stemAll, Method analyze) {

    Object load(byte[] affix, byte[] words, boolean partial) throws IOException {
      try {
        return load.invoke(null, affix, words, partial);
      } catch (InvocationTargetException e) {
        if (e.getCause() instanceof IOException io) {
          throw io;
        }
        throw new IllegalStateException("HunspellDictionary.load", e.getCause());
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(e);
      }
    }

    Object call(Method m, Object... args) {
      try {
        return m.invoke(null, args);
      } catch (InvocationTargetException e) {
        if (e.getCause() instanceof RuntimeException r) {
          throw r;
        }
        throw new IllegalStateException(m.getName(), e.getCause());
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(e);
      }
    }

    /** Checks the jar the system property names and loads the bridge from it. */
    static Bridge open() throws IOException {
      String jar = System.getProperty(JAR_PROPERTY);
      if (jar == null) {
        throw new IllegalStateException("-D" + JAR_PROPERTY + " must name the jar of the Hunspell classes");
      }
      Path path = Path.of(jar);
      String commit = Files.readString(path.resolveSibling("commit"), StandardCharsets.UTF_8).strip();
      if (!commit.equals(BRANCH_COMMIT)) {
        throw new IllegalStateException(path + " was built from " + commit + ", not " + BRANCH_COMMIT);
      }
      try (JarFile file = new JarFile(path.toFile())) {
        for (String c : List.of(BRIDGE, PACKAGE + "HunspellStemmer", PACKAGE + "HunspellDictionary",
            PACKAGE + "HunspellAffixTable", PACKAGE + "HunspellCompoundRule", PACKAGE + "HunspellConversion",
            PACKAGE + "AffixCondition")) {
          if (file.getEntry(c.replace('.', '/') + ".class") == null) {
            throw new IllegalStateException(path + " does not hold " + c);
          }
        }
      }
      try {
        Class<?> bridge = Class.forName(BRIDGE, true, new BranchFirst(path.toUri().toURL()));
        Class<?> stemmer = Class.forName(PACKAGE + "HunspellStemmer", true, bridge.getClassLoader());
        if (stemmer.getClassLoader() != bridge.getClassLoader()) {
          throw new IllegalStateException("HunspellStemmer was not loaded from the branch's jar");
        }
        return new Bridge(bridge.getMethod("load", byte[].class, byte[].class, boolean.class),
            bridge.getMethod("unsupported", Object.class), bridge.getMethod("stem", Object.class, String.class),
            bridge.getMethod("stemAll", Object.class, String.class),
            bridge.getMethod("analyze", Object.class, String.class));
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException("HunspellBridge", e);
      }
    }
  }

  /** Loads the classes of the Hunspell package from the branch's jar before asking the generator's loader. */
  private static final class BranchFirst extends URLClassLoader {
    BranchFirst(URL jar) {
      super(new URL[] {jar}, HunspellGolden.class.getClassLoader());
    }

    @Override
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
      if (!name.startsWith(PACKAGE) || name.indexOf('.', PACKAGE.length()) >= 0) {
        return super.loadClass(name, resolve);
      }
      synchronized (getClassLoadingLock(name)) {
        Class<?> c = findLoadedClass(name);
        if (c == null) {
          c = findClass(name);
        }
        if (resolve) {
          resolveClass(c);
        }
        return c;
      }
    }
  }

  @Override
  public String annotator() {
    return "hunspell";
  }

  @Override
  public List<ModelRef> models() {
    return models;
  }

  @Override
  public String modelDir() {
    return run;
  }

  @Override
  public String annotatorClass() {
    return "opennlp.tools.stemmer.hunspell.HunspellStemmer";
  }

  @Override
  public String method() {
    return "stemAll";
  }

  @Override
  public Map<String, String> config() {
    return Map.of("mode", run.endsWith("+partial") ? "ALLOW_PARTIAL" : "STRICT", "branchCommit", BRANCH_COMMIT);
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET, Runs.STEM);
  }

  @Override
  public boolean accepts(Input input) {
    if (input.set().equals(Runs.STEM)) {
      return input.id().equals(stemSample);
    }
    String own = run.endsWith("+partial") ? run.substring(0, run.length() - "+partial".length()) : run;
    return input.set().equals(SET) && (input.id().equals(EDGE) || input.id().equals(own));
  }

  @Override
  public List<Layer> layers(Input input) {
    Table load = new Table("turbo:hunspell-load", "thrown:str", "message:str");
    load.row(loadThrown, loadMessage);
    Table unsupported = new Table("turbo:hunspell-unsupported", "directive:str", "line:u32");
    Table words = new Table("turbo:hunspell-words", "word:str", "stem:str", "thrown:str");
    Table stems = new Table("turbo:hunspell-stems", "word:u32", "stem:str");
    Table analyses = new Table("turbo:hunspell-analyses", "word:u32", "analysis:str");
    if (dictionary != null) {
      for (String s : castStrings(bridge.call(bridge.unsupported(), dictionary))) {
        int tab = s.indexOf('\t');
        unsupported.row(s.substring(0, tab), Long.parseLong(s.substring(tab + 1)));
      }
      for (String line : input.lines()) {
        if (line.startsWith("#")) {
          continue;
        }
        String word = EscapedLines.unescape(line);
        int row = words.size();
        String stem;
        String thrown = "";
        List<String> all = List.of();
        List<String> found = List.of();
        try {
          stem = (String) bridge.call(bridge.stem(), dictionary, word);
          all = castStrings(bridge.call(bridge.stemAll(), dictionary, word));
          found = castStrings(bridge.call(bridge.analyze(), dictionary, word));
        } catch (RuntimeException e) {
          stem = "";
          thrown = e.getClass().getName();
          all = List.of();
          found = List.of();
        }
        for (String s : all) {
          stems.row(row, TrainGolden.lossless(s));
        }
        for (String a : found) {
          analyses.row(row, TrainGolden.lossless(a));
        }
        words.row(TrainGolden.lossless(word), TrainGolden.lossless(stem), thrown);
      }
    }
    return List.of(load.layer(), unsupported.layer(), words.layer(), stems.layer(), analyses.layer());
  }

  /** A list the bridge returned, whose elements are strings. */
  private static List<String> castStrings(Object list) {
    List<String> out = new ArrayList<>();
    for (Object o : (List<?>) list) {
      out.add((String) o);
    }
    return out;
  }
}
