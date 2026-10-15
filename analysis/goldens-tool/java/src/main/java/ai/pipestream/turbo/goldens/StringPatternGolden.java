package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarFile;

/**
 * What {@code StringPattern.recognize} and {@code FeatureGeneratorUtil.tokenFeature} return for
 * tokens, in two runs: {@code utf16-units}, the classes of the OpenNLP commit the image is built
 * from, which look at a token one UTF-16 unit at a time, and {@code code-points}, the same two
 * classes of the fork's {@code ready/string-pattern-unicode} branch, which look at code points.
 * The branch's classes are compiled into a jar of their own in the image (the container's
 * Dockerfile) and loaded here in a class loader that looks in that jar before the generator's
 * class path, so the two versions of the same class names run side by side.
 *
 * <p>The input set {@code sequence}: on {@code tokens} the tokens are the input's lines,
 * unescaped ({@link EscapedLines}; the file holds lone surrogates that way), then every code point
 * of a few ranges of scripts, digits and symbols, the four surrogate bounds alone, and a capital
 * letter followed by a period for Latin, Greek and mathematical capitals; on {@code sentences}
 * they are the tokens of the sentences. A token is recorded once, the first time it comes.</p>
 *
 * <p>The layer {@code turbo:string-patterns}, anchored to the document, has one row per token:
 * the token ({@link TrainGolden#lossless}, as it can be half a surrogate pair), each flag of the
 * pattern as 0 or 1, {@code digits()} and the class {@code tokenFeature} returns.</p>
 */
final class StringPatternGolden implements AnnotatorGolden {

  /** The input set of the sequence goldens: the sentences and the pattern tokens. */
  static final String SET = Runs.SEQUENCE;

  /** The system property naming the branch's jar, which the image's entrypoint sets. */
  static final String JAR_PROPERTY = "turbo.goldens.stringPatternJar";

  /** The branch commit the jar must be built from, which the image writes beside it. */
  static final String BRANCH_COMMIT = "b2473ca80d65ae2ef10a24b08b1d0ae6a9231109";

  /** The classes the jar must hold, which this loader takes from it first. */
  private static final Set<String> BRANCH_CLASSES = Set.of("opennlp.tools.util.featuregen.StringPattern",
      "opennlp.tools.util.featuregen.FeatureGeneratorUtil", "opennlp.tools.util.ParamChecks");

  private static final String[] FLAGS = {"isAllLetter", "isInitialCapitalLetter", "isAllCapitalLetter",
      "isAllLowerCaseLetter", "isAllDigit", "isAllHiragana", "isAllKatakana", "containsPeriod", "containsComma",
      "containsSlash", "containsDigit", "containsHyphen", "containsLetters"};

  /** The flag columns, in the order of {@link #FLAGS}. */
  private static final String[] FLAG_COLUMNS = {"all-letter", "initial-capital", "all-capital", "all-lower-case",
      "all-digit", "all-hiragana", "all-katakana", "period", "comma", "slash", "digit", "hyphen", "letters"};

  /** The code point ranges whose every character is a token of {@code tokens}. */
  private static final int[][] RANGES = {
      {0x20, 0x2FF}, {0x370, 0x3FF}, {0x400, 0x45F}, {0x590, 0x5FF}, {0x600, 0x6FF},
      {0x900, 0x97F}, {0x1E00, 0x1EFF}, {0x2000, 0x206F}, {0x2150, 0x218F}, {0x2460, 0x24FF},
      {0x3000, 0x30FF}, {0x4E00, 0x4E0F}, {0xAC00, 0xAC0F}, {0xFF00, 0xFF65},
      {0x10400, 0x1044F}, {0x1D400, 0x1D433}, {0x1F600, 0x1F60F}, {0x20000, 0x20007}
  };

  /** The capitals each followed by a period, for the {@code cp} class. */
  private static final int[][] CAPITALS = {{0x41, 0x5A}, {0xC0, 0xDE}, {0x391, 0x3A9}, {0x1D400, 0x1D433}};

  private final String mode;
  private final Method recognize;
  private final Method tokenFeature;
  private final Method digits;
  private final Method[] flags = new Method[FLAGS.length];

  private StringPatternGolden(String mode, ClassLoader loader) throws ReflectiveOperationException {
    this.mode = mode;
    Class<?> pattern = Class.forName("opennlp.tools.util.featuregen.StringPattern", true, loader);
    Class<?> util = Class.forName("opennlp.tools.util.featuregen.FeatureGeneratorUtil", true, loader);
    if (pattern.getClassLoader() != loader && loader != StringPatternGolden.class.getClassLoader()) {
      throw new IllegalStateException("StringPattern was not loaded from the branch's jar");
    }
    recognize = pattern.getMethod("recognize", String.class);
    tokenFeature = util.getMethod("tokenFeature", String.class);
    digits = pattern.getMethod("digits");
    for (int i = 0; i < FLAGS.length; i++) {
      flags[i] = pattern.getMethod(FLAGS[i]);
    }
  }

  /**
   * Both runs: the generator's own classes, and the branch's from the jar the system property
   * names, which must hold the three classes and have the branch commit written beside it.
   */
  static List<StringPatternGolden> all() throws IOException {
    String jar = System.getProperty(JAR_PROPERTY);
    if (jar == null) {
      throw new IllegalStateException("-D" + JAR_PROPERTY + " must name the jar of the code point StringPattern");
    }
    Path path = Path.of(jar);
    String commit = Files.readString(path.resolveSibling("commit"), StandardCharsets.UTF_8).strip();
    if (!commit.equals(BRANCH_COMMIT)) {
      throw new IllegalStateException(path + " was built from " + commit + ", not " + BRANCH_COMMIT);
    }
    try (JarFile file = new JarFile(path.toFile())) {
      for (String c : BRANCH_CLASSES) {
        if (file.getEntry(c.replace('.', '/') + ".class") == null) {
          throw new IllegalStateException(path + " does not hold " + c);
        }
      }
    }
    try {
      return List.of(new StringPatternGolden("utf16-units", StringPatternGolden.class.getClassLoader()),
          new StringPatternGolden("code-points", new BranchFirst(path.toUri().toURL())));
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("StringPattern", e);
    }
  }

  /** Loads the branch's classes from its jar before asking the generator's class loader. */
  private static final class BranchFirst extends URLClassLoader {
    BranchFirst(URL jar) {
      super(new URL[] {jar}, StringPatternGolden.class.getClassLoader());
    }

    @Override
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
      if (!BRANCH_CLASSES.contains(name)) {
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
    return "string-pattern";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return mode;
  }

  @Override
  public String annotatorClass() {
    return "opennlp.tools.util.featuregen.StringPattern";
  }

  @Override
  public String method() {
    return "recognize";
  }

  @Override
  public Map<String, String> config() {
    return mode.equals("code-points") ? Map.of("mode", mode, "branchCommit", BRANCH_COMMIT) : Map.of("mode", mode);
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(SET) && (input.id().equals("tokens") || input.id().equals("sentences"));
  }

  @Override
  public List<Layer> layers(Input input) {
    String[] columns = new String[FLAG_COLUMNS.length + 3];
    columns[0] = "token:str";
    for (int i = 0; i < FLAG_COLUMNS.length; i++) {
      columns[i + 1] = FLAG_COLUMNS[i] + ":u32";
    }
    columns[columns.length - 2] = "digits:u32";
    columns[columns.length - 1] = "class:str";
    Table t = new Table("turbo:string-patterns", columns);
    for (String token : tokens(input)) {
      Object[] row = new Object[columns.length];
      try {
        Object p = recognize.invoke(null, token);
        row[0] = TrainGolden.lossless(token);
        for (int i = 0; i < flags.length; i++) {
          row[i + 1] = Table.flag((Boolean) flags[i].invoke(p));
        }
        row[row.length - 2] = (long) (Integer) digits.invoke(p);
        row[row.length - 1] = (String) tokenFeature.invoke(null, token);
      } catch (IllegalAccessException | InvocationTargetException e) {
        throw new IllegalStateException(input.inputId() + ": StringPattern on " + TrainGolden.lossless(token), e);
      }
      t.row(row);
    }
    return List.of(t.layer());
  }

  /** The tokens of the input, in the order they first come. */
  static Set<String> tokens(Input input) {
    Set<String> tokens = new LinkedHashSet<>();
    if (input.id().equals("sentences")) {
      for (String[] sentence : Sentences.of(input)) {
        tokens.addAll(List.of(sentence));
      }
      return tokens;
    }
    tokens.addAll(EscapedLines.read(input));
    for (int[] range : RANGES) {
      for (int cp = range[0]; cp <= range[1]; cp++) {
        tokens.add(new String(Character.toChars(cp)));
      }
    }
    for (int cp : new int[] {0xD800, 0xDBFF, 0xDC00, 0xDFFF}) {
      tokens.add(String.valueOf((char) cp));
    }
    List<String> capitals = new ArrayList<>();
    for (int[] range : CAPITALS) {
      for (int cp = range[0]; cp <= range[1]; cp++) {
        capitals.add(new String(Character.toChars(cp)));
      }
    }
    for (String c : capitals) {
      tokens.add(c + ".");
    }
    return tokens;
  }
}
