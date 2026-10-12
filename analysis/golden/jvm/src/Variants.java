import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.tokenize.DetokenizationDictionary;
import opennlp.tools.tokenize.Detokenizer;
import opennlp.tools.tokenize.DictionaryDetokenizer;
import opennlp.tools.tokenize.TokenizerFactory;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.Span;

/**
 * Writes the golden files for the cases the plain {@link Golden} run does not reach: a
 * {@code TokenizerME} with an abbreviation dictionary, one that splits between the halves of a
 * surrogate pair, one with the alphanumeric optimization on, and the dictionary detokenizer.
 *
 * <p>Usage: {@code Variants <opennlp-commit> <variants-dir> <inputs-dir> <models-dir> <analysis-dir>}.
 * {@code <variants-dir>} is {@code analysis/golden/jvm/variants}. Output goes under
 * {@code <analysis-dir>/tokenize/tests/golden/variants} and {@code .../golden/detokenizer}. The
 * models directory holds the five UD tokenizer models ({@code Golden} finds them by manifest).
 */
public final class Variants {

  private static final String SPLIT_MARKER = "<SPLIT>";

  /** Inputs of the abbreviation runs for which every context is written. */
  private static final Set<String> ABBREVIATION_CONTEXTS =
      Set.of("fixture-sentence", "known-abbreviations", "dot-count-jump", "abbreviation-units");

  private Variants() {
  }

  public static void main(String[] args) throws IOException {
    if (args.length != 5) {
      System.err.println("usage: Variants <opennlp-commit> <variants-dir> <inputs-dir> <models-dir> <analysis-dir>");
      System.exit(2);
    }
    Path variants = Path.of(args[1]);
    List<Inputs.Input> inputs = Inputs.load(Path.of(args[2]));
    List<ModelFile> models = new ArrayList<>();
    try (Stream<Path> s = Files.list(Path.of(args[3]))) {
      for (Path p : s.filter(f -> f.getFileName().toString().endsWith(".bin")).sorted().toList()) {
        ModelFile m = ModelFile.open(p);
        if (m.component().equals("TokenizerME")) {
          models.add(m);
        }
      }
    }
    Path golden = Path.of(args[4], "tokenize", "tests", "golden");
    Path variantsOut = golden.resolve("variants");
    RunInfo run = new RunInfo(args[0], System.getProperty("java.version"), "unicode");

    abbreviations(variantsOut, run, variants, inputs, models);
    surrogates(variantsOut, run, Inputs.load(variants.resolve("surrogate")), models);
    alphaNumeric(variantsOut, run, inputs, Inputs.load(variants.resolve("alphanumeric")), models);
    detokenizer(golden.resolve("detokenizer"), run, variants.resolve("detokenizer"));

    System.out.println(inputs.size() + " inputs, " + models.size() + " models, " + Golden.problemCount() + " problems");
    Golden.printProblems();
  }

  // ---- TokenizerME with an abbreviation dictionary ----

  private static void abbreviations(Path out, RunInfo run, Path variants, List<Inputs.Input> inputs,
      List<ModelFile> models) throws IOException {
    Path dir = variants.resolve("abbreviations");
    List<Inputs.Input> all = new ArrayList<>(Inputs.load(dir));
    all.addAll(inputs);
    Set<String> contextInputs = ABBREVIATION_CONTEXTS;

    // The model OpenNLP trained on six sentences, with its own case-insensitive dictionary.
    ModelFile fixture = ModelFile.open(dir.resolve("tokenizer-abbreviations.bin"));
    TokenizerModel fixtureModel = new TokenizerModel(fixture.path().toFile());
    writeTokenizer(out.resolve("abbreviations-fixture"), "tokenizer-me-abbreviations-fixture", run, all, fixture,
        fixtureModel, "abbreviations: in the model file");
    ContextRuns.write(out.resolve("contexts"), "abbreviations-fixture", run, all, contextInputs, fixture,
        fixtureModel);

    // The English UD model with the dictionary of english-dictionary.xml added.
    ModelFile english = byLanguage(models, "en");
    Path xml = dir.resolve("english-dictionary.xml");
    Dictionary dictionary;
    try (InputStream in = Files.newInputStream(xml)) {
      dictionary = new Dictionary(in);
    }
    TokenizerModel base = new TokenizerModel(english.path().toFile());
    TokenizerModel englishModel = new TokenizerModel(base.getMaxentModel(), new HashMap<>(),
        new TokenizerFactory(base.getLanguage(), dictionary, base.getFactory().isUseAlphaNumericOptimization(),
            base.getFactory().getAlphaNumericPattern()));
    String note = "abbreviations: " + xml.getFileName() + " sha256 " + sha256(xml);
    writeTokenizer(out.resolve("abbreviations-en"), "tokenizer-me-abbreviations-en", run, all, english,
        englishModel, note);
    ContextRuns.write(out.resolve("contexts"), "abbreviations-en", run, all, contextInputs, english, englishModel);
  }

  // ---- split between the halves of a surrogate pair ----

  private static void surrogates(Path out, RunInfo run, List<Inputs.Input> inputs, List<ModelFile> models)
      throws IOException {
    for (ModelFile model : models) {
      TokenizerModel tm = new TokenizerModel(model.path().toFile());
      writeTokenizer(out.resolve("surrogate-" + model.language()), "tokenizer-me-" + model.language(), run, inputs,
          model, tm, null);
    }
  }

  // ---- alphanumeric optimization on ----

  private static void alphaNumeric(Path out, RunInfo run, List<Inputs.Input> inputs, List<Inputs.Input> extra,
      List<ModelFile> models) throws IOException {
    List<Inputs.Input> all = new ArrayList<>(inputs);
    all.addAll(extra);
    for (ModelFile model : models) {
      TokenizerModel base = new TokenizerModel(model.path().toFile());
      TokenizerFactory f = base.getFactory();
      TokenizerModel on = new TokenizerModel(base.getMaxentModel(), new HashMap<>(),
          new TokenizerFactory(base.getLanguage(), base.getAbbreviations(), true, f.getAlphaNumericPattern()));
      if (!on.getFactory().isUseAlphaNumericOptimization()) {
        throw new IllegalStateException("the optimization did not turn on for " + model.fileName());
      }
      String note = "effective-alpha-numeric-pattern: "
          + TextEscape.escape(ContextRuns.effectivePattern(on.getFactory()).pattern());
      writeTokenizer(out.resolve("alphanumeric-" + model.language()), "tokenizer-me-" + model.language(), run, all,
          model, on, note);
      Set<String> names = new java.util.HashSet<>(ContextRuns.INPUT_NAMES);
      names.add("mixed");
      ContextRuns.write(out.resolve("contexts"), "alphanumeric-" + model.language(), run, all, names, model, on);
    }
  }

  /**
   * One file per input for a TokenizerME built from {@code tokenizerModel}. {@code model} is the
   * file the maxent model came from. {@code note} is one more header line ({@code key: value}), or null.
   */
  private static void writeTokenizer(Path dir, String component, RunInfo run, List<Inputs.Input> inputs,
      ModelFile model, TokenizerModel tokenizerModel, String note) throws IOException {
    TokenizerME tokenizer = new TokenizerME(tokenizerModel);
    for (Inputs.Input in : inputs) {
      GoldenFile file = new GoldenFile(component, run, in.text());
      ModelHeader.write(file, model, tokenizerModel);
      if (note != null) {
        file.line(note);
      }
      file.input();
      try {
        Span[] spans = tokenizer.tokenizePos(in.text());
        double[] probs = tokenizer.probs();
        if (spans.length != probs.length) {
          throw new IllegalStateException(spans.length + " spans but " + probs.length + " probabilities");
        }
        for (int i = 0; i < spans.length; i++) {
          file.span(spans[i], probs[i], null);
        }
      } catch (RuntimeException e) {
        file.error(e);
        Golden.problem(dir.getFileName() + "/" + in.name() + ": " + e);
      }
      file.write(dir.resolve(in.name() + ".txt"));
    }
  }

  // ---- DictionaryDetokenizer ----

  /**
   * For every file in {@code detokenizer/cases} (one token per line; the part of the name before the
   * first {@code -} names the dictionary {@code <name>-detokenizer.xml}) writes the operation of each
   * token and the text with and without the split marker.
   */
  private static void detokenizer(Path out, RunInfo run, Path dir) throws IOException {
    List<Path> cases;
    try (Stream<Path> s = Files.list(dir.resolve("cases"))) {
      cases = s.filter(p -> p.getFileName().toString().endsWith(".txt")).sorted().toList();
    }
    for (Path casePath : cases) {
      String name = casePath.getFileName().toString().replaceAll("\\.txt$", "");
      Path xml = dir.resolve("dictionaries").resolve(name.substring(0, name.indexOf('-')) + "-detokenizer.xml");
      DetokenizationDictionary dictionary;
      try (InputStream in = Files.newInputStream(xml)) {
        dictionary = new DetokenizationDictionary(in);
      }
      Detokenizer detokenizer = new DictionaryDetokenizer(dictionary);
      String content = new String(Files.readAllBytes(casePath), StandardCharsets.UTF_8);
      String[] tokens = content.isEmpty() ? new String[0] : content.replaceAll("\n$", "").split("\n", -1);

      GoldenFile file = new GoldenFile("detokenizer", run, "");
      file.header("dictionary", xml.getFileName().toString());
      file.header("dictionary-sha256", sha256(xml));
      file.header("split-marker", SPLIT_MARKER);
      file.line("tokens: " + tokens.length);
      try {
        Detokenizer.DetokenizationOperation[] ops = detokenizer.detokenize(tokens);
        for (int i = 0; i < tokens.length; i++) {
          file.line("token: " + TextEscape.escape(tokens[i]) + " " + ops[i].name());
        }
        file.line("text: " + TextEscape.escape(detokenizer.detokenize(tokens, null)));
        file.line("text-marked: " + TextEscape.escape(detokenizer.detokenize(tokens, SPLIT_MARKER)));
      } catch (RuntimeException e) {
        file.error(e);
        Golden.problem("detokenizer/" + name + ": " + e);
      }
      file.write(out.resolve(name + ".txt"));
    }
  }

  private static ModelFile byLanguage(List<ModelFile> models, String language) {
    return models.stream().filter(m -> m.language().equals(language)).findFirst()
        .orElseThrow(() -> new IllegalStateException("no TokenizerME model for language " + language));
  }

  private static String sha256(Path path) throws IOException {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)));
    } catch (java.security.NoSuchAlgorithmException e) {
      throw new IllegalStateException(e);
    }
  }
}
