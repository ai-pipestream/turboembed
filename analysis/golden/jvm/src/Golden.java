import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import opennlp.tools.util.WhitespaceMode;

/**
 * Writes the golden files the Rust ports are tested against.
 *
 * <p>Usage: {@code Golden <opennlp-commit> <inputs-dir> <models-dir> <analysis-dir>}. The program
 * reads every {@code .txt} file of the inputs directory, runs each component and writes under
 * {@code <analysis-dir>/tokenize/tests/golden} and {@code <analysis-dir>/sentdetect/tests/golden}.
 * Every model file ({@code .bin}) in the models directory is used by its manifest: a
 * {@code TokenizerME} model gets tokenizer and context files, a {@code SentenceDetectorME} model
 * gets sentence files. Running it again with more models adds their files.
 */
public final class Golden {

  private static final List<String> PROBLEMS = new ArrayList<>();

  private Golden() {
  }

  public static void main(String[] args) throws IOException {
    if (args.length != 4) {
      System.err.println("usage: Golden <opennlp-commit> <inputs-dir> <models-dir> <analysis-dir>");
      System.exit(2);
    }
    String commit = args[0];
    List<Inputs.Input> inputs = Inputs.load(Path.of(args[1]));
    List<ModelFile> models = loadModels(Path.of(args[2]));
    Path tokenizeGolden = Path.of(args[3], "tokenize", "tests", "golden");
    Path sentGolden = Path.of(args[3], "sentdetect", "tests", "golden");

    RunInfo unicode = new RunInfo(commit, System.getProperty("java.version"), "unicode");
    RunInfo legacy = unicode.withWhitespace("legacy");

    // The default mode first, then the legacy mode for the components whose output depends on it.
    WhitespaceMode.setActive(WhitespaceMode.UNICODE);
    TokenizerRuns.writePlain(tokenizeGolden, unicode, inputs, true);
    SentenceRuns.writeNewline(sentGolden, unicode, inputs);
    for (ModelFile model : models) {
      switch (model.component()) {
        case "TokenizerME" -> {
          TokenizerRuns.writeMaxent(tokenizeGolden, unicode, inputs, model);
          ContextRuns.write(tokenizeGolden.resolve("contexts"), unicode, inputs, model);
        }
        case "SentenceDetectorME" -> SentenceRuns.writeMaxent(sentGolden, unicode, inputs, model);
        default -> problem(model.fileName() + ": unknown component '" + model.component() + "', skipped");
      }
    }

    WhitespaceMode.setActive(WhitespaceMode.LEGACY);
    try {
      TokenizerRuns.writePlain(tokenizeGolden, legacy, inputs, false);
      SentenceRuns.writeNewline(sentGolden, legacy, inputs);
    } finally {
      WhitespaceMode.setActive(WhitespaceMode.UNICODE);
    }

    System.out.println(inputs.size() + " inputs, " + models.size() + " models, " + PROBLEMS.size() + " problems");
    for (String p : PROBLEMS) {
      System.out.println("problem: " + p);
    }
  }

  /** The directory name for a component: legacy runs get {@code -legacy} appended. */
  static String componentDir(String component, RunInfo run) {
    return run.whitespace().equals("legacy") ? component + "-legacy" : component;
  }

  /** Records something that looked wrong; printed at the end. */
  static void problem(String text) {
    PROBLEMS.add(text);
  }

  private static List<ModelFile> loadModels(Path dir) throws IOException {
    List<ModelFile> models = new ArrayList<>();
    try (Stream<Path> s = Files.list(dir)) {
      for (Path p : s.filter(f -> f.getFileName().toString().endsWith(".bin")).sorted().toList()) {
        models.add(ModelFile.open(p));
      }
    }
    return models;
  }
}
