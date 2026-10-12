import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Function;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.tokenize.uax29.WordToken;
import opennlp.tools.tokenize.uax29.WordTokenizer;
import opennlp.tools.util.Span;

/** Writes the tokenizer golden files: SimpleTokenizer, WhitespaceTokenizer, the UAX #29 tokenizer and TokenizerME. */
final class TokenizerRuns {

  private TokenizerRuns() {
  }

  /** The components that carry no model, in the order they are written. */
  static void writePlain(Path goldenDir, RunInfo run, List<Inputs.Input> inputs, boolean includeUax29) throws IOException {
    plain(goldenDir, run, inputs, "simple", t -> SimpleTokenizer.INSTANCE.tokenizePos(t));
    plain(goldenDir, run, inputs, "simple-newlines", TokenizerRuns::simpleKeepingNewLines);
    plain(goldenDir, run, inputs, "whitespace", t -> WhitespaceTokenizer.INSTANCE.tokenizePos(t));
    plain(goldenDir, run, inputs, "whitespace-newlines", t -> WhitespaceTokenizer.newInstance(true).tokenizePos(t));
    if (includeUax29) {
      uax29(goldenDir, run, inputs);
    }
  }

  /** SimpleTokenizer has only a shared instance, so the flag is set for one call and restored. */
  private static Span[] simpleKeepingNewLines(String text) {
    SimpleTokenizer.INSTANCE.setKeepNewLines(true);
    try {
      return SimpleTokenizer.INSTANCE.tokenizePos(text);
    } finally {
      SimpleTokenizer.INSTANCE.setKeepNewLines(false);
    }
  }

  private static void plain(Path goldenDir, RunInfo run, List<Inputs.Input> inputs, String component,
      Function<String, Span[]> tokenize) throws IOException {
    for (Inputs.Input in : inputs) {
      GoldenFile file = new GoldenFile(component, run, in.text()).input();
      try {
        for (Span s : tokenize.apply(in.text())) {
          file.span(s, null, null);
        }
      } catch (RuntimeException e) {
        file.error(e);
        Golden.problem(component + "/" + in.name() + ": " + e);
      }
      file.write(goldenDir.resolve(Golden.componentDir(component, run)).resolve(in.name() + ".txt"));
    }
  }

  private static void uax29(Path goldenDir, RunInfo run, List<Inputs.Input> inputs) throws IOException {
    WordTokenizer tokenizer = new WordTokenizer();
    for (Inputs.Input in : inputs) {
      GoldenFile file = new GoldenFile("uax29", run, in.text()).input();
      try {
        for (WordToken t : tokenizer.tokenizeTyped(in.text())) {
          file.span(t.span(), null, t.type().name());
        }
      } catch (RuntimeException e) {
        file.error(e);
        Golden.problem("uax29/" + in.name() + ": " + e);
      }
      file.write(goldenDir.resolve("uax29").resolve(in.name() + ".txt"));
    }
  }

  /** One file per input for TokenizerME over the whole input, with the probability of every token. */
  static void writeMaxent(Path goldenDir, RunInfo run, List<Inputs.Input> inputs, ModelFile model) throws IOException {
    TokenizerModel tokenizerModel = new TokenizerModel(model.path().toFile());
    TokenizerME tokenizer = new TokenizerME(tokenizerModel);
    String component = "tokenizer-me-" + model.language();
    for (Inputs.Input in : inputs) {
      GoldenFile file = new GoldenFile(component, run, in.text());
      ModelHeader.write(file, model, tokenizerModel);
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
        Golden.problem(component + "/" + in.name() + ": " + e);
      }
      file.write(goldenDir.resolve(component).resolve(in.name() + ".txt"));
    }
  }
}
