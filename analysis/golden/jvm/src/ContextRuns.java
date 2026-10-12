import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.tokenize.TokenContextGenerator;
import opennlp.tools.tokenize.TokenizerFactory;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.tokenize.lang.Factory;
import opennlp.tools.util.Span;
import opennlp.tools.util.StringList;

/**
 * Writes the feature files: for each input, every context {@code DefaultTokenContextGenerator}
 * produced and the two probabilities the model returned.
 *
 * <p>The loop is the one in {@code TokenizerME.tokenizePos}, including the alphanumeric and
 * single code unit short cuts and the jump over the dots of an abbreviation, and it calls the same
 * context generator and model. It is replicated because {@code TokenizerME} does not expose the
 * contexts. The generator rebuilds the spans from the same decisions and reports a problem if they
 * differ from what {@code TokenizerME} returns for the input, so the two loops cannot drift apart
 * unnoticed.
 */
final class ContextRuns {

  /** The inputs the context files cover. */
  static final Set<String> INPUT_NAMES = Set.of("spanish", "french", "numbers", "urls-emails", "supplementary");

  private ContextRuns() {
  }

  static void write(Path contextDir, RunInfo run, List<Inputs.Input> inputs, ModelFile model) throws IOException {
    write(contextDir, model.language(), run, inputs, INPUT_NAMES, model, new TokenizerModel(model.path().toFile()));
  }

  /**
   * Writes the feature file {@code <name>.txt} for the inputs whose names are in {@code names}, using
   * {@code tokenizerModel} (which can be a model built from {@code model} with other settings).
   */
  static void write(Path contextDir, String name, RunInfo run, List<Inputs.Input> inputs, Set<String> names,
      ModelFile model, TokenizerModel tokenizerModel) throws IOException {
    TokenizerFactory factory = tokenizerModel.getFactory();
    MaxentModel maxent = tokenizerModel.getMaxentModel();
    TokenContextGenerator cg = factory.getContextGenerator();
    Dictionary abbreviations = tokenizerModel.getAbbreviations();
    Pattern alphaNumeric = factory.isUseAlphaNumericOptimization() ? effectivePattern(factory) : null;
    int splitIndex = maxent.getIndex(TokenizerME.SPLIT);
    int noSplitIndex = maxent.getIndex(TokenizerME.NO_SPLIT);
    TokenizerME reference = new TokenizerME(tokenizerModel);

    GoldenFile file = new GoldenFile("tokenizer-me-" + name + "-contexts", run, "");
    ModelHeader.write(file, model, tokenizerModel);
    file.line("outcomes: " + TokenizerME.SPLIT + " " + TokenizerME.NO_SPLIT);

    for (Inputs.Input in : inputs) {
      if (!names.contains(in.name())) {
        continue;
      }
      String text = in.text();
      file.line("input: " + TextEscape.escape(text));
      List<Span> rebuilt = new ArrayList<>();
      for (Span unit : WhitespaceTokenizer.INSTANCE.tokenizePos(text)) {
        String tok = text.substring(unit.getStart(), unit.getEnd());
        if (tok.length() < 2 || (alphaNumeric != null && alphaNumeric.matcher(tok).matches())) {
          rebuilt.add(unit);
          continue;
        }
        file.line("unit: " + unit.getStart() + " " + unit.getEnd());
        int start = unit.getStart();
        int end = unit.getEnd();
        for (int j = start + 1; j < end; j++) {
          int at = j - unit.getStart();
          String[] context = cg.getContext(tok, at);
          double[] probs = maxent.eval(context);
          file.line("at: " + at);
          file.line("features: " + String.join("\t", Arrays.stream(context).map(TextEscape::escape).toList()));
          file.line("probs: " + Double.toString(probs[splitIndex]) + " " + Double.toString(probs[noSplitIndex]));
          if (TokenizerME.SPLIT.equals(maxent.getBestOutcome(probs))) {
            if (isAbbreviation(abbreviations, tok)) {
              rebuilt.add(new Span(start, end));
              j = j + (int) tok.codePoints().filter(ch -> ch == '.').count();
              start = j + 1;
            } else {
              rebuilt.add(new Span(start, j));
              start = j;
            }
          }
        }
        if (start < end) {
          rebuilt.add(new Span(start, end));
        }
      }
      Span[] expected = reference.tokenizePos(text);
      if (!Arrays.equals(expected, rebuilt.toArray(new Span[0]))) {
        Golden.problem("contexts/" + name + "/" + in.name()
            + ": rebuilt spans differ from TokenizerME.tokenizePos");
      }
    }
    file.write(contextDir.resolve("tokenizer-me-" + name + ".txt"));
  }

  /**
   * The pattern {@code TokenizerME} uses when the optimization is on: the factory's pattern, and
   * {@code Factory.DEFAULT_ALPHANUMERIC} if there is none. {@code TokenizerFactory.getAlphaNumericPattern}
   * itself falls back to the pattern of the model's language, so the second case does not arise for a
   * model.
   */
  static Pattern effectivePattern(TokenizerFactory factory) {
    Pattern p = factory.getAlphaNumericPattern();
    return p == null ? Factory.DEFAULT_ALPHANUMERIC : p;
  }

  private static boolean isAbbreviation(Dictionary dictionary, String token) {
    return dictionary != null && dictionary.contains(new StringList(token));
  }
}
