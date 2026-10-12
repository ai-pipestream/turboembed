package ai.pipestream.turbo.goldens;

import opennlp.tools.tokenize.TokenContextGenerator;
import opennlp.tools.tokenize.TokenizerFactory;
import opennlp.tools.tokenize.TokenizerModel;

/**
 * A {@link TokenizerFactory} whose context generator is the one the plain factory makes, wrapped
 * so that each request ({@code getContext(token, index)}, where {@code token} is one whitespace
 * separated token and {@code index} a UTF-16 index into it) is reported to the model's
 * {@link MaxentRecorder} together with the array it returned. Everything else is the plain
 * factory's.
 *
 * <p>{@code TokenizerModel} makes its own instance of this class through {@code ExtensionLoader},
 * as {@link RecordingSentenceDetectorFactory} describes.</p>
 */
public final class RecordingTokenizerFactory extends TokenizerFactory {

  /** The constructor {@code ExtensionLoader} calls. */
  public RecordingTokenizerFactory() {}

  /** A factory with the settings of a loaded model's factory, to build a recording model from. */
  RecordingTokenizerFactory(TokenizerFactory loaded) {
    super(loaded.getLanguageCode(), loaded.getAbbreviationDictionary(), loaded.isUseAlphaNumericOptimization(),
        loaded.getAlphaNumericPattern());
  }

  @Override
  public TokenContextGenerator getContextGenerator() {
    TokenContextGenerator plain = super.getContextGenerator();
    if (!(artifactProvider instanceof TokenizerModel m && m.getMaxentModel() instanceof MaxentRecorder recorder)) {
      throw new IllegalStateException("a recording tokenizer factory needs a tokenizer model whose maxent model"
          + " is a MaxentRecorder");
    }
    return (token, index) -> {
      String[] context = plain.getContext(token, index);
      recorder.contextRequested(new ContextRequest(token, index), context);
      return context;
    };
  }
}
