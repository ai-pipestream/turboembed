package ai.pipestream.turbo.goldens;

import opennlp.tools.lemmatizer.LemmatizerContextGenerator;
import opennlp.tools.lemmatizer.LemmatizerFactory;
import opennlp.tools.lemmatizer.LemmatizerModel;

/**
 * A {@link LemmatizerFactory} whose context generator is the one the plain factory makes, wrapped
 * so that each request the beam search makes ({@code getContext(index, tokens, lemmas,
 * additionalContext)}) is reported to the model's {@link MaxentRecorder} together with the array
 * it returned. The request's text is {@code tokens[index]}, the very string object the lemmatizer
 * was given, so the dump can tell by identity which token of which sentence the decision was
 * about. The sequence validator is the plain factory's, which accepts every class.
 *
 * <p>{@code LemmatizerModel} makes its own instance of this class through {@code ExtensionLoader},
 * as {@link RecordingSentenceDetectorFactory} describes.</p>
 */
public final class RecordingLemmatizerFactory extends LemmatizerFactory {

  /** The constructor {@code ExtensionLoader} calls. */
  public RecordingLemmatizerFactory() {}

  @Override
  public LemmatizerContextGenerator getContextGenerator() {
    LemmatizerContextGenerator plain = super.getContextGenerator();
    if (!(artifactProvider instanceof LemmatizerModel m
        && m.getArtifact(LemmaGolden.CLASSIFIER_ENTRY) instanceof MaxentRecorder recorder)) {
      throw new IllegalStateException("a recording lemmatizer factory needs a lemmatizer model whose classifier"
          + " is a MaxentRecorder");
    }
    return new LemmatizerContextGenerator() {
      @Override
      public String[] getContext(int index, String[] tokens, String[] lemmas, Object[] additionalContext) {
        String[] context = plain.getContext(index, tokens, lemmas, additionalContext);
        recorder.contextRequested(new ContextRequest(tokens[index], index), context);
        return context;
      }

      @Override
      public String[] getContext(int index, String[] tokens, String[] tags, String[] lemmas) {
        String[] context = plain.getContext(index, tokens, tags, lemmas);
        recorder.contextRequested(new ContextRequest(tokens[index], index), context);
        return context;
      }
    };
  }
}
