package ai.pipestream.turbo.goldens;

import opennlp.tools.sentdetect.SDContextGenerator;
import opennlp.tools.sentdetect.SentenceDetectorFactory;
import opennlp.tools.sentdetect.SentenceModel;

/**
 * A {@link SentenceDetectorFactory} whose context generator is the one the plain factory makes,
 * wrapped so that each request ({@code getContext(text, position)}) is reported to the model's
 * {@link MaxentRecorder} together with the array it returned. Everything else is the plain
 * factory's.
 *
 * <p>{@code SentenceModel} records this class's name in its manifest and makes its own instance
 * through {@code ExtensionLoader}, which needs a public class with a public no-argument
 * constructor in a package registered with {@link RecordingModels#allowFactories()}. That
 * instance reads its settings from the model as the plain factory does and finds the recorder
 * as the model's maxent model.</p>
 */
public final class RecordingSentenceDetectorFactory extends SentenceDetectorFactory {

  /** The constructor {@code ExtensionLoader} calls. */
  public RecordingSentenceDetectorFactory() {}

  /** A factory with the settings of a loaded model's factory, to build a recording model from. */
  RecordingSentenceDetectorFactory(SentenceDetectorFactory loaded) {
    super(loaded.getLanguageCode(), loaded.isUseTokenEnd(), loaded.getAbbreviationDictionary(),
        loaded.getEOSCharacters());
  }

  @Override
  public SDContextGenerator getSDContextGenerator() {
    SDContextGenerator plain = super.getSDContextGenerator();
    if (!(artifactProvider instanceof SentenceModel m && m.getMaxentModel() instanceof MaxentRecorder recorder)) {
      throw new IllegalStateException("a recording sentence factory needs a sentence model whose maxent model"
          + " is a MaxentRecorder");
    }
    return (text, position) -> {
      String[] context = plain.getContext(text, position);
      recorder.contextRequested(new ContextRequest(text, position), context);
      return context;
    };
  }
}
