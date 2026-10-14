package ai.pipestream.turbo.goldens;

import java.util.Map;
import opennlp.tools.postag.POSContextGenerator;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerFactory;

/**
 * A {@link POSTaggerFactory} whose context generator is the one the plain factory makes, wrapped
 * so that each request ({@code getContext(index, tokens, tags, additionalContext)}) is reported
 * to the model's {@link MaxentRecorder} together with the array it returned. The request's text
 * is {@code tokens[index]}, the very string object the tagger was given, so the dump can tell by
 * identity which token of which sentence the decision was about. Everything else is the plain
 * factory's: the descriptor and the tag dictionary come from the recording model, which carries
 * the loaded model's.
 *
 * <p>{@code POSModel} makes its own instance of this class through {@code ExtensionLoader}, as
 * {@link RecordingSentenceDetectorFactory} describes.</p>
 */
public final class RecordingPOSTaggerFactory extends POSTaggerFactory {

  /** The constructor {@code ExtensionLoader} calls. */
  public RecordingPOSTaggerFactory() {}

  /**
   * A factory with the loaded model's descriptor and tag dictionary, to build a recording model
   * from. The resources map is empty: the published models' descriptors name no resource.
   */
  RecordingPOSTaggerFactory(POSModel loaded) {
    super(PosGolden.descriptor(loaded), Map.of(), loaded.getFactory().getTagDictionary());
  }

  @Override
  public POSContextGenerator getPOSContextGenerator(int cacheSize) {
    POSContextGenerator plain = super.getPOSContextGenerator(cacheSize);
    if (!(artifactProvider instanceof POSModel m
        && m.getArtifact(PosGolden.CLASSIFIER_ENTRY) instanceof MaxentRecorder recorder)) {
      throw new IllegalStateException("a recording part-of-speech factory needs a part-of-speech model whose"
          + " classifier is a MaxentRecorder");
    }
    return (index, tokens, tags, additionalContext) -> {
      String[] context = plain.getContext(index, tokens, tags, additionalContext);
      recorder.contextRequested(new ContextRequest(tokens[index], index), context);
      return context;
    };
  }
}
