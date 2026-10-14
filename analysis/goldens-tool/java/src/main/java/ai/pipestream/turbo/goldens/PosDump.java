package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerFactory;
import opennlp.tools.postag.POSTaggerME;

/**
 * The part-of-speech tagger's decision points: every context {@code BeamSearch} asks the model
 * about while {@link POSTaggerME#tag(String[])} tags a sentence, one evaluation per token per
 * path of the beam, in the order the search made them.
 *
 * <p>The search asks for a token several times, once per partial sequence it extends, with
 * different previous tags each time, so the decisions cannot be listed from outside. The context
 * generator is wrapped instead ({@link RecordingPOSTaggerFactory}): it reports each
 * {@code getContext(index, tokens, tags, additionalContext)} call with the array it returned,
 * which the search passes straight to {@code eval}. The request's text is {@code tokens[index]},
 * the string object of that token; the run gives each sentence's token array as the tagger
 * received it, so the token is found by identity and the point's position is that token's start
 * in the input. The previous tags are not in the dump as such: the context strings carry them,
 * as the model saw them.</p>
 */
final class PosDump implements AnnotatorDump {

  private final PosGolden golden;
  private final MaxentRecorder recorder;
  private final POSTaggerME tagger;

  private PosDump(PosGolden golden, MaxentRecorder recorder, POSTaggerME tagger) {
    this.golden = golden;
    this.recorder = recorder;
    this.tagger = tagger;
  }

  /**
   * Builds a part-of-speech model with the loaded model's beam size, manifest entries, descriptor
   * and tag dictionary over a recorder of its classifier, through
   * {@link POSModel#POSModel(String, opennlp.tools.ml.model.MaxentModel, int, Map, POSTaggerFactory)},
   * and the tagger over it asked for the golden's format.
   */
  static PosDump of(PosGolden golden) {
    RecordingModels.allowFactories();
    POSModel loaded = golden.loaded();
    RecordingModels.checkFactory(loaded.getFactory(), POSTaggerFactory.class);
    MaxentRecorder recorder = MaxentRecorder.wrap(PosGolden.classifier(loaded));
    POSModel recording = new POSModel(loaded.getLanguage(), recorder, PosGolden.beamSize(loaded),
        RecordingModels.entriesFor(loaded), new RecordingPOSTaggerFactory(loaded));
    RecordingModels.checkManifest(loaded, recording, RecordingPOSTaggerFactory.class);
    return new PosDump(golden, recorder, new POSTaggerME(recording, golden.format()));
  }

  @Override
  public AnnotatorGolden golden() {
    return golden;
  }

  @Override
  public Run run(Input input) throws AnnotatorException {
    PosGolden.Run run;
    try {
      run = PosGolden.run(tagger, golden.pipeline(), input);
    } catch (AnnotatorException e) {
      // What the model was asked before the annotator threw belongs to no dump.
      recorder.discard();
      throw e;
    }
    // Each token object to its index in its sentence and its UTF-16 start in the input.
    Map<String, int[]> tokens = new IdentityHashMap<>();
    for (int s = 0; s < run.sentenceTokens().size(); s++) {
      String[] words = run.sentenceTokens().get(s);
      int[] starts = run.tokenStarts().get(s);
      for (int i = 0; i < words.length; i++) {
        if (tokens.put(words[i], new int[] {i, starts[i]}) != null) {
          throw new IllegalStateException(input.inputId() + ": two tokens are one string object");
        }
      }
    }
    List<DecisionPoint> points = new ArrayList<>();
    for (Evaluation e : recorder.take()) {
      ContextRequest r = e.request();
      if (r == null || e.values() != null) {
        throw new IllegalStateException(input.inputId()
            + ": an eval call that is not a context request without values");
      }
      int[] token = tokens.get(r.text());
      if (token == null || token[0] != r.index()) {
        throw new IllegalStateException(input.inputId() + ": a context request about index " + r.index()
            + " is not about a token of its sentence");
      }
      points.add(DecisionPoint.at(input.offsets(), token[1], e));
    }
    return new Run(recorder.outcomeLabels(), points, run.layers());
  }
}
