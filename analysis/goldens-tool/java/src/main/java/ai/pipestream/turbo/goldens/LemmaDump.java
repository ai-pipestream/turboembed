package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import opennlp.tools.lemmatizer.LemmatizerFactory;
import opennlp.tools.lemmatizer.LemmatizerME;
import opennlp.tools.lemmatizer.LemmatizerModel;

/**
 * The maxent lemmatizer's decision points: every context {@code BeamSearch} asks the model about
 * while {@link LemmatizerME#lemmatize(String[], String[])} lemmatizes a sentence, one evaluation
 * per token per path of the beam, in the order the search made them. The context generator is
 * wrapped ({@link RecordingLemmatizerFactory}) as {@link PosDump} describes; the point's position
 * is the start of the token the request was about.
 */
final class LemmaDump implements AnnotatorDump {

  private final LemmaGolden golden;
  private final MaxentRecorder recorder;
  private final LemmatizerME lemmatizer;

  private LemmaDump(LemmaGolden golden, MaxentRecorder recorder, LemmatizerME lemmatizer) {
    this.golden = golden;
    this.recorder = recorder;
    this.lemmatizer = lemmatizer;
  }

  /**
   * Builds a lemmatizer model with the loaded model's beam size and manifest entries over a
   * recorder of its classifier, through
   * {@link LemmatizerModel#LemmatizerModel(String, opennlp.tools.ml.model.MaxentModel, int, Map, LemmatizerFactory)},
   * and the lemmatizer over it.
   */
  static LemmaDump of(LemmaGolden golden) {
    RecordingModels.allowFactories();
    LemmatizerModel loaded = golden.loaded();
    RecordingModels.checkFactory(loaded.getFactory(), LemmatizerFactory.class);
    MaxentRecorder recorder = MaxentRecorder.wrap(LemmaGolden.classifier(loaded));
    LemmatizerModel recording = new LemmatizerModel(loaded.getLanguage(), recorder, LemmaGolden.beamSize(loaded),
        LemmaGolden.entriesFor(loaded), new RecordingLemmatizerFactory());
    RecordingModels.checkManifest(loaded, recording, RecordingLemmatizerFactory.class);
    return new LemmaDump(golden, recorder, new LemmatizerME(recording));
  }

  @Override
  public AnnotatorGolden golden() {
    return golden;
  }

  @Override
  public Run run(Input input) throws AnnotatorException {
    LemmaGolden.Run run;
    try {
      run = LemmaGolden.run(lemmatizer, golden.pipeline(), input, true);
    } catch (AnnotatorException e) {
      recorder.discard();
      throw e;
    }
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
