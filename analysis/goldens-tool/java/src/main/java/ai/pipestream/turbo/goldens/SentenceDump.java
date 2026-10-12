package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.List;
import opennlp.tools.sentdetect.SentenceDetectorFactory;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;

/**
 * The sentence detector's decision points: every candidate end of sentence
 * {@link SentenceDetectorME#sentPosDetect(CharSequence)} asks its model about.
 *
 * <p>The detector skips candidates depending on the decisions before them, so the positions
 * cannot be listed from outside. The context generator is wrapped instead
 * ({@link RecordingSentenceDetectorFactory}): it reports each {@code getContext(text, position)}
 * call, and {@code sentPosDetect} passes that array straight to {@code eval}. A point's position
 * is that {@code position}, the UTF-16 index of the end of sentence character the scanner found,
 * converted to its UTF-8 byte offset.</p>
 */
final class SentenceDump implements AnnotatorDump {

  private final SentenceGolden golden;
  private final MaxentRecorder recorder;
  private final SentenceDetectorME detector;

  private SentenceDump(SentenceGolden golden, MaxentRecorder recorder, SentenceDetectorME detector) {
    this.golden = golden;
    this.recorder = recorder;
    this.detector = detector;
  }

  /**
   * Builds a sentence model with the loaded model's manifest entries and factory settings over a
   * recorder of its maxent model, through
   * {@link SentenceModel#SentenceModel(String, opennlp.tools.ml.model.MaxentModel, java.util.Map, SentenceDetectorFactory)}.
   */
  static SentenceDump of(SentenceGolden golden) {
    RecordingModels.allowFactories();
    SentenceModel loaded = golden.loaded();
    RecordingModels.checkFactory(loaded.getFactory(), SentenceDetectorFactory.class);
    MaxentRecorder recorder = MaxentRecorder.wrap(loaded.getMaxentModel());
    SentenceModel recording = new SentenceModel(loaded.getLanguage(), recorder, RecordingModels.entriesFor(loaded),
        new RecordingSentenceDetectorFactory(loaded.getFactory()));
    RecordingModels.checkManifest(loaded, recording, RecordingSentenceDetectorFactory.class);
    return new SentenceDump(golden, recorder, new SentenceDetectorME(recording));
  }

  @Override
  public AnnotatorGolden golden() {
    return golden;
  }

  @Override
  public Run run(Input input) throws AnnotatorException {
    List<Layer> layers;
    try {
      layers = SentenceGolden.layers(detector, input);
    } catch (AnnotatorException e) {
      // What the model was asked before the annotator threw belongs to no dump.
      recorder.discard();
      throw e;
    }
    List<DecisionPoint> points = new ArrayList<>();
    for (Evaluation e : recorder.take()) {
      ContextRequest r = e.request();
      if (r == null || r.text() != input.text() || e.values() != null) {
        throw new IllegalStateException("input " + input.inputId()
            + ": an eval call that is not a context request about the input text without values");
      }
      points.add(DecisionPoint.at(input.offsets(), r.index(), e));
    }
    return new Run(recorder.outcomeLabels(), points, layers);
  }
}
