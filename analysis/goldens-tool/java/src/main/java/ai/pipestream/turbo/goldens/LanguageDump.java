package ai.pipestream.turbo.goldens;

import java.util.List;
import opennlp.tools.langdetect.LanguageDetectorFactory;
import opennlp.tools.langdetect.LanguageDetectorME;
import opennlp.tools.langdetect.LanguageDetectorModel;

/**
 * The language detector's decision point: the one {@code eval} call
 * {@link LanguageDetectorME#predictLanguages(CharSequence)} makes for the whole document, at
 * position 0. No context generator is wrapped, as there is nothing to pair.
 *
 * <p>The call is {@code eval(String[], float[])}. Its contexts are the distinct n-grams the
 * context generator produced, in the iteration order of the {@code HashMap} that counted them,
 * and each value is 1 (the detector does not pass the counts, see OPENNLP-1261 in its source).
 * The dump holds them exactly as {@code eval} received them.</p>
 */
final class LanguageDump implements AnnotatorDump {

  private final LanguageGolden golden;
  private final MaxentRecorder recorder;
  private final LanguageDetectorME detector;

  private LanguageDump(LanguageGolden golden, MaxentRecorder recorder, LanguageDetectorME detector) {
    this.golden = golden;
    this.recorder = recorder;
    this.detector = detector;
  }

  /**
   * Builds a language detector model with the loaded model's manifest entries over a recorder of
   * its maxent model, through
   * {@link LanguageDetectorModel#LanguageDetectorModel(opennlp.tools.ml.model.MaxentModel, java.util.Map, LanguageDetectorFactory)}
   * with a plain {@link LanguageDetectorFactory}, the loaded model's factory class.
   */
  static LanguageDump of(LanguageGolden golden) {
    LanguageDetectorModel loaded = golden.loaded();
    RecordingModels.checkFactory(loaded.getFactory(), LanguageDetectorFactory.class);
    MaxentRecorder recorder = MaxentRecorder.wrap(loaded.getMaxentModel());
    LanguageDetectorModel recording = new LanguageDetectorModel(recorder, RecordingModels.entriesFor(loaded),
        new LanguageDetectorFactory());
    RecordingModels.checkManifest(loaded, recording, LanguageDetectorFactory.class);
    return new LanguageDump(golden, recorder, new LanguageDetectorME(recording));
  }

  @Override
  public AnnotatorGolden golden() {
    return golden;
  }

  @Override
  public Run run(Input input) throws AnnotatorException {
    List<Layer> layers;
    try {
      layers = LanguageGolden.layers(detector, input);
    } catch (AnnotatorException e) {
      // What the model was asked before the annotator threw belongs to no dump.
      recorder.discard();
      throw e;
    }
    List<Evaluation> evaluations = recorder.take();
    if (evaluations.size() != 1 || evaluations.get(0).request() != null) {
      throw new IllegalStateException("input " + input.inputId() + ": predictLanguages made "
          + evaluations.size() + " eval calls, one was expected");
    }
    return new Run(recorder.outcomeLabels(), List.of(DecisionPoint.document(evaluations.get(0))), layers);
  }
}
