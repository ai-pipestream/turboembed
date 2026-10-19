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
 *
 * <p>When the model's best outcome is a split, the detector asks {@code isAcceptableBreak}
 * whether the abbreviation dictionary allows it, and splits only if it does. The recording
 * detector is a subclass that records each answer ({@link Detector}), and the point carries it.
 * Every such call must be about the candidate just evaluated, and each point whose best outcome
 * is a split must have exactly one; this is checked.</p>
 */
final class SentenceDump implements AnnotatorDump {

  /** A detector that records what {@code isAcceptableBreak} returned, for which candidate. */
  static final class Detector extends SentenceDetectorME {

    /** One call: the text and the candidate index it was about, and the answer. */
    record Break(CharSequence text, int candidate, boolean acceptable) {}

    private final List<Break> breaks = new ArrayList<>();

    Detector(SentenceModel model) {
      super(model);
    }

    @Override
    protected boolean isAcceptableBreak(CharSequence s, int fromIndex, int candidateIndex) {
      boolean ok = super.isAcceptableBreak(s, fromIndex, candidateIndex);
      breaks.add(new Break(s, candidateIndex, ok));
      return ok;
    }

    /** Returns the calls recorded since the last call and forgets them. */
    List<Break> take() {
      List<Break> out = List.copyOf(breaks);
      breaks.clear();
      return out;
    }
  }

  private final SentenceGolden golden;
  private final MaxentRecorder recorder;
  private final Detector detector;

  private SentenceDump(SentenceGolden golden, MaxentRecorder recorder, Detector detector) {
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
    return new SentenceDump(golden, recorder, new Detector(recording));
  }

  @Override
  public AnnotatorGolden golden() {
    return golden;
  }

  @Override
  public Run run(Input input) throws AnnotatorException {
    List<Layer> layers;
    try {
      layers = SentenceGolden.layers(detector, golden.whitespace(), input);
    } catch (AnnotatorException e) {
      // What the model was asked before the annotator threw belongs to no dump.
      recorder.discard();
      detector.take();
      throw e;
    }
    List<Detector.Break> breaks = detector.take();
    int nextBreak = 0;
    List<DecisionPoint> points = new ArrayList<>();
    for (Evaluation e : recorder.take()) {
      ContextRequest r = e.request();
      if (r == null || r.text() != input.text() || e.values() != null) {
        throw new IllegalStateException("input " + input.inputId()
            + ": an eval call that is not a context request about the input text without values");
      }
      DecisionPoint point = DecisionPoint.at(input.offsets(), r.index(), e);
      if (recorder.getBestOutcome(e.outcomes()).equals(SentenceDetectorME.SPLIT)) {
        if (nextBreak == breaks.size()) {
          throw new IllegalStateException("input " + input.inputId() + ": no isAcceptableBreak call for the split at "
              + r.index());
        }
        Detector.Break b = breaks.get(nextBreak++);
        if (b.text() != input.text() || b.candidate() != r.index()) {
          throw new IllegalStateException("input " + input.inputId() + ": isAcceptableBreak was asked about "
              + b.candidate() + " after the split at " + r.index());
        }
        point = point.withAcceptable(b.acceptable());
      }
      points.add(point);
    }
    if (nextBreak != breaks.size()) {
      throw new IllegalStateException("input " + input.inputId() + ": " + (breaks.size() - nextBreak)
          + " isAcceptableBreak calls without a split before them");
    }
    return new Run(recorder.outcomeLabels(), points, layers);
  }
}
