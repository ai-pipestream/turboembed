package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.util.Span;
import opennlp.tools.util.WhitespaceMode;

/**
 * The sentence detector's golden: {@code opennlp:sentences} from
 * {@link SentenceDetectorME#sentPosDetect(CharSequence)}, then {@code turbo:utf16<opennlp:sentences>}
 * when a sentence starts or ends inside a surrogate pair ({@link SpanLayers}), then its confidence
 * layer with each returned span's probability. The probabilities are checked bit for bit against
 * {@code getSentenceProbabilities()}, which must report the same values.
 *
 * <p>The detector runs in one whitespace mode ({@link Whitespace}); a run in the legacy mode has
 * the {@code config} record {@code whitespace legacy} and the directory {@code <model>+legacy}.</p>
 */
final class SentenceGolden implements AnnotatorGolden {

  private final ModelRef model;
  private final SentenceModel loaded;
  private final SentenceDetectorME detector;
  private final WhitespaceMode whitespace;
  private final Set<String> inputSets;

  private SentenceGolden(ModelRef model, SentenceModel loaded, WhitespaceMode whitespace, Set<String> inputSets) {
    this.model = model;
    this.loaded = loaded;
    this.detector = new SentenceDetectorME(loaded);
    this.whitespace = whitespace;
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Loads the sentence model from {@code file}, hashing exactly the bytes loaded. */
  static SentenceGolden load(Path file, WhitespaceMode whitespace, Set<String> inputSets) throws IOException {
    ModelFile f = ModelFile.read(file);
    return new SentenceGolden(f.ref(), new SentenceModel(f.stream()), whitespace, inputSets);
  }

  /** The loaded model, which {@link SentenceDump} builds its recording model from. */
  SentenceModel loaded() {
    return loaded;
  }

  /** The whitespace mode the detector runs in. */
  WhitespaceMode whitespace() {
    return whitespace;
  }

  @Override
  public String annotator() {
    return "sentdetect";
  }

  @Override
  public List<ModelRef> models() {
    return List.of(model);
  }

  @Override
  public String modelDir() {
    return model.baseName() + Whitespace.dirSuffix(whitespace);
  }

  @Override
  public String annotatorClass() {
    return SentenceDetectorME.class.getName();
  }

  @Override
  public String method() {
    return "sentPosDetect";
  }

  @Override
  public Map<String, String> config() {
    return Whitespace.config(whitespace);
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    return layers(detector, whitespace, input);
  }

  /**
   * The layers {@code detector} produces for the input in {@code whitespace}; {@link SentenceDump}
   * runs it on its recording detector.
   */
  @SuppressWarnings("removal")
  static List<Layer> layers(SentenceDetectorME detector, WhitespaceMode whitespace, Input input)
      throws AnnotatorException {
    Span[] spans = Whitespace.in(whitespace, () -> AnnotatorException.call(() -> detector.sentPosDetect(input.text())));
    double[] reported = detector.getSentenceProbabilities();
    double[] probs = new double[spans.length];
    for (int i = 0; i < spans.length; i++) {
      probs[i] = spans[i].getProb();
    }
    Probabilities.checkSame(input.inputId(), "getSentenceProbabilities", probs, reported);
    List<Layer> layers = new ArrayList<>(SpanLayers.of("opennlp:sentences", input.offsets(), spans));
    layers.add(new Layer("turbo:confidence<opennlp:sentences>", "aligned:opennlp:sentences",
        new Column.F64("value", probs)));
    return layers;
  }
}
