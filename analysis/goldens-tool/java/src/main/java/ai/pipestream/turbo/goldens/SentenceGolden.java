package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.util.Span;

/**
 * The sentence detector's golden: {@code opennlp:sentences} from
 * {@link SentenceDetectorME#sentPosDetect(CharSequence)}, then its confidence layer with each
 * returned span's probability. The probabilities are checked bit for bit against
 * {@code getSentenceProbabilities()}, which must report the same values.
 */
final class SentenceGolden implements AnnotatorGolden {

  private final ModelRef model;
  private final SentenceDetectorME detector;

  private SentenceGolden(ModelRef model, SentenceDetectorME detector) {
    this.model = model;
    this.detector = detector;
  }

  /** Loads the sentence model from {@code file}, hashing exactly the bytes loaded. */
  static SentenceGolden load(Path file) throws IOException {
    byte[] bytes = Files.readAllBytes(file);
    SentenceModel m = new SentenceModel(new ByteArrayInputStream(bytes));
    return new SentenceGolden(new ModelRef(file.getFileName().toString(), Sha.sha512(bytes)),
        new SentenceDetectorME(m));
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
  public String annotatorClass() {
    return SentenceDetectorME.class.getName();
  }

  @Override
  public String method() {
    return "sentPosDetect";
  }

  @Override
  @SuppressWarnings("removal")
  public List<Layer> layers(Input input) {
    Span[] spans = detector.sentPosDetect(input.text());
    double[] reported = detector.getSentenceProbabilities();
    long[] starts = new long[spans.length];
    long[] ends = new long[spans.length];
    double[] probs = new double[spans.length];
    for (int i = 0; i < spans.length; i++) {
      Utf8Offsets.ByteSpan b = input.offsets().span(spans[i].getStart(), spans[i].getEnd());
      starts[i] = b.start();
      ends[i] = b.end();
      probs[i] = spans[i].getProb();
    }
    Probabilities.checkSame(input.inputId(), "getSentenceProbabilities", probs, reported);
    return List.of(
        new Layer("opennlp:sentences", "text", new Column.U32("start", starts), new Column.U32("end", ends)),
        new Layer("turbo:confidence<opennlp:sentences>", "aligned:opennlp:sentences",
            new Column.F64("value", probs)));
  }
}
