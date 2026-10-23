package ai.pipestream.turbo.goldens;

import ai.onnxruntime.OrtException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.dl.vectors.SentenceVectorsDL;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.util.Span;

/**
 * OpenNLP's {@link SentenceVectorsDL} on each sentence the English sentence model finds: the
 * sentences as {@code opennlp:sentences}, then each one's vector as {@code
 * turbo:vector<opennlp:sentences>}, the 32-bit pattern of every float in hex, space separated.
 *
 * <p>The ONNX model is a mean-pooled copy of the upstream all-MiniLM-L6-v2 export (made by
 * {@code container/mean_pool_onnx.py}): {@link SentenceVectorsDL#embed(CharSequence)} returns the
 * first row of the graph's first output, which in that copy is the mean of the last hidden states
 * over the sentence's tokens, [CLS] and [SEP] included.
 */
final class SentenceVectorsGolden implements AnnotatorGolden {

  private final ModelRef onnx;
  private final ModelRef vocabulary;
  private final ModelRef sentences;
  private final SentenceDetectorME detector;
  private final SentenceVectorsDL vectors;
  private final Set<String> inputSets;

  private SentenceVectorsGolden(ModelRef onnx, ModelRef vocabulary, ModelRef sentences, SentenceDetectorME detector,
      SentenceVectorsDL vectors, Set<String> inputSets) {
    this.onnx = onnx;
    this.vocabulary = vocabulary;
    this.sentences = sentences;
    this.detector = detector;
    this.vectors = vectors;
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Loads the pooled ONNX model, the vocabulary and the sentence model, hashing each file. */
  static SentenceVectorsGolden load(Path onnx, Path vocabulary, Path sentenceModel, Set<String> inputSets)
      throws IOException {
    ModelFile o = ModelFile.read(onnx);
    ModelFile v = ModelFile.read(vocabulary);
    ModelFile s = ModelFile.read(sentenceModel);
    try {
      return new SentenceVectorsGolden(o.ref(), v.ref(), s.ref(), new SentenceDetectorME(new SentenceModel(s.stream())),
          new SentenceVectorsDL(onnx.toFile(), vocabulary.toFile()), inputSets);
    } catch (OrtException e) {
      throw new IOException(onnx + ": " + e.getMessage(), e);
    }
  }

  @Override
  public String annotator() {
    return "sentvectors";
  }

  @Override
  public List<ModelRef> models() {
    return List.of(onnx, vocabulary, sentences);
  }

  /** The ONNX file's name without {@code .onnx}. */
  @Override
  public String modelDir() {
    String name = onnx.fileName();
    return name.endsWith(".onnx") ? name.substring(0, name.length() - ".onnx".length()) : name;
  }

  @Override
  public String annotatorClass() {
    return SentenceVectorsDL.class.getName();
  }

  @Override
  public String method() {
    return "embed";
  }

  @Override
  public Map<String, String> config() {
    return Map.of("lowerCase", "true");
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    Span[] spans = AnnotatorException.call(() -> detector.sentPosDetect(input.text()));
    String[] hex = new String[spans.length];
    for (int i = 0; i < spans.length; i++) {
      String sentence = input.text().substring(spans[i].getStart(), spans[i].getEnd());
      float[] v = AnnotatorException.call(() -> vectors.embed(sentence));
      StringBuilder b = new StringBuilder(v.length * 9);
      for (float x : v) {
        if (!b.isEmpty()) {
          b.append(' ');
        }
        b.append(String.format("%08x", Float.floatToRawIntBits(x)));
      }
      hex[i] = b.toString();
    }
    List<Layer> layers = new ArrayList<>(SpanLayers.of("opennlp:sentences", input.offsets(), spans));
    layers.add(new Layer("turbo:vector<opennlp:sentences>", "aligned:opennlp:sentences", new Column.Str("f32", hex)));
    return layers;
  }
}
