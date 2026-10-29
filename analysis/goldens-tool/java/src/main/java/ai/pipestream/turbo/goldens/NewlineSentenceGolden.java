package ai.pipestream.turbo.goldens;

import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.sentdetect.NewlineSentenceDetector;
import opennlp.tools.util.Span;

/**
 * The golden of {@link NewlineSentenceDetector}, which has no model: {@code opennlp:sentences}
 * from {@code sentPosDetect}, then {@code turbo:utf16<opennlp:sentences>} when a sentence cuts a
 * surrogate pair ({@link SpanLayers}). The directory is {@code newline}.
 */
final class NewlineSentenceGolden implements AnnotatorGolden {

  private final NewlineSentenceDetector detector = new NewlineSentenceDetector();
  private final Set<String> inputSets;

  NewlineSentenceGolden(Set<String> inputSets) {
    this.inputSets = Set.copyOf(inputSets);
  }

  @Override
  public String annotator() {
    return "sentdetect";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return "newline";
  }

  @Override
  public String annotatorClass() {
    return NewlineSentenceDetector.class.getName();
  }

  @Override
  public String method() {
    return "sentPosDetect";
  }

  @Override
  public Map<String, String> config() {
    return Map.of();
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    Span[] spans = AnnotatorException.call(() -> detector.sentPosDetect(input.text()));
    return SpanLayers.of("opennlp:sentences", input.offsets(), spans);
  }
}
