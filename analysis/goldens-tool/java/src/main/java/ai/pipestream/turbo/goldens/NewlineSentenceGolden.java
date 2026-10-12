package ai.pipestream.turbo.goldens;

import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.sentdetect.NewlineSentenceDetector;
import opennlp.tools.util.Span;
import opennlp.tools.util.WhitespaceMode;

/**
 * The golden of {@link NewlineSentenceDetector}, which has no model: {@code opennlp:sentences}
 * from {@code sentPosDetect}, then {@code turbo:utf16<opennlp:sentences>} when a sentence cuts a
 * surrogate pair ({@link SpanLayers}). It runs in one whitespace mode; the directory is
 * {@code newline}, or {@code newline+legacy} with the {@code config} record
 * {@code whitespace legacy}.
 */
final class NewlineSentenceGolden implements AnnotatorGolden {

  private final NewlineSentenceDetector detector = new NewlineSentenceDetector();
  private final WhitespaceMode whitespace;
  private final Set<String> inputSets;

  NewlineSentenceGolden(WhitespaceMode whitespace, Set<String> inputSets) {
    this.whitespace = whitespace;
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
    return "newline" + Whitespace.dirSuffix(whitespace);
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
    return Whitespace.config(whitespace);
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    Span[] spans = Whitespace.in(whitespace, () -> AnnotatorException.call(() -> detector.sentPosDetect(input.text())));
    return SpanLayers.of("opennlp:sentences", input.offsets(), spans);
  }
}
