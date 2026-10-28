package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.tokenize.lattice.UnigramSegmenter;
import opennlp.tools.util.Span;

/**
 * The golden of OpenNLP's {@link UnigramSegmenter}: the words of a frequency lexicon that give the
 * highest sum of log probabilities over each whitespace-free stretch of the text, an unlisted
 * character being a word of its own.
 *
 * <p>The lexicon is a UTF-8 file of one word, its count and an optional tag per line; its file is
 * the golden's {@code model} record and names the directory. {@code StringUtil.isWhitespace}
 * decides both where a lexicon line's word ends and where the text is split.</p>
 *
 * <p>The layer is {@code opennlp:tokens}, one row per word. Every word is whole code points (the
 * step over an unlisted character takes a whole code point, and the lexicon words are whole), so
 * no row cuts a character.</p>
 */
final class UnigramGolden implements AnnotatorGolden {

  private final ModelRef lexicon;
  private final UnigramSegmenter segmenter;
  private final Set<String> inputSets;

  private UnigramGolden(ModelRef lexicon, UnigramSegmenter segmenter, Set<String> inputSets) {
    this.lexicon = lexicon;
    this.segmenter = segmenter;
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Loads the lexicon from {@code file}, hashing exactly the bytes loaded. */
  static UnigramGolden load(Path file, Set<String> inputSets) throws IOException {
    ModelFile f = ModelFile.read(file);
    UnigramSegmenter segmenter = UnigramSegmenter.load(f.stream(), StandardCharsets.UTF_8);
    return new UnigramGolden(f.ref(), segmenter, inputSets);
  }

  @Override
  public String annotator() {
    return "unigram";
  }

  @Override
  public List<ModelRef> models() {
    return List.of(lexicon);
  }

  @Override
  public String modelDir() {
    return lexicon.baseName();
  }

  @Override
  public String annotatorClass() {
    return UnigramSegmenter.class.getName();
  }

  @Override
  public String method() {
    return "tokenizePos";
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
    Span[] spans = AnnotatorException.call(() -> segmenter.tokenizePos(input.text()));
    List<Layer> layers = SpanLayers.of("opennlp:tokens", input.offsets(), spans);
    if (layers.size() != 1) {
      throw new IllegalStateException("input " + input.inputId() + ": a word cuts a character");
    }
    return layers;
  }
}
