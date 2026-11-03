package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The unigram segmenter's golden, on a lexicon of a few words. */
class UnigramGoldenTest {

  private static final String LEXICON = "北京 3 ns\n大学 2 n\n";

  private static Input input(String text) {
    return new Input("s", "x", text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets("s/x", text));
  }

  private static long[] u32(Layer layer, int column) {
    return ((Column.U32) layer.columns().get(column)).values();
  }

  private static UnigramGolden load(Path dir) throws IOException {
    Path file = dir.resolve("tiny-lexicon.txt");
    Files.writeString(file, LEXICON, StandardCharsets.UTF_8);
    return UnigramGolden.load(file, Set.of("s"));
  }

  @Test
  void theLexiconNamesTheDirectory(@TempDir Path dir) throws IOException {
    UnigramGolden g = load(dir);
    assertEquals("unigram", g.annotator());
    assertEquals("tiny-lexicon", g.modelDir());
    assertEquals("opennlp.tools.tokenize.lattice.UnigramSegmenter", g.annotatorClass());
    assertEquals(Map.of(), g.config());
  }

  @Test
  void theWordsHaveByteSpans(@TempDir Path dir) throws IOException, AnnotatorException {
    // Each character is three bytes; the unlisted U+3002 is a word of its own.
    List<Layer> layers = load(dir).layers(input("北京大学。 大学"));
    assertEquals(1, layers.size());
    Layer l = layers.get(0);
    assertEquals("turbo:tokens", l.id());
    assertArrayEquals(new long[] {0, 6, 12, 16}, u32(l, 0));
    assertArrayEquals(new long[] {6, 12, 15, 22}, u32(l, 1));
  }

  @Test
  void unicodeWhitespaceDecidesTheStretches(@TempDir Path dir) throws IOException, AnnotatorException {
    // U+0085 is White_Space and splits the text; U+001C is not, so it is an unlisted word.
    assertEquals(2, load(dir).layers(input("北京\u0085大学")).get(0).rows());
    assertEquals(3, load(dir).layers(input("北京\u001c大学")).get(0).rows());
  }
}
