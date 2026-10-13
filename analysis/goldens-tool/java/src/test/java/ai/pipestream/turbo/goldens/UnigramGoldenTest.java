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
import opennlp.tools.util.WhitespaceMode;
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

  private static UnigramGolden load(Path dir, WhitespaceMode mode) throws IOException {
    Path file = dir.resolve("tiny-lexicon.txt");
    Files.writeString(file, LEXICON, StandardCharsets.UTF_8);
    return UnigramGolden.load(file, mode, Set.of("s"));
  }

  @Test
  void theLexiconNamesTheDirectoryAndTheModeIsConfig(@TempDir Path dir) throws IOException {
    UnigramGolden g = load(dir, WhitespaceMode.UNICODE);
    assertEquals("unigram", g.annotator());
    assertEquals("tiny-lexicon", g.modelDir());
    assertEquals("opennlp.tools.tokenize.lattice.UnigramSegmenter", g.annotatorClass());
    assertEquals(Map.of(), g.config());
    UnigramGolden legacy = load(dir, WhitespaceMode.LEGACY);
    assertEquals("tiny-lexicon+legacy", legacy.modelDir());
    assertEquals(Map.of("whitespace", "legacy"), legacy.config());
  }

  @Test
  void theWordsHaveByteSpans(@TempDir Path dir) throws IOException, AnnotatorException {
    // Each character is three bytes; the unlisted U+3002 is a word of its own.
    List<Layer> layers = load(dir, WhitespaceMode.UNICODE).layers(input("北京大学。 大学"));
    assertEquals(1, layers.size());
    Layer l = layers.get(0);
    assertEquals("opennlp:tokens", l.id());
    assertArrayEquals(new long[] {0, 6, 12, 16}, u32(l, 0));
    assertArrayEquals(new long[] {6, 12, 15, 22}, u32(l, 1));
  }

  @Test
  void theModeDecidesTheStretches(@TempDir Path dir) throws IOException, AnnotatorException {
    // U+0085 is whitespace in the Unicode mode only.
    String text = "北京\u0085大学";
    assertEquals(2, load(dir, WhitespaceMode.UNICODE).layers(input(text)).get(0).rows());
    assertEquals(3, load(dir, WhitespaceMode.LEGACY).layers(input(text)).get(0).rows());
    assertEquals(WhitespaceMode.UNICODE, WhitespaceMode.current());
  }
}
