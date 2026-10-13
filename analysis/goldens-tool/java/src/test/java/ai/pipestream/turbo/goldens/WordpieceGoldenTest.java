package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The WordPiece encoder's golden, on a vocabulary of a few pieces. */
class WordpieceGoldenTest {

  private static final String VOCABULARY =
      String.join("\n", "[PAD]", "[UNK]", "[CLS]", "[SEP]", "cafe", "un", "##aff", "##able", "!", "東") + "\n";

  private static Input input(String text) {
    return new Input("s", "x", text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets("s/x", text));
  }

  private static long[] u32(Layer layer, int column) {
    return ((Column.U32) layer.columns().get(column)).values();
  }

  private static WordpieceGolden load(Path dir) throws IOException {
    Path file = dir.resolve("tiny-vocab.txt");
    Files.writeString(file, VOCABULARY, StandardCharsets.UTF_8);
    return WordpieceGolden.load(file, Set.of("s"));
  }

  @Test
  void theVocabularyNamesTheDirectoryAndTheOptionsAreConfig(@TempDir Path dir) throws IOException {
    WordpieceGolden g = load(dir);
    assertEquals("subwords", g.annotator());
    assertEquals("tiny-vocab", g.modelDir());
    assertEquals("opennlp.tools.tokenize.WordpieceEncoder", g.annotatorClass());
    assertEquals(Map.of("lowerCase", "true", "maxWordCodePoints", "100"), g.config());
  }

  @Test
  void thePiecesOfTheTextHaveByteSpansAndTheSpecialTokensAreLeftOut(@TempDir Path dir)
      throws IOException, AnnotatorException {
    // "é" is two bytes and is stripped to "e"; "東" is three bytes and isolated.
    List<Layer> layers = load(dir).layers(input("Café Unaffable!東"));
    assertEquals(1, layers.size());
    Layer l = layers.get(0);
    assertEquals("opennlp:subwords", l.id());
    assertEquals("text", l.anchor());
    assertArrayEquals(new long[] {0, 6, 8, 11, 15, 16}, u32(l, 0));
    assertArrayEquals(new long[] {5, 8, 11, 15, 16, 19}, u32(l, 1));
    assertArrayEquals(new long[] {4, 5, 6, 7, 8, 9}, u32(l, 2));
  }

  @Test
  void anEmptyTextHasNoPieces(@TempDir Path dir) throws IOException, AnnotatorException {
    assertEquals(0, load(dir).layers(input("")).get(0).rows());
  }

  @Test
  void aVocabularyWithoutAFinalLineFeedIsRefused() {
    byte[] bytes = "[UNK]\n[CLS]".getBytes(StandardCharsets.UTF_8);
    assertThrows(IOException.class, () -> WordpieceGolden.lines(Path.of("v.txt"), bytes));
  }
}
