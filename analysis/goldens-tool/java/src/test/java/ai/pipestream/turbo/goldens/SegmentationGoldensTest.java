package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The goldens of the tokenizers without a model, the newline detector and the detokenizer. */
class SegmentationGoldensTest {

  private static Input input(String set, String id, String text) {
    String inputId = set + "/" + id;
    return new Input(set, id, text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets(inputId, text));
  }

  private static long[] u32(Layer layer, int column) {
    return ((Column.U32) layer.columns().get(column)).values();
  }

  private static String[] str(Layer layer, int column) {
    return ((Column.Str) layer.columns().get(column)).values();
  }

  @Test
  void eachRunHasItsOwnDirectoryAndConfig() {
    var plain = new PlainTokenGolden(PlainTokenGolden.Kind.SIMPLE, false, Set.of("e"));
    assertEquals("simple", plain.modelDir());
    assertEquals(Map.of(), plain.config());
    var both = new PlainTokenGolden(PlainTokenGolden.Kind.WHITESPACE, true, Set.of("e"));
    assertEquals("whitespace+keep-new-lines", both.modelDir());
    assertEquals(Map.of("keepNewLines", "true"), both.config());
    assertEquals("opennlp.tools.tokenize.WhitespaceTokenizer", both.annotatorClass());
    assertEquals("newline", new NewlineSentenceGolden(Set.of("e")).modelDir());
    assertEquals("uax29", new WordTokenGolden(Set.of("e")).modelDir());
  }

  @Test
  void theSimpleTokenizerCutsACharacterOutsideTheBmpAndKeepingNewLinesThrowsBeforeAnyToken()
      throws AnnotatorException {
    var simple = new PlainTokenGolden(PlainTokenGolden.Kind.SIMPLE, false, Set.of("e"));
    List<Layer> layers = simple.layers(input("e", "x", "a😀"));
    assertEquals(List.of("turbo:tokens", "turbo:utf16<turbo:tokens>"), layers.stream().map(Layer::id).toList());
    assertArrayEquals(new long[] {0, 1, 1}, u32(layers.get(0), 0));
    assertArrayEquals(new long[] {1, 5, 5}, u32(layers.get(0), 1));
    assertArrayEquals(new long[] {0, 1, 1}, u32(layers.get(1), 2));

    var keep = new PlainTokenGolden(PlainTokenGolden.Kind.SIMPLE, true, Set.of("e"));
    assertThrows(AnnotatorException.class, () -> keep.layers(input("e", "x", "\nIt rained.")));
  }

  @Test
  void informationSeparatorsAreTextAndNextLineIsWhitespace() throws AnnotatorException {
    // U+001C is not White_Space, so it stays inside the token; U+0085 is, so it splits.
    var tokenizer = new PlainTokenGolden(PlainTokenGolden.Kind.WHITESPACE, false, Set.of("e"));
    assertEquals(1, tokenizer.layers(input("e", "x", "a\u001cb")).get(0).rows());
    assertEquals(2, tokenizer.layers(input("e", "x", "a\u0085b")).get(0).rows());
  }

  @Test
  void theWordTokenizerWritesTheWordTypeOfEachToken() throws AnnotatorException {
    List<Layer> layers = new WordTokenGolden(Set.of("e")).layers(input("e", "x", "Hi 42 東"));
    assertEquals(List.of("turbo:tokens", "turbo:word-types"), layers.stream().map(Layer::id).toList());
    assertEquals("aligned:turbo:tokens", layers.get(1).anchor());
    assertArrayEquals(new String[] {"ALPHANUMERIC", "NUMERIC", "IDEOGRAPHIC"}, str(layers.get(1), 0));
  }

  @Test
  void aDetokenizerInputIsOneTokenPerLine(@TempDir Path dir) throws IOException, AnnotatorException {
    Path xml = dir.resolve("t-detokenizer.xml");
    Files.writeString(xml, """
        <?xml version="1.0" encoding="UTF-8"?>
        <dictionary>
        <entry operation="MOVE_LEFT"><token>,</token></entry>
        </dictionary>
        """);
    DetokenizerGolden golden = DetokenizerGolden.load(xml, Set.of("d"));
    assertEquals("t-detokenizer", golden.modelDir());
    assertTrue(golden.accepts(input("d", "t-case", "")));
    assertFalse(golden.accepts(input("d", "u-case", "")));
    assertFalse(golden.accepts(input("e", "t-case", "")));

    List<Layer> layers = golden.layers(input("d", "t-case", "a\né\n,\n"));
    assertEquals(List.of("turbo:tokens", "turbo:detokenization", "turbo:detokenized"),
        layers.stream().map(Layer::id).toList());
    assertArrayEquals(new long[] {0, 2, 5}, u32(layers.get(0), 0));
    assertArrayEquals(new long[] {1, 4, 6}, u32(layers.get(0), 1));
    assertArrayEquals(new String[] {"NO_OPERATION", "NO_OPERATION", "MERGE_TO_LEFT"}, str(layers.get(1), 0));
    assertArrayEquals(new String[] {"a é,"}, str(layers.get(2), 0));
    assertArrayEquals(new String[] {"a é<SPLIT>,"}, str(layers.get(2), 1));

    assertEquals(0, golden.layers(input("d", "t-empty", "")).get(0).rows());
    assertThrows(IllegalArgumentException.class, () -> golden.layers(input("d", "t-open", "a\nb")));
    assertThrows(IllegalArgumentException.class, () -> DetokenizerGolden.load(dir.resolve("t.xml"), Set.of("d")));
  }
}
