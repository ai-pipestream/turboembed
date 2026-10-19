package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.util.WhitespaceMode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The lattice tokenizer's golden, on a dictionary of a few words. */
class LatticeGoldenTest {

  private static Input input(String text) {
    return new Input("s", "x", text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets("s/x", text));
  }

  private static Path dictionary(Path dir, Charset charset) throws IOException {
    Files.createDirectories(dir);
    Files.writeString(dir.resolve("matrix.def"), "1 1\n0 0 0\n", charset);
    Files.writeString(dir.resolve("b.csv"), "東京,0,0,3000,名詞,\"地域,一般\"\n", charset);
    Files.writeString(dir.resolve("a.csv"), "東,0,0,6000,名詞\n京都,0,0,3000,名詞\n", charset);
    Files.writeString(dir.resolve("char.def"), "DEFAULT 0 1 0\nKANJI 0 0 2\n0x4E00..0x9FFF KANJI\n", charset);
    Files.writeString(dir.resolve("unk.def"), "DEFAULT,0,0,9000,記号\nKANJI,0,0,8000,名詞,未知\n", charset);
    return dir;
  }

  @Test
  void theDirectoryNamesTheRunAndTheManifestIsItsHash(@TempDir Path tmp) throws IOException {
    Path dir = dictionary(tmp.resolve("tiny-dic"), StandardCharsets.UTF_8);
    LatticeGolden g = LatticeGolden.load(dir, WhitespaceMode.UNICODE, Set.of("s"));
    assertEquals("lattice", g.annotator());
    assertEquals("tiny-dic", g.modelDir());
    assertEquals("opennlp.tools.tokenize.lattice.LatticeTokenizer", g.annotatorClass());
    assertEquals(Map.of(), g.config());
    List<String> order = LatticeGolden.files(dir).stream().map(p -> p.getFileName().toString()).toList();
    assertEquals(List.of("matrix.def", "a.csv", "b.csv", "char.def", "unk.def"), order);
    StringBuilder m = new StringBuilder();
    for (Path f : LatticeGolden.files(dir)) {
      m.append(f.getFileName()).append('\t').append(Sha.sha512(Files.readAllBytes(f))).append('\n');
    }
    assertEquals(new ModelRef("tiny-dic", Sha.sha512(m.toString().getBytes(StandardCharsets.UTF_8))),
        g.models().get(0));
    LatticeGolden legacy = LatticeGolden.load(dir, WhitespaceMode.LEGACY, Set.of("s"));
    assertEquals("tiny-dic+legacy", legacy.modelDir());
    assertEquals(Map.of("whitespace", "legacy"), legacy.config());
  }

  @Test
  void theMorphemesHaveByteSpansAndFeatures(@TempDir Path tmp) throws IOException, AnnotatorException {
    LatticeGolden g = LatticeGolden.load(dictionary(tmp.resolve("d"), StandardCharsets.UTF_8),
        WhitespaceMode.UNICODE, Set.of("s"));
    List<Layer> layers = g.layers(input("東京 峠x"));
    assertEquals(2, layers.size());
    assertEquals("opennlp:tokens", layers.get(0).id());
    assertArrayEquals(new long[] {0, 7, 10}, ((Column.U32) layers.get(0).columns().get(0)).values());
    assertArrayEquals(new long[] {6, 10, 11}, ((Column.U32) layers.get(0).columns().get(1)).values());
    Layer morphemes = layers.get(1);
    assertEquals("turbo:morphemes", morphemes.id());
    assertEquals("aligned:opennlp:tokens", morphemes.anchor());
    assertArrayEquals(new long[] {0, 1, 1}, ((Column.U32) morphemes.columns().get(0)).values());
    assertArrayEquals(new String[] {"名詞,地域,一般", "名詞,未知", "記号"},
        ((Column.Str) morphemes.columns().get(1)).values());
  }

  @Test
  void aRecodedCopyMustReadTheSame(@TempDir Path tmp) throws IOException, AnnotatorException {
    Charset eucJp = Charset.forName("EUC-JP");
    Path original = dictionary(tmp.resolve("orig"), eucJp);
    Path copy = dictionary(tmp.resolve("orig.utf8"), StandardCharsets.UTF_8);
    LatticeGolden g = LatticeGolden.loadRecoded(original, eucJp, copy, WhitespaceMode.UNICODE, Set.of("s"));
    assertEquals(2, g.models().size());
    assertEquals("orig", g.modelDir());
    assertEquals("orig.utf8", g.models().get(1).fileName());
    // 東 6000 + 京都 3000 is cheaper than 東京 3000 + an unknown 都 8000.
    assertEquals(2, g.layers(input("東京都")).get(0).rows());
    Files.writeString(copy.resolve("a.csv"), "東,0,0,6001,名詞\n京都,0,0,3000,名詞\n", StandardCharsets.UTF_8);
    assertThrows(IllegalStateException.class,
        () -> LatticeGolden.loadRecoded(original, eucJp, copy, WhitespaceMode.UNICODE, Set.of("s")));
  }

  @Test
  void aMissingTemplateIsTheAnnotatorsError(@TempDir Path tmp) throws IOException {
    Path dir = dictionary(tmp.resolve("d"), StandardCharsets.UTF_8);
    Files.writeString(dir.resolve("unk.def"), "KANJI,0,0,8000,名詞\n", StandardCharsets.UTF_8);
    LatticeGolden g = LatticeGolden.load(dir, WhitespaceMode.UNICODE, Set.of("s"));
    AnnotatorException e = assertThrows(AnnotatorException.class, () -> g.layers(input("東x")));
    assertEquals("java.lang.IllegalStateException", e.exceptionClass());
  }
}
