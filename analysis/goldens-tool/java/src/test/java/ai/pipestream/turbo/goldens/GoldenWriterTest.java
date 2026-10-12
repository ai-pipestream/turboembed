package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.List;
import org.junit.jupiter.api.Test;

class GoldenWriterTest {

  @Test
  void escapesOnlyTheListedCharacters() {
    assertEquals("a\\tb\\nc\\rd\\\\e", GoldenWriter.escape("a\tb\nc\rd\\e"));
    assertEquals("\\u0000\\u001f\\u0007", GoldenWriter.escape("\u0000\u001f\u0007"));
    // DEL, non-ASCII text, characters outside the BMP and quotes stay as they are.
    String kept = "\u007f é 中 😀 \"'";
    assertEquals(kept, GoldenWriter.escape(kept));
  }

  @Test
  void f64IsTheRawBitsThenTheDecimal() {
    assertEquals("3fe0000000000000 0.5", GoldenWriter.f64(0.5));
    assertEquals("0000000000000000 0.0", GoldenWriter.f64(0.0));
    assertEquals("8000000000000000 -0.0", GoldenWriter.f64(-0.0));
    assertEquals("3ff0000000000000 1.0", GoldenWriter.f64(1.0));
    assertEquals("0000000000000001 4.9E-324", GoldenWriter.f64(Double.MIN_VALUE));
  }

  @Test
  void u32RefusesValuesOutsideItsRange() {
    assertEquals("0", GoldenWriter.u32(0));
    assertEquals("4294967295", GoldenWriter.u32(0xffff_ffffL));
    assertThrows(IllegalArgumentException.class, () -> GoldenWriter.u32(-1));
    assertThrows(IllegalArgumentException.class, () -> GoldenWriter.u32(0x1_0000_0000L));
  }

  @Test
  void writesTheWholeFileByteForByte() {
    Header h = new Header("92d486d05151ef00b6d145c405046aa3a33b5a48", "3.0.0-PRE-SNAPSHOT",
        new Jdk("21.0.1", "21.0.1+12", "Vendor", "15.0"), "en/x", 6, 6, "ab", List.of(new ModelRef("m.bin", "cd")),
        "a.B", "run");
    List<Layer> layers = List.of(
        new Layer("opennlp:tokens", "text", new Column.U32("start", new long[] {0, 3}),
            new Column.U32("end", new long[] {2, 6})),
        new Layer("opennlp:language", "document", new Column.Str("code", new String[] {"e\tn"}),
            new Column.F64("confidence", new double[] {0.5})));
    String expected = """
        golden 1
        generator\tturbo-goldens-java\t1
        opennlp\t92d486d05151ef00b6d145c405046aa3a33b5a48\t3.0.0-PRE-SNAPSHOT
        jdk\t21.0.1\t21.0.1+12\tVendor
        unicode\t15.0
        input\ten/x\t6\t6\tab
        model\tm.bin\tcd
        annotator\ta.B\trun
        layer\topennlp:tokens\ttext\t2\tstart:u32\tend:u32
        0\t2
        3\t6
        end\topennlp:tokens
        layer\topennlp:language\tdocument\t1\tcode:str\tconfidence:f64
        e\\tn\t3fe0000000000000 0.5
        end\topennlp:language
        """;
    assertEquals(expected, new String(GoldenWriter.write(h, layers), StandardCharsets.UTF_8));
  }

  @Test
  void configRecordsAreWrittenSortedByKeyAfterTheAnnotator() {
    Header h = new Header("92d486d05151ef00b6d145c405046aa3a33b5a48", "3.0.0-SNAPSHOT",
        new Jdk("21.0.1", "21.0.1+12", "Vendor", "15.0"), "norm/x", 1, 1, "ab", List.of(), "a.B", "run",
        java.util.Map.of("z", "1", "a", "t\tab"));
    String file = new String(GoldenWriter.write(h, List.of(new Layer("turbo:view", "text",
        new Column.U32("start", new long[0]), new Column.U32("end", new long[0]), new Column.Str("with", new String[0])))),
        StandardCharsets.UTF_8);
    assertTrue(file.contains("annotator\ta.B\trun\nconfig\ta\tt\\tab\nconfig\tz\t1\nlayer\t"), file);
  }

  @Test
  void aLayerWithColumnsOfDifferentLengthsIsRefused() {
    assertThrows(IllegalArgumentException.class, () -> new Layer("x", "text",
        new Column.U32("start", new long[] {0}), new Column.U32("end", new long[] {1, 2})));
  }

  @Test
  void theUnicodeTableFailsOnAnUnknownJdk() {
    assertEquals("15.0", Jdk.unicodeVersion(21));
    assertEquals("16.0", Jdk.unicodeVersion(24));
    assertThrows(IllegalStateException.class, () -> Jdk.unicodeVersion(25));
  }
}
