package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.charset.StandardCharsets;
import java.util.List;
import org.junit.jupiter.api.Test;

class DumpWriterTest {

  private static final Header HEADER = new Header("92d486d05151ef00b6d145c405046aa3a33b5a48", "3.0.0-PRE-SNAPSHOT",
      new Jdk("21.0.1", "21.0.1+12", "Vendor", "15.0"), "en/x", 9, 8, "ab", List.of(new ModelRef("m.bin", "cd")),
      "a.B", "run");

  @Test
  void writesTheWholeFileByteForByte() {
    List<DecisionPoint> points = List.of(
        new DecisionPoint(2, 0, new String[] {"eos=.", "x=a\tb"}, null, new double[] {0.25, 0.75}),
        new DecisionPoint(5, 1, new String[] {"p=\ud83d", "n"}, new float[] {1f, 0.5f}, new double[] {0.5, 0.5}));
    String expected = """
        dump 1
        generator\tturbo-goldens-java\t1
        opennlp\t92d486d05151ef00b6d145c405046aa3a33b5a48\t3.0.0-PRE-SNAPSHOT
        jdk\t21.0.1\t21.0.1+12\tVendor
        unicode\t15.0
        input\ten/x\t9\t8\tab
        model\tm.bin\tcd
        annotator\ta.B\trun
        point\t0\t2\t2\t2\t0\tnone
        context\teos=.
        context\tx=a\\tb
        outcome\ts\t3fd0000000000000 0.25
        outcome\tn\t3fe8000000000000 0.75
        end\t0
        point\t1\t5\t2\t2\t1\tgiven
        context\tp=\\ud83d\t3ff0000000000000 1.0
        context\tn\t3fe0000000000000 0.5
        outcome\ts\t3fe0000000000000 0.5
        outcome\tn\t3fe0000000000000 0.5
        end\t1
        """;
    assertEquals(expected, new String(DumpWriter.write(HEADER, new String[] {"s", "n"}, points), StandardCharsets.UTF_8));
  }

  @Test
  void aDumpWithoutPointsIsItsHeader() {
    String written = new String(DumpWriter.write(HEADER, new String[] {"s", "n"}, List.of()), StandardCharsets.UTF_8);
    assertEquals("dump 1\n", written.substring(0, 7));
    assertEquals("annotator\ta.B\trun\n", written.substring(written.lastIndexOf("annotator")));
  }

  @Test
  void aPointWithAnotherNumberOfOutcomesThanTheModelIsRefused() {
    List<DecisionPoint> points = List.of(new DecisionPoint(0, 0, new String[] {"a"}, null, new double[] {1.0}));
    assertThrows(IllegalArgumentException.class, () -> DumpWriter.write(HEADER, new String[] {"s", "n"}, points));
  }

  @Test
  void valuesMustMatchTheContextsAndTheUnitIsZeroOrOne() {
    assertThrows(IllegalArgumentException.class,
        () -> new DecisionPoint(0, 0, new String[] {"a", "b"}, new float[] {1f}, new double[] {1.0}));
    assertThrows(IllegalArgumentException.class,
        () -> new DecisionPoint(0, 2, new String[] {"a"}, null, new double[] {1.0}));
  }
}
