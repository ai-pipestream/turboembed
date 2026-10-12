package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import opennlp.tools.ml.maxent.GISModel;
import opennlp.tools.ml.model.Context;
import opennlp.tools.sentdetect.SentenceDetectorFactory;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerFactory;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.model.BaseModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MaxentRecorderTest {

  /**
   * Two outcomes, {@code s} and {@code n}. The predicate {@code even} has weight 0 for both, so a
   * context of it alone gives 0.5 and 0.5 exactly; {@code eos=.} has weight ln 3 for {@code s}
   * only, so it gives 3/4 and 1/4.
   */
  private static GISModel sentenceWeights() {
    return new GISModel(
        new Context[] {new Context(new int[] {0, 1}, new double[] {0.0, 0.0}),
            new Context(new int[] {0}, new double[] {Math.log(3)})},
        new String[] {"even", "eos=."}, new String[] {"s", "n"});
  }

  @Test
  void knownWeightsGiveKnownProbabilitiesAndEveryCallIsRecordedInOrder() {
    MaxentRecorder r = MaxentRecorder.wrap(sentenceWeights());
    String[] first = {"even"};
    double[] p = r.eval(first);
    assertArrayEquals(new double[] {0.5, 0.5}, p);
    String[] second = {"unknown", "eos=."};
    double[] q = r.eval(second);
    assertEquals(0.75, q[0], 1e-15);
    assertEquals(0.25, q[1], 1e-15);
    float[] values = {2f};
    r.eval(new String[] {"eos=."}, values);
    // Changing the arrays afterwards does not change what was recorded.
    first[0] = "changed";
    p[0] = -1;
    values[0] = -1;

    List<Evaluation> calls = r.take();
    assertEquals(3, calls.size());
    assertArrayEquals(new String[] {"even"}, calls.get(0).contexts());
    assertArrayEquals(new double[] {0.5, 0.5}, calls.get(0).outcomes());
    assertNull(calls.get(0).values());
    assertNull(calls.get(0).request());
    assertArrayEquals(new String[] {"unknown", "eos=."}, calls.get(1).contexts());
    assertArrayEquals(q, calls.get(1).outcomes());
    assertArrayEquals(new float[] {2f}, calls.get(2).values());
    assertEquals(0.9, calls.get(2).outcomes()[0], 1e-15);
    assertEquals(List.of(), r.take());
  }

  @Test
  void everyOtherCallAnswersAsTheLoadedModel() {
    GISModel m = sentenceWeights();
    MaxentRecorder r = MaxentRecorder.wrap(m);
    assertEquals(2, r.getNumOutcomes());
    assertEquals(1, r.getIndex("n"));
    assertEquals(-1, r.getIndex("x"));
    assertEquals("n", r.getOutcome(1));
    assertEquals("s", r.getBestOutcome(new double[] {0.75, 0.25}));
    assertEquals(m.getAllOutcomes(new double[] {0.75, 0.25}), r.getAllOutcomes(new double[] {0.75, 0.25}));
    assertEquals(m.getModelType(), r.getModelType());
    assertArrayEquals(new String[] {"s", "n"}, r.outcomeLabels());
  }

  @Test
  void aRequestIsPairedWithTheEvalOfItsArrayOnly() {
    MaxentRecorder r = MaxentRecorder.wrap(sentenceWeights());
    String[] context = {"even"};
    ContextRequest request = new ContextRequest("text", 2);
    r.contextRequested(request, context);
    r.eval(context);
    assertSame(request, r.take().get(0).request());

    r.contextRequested(request, context);
    assertThrows(IllegalStateException.class, () -> r.eval(new String[] {"even"}));

    MaxentRecorder twice = MaxentRecorder.wrap(sentenceWeights());
    twice.contextRequested(request, context);
    assertThrows(IllegalStateException.class, () -> twice.contextRequested(request, context));

    MaxentRecorder never = MaxentRecorder.wrap(sentenceWeights());
    never.contextRequested(request, context);
    assertThrows(IllegalStateException.class, never::take);

    // After an annotator threw, what it recorded is discarded, a pending request too.
    MaxentRecorder threw = MaxentRecorder.wrap(sentenceWeights());
    threw.contextRequested(request, context);
    threw.eval(context);
    threw.contextRequested(request, context);
    threw.discard();
    assertEquals(List.of(), threw.take());
  }

  private static Path save(BaseModel model, Path file) throws IOException {
    try (OutputStream out = Files.newOutputStream(file)) {
      model.serialize(out);
    }
    return file;
  }

  private static Input input(String text) {
    return new Input("t", "x", text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets("t/x", text));
  }

  @Test
  void theSentenceDumpHasEveryCandidateWithItsContextsAtItsByteOffset(@TempDir Path dir)
      throws IOException, AnnotatorException {
    char[] eos = {'.'};
    SentenceModel model = new SentenceModel("en", sentenceWeights(), Map.of(),
        new SentenceDetectorFactory("en", true, null, eos));
    SentenceGolden golden = SentenceGolden.load(save(model, dir.resolve("s.bin")));
    // The full stops are at UTF-16 indices 6 and 14 and at UTF-8 byte offsets 8 and 18.
    Input in = input("Ünï ok. 😀 Yes.");
    AnnotatorDump.Run run = SentenceDump.of(golden).run(in);

    assertArrayEquals(GoldenWriter.write(header(golden), golden.layers(in)),
        GoldenWriter.write(header(golden), run.layers()));
    assertArrayEquals(new String[] {"s", "n"}, run.outcomeLabels());
    assertEquals(2, run.points().size());
    var plain = new SentenceDetectorFactory("en", true, null, eos).getSDContextGenerator();
    int[] utf16 = {6, 14};
    long[] utf8 = {8, 18};
    for (int i = 0; i < 2; i++) {
      DecisionPoint p = run.points().get(i);
      assertEquals(utf8[i], p.position());
      assertEquals(0, p.unit());
      assertArrayEquals(plain.getContext(in.text(), utf16[i]), p.contexts());
      assertEquals(0.75, p.outcomes()[0], 1e-15);
      assertEquals(0.25, p.outcomes()[1], 1e-15);
    }
  }

  @Test
  void theTokenDumpPairsRequestsWithTheirWhitespaceTokens(@TempDir Path dir)
      throws IOException, AnnotatorException {
    // No predicate of the tokenizer's contexts is known, so every decision is 0.5 and 0.5, and the
    // first outcome, F (no split), wins.
    GISModel weights = new GISModel(new Context[] {new Context(new int[] {0, 1}, new double[] {0.0, 0.0})},
        new String[] {"never"}, new String[] {"F", "T"});
    TokenizerModel model = new TokenizerModel(weights, Map.of(), new TokenizerFactory("en", null, false, null));
    TokenGolden golden = TokenGolden.load(save(model, dir.resolve("t.bin")));
    // "x" is too short to be split. The two tokens "a😀" have the same text; each is asked about
    // index 1 (before the emoji) and index 2 (between its halves). "b." is asked about index 1.
    Input in = input("x a😀 b. a😀");
    AnnotatorDump.Run run = TokenDump.of(golden).run(in);

    assertArrayEquals(GoldenWriter.write(header(golden), golden.layers(in)),
        GoldenWriter.write(header(golden), run.layers()));
    long[] position = {3, 3, 9, 12, 12};
    int[] unit = {0, 1, 0, 0, 1};
    String[] token = {"a😀", "a😀", "b.", "a😀", "a😀"};
    int[] index = {1, 2, 1, 1, 2};
    var plain = new TokenizerFactory("en", null, false, null).getContextGenerator();
    assertEquals(position.length, run.points().size());
    for (int i = 0; i < position.length; i++) {
      DecisionPoint p = run.points().get(i);
      assertEquals(position[i], p.position(), "point " + i);
      assertEquals(unit[i], p.unit(), "point " + i);
      assertArrayEquals(plain.getContext(token[i], index[i]), p.contexts(), "point " + i);
      assertArrayEquals(new double[] {0.5, 0.5}, p.outcomes());
    }
  }

  private static Header header(AnnotatorGolden a) {
    return new Header("92d486d05151ef00b6d145c405046aa3a33b5a48", "v", new Jdk("21", "21", "V", "15.0"), "t/x", 1,
        1, "ab", a.models(), a.annotatorClass(), a.method());
  }
}
