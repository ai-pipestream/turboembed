package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** {@link Main.Options}: the options after the positional arguments, and the runs they select. */
class OptionsTest {

  /** An annotator of a given directory and model directory that runs on nothing. */
  private record Named(String annotator, String modelDir) implements AnnotatorGolden {
    @Override
    public List<ModelRef> models() {
      return List.of();
    }

    @Override
    public String annotatorClass() {
      return "test.Named";
    }

    @Override
    public String method() {
      return "run";
    }

    @Override
    public Set<String> inputSets() {
      return Set.of();
    }

    @Override
    public List<Layer> layers(Input input) {
      return List.of();
    }
  }

  private record NamedDump(AnnotatorGolden golden) implements AnnotatorDump {
    @Override
    public Run run(Input input) {
      return new Run(new String[0], List.of(), List.of());
    }
  }

  private static Runs.All all() {
    Named sentences = new Named("sentdetect", "en-plain");
    Named tokens = new Named("tokenize", "simple");
    Named training = new Named("train", "sentdetect-one-cutoff5");
    return new Runs.All(List.of(sentences, tokens, training), List.of(new NamedDump(sentences)));
  }

  @Test
  void noOptionsKeepEveryRunAndWriteNoDumps() {
    Main.Options o = Main.Options.parse(List.of());
    assertFalse(o.dumps());
    Runs.All kept = o.select(all());
    assertEquals(3, kept.annotators().size());
    assertEquals(1, kept.recorders().size());
  }

  @Test
  void dumpsIsStillAccepted() {
    assertTrue(Main.Options.parse(List.of("--dumps")).dumps());
  }

  @Test
  void skippedAnnotatorsLeaveWithTheirRecorders() {
    Main.Options o = Main.Options.parse(List.of("--skip-annotators", "train,sentdetect"));
    Runs.All kept = o.select(all());
    assertEquals(List.of("tokenize"), kept.annotators().stream().map(AnnotatorGolden::annotator).toList());
    assertTrue(kept.recorders().isEmpty());
  }

  @Test
  void onlyRunsKeepsTheNamedRuns() {
    Main.Options o = Main.Options.parse(List.of("--only-runs", "sentdetect/en-plain,train/sentdetect-one-cutoff5"));
    Runs.All kept = o.select(all());
    assertEquals(List.of("sentdetect", "train"), kept.annotators().stream().map(AnnotatorGolden::annotator).toList());
    assertEquals(1, kept.recorders().size());
  }

  @Test
  void bothFiltersApply() {
    Main.Options o = Main.Options.parse(List.of("--only-runs", "sentdetect/en-plain,tokenize/simple",
        "--skip-annotators", "tokenize"));
    assertEquals(List.of("sentdetect"), o.select(all()).annotators().stream().map(AnnotatorGolden::annotator).toList());
  }

  @Test
  void aNameThatMatchesNothingIsAnError() {
    assertThrows(IllegalArgumentException.class,
        () -> Main.Options.parse(List.of("--skip-annotators", "formats")).select(all()));
    assertThrows(IllegalArgumentException.class,
        () -> Main.Options.parse(List.of("--only-runs", "tokenize/whitespace")).select(all()));
  }

  @Test
  void malformedOptionsAreRefused() {
    assertThrows(IllegalArgumentException.class, () -> Main.Options.parse(List.of("--skip-annotators")));
    assertThrows(IllegalArgumentException.class, () -> Main.Options.parse(List.of("--only-runs", "tokenize")));
    assertThrows(IllegalArgumentException.class, () -> Main.Options.parse(List.of("--only-runs", "a/b/c")));
    assertThrows(IllegalArgumentException.class, () -> Main.Options.parse(List.of("--skip-annotators", "a,,b")));
    assertThrows(IllegalArgumentException.class, () -> Main.Options.parse(List.of("--verbose")));
  }
}
