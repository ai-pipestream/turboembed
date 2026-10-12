import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.sentdetect.EndOfSentenceScanner;
import opennlp.tools.sentdetect.SDContextGenerator;
import opennlp.tools.sentdetect.SentenceDetectorFactory;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.util.Span;
import opennlp.tools.util.WhitespaceMode;
import opennlp.tools.util.normalizer.CharClass;

/**
 * Writes the golden files of the fixture sentence models: for every model in the models directory and
 * every input, the spans and probabilities {@code SentenceDetectorME} returns, and one context file per
 * model with every context the detector evaluated.
 *
 * <p>The context loop is a copy of {@code SentenceDetectorME.sentPosDetect}: the delimiter-run skip with
 * its upper case exception, the skip of candidates before the last sentence start, the model call, the
 * abbreviation check (called through a subclass, since {@code isAcceptableBreak} is protected) and the
 * sentence start placement. It rebuilds the spans from its own decisions and reports a problem when they
 * differ from what {@code sentPosDetect} returns, so the copy cannot drift from the original unnoticed.
 *
 * <p>Every file is written twice: in the default {@code unicode} whitespace mode, and in the
 * {@code legacy} mode into directories with {@code -legacy} appended.
 *
 * <p>Compiled together with {@code GoldenFile}, {@code TextEscape}, {@code Utf8Offsets}, {@code RunInfo},
 * {@code Inputs}, {@code ModelFile} and {@code ModelHeader} from {@code analysis/golden/jvm/src}.
 *
 * <p>Usage: {@code java GoldenSentences <opennlp-commit> <inputs-dir> <models-dir> <golden-dir>}
 */
public final class GoldenSentences {

  private static final CharClass WHITESPACE = CharClass.whitespace();
  private static final List<String> PROBLEMS = new ArrayList<>();

  private GoldenSentences() {
  }

  /** Exposes the protected abbreviation check. */
  private static final class Detector extends SentenceDetectorME {
    Detector(SentenceModel model) {
      super(model);
    }

    boolean acceptable(CharSequence s, int fromIndex, int candidateIndex) {
      return isAcceptableBreak(s, fromIndex, candidateIndex);
    }
  }

  public static void main(String[] args) throws IOException {
    if (args.length != 4) {
      System.err.println("usage: GoldenSentences <opennlp-commit> <inputs-dir> <models-dir> <golden-dir>");
      System.exit(2);
    }
    List<Inputs.Input> inputs = Inputs.load(Path.of(args[1]));
    List<Path> models;
    try (Stream<Path> s = Files.list(Path.of(args[2]))) {
      models = s.filter(p -> p.getFileName().toString().endsWith(".bin")).sorted().toList();
    }
    Path golden = Path.of(args[3]);
    RunInfo unicode = new RunInfo(args[0], System.getProperty("java.version"), "unicode");
    for (RunInfo run : List.of(unicode, unicode.withWhitespace("legacy"))) {
      WhitespaceMode.setActive(run.whitespace().equals("legacy") ? WhitespaceMode.LEGACY : WhitespaceMode.UNICODE);
      for (Path path : models) {
        write(golden, run, inputs, ModelFile.open(path));
      }
    }
    WhitespaceMode.setActive(WhitespaceMode.UNICODE);
    System.out.println(inputs.size() + " inputs, " + models.size() + " models, " + PROBLEMS.size() + " problems");
    for (String p : PROBLEMS) {
      System.out.println("problem: " + p);
    }
  }

  static void write(Path golden, RunInfo run, List<Inputs.Input> inputs, ModelFile model) throws IOException {
    SentenceModel sentenceModel = new SentenceModel(model.path().toFile());
    Detector detector = new Detector(sentenceModel);
    String name = model.fileName().substring(0, model.fileName().length() - ".bin".length());
    String component = "sentdetect-me-fixture-" + name;
    String suffix = run.whitespace().equals("legacy") ? "-legacy" : "";

    GoldenFile contexts = new GoldenFile(component + "-contexts", run, "");
    ModelHeader.write(contexts, model);
    contexts.line("outcomes: " + SentenceDetectorME.SPLIT + " " + SentenceDetectorME.NO_SPLIT);

    for (Inputs.Input in : inputs) {
      GoldenFile file = new GoldenFile(component, run, in.text());
      ModelHeader.write(file, model);
      file.input();
      Span[] spans = detector.sentPosDetect(in.text());
      double[] probs = detector.probs();
      for (int i = 0; i < spans.length; i++) {
        file.span(spans[i], probs[i], null);
      }
      file.write(golden.resolve(component + suffix).resolve(in.name() + ".txt"));

      contexts.line("input: " + TextEscape.escape(in.text()));
      List<Span> rebuilt = contextLoop(contexts, detector, sentenceModel, in.text());
      List<Span> expected = new ArrayList<>();
      for (int i = 0; i < spans.length; i++) {
        expected.add(new Span(spans[i], probs[i]));
      }
      if (!sameSpans(rebuilt, expected)) {
        PROBLEMS.add(component + suffix + "/" + in.name() + ": context loop gave " + rebuilt + ", sentPosDetect "
            + Arrays.toString(spans) + " " + Arrays.toString(probs));
      }
    }
    contexts.write(golden.resolve("contexts").resolve(component + suffix + ".txt"));
  }

  /**
   * The loop of {@code sentPosDetect}, writing an {@code at:}, {@code features:}, {@code probs:} and
   * {@code decision:} line for every candidate the model is asked about. The decision is {@code split},
   * {@code no-split} (the model's best outcome is not a split) or {@code vetoed} (the abbreviation
   * dictionary refused the split). Returns the spans rebuilt from those decisions.
   */
  static List<Span> contextLoop(GoldenFile out, Detector detector, SentenceModel model, String s) {
    SentenceDetectorFactory factory = model.getFactory();
    MaxentModel maxent = model.getMaxentModel();
    SDContextGenerator cgen = factory.getSDContextGenerator();
    EndOfSentenceScanner scanner = factory.getEndOfSentenceScanner();
    boolean useTokenEnd = factory.isUseTokenEnd();
    int split = maxent.getIndex(SentenceDetectorME.SPLIT);
    int noSplit = maxent.getIndex(SentenceDetectorME.NO_SPLIT);

    List<Integer> enders = scanner.getPositions(s);
    List<Integer> positions = new ArrayList<>();
    List<Double> probs = new ArrayList<>();
    for (int i = 0, end = enders.size(), index = 0; i < end; i++) {
      int cint = enders.get(i);
      int fws = firstWs(s, cint + 1);
      if (i + 1 < end && enders.get(i + 1) < fws) {
        int next = cint + 1;
        if (next >= s.length() || !Character.isUpperCase(s.charAt(next))) {
          continue;
        }
      }
      if (!positions.isEmpty() && cint < positions.get(positions.size() - 1)) {
        continue;
      }
      String[] context = cgen.getContext(s, cint);
      double[] p = maxent.eval(context);
      String best = maxent.getBestOutcome(p);
      out.line("at: " + cint);
      out.line("features: " + String.join("\t", Arrays.stream(context).map(TextEscape::escape).toList()));
      out.line("probs: " + Double.toString(p[split]) + " " + Double.toString(p[noSplit]));
      if (!best.equals(SentenceDetectorME.SPLIT)) {
        out.line("decision: no-split");
        continue;
      }
      if (!detector.acceptable(s, index, cint)) {
        out.line("decision: vetoed");
        continue;
      }
      out.line("decision: split");
      if (index != cint) {
        positions.add(useTokenEnd ? firstNonWs(s, firstWs(s, cint + 1)) : firstNonWs(s, cint + 1));
        probs.add(p[maxent.getIndex(best)]);
      }
      index = cint + 1;
    }

    List<Span> spans = new ArrayList<>();
    int start = 0;
    for (int i = 0; i < positions.size(); i++) {
      addTrimmed(spans, s, start, positions.get(i), probs.get(i));
      start = positions.get(i);
    }
    if (positions.isEmpty() || positions.get(positions.size() - 1) != s.length()) {
      addTrimmed(spans, s, start, s.length(), 1d);
    }
    return spans;
  }

  static int firstWs(CharSequence s, int pos) {
    while (pos < s.length()) {
      int cp = Character.codePointAt(s, pos);
      if (WHITESPACE.contains(cp)) {
        break;
      }
      pos += Character.charCount(cp);
    }
    return pos;
  }

  static int firstNonWs(CharSequence s, int pos) {
    while (pos < s.length()) {
      int cp = Character.codePointAt(s, pos);
      if (!WHITESPACE.contains(cp)) {
        break;
      }
      pos += Character.charCount(cp);
    }
    return pos;
  }

  static void addTrimmed(List<Span> spans, CharSequence s, int start, int end, double prob) {
    while (start < end && WHITESPACE.contains(Character.codePointAt(s, start))) {
      start += Character.charCount(Character.codePointAt(s, start));
    }
    while (end > start && WHITESPACE.contains(Character.codePointBefore(s, end))) {
      end -= Character.charCount(Character.codePointBefore(s, end));
    }
    if (end > start) {
      spans.add(new Span(start, end, null, prob));
    }
  }

  static boolean sameSpans(List<Span> a, List<Span> b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (int i = 0; i < a.size(); i++) {
      Span x = a.get(i);
      Span y = b.get(i);
      if (x.getStart() != y.getStart() || x.getEnd() != y.getEnd() || x.getProb() != y.getProb()) {
        return false;
      }
    }
    return true;
  }
}
