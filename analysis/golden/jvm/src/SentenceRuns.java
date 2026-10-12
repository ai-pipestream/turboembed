import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import opennlp.tools.sentdetect.NewlineSentenceDetector;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.util.Span;

/** Writes the sentence detector golden files: NewlineSentenceDetector and, when a model is given, SentenceDetectorME. */
final class SentenceRuns {

  private SentenceRuns() {
  }

  static void writeNewline(Path goldenDir, RunInfo run, List<Inputs.Input> inputs) throws IOException {
    NewlineSentenceDetector detector = new NewlineSentenceDetector();
    for (Inputs.Input in : inputs) {
      GoldenFile file = new GoldenFile("newline", run, in.text()).input();
      try {
        for (Span s : detector.sentPosDetect(in.text())) {
          file.span(s, null, null);
        }
      } catch (RuntimeException e) {
        file.error(e);
        Golden.problem("newline/" + in.name() + ": " + e);
      }
      file.write(goldenDir.resolve(Golden.componentDir("newline", run)).resolve(in.name() + ".txt"));
    }
  }

  /** One file per input for SentenceDetectorME over the whole input, with each sentence's probability. */
  static void writeMaxent(Path goldenDir, RunInfo run, List<Inputs.Input> inputs, ModelFile model) throws IOException {
    SentenceDetectorME detector = new SentenceDetectorME(new SentenceModel(model.path().toFile()));
    String component = "sentdetect-me-" + model.language();
    for (Inputs.Input in : inputs) {
      GoldenFile file = new GoldenFile(component, run, in.text());
      ModelHeader.write(file, model);
      file.input();
      try {
        Span[] spans = detector.sentPosDetect(in.text());
        double[] probs = detector.probs();
        if (spans.length != probs.length) {
          throw new IllegalStateException(spans.length + " spans but " + probs.length + " probabilities");
        }
        for (int i = 0; i < spans.length; i++) {
          file.span(spans[i], probs[i], null);
        }
      } catch (RuntimeException e) {
        file.error(e);
        Golden.problem(component + "/" + in.name() + ": " + e);
      }
      file.write(goldenDir.resolve(component).resolve(in.name() + ".txt"));
    }
  }
}
