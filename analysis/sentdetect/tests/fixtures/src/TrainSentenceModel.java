import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.sentdetect.SentenceDetectorFactory;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.sentdetect.SentenceSampleStream;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.Parameters;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.StringList;
import opennlp.tools.util.TrainingParameters;

/**
 * Trains the two small English sentence models the Rust tests use.
 *
 * <p>{@code en-abbrev.bin} has a case-insensitive abbreviation dictionary and {@code useTokenEnd=true};
 * {@code en-plain.bin} has no dictionary and {@code useTokenEnd=false}. Both are GIS maxent models
 * trained with cutoff 1 and 100 iterations on the sentences file (one sentence per line, an empty line
 * between documents). The end-of-sentence characters are the English default, which the factory
 * writes to the manifest.
 *
 * <p>Usage: {@code java TrainSentenceModel <sentences.txt> <output directory>}
 */
public final class TrainSentenceModel {

  /** The dictionary entries. "et al." is one entry of two tokens; only its first token is an abbreviation to the detector. */
  static final String[][] ABBREVIATIONS = {
    {"Mr."}, {"Mrs."}, {"Dr."}, {"Prof."}, {"St."}, {"Inc."}, {"U.S."}, {"e.g."}, {"i.e."}, {"etc."},
    {"a.m."}, {"p.m."}, {"et", "al."},
  };

  private TrainSentenceModel() {
  }

  public static void main(String[] args) throws IOException {
    Path sentences = Path.of(args[0]);
    Path out = Path.of(args[1]);
    Files.createDirectories(out);

    Dictionary abbreviations = new Dictionary(false);
    for (String[] entry : ABBREVIATIONS) {
      abbreviations.put(new StringList(entry));
    }
    train(sentences, new SentenceDetectorFactory("en", true, abbreviations, null), out.resolve("en-abbrev.bin"));
    train(sentences, new SentenceDetectorFactory("en", false, null, null), out.resolve("en-plain.bin"));
  }

  static void train(Path sentences, SentenceDetectorFactory factory, Path file) throws IOException {
    TrainingParameters params = new TrainingParameters();
    params.put(Parameters.ALGORITHM_PARAM, "MAXENT");
    params.put(Parameters.TRAINER_TYPE_PARAM, "Event");
    params.put(Parameters.ITERATIONS_PARAM, 100);
    params.put(Parameters.CUTOFF_PARAM, 1);
    try (ObjectStream<String> lines = new PlainTextByLineStream(
        () -> new FileInputStream(sentences.toFile()), StandardCharsets.UTF_8);
        SentenceSampleStream samples = new SentenceSampleStream(lines)) {
      SentenceModel model = SentenceDetectorME.train("en", samples, factory, params);
      try (OutputStream os = Files.newOutputStream(file)) {
        model.serialize(os);
      }
    }
  }
}
