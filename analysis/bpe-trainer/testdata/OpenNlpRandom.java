import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Random;
import opennlp.tools.tokenize.BPEModel;
import opennlp.tools.tokenize.BPETokenizer.SymbolPair;
import opennlp.tools.tokenize.BPETokenizerTrainer;

/**
 * Writes opennlp-random.jsonl: OpenNLP's BPETokenizerTrainer on random corpora, one JSON line per
 * corpus with its text, the number of merges asked for and the merges made.
 *
 * <p>Run from the repository root with the opennlp-runtime and opennlp-api jars of the commit
 * docs/bpe-trainer.md names on the class path:</p>
 *
 * <pre>
 * java -cp opennlp-runtime.jar:opennlp-api.jar \
 *   analysis/bpe-trainer/testdata/OpenNlpRandom.java \
 *   &gt; analysis/bpe-trainer/testdata/opennlp-random.jsonl
 * </pre>
 *
 * <p>The corpora are made from a seeded java.util.Random, so a run writes the same file. Letters
 * come from a small alphabet, so pairs tie often, and the larger corpora fill maps of 64 bins
 * and more. Every character is in the Basic Multilingual Plane.</p>
 */
public final class OpenNlpRandom {

  private static final String LETTERS = "abcdefghijklmnopqrstuvwxyzé中ж";
  private static final String[] SPACES = {" ", " ", " ", "  ", "\t", "\n"};

  public static void main(String[] args) {
    PrintStream out = new PrintStream(System.out, true, StandardCharsets.UTF_8);
    Random random = new Random(194);
    for (int run = 0; run < 160; run++) {
      boolean large = run % 4 == 3;
      int alphabet = 1 + random.nextInt(large ? LETTERS.length() : 5);
      int words = 1 + random.nextInt(large ? 3000 : 60);
      int longest = 1 + random.nextInt(large ? 9 : 14);
      StringBuilder text = new StringBuilder();
      for (int w = 0; w < words; w++) {
        if (w > 0) {
          text.append(SPACES[random.nextInt(SPACES.length)]);
        }
        int length = 1 + random.nextInt(longest);
        for (int i = 0; i < length; i++) {
          text.append(LETTERS.charAt(random.nextInt(alphabet)));
        }
      }
      int numMerges = 1 + random.nextInt(large ? 400 : 120);
      BPEModel model = new BPETokenizerTrainer().train(List.of(text.toString()), numMerges, "en");
      StringBuilder line = new StringBuilder();
      line.append("{\"num_merges\":").append(numMerges).append(",\"text\":").append(json(text.toString()));
      line.append(",\"merges\":[");
      List<SymbolPair> merges = model.getMerges();
      for (int i = 0; i < merges.size(); i++) {
        if (i > 0) {
          line.append(',');
        }
        line.append('[').append(json(merges.get(i).left())).append(',').append(json(merges.get(i).right()))
            .append(']');
      }
      line.append("]}");
      out.println(line);
    }
  }

  private static String json(String s) {
    return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\t", "\\t")
        + "\"";
  }
}
