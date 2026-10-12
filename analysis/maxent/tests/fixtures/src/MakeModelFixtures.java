import java.io.ByteArrayInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Pattern;

import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.ml.TrainerFactory;
import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.Event;
import opennlp.tools.ml.model.GenericModelWriter;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.tokenize.TokenSample;
import opennlp.tools.tokenize.TokenizerFactory;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Parameters;
import opennlp.tools.util.TrainingParameters;

/**
 * Trains a GIS, a QN and a perceptron model on a handful of events and
 * writes each in OpenNLP's binary model format, then a tiny TokenizerME
 * model file with an abbreviation dictionary, serialized by OpenNLP itself.
 * For each model it evaluates a list of contexts and writes the
 * probabilities with Double.toString, so the Rust side can compare them.
 */
public final class MakeModelFixtures {

  /** Features with a NUL, a non-ASCII character and a character outside the BMP. */
  static final String NUL = "w=\u0000";
  static final String ACCENT = "w=été";
  static final String EMOJI = "w=😀";

  static List<Event> events() {
    List<Event> e = new ArrayList<>();
    e.add(new Event("yes", new String[] {"w=a", "len=1", "cap"}));
    e.add(new Event("yes", new String[] {"w=a", "len=2"}));
    e.add(new Event("yes", new String[] {"w=b", "cap", ACCENT}));
    e.add(new Event("no", new String[] {"w=c", "len=1"}));
    e.add(new Event("no", new String[] {"w=c", "len=3", NUL}));
    e.add(new Event("no", new String[] {"w=d", "len=3"}));
    e.add(new Event("maybe", new String[] {"w=b", "len=2", EMOJI}));
    e.add(new Event("maybe", new String[] {"w=d", "cap", EMOJI}));
    e.add(new Event("maybe", new String[] {"w=a", "len=3", NUL, ACCENT}));
    return e;
  }

  static final String[][] CONTEXTS = {
      {},
      {"w=a"},
      {"w=a", "len=1", "cap"},
      {"w=c", "len=3"},
      {"w=b", "len=2", EMOJI},
      {NUL},
      {ACCENT, "cap"},
      {"unknown", "w=d"},
      {"w=a", "w=a"},
      {"w=d", "cap", EMOJI, NUL, ACCENT, "len=2"},
  };

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0], "models");
    Files.createDirectories(dir);
    try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(dir.resolve("expected.txt"),
        StandardCharsets.UTF_8))) {
      out.println("# model <TAB> features separated by spaces, \\uXXXX escaped <TAB> probabilities in outcome order");
      for (String algorithm : new String[] {"MAXENT", "MAXENT_QN", "PERCEPTRON"}) {
        TrainingParameters p = new TrainingParameters();
        p.put(Parameters.ALGORITHM_PARAM, algorithm);
        p.put(Parameters.ITERATIONS_PARAM, 100);
        p.put(Parameters.CUTOFF_PARAM, 0);
        MaxentModel model = TrainerFactory.getEventTrainer(p, new HashMap<>())
            .train(ObjectStreamUtils.createObjectStream(events()));
        String file = switch (algorithm) {
          case "MAXENT" -> "gis.model";
          case "MAXENT_QN" -> "qn.model";
          default -> "perceptron.model";
        };
        try (OutputStream os = Files.newOutputStream(dir.resolve(file));
             DataOutputStream dos = new DataOutputStream(os)) {
          new GenericModelWriter((AbstractModel) model, dos).persist();
        }
        out.print("# " + file + " outcomes:");
        for (int i = 0; i < model.getNumOutcomes(); i++) {
          out.print(" " + model.getOutcome(i));
        }
        out.println();
        for (String[] context : CONTEXTS) {
          print(out, file, context, model.eval(context));
        }
      }
      tokenizer(dir, out);
    }
  }

  /** A TokenizerME model file as OpenNLP writes it: a zip with a manifest, the model and a dictionary. */
  static void tokenizer(Path dir, PrintWriter out) throws IOException {
    String xml = "<dictionary case_sensitive=\"false\">"
        + "<entry><token>Dr.</token></entry><entry><token>e.g.</token></entry>"
        + "<entry><token>U.S.</token></entry></dictionary>";
    Dictionary abbreviations = new Dictionary(new ByteArrayInputStream(xml.getBytes(StandardCharsets.UTF_8)));
    List<TokenSample> samples = new ArrayList<>();
    for (String s : new String[] {
        "Dr. Smith<SPLIT>, e.g. here<SPLIT>.", "It<SPLIT>'s ok<SPLIT>.", "The U.S. team won<SPLIT>!",
        "Call me<SPLIT>, please<SPLIT>.", "(<SPLIT>yes<SPLIT>)", "\"<SPLIT>Hi<SPLIT>\"<SPLIT>, she said<SPLIT>."}) {
      samples.add(TokenSample.parse(s, "<SPLIT>"));
    }
    TrainingParameters p = new TrainingParameters();
    p.put(Parameters.ALGORITHM_PARAM, "MAXENT");
    p.put(Parameters.ITERATIONS_PARAM, 50);
    p.put(Parameters.CUTOFF_PARAM, 0);
    TokenizerFactory factory = new TokenizerFactory("en", abbreviations, false, Pattern.compile("^[A-Za-z0-9]+$"));
    TokenizerModel model = TokenizerME.train(ObjectStreamUtils.createObjectStream(samples), factory, p);
    try (OutputStream os = Files.newOutputStream(dir.resolve("tokenizer.bin"))) {
      model.serialize(os);
    }
    MaxentModel maxent = model.getMaxentModel();
    out.print("# tokenizer.bin token.model outcomes:");
    for (int i = 0; i < maxent.getNumOutcomes(); i++) {
      out.print(" " + maxent.getOutcome(i));
    }
    out.println();
    String[][] contexts = {
        {}, {"p=D", "s=r.", "p1=D", "p1_alpha", "s1=r", "s1_alpha"},
        {"p=Smith", "s=,", "p1=h", "s1=,", "p1s1=h_,"}, {"abb", "p=e.g", "s=."}, {"nothing-known"}};
    for (String[] context : contexts) {
      print(out, "tokenizer.bin", context, maxent.eval(context));
    }
  }

  static void print(PrintWriter out, String file, String[] context, double[] probs) {
    StringBuilder sb = new StringBuilder(file).append('\t');
    for (int i = 0; i < context.length; i++) {
      if (i > 0) {
        sb.append(' ');
      }
      for (char c : context[i].toCharArray()) {
        if (c < 0x21 || c > 0x7e || c == '\\') {
          sb.append(String.format("\\u%04X", (int) c));
        } else {
          sb.append(c);
        }
      }
    }
    sb.append('\t');
    for (int i = 0; i < probs.length; i++) {
      if (i > 0) {
        sb.append(' ');
      }
      sb.append(Double.toString(probs[i]));
    }
    out.println(sb);
  }
}
