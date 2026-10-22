/*
 * Licensed under the Apache License, Version 2.0.
 */

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import opennlp.dl.InferenceOptions;
import opennlp.dl.namefinder.NameFinderDL;
import opennlp.tools.sentdetect.SentenceDetector;
import opennlp.tools.util.Span;

/**
 * The names OpenNLP's NameFinderDL finds in each line of a cases file, as
 * JSON: the golden turbo-namefind-dl's parity test compares with.
 *
 * Each line is one input, its tokens separated by single spaces, so the
 * text NameFinderDL joins is the line itself. The whole line is one
 * sentence, and the inference options are the defaults (chunks of 250
 * words sharing 50, no lower casing, no folding).
 *
 * Arguments: the ONNX model, its vocab.txt, the cases file and the JSON
 * file to write.
 */
public final class NameFinderDlGolden {

  private static final Map<Integer, String> LABELS = new LinkedHashMap<>();

  static {
    String[] names = {"O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"};
    for (int i = 0; i < names.length; i++) {
      LABELS.put(i, names[i]);
    }
  }

  /** The whole text, one sentence. */
  private static final SentenceDetector WHOLE = new SentenceDetector() {
    @Override
    public String[] sentDetect(CharSequence s) {
      return new String[] {s.toString()};
    }

    @Override
    public Span[] sentPosDetect(CharSequence s) {
      return new Span[] {new Span(0, s.length())};
    }
  };

  public static void main(String[] args) throws Exception {
    List<String> lines = Files.readAllLines(Path.of(args[2]), StandardCharsets.UTF_8);
    StringBuilder out = new StringBuilder("{\"cases\": [\n");
    try (NameFinderDL finder = new NameFinderDL(new File(args[0]), new File(args[1]), LABELS,
        new InferenceOptions(), WHOLE)) {
      for (int i = 0; i < lines.size(); i++) {
        String line = lines.get(i);
        Span[] spans = finder.find(line.split(" ", -1));
        out.append("  {\"text\": ").append(quote(line)).append(", \"spans\": [");
        for (int k = 0; k < spans.length; k++) {
          Span s = spans[k];
          out.append(k == 0 ? "" : ", ")
              .append("{\"start\": ").append(s.getStart())
              .append(", \"end\": ").append(s.getEnd())
              .append(", \"type\": ").append(quote(s.getType()))
              .append(", \"prob\": ").append(s.getProb()).append('}');
        }
        out.append("]}").append(i + 1 < lines.size() ? ",\n" : "\n");
      }
    }
    out.append("]}\n");
    Files.writeString(Path.of(args[3]), out, StandardCharsets.UTF_8);
  }

  private static String quote(String s) {
    StringBuilder b = new StringBuilder("\"");
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c == '"' || c == '\\') {
        b.append('\\').append(c);
      } else if (c < 0x20) {
        b.append(String.format("\\u%04x", (int) c));
      } else {
        b.append(c);
      }
    }
    return b.append('"').toString();
  }
}
