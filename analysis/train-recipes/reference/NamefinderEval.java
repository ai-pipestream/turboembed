import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import opennlp.tools.formats.Conll02NameSampleStream;
import opennlp.tools.formats.Conll03NameSampleStream;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;

/** Local-only reference producer for docs/namefinder-evaluation.md. */
public final class NamefinderEval {
  private NamefinderEval() {}

  private static String quote(String s) {
    if (s == null) return "null";
    StringBuilder b = new StringBuilder("\"");
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c == '\\' || c == '"') b.append('\\').append(c);
      else if (c < 32 || c == '\u2014') b.append(String.format("\\u%04x", (int) c));
      else b.append(c);
    }
    return b.append('"').toString();
  }

  private static String sha(byte[] bytes) throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
  }

  private static int types(String name) {
    int per = Conll02NameSampleStream.GENERATE_PERSON_ENTITIES;
    int org = Conll02NameSampleStream.GENERATE_ORGANIZATION_ENTITIES;
    int loc = Conll02NameSampleStream.GENERATE_LOCATION_ENTITIES;
    int misc = Conll02NameSampleStream.GENERATE_MISC_ENTITIES;
    return switch (name) {
      case "per" -> per;
      case "org" -> org;
      case "loc" -> loc;
      case "misc" -> misc;
      case "all" -> per | org | loc | misc;
      default -> throw new IllegalArgumentException("types must be all, per, org, loc or misc");
    };
  }

  private static String mention(Span span) {
    return "{\"start\":" + span.getStart() + ",\"end\":" + span.getEnd()
        + ",\"label\":" + quote(span.getType()) + "}";
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 4 || args[3].isBlank()) {
      throw new IllegalArgumentException("usage: NamefinderEval MODEL CORPUS TYPES REFERENCE_REVISION");
    }
    int types = types(args[2]);
    byte[] modelBytes = Files.readAllBytes(Path.of(args[0]));
    byte[] corpusBytes = Files.readAllBytes(Path.of(args[1]));
    // Reject malformed UTF-8 rather than Java's replacement decoding.
    StandardCharsets.UTF_8.newDecoder().decode(java.nio.ByteBuffer.wrap(corpusBytes));
    NameFinderME finder = new NameFinderME(new TokenNameFinderModel(new ByteArrayInputStream(modelBytes)));
    StringBuilder out = new StringBuilder("{\"format_version\":1,\"engine\":")
        .append(quote("opennlp:" + args[3]))
        .append(",\"model_sha256\":").append(quote(sha(modelBytes)))
        .append(",\"corpus_sha256\":").append(quote(sha(corpusBytes)))
        .append(",\"types\":").append(quote(args[2])).append(",\"samples\":[");
    int count = 0;
    try (var lines = new PlainTextByLineStream(() -> new ByteArrayInputStream(corpusBytes), StandardCharsets.UTF_8);
         var samples = new Conll03NameSampleStream(Conll03NameSampleStream.LANGUAGE.EN, lines, types)) {
      NameSample sample;
      while ((sample = samples.read()) != null) {
        if (sample.isClearAdaptiveDataSet()) finder.clearAdaptiveData();
        Span[] predicted = finder.find(sample.getSentence());
        double[] probabilities = finder.probs(predicted);
        if (count++ != 0) out.append(',');
        out.append("{\"tokens\":").append(sample.getSentence().length)
            .append(",\"clear_adaptive_data\":").append(sample.isClearAdaptiveDataSet())
            .append(",\"gold\":[");
        Span[] gold = sample.getNames();
        for (int i = 0; i < gold.length; i++) {
          if (i != 0) out.append(',');
          out.append(mention(gold[i]));
        }
        out.append("],\"predicted\":[");
        for (int i = 0; i < predicted.length; i++) {
          if (!Double.isFinite(probabilities[i]) || probabilities[i] < 0 || probabilities[i] > 1) {
            throw new IllegalStateException("invalid name probability");
          }
          if (i != 0) out.append(',');
          out.append("{\"mention\":").append(mention(predicted[i]))
              .append(",\"probability\":").append(probabilities[i]).append('}');
        }
        out.append("]}");
      }
    }
    if (count == 0) throw new IllegalArgumentException("empty corpus");
    System.out.println(out.append("]}"));
  }
}
