package ai.pipestream.turbo.goldens;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.util.normalizer.AlignedText;
import opennlp.tools.util.normalizer.Alignment;
import opennlp.tools.util.normalizer.FullCaseFoldCharSequenceNormalizer;
import opennlp.tools.util.normalizer.GermanUmlautCharSequenceNormalizer;
import opennlp.tools.util.normalizer.WhitespaceCharSequenceNormalizer;

/** Three real normalizers, with OpenNLP's own composition back to the source. */
final class NormalizerChainGolden implements AnnotatorGolden {
  @Override public String annotator() { return "normalize"; }
  @Override public String modelDir() { return "umlaut-full-case-whitespace-chain"; }
  @Override public List<ModelRef> models() { return List.of(); }
  @Override public String annotatorClass() { return Alignment.class.getName(); }
  @Override public String method() { return "andThen"; }
  @Override public Set<String> inputSets() { return Set.of(NormalizerGolden.INPUT_SET); }
  @Override public Map<String, String> config() {
    return Map.of("steps", "GermanUmlautCharSequenceNormalizer,FullCaseFoldCharSequenceNormalizer,WhitespaceCharSequenceNormalizer");
  }

  @Override public List<Layer> layers(Input input) {
    String text = input.text();
    AlignedText first = GermanUmlautCharSequenceNormalizer.getInstance().normalizeAligned(text);
    AlignedText second = FullCaseFoldCharSequenceNormalizer.getInstance().normalizeAligned(first.normalized());
    AlignedText third = WhitespaceCharSequenceNormalizer.getInstance().normalizeAligned(second.normalized());
    Alignment combined = first.alignment().andThen(second.alignment()).andThen(third.alignment());
    List<Edit> edits = Edit.fromAlignment(text, new AlignedText(text, third.normalized(), combined));
    if (!Edit.apply(text, edits).equals(third.normalizedString())) {
      throw new IllegalStateException("composed edits do not reproduce the final text: " + input.inputId());
    }
    long[] starts = new long[edits.size()];
    long[] ends = new long[edits.size()];
    String[] with = new String[edits.size()];
    for (int i = 0; i < edits.size(); i++) {
      Edit e = edits.get(i);
      Utf8Offsets.ByteSpan span = input.offsets().span(e.start(), e.end());
      starts[i] = span.start();
      ends[i] = span.end();
      with[i] = e.with();
    }
    return List.of(new Layer("turbo:view", "text", new Column.U32("start", starts),
        new Column.U32("end", ends), new Column.Str("with", with)));
  }

  /** Standalone reproduction using the pinned generator jar, without other models. */
  public static void main(String[] args) throws Exception {
    if (args.length != 3) {
      throw new IllegalArgumentException("usage: INPUTS OUTPUT OPENNLP_COMMIT");
    }
    if (args[2].length() != 40 || !args[2].chars().allMatch(c -> c >= '0' && c <= '9' || c >= 'a' && c <= 'f')) {
      throw new IllegalArgumentException("OPENNLP_COMMIT must be 40 lowercase hex digits");
    }
    var run = new NormalizerChainGolden();
    Path out = Path.of(args[1], run.annotator(), run.modelDir(), NormalizerGolden.INPUT_SET);
    Files.createDirectories(out);
    try (var files = Files.list(Path.of(args[0], NormalizerGolden.INPUT_SET))) {
      for (Path path : files.filter(p -> p.toString().endsWith(".txt")).sorted().toList()) {
        String name = path.getFileName().toString();
        Input input = Input.read(path, NormalizerGolden.INPUT_SET, name.substring(0, name.length() - 4));
        Header header = new Header(args[2], OpennlpVersion.read(), Jdk.current(), input.inputId(),
            input.bytes().length, input.text().length(), Sha.sha256(input.bytes()), run.models(),
            run.annotatorClass(), run.method(), run.config());
        Files.write(out.resolve(input.id() + ".golden"), GoldenWriter.write(header, run.layers(input)));
      }
    }
  }
}
