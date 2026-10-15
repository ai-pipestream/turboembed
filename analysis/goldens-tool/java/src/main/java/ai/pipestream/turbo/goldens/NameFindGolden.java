package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.NameSampleDataStream;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;

/**
 * {@link NameFinderME} over a model file the Rust trainer wrote, so that the file is shown to load
 * in the JVM and the Rust name finder can be compared with what the JVM finds with it.
 *
 * <p>The input is read with {@link NameSampleDataStream}; each sample's tokens are given to
 * {@code find} in order, after {@code clearAdaptiveData} when the sample follows an empty line,
 * as {@code TokenNameFinderEvaluator} runs a name finder. A line the stream cannot read is
 * skipped.</p>
 *
 * <p>Layers, all anchored on the document: {@code turbo:name-found}: each name {@code find}
 * returned, the index of its sentence among the samples read, its start and end token, its type
 * and its probability; {@code turbo:name-probs}: the probability of every outcome of each
 * sentence's best sequence ({@code probs()}), by sentence and token.</p>
 */
final class NameFindGolden implements AnnotatorGolden {

  private final ModelFile file;
  private final TokenNameFinderModel model;
  private final Set<String> inputSets;

  private NameFindGolden(ModelFile file, Set<String> inputSets) throws IOException {
    this.file = file;
    this.model = new TokenNameFinderModel(file.stream());
    this.inputSets = Set.copyOf(inputSets);
  }

  static NameFindGolden load(Path fixture, Set<String> inputSets) throws IOException {
    return new NameFindGolden(ModelFile.read(fixture), inputSets);
  }

  @Override
  public String annotator() {
    return NameEventGolden.ANNOTATOR;
  }

  @Override
  public List<ModelRef> models() {
    return List.of(file.ref());
  }

  @Override
  public String annotatorClass() {
    return "opennlp.tools.namefind.NameFinderME";
  }

  @Override
  public String method() {
    return "find";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> c = new TreeMap<>();
    c.put("run", "find");
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public boolean accepts(Input input) {
    return inputSets.contains(input.set()) && input.id().startsWith("name");
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    NameFinderME finder = new NameFinderME(model);
    InputStreamFactory in = () -> new ByteArrayInputStream(input.bytes());
    List<Long> sentence = new ArrayList<>();
    List<Long> start = new ArrayList<>();
    List<Long> end = new ArrayList<>();
    List<String> type = new ArrayList<>();
    List<Double> prob = new ArrayList<>();
    List<Long> probSentence = new ArrayList<>();
    List<Long> probToken = new ArrayList<>();
    List<Double> probs = new ArrayList<>();
    long index = 0;
    try (ObjectStream<NameSample> samples =
        new NameSampleDataStream(new PlainTextByLineStream(in, StandardCharsets.UTF_8))) {
      for (int reads = 0; reads < 100_000; reads++) {
        NameSample s;
        try {
          s = samples.read();
        } catch (IOException | RuntimeException e) {
          continue;
        }
        if (s == null) {
          break;
        }
        if (s.isClearAdaptiveDataSet()) {
          finder.clearAdaptiveData();
        }
        Span[] found = finder.find(s.getSentence());
        for (Span span : found) {
          sentence.add(index);
          start.add((long) span.getStart());
          end.add((long) span.getEnd());
          type.add(span.getType() == null ? "" : span.getType());
          prob.add(span.getProb());
        }
        double[] p = finder.probs();
        for (int t = 0; t < p.length; t++) {
          probSentence.add(index);
          probToken.add((long) t);
          probs.add(p[t]);
        }
        index++;
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the stream failed", e);
    }
    List<Layer> layers = new ArrayList<>();
    layers.add(new Layer("turbo:name-found", "document", new Column.U32("sentence", longs(sentence)),
        new Column.U32("start", longs(start)), new Column.U32("end", longs(end)),
        new Column.Str("type", type.toArray(new String[0])), new Column.F64("prob", doubles(prob))));
    layers.add(new Layer("turbo:name-probs", "document", new Column.U32("sentence", longs(probSentence)),
        new Column.U32("token", longs(probToken)), new Column.F64("prob", doubles(probs))));
    return layers;
  }

  private static long[] longs(List<Long> values) {
    return values.stream().mapToLong(Long::longValue).toArray();
  }

  private static double[] doubles(List<Double> values) {
    return values.stream().mapToDouble(Double::doubleValue).toArray();
  }
}
