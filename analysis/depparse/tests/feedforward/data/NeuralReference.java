package opennlp.tools.depparse;

import java.nio.file.Files;
import java.nio.file.Path;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import opennlp.tools.util.ObjectStreamUtils;

/** Executable fixture generator against the unmodified pinned Java sources. */
public final class NeuralReference {
  static final String[] WORDS = {"I", "see", "birds"};
  static final String[] TAGS = {"PRON", "VERB", "NOUN"};
  static final DependencyGraph GOLD = DependencyGraph.of(new int[] {1, -1, 1},
      new String[] {"nsubj", "root", "obj"});
  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    List<DependencySample> corpus = List.of(new DependencySample(WORDS, TAGS, GOLD),
        new DependencySample(new String[] {"birds", "fly"}, new String[] {"NOUN", "VERB"},
          DependencyGraph.of(new int[] {1, -1}, new String[] {"nsubj", "root"})));
    for (boolean dropout : List.of(false, true)) {
      String name = dropout ? "dropout" : "plain";
      var settings = new FeedforwardDependencyTrainer.Settings(3, 5, 3, 4, 0.02,
          dropout ? 0.001 : 0, dropout ? 0.3 : 0, 1, 17);
      var model = FeedforwardDependencyTrainer.train(ObjectStreamUtils.createObjectStream(corpus), settings);
      try (OutputStream stream = Files.newOutputStream(out.resolve(name + ".bin"))) {
        model.serialize(stream);
      }
      List<String> rows = new ArrayList<>();
      ArcStandardState state = new ArcStandardState(WORDS.length);
      for (Transition transition : ArcStandardOracle.transitions(GOLD)) {
        String[] features = FeedforwardContext.extract(state, WORDS, TAGS);
        rows.add("C\t" + String.join("\t", Arrays.stream(features).map(s -> s == null ? "<NULL>" : s).toList()));
        int[] ids = model.featureIds(features);
        rows.add("I\t" + Arrays.toString(ids));
        rows.add("S\t" + Arrays.toString(model.score(ids)));
        state.apply(transition);
      }
      for (int beam : List.of(1, 4)) {
        var graph = new FeedforwardDependencyParser(model, beam).parse(WORDS, TAGS);
        for (int i = 0; i < graph.size(); i++) {
          rows.add("G\t" + beam + "\t" + i + "\t" + graph.headOf(i) + "\t" + graph.relationOf(i));
        }
      }
      Files.write(out.resolve(name + ".tsv"), rows);
    }
    List<String> normalizations = new ArrayList<>();
    for (String word : List.of("İ", "ΟΔΟΣ", "Σ", "中Σ", "ΟΣ’Α", "Ο’Σ", "*ROOT*", "😀ABC")) {
      normalizations.add(word + "\t" + FeedforwardDependencyModel.normalize(word));
    }
    Files.write(out.resolve("normalize.tsv"), normalizations);
  }
}
