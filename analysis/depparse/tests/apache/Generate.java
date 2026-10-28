package opennlp.tools.depparse;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import opennlp.tools.formats.conllu.*;
import opennlp.tools.util.*;

/** Actual Apache trainer output on the attributed, checked-in UD excerpts. */
public final class Generate {
  static List<DependencySample> samples(String... paths) throws Exception {
    List<DependencySample> out = new ArrayList<>();
    for (String path : paths) {
      try (var stream = new ConlluDependencySampleStream(
          new MarkableFileInputStreamFactory(new File(path)), ConlluTagset.U)) {
        DependencySample sample;
        while ((sample = stream.read()) != null) out.add(sample);
      }
    }
    return out;
  }
  static ObjectStream<DependencySample> stream(List<DependencySample> samples) {
    return new ObjectStream<>() {
      int index;
      public DependencySample read() { return index < samples.size() ? samples.get(index++) : null; }
      public void reset() { index = 0; }
      public void close() {}
    };
  }
  static void write(FeedforwardDependencyModel model, List<DependencySample> samples,
      Path directory, String name) throws Exception {
    model.serialize(directory.resolve(name + ".bin"));
    try (var out = Files.newBufferedWriter(directory.resolve(name + ".tsv"))) {
      int sentence = 0;
      for (var sample : samples) {
        for (int beam : new int[] {1, 4}) {
          var tree = new FeedforwardDependencyParser(model, beam).parse(sample.getTokens(), sample.getTags());
          for (int i = 0; i < tree.size(); i++) {
            out.write(beam + "\t" + sentence + "\t" + i + "\t" + tree.headOf(i)
                + "\t" + tree.relationOf(i) + "\n");
          }
        }
        sentence++;
      }
    }
    System.out.println(name + " defaultBeam=" + model.getBeamSize() + " sentences=" + samples.size());
  }
  public static void main(String[] args) throws Exception {
    var data = samples(args[1], args[2]);
    Path directory = Path.of(args[0]); Files.createDirectories(directory);
    var linearParams = TrainingParameters.defaultParams();
    linearParams.put("Iterations", 5); linearParams.put("Cutoff", 1); linearParams.put("Threads", 1);
    var linear = DependencyParserME.train("und", stream(data), linearParams);
    linear.serialize(directory.resolve("linear.bin"));
    try (var out = Files.newBufferedWriter(directory.resolve("linear.tsv"))) {
      int sentence = 0;
      var parser = new DependencyParserME(linear);
      for (var sample : data) {
        var tree = parser.parse(sample.getTokens(), sample.getTags());
        for (int i = 0; i < tree.size(); i++) {
          out.write(sentence + "\t" + i + "\t" + tree.headOf(i) + "\t" + tree.relationOf(i) + "\n");
        }
        sentence++;
      }
    }
    var p = new TrainingParameters();
    p.put("EmbeddingSize", 3); p.put("HiddenSize", 5); p.put("Iterations", 3);
    p.put("BatchSize", 4); p.put("Cutoff", 1); p.put("Dropout", 0.0); p.put("Seed", 17);
    var model = FeedforwardDependencyTrainer.train("und", stream(data), p);
    write(model, data, directory, "apache");
    p.put("Iterations", 1); p.put("BeamSize", 4);
    var refined = FeedforwardDependencyTrainer.refine(model, stream(data), p);
    write(refined, data, directory, "apache-refined");
  }
}
