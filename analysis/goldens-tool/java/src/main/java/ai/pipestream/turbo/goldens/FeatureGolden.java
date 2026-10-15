package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.postag.DefaultPOSContextGenerator;
import opennlp.tools.util.featuregen.AdaptiveFeatureGenerator;
import opennlp.tools.util.featuregen.GeneratorFactory;
import opennlp.tools.util.model.ArtifactSerializer;

/**
 * The features OpenNLP's feature generators give every token of the {@code sequence/sentences}
 * input: one run per descriptor of the fixtures' {@code featuregen} directory (OpenNLP's default
 * part-of-speech and name finder descriptors, one with every factory and one with nested, cached
 * and windowed generators), each built by {@code GeneratorFactory.create} with the resources it
 * names read from the same directory, and two runs of {@code DefaultPOSContextGenerator}, without
 * and with the names dictionary as its n-gram dictionary.
 *
 * <p>The sentences are one document: a descriptor's generator has its adaptive data cleared once,
 * and after each sentence {@code updateAdaptiveData} is called with the outcomes that were passed
 * as the previous outcomes ({@link Sentences#syntheticOutcomes}). The layers, anchored to the
 * document: {@code turbo:sentence-tokens} (each token with its sentence and the outcome passed for
 * it) and {@code turbo:features} (each feature, with its sentence and token, in the order the
 * generator added it; {@link TrainGolden#lossless}, as a prefix or suffix can be half a
 * surrogate pair).</p>
 */
final class FeatureGolden implements AnnotatorGolden {

  /** The fixtures directory of the descriptors and the resources they name. */
  static final String DIR = Runs.FEATUREGEN;

  /** The descriptors, in the fixtures' {@code featuregen} directory. */
  static final List<String> DESCRIPTORS = List.of("pos-default.xml", "ner-default.xml", "everything.xml",
      "nesting.xml");

  /** The dictionary the second {@code DefaultPOSContextGenerator} run uses. */
  static final String NAMES_DICTIONARY = "names.dictionary";

  private final String dir;
  private final ModelFile descriptor;
  private final List<ModelFile> resources;
  private final Dictionary ngram;

  private FeatureGolden(String dir, ModelFile descriptor, List<ModelFile> resources, Dictionary ngram) {
    this.dir = dir;
    this.descriptor = descriptor;
    this.resources = resources;
    this.ngram = ngram;
  }

  static List<FeatureGolden> all(Path fixtures) throws IOException {
    Path dir = fixtures.resolve(DIR);
    List<FeatureGolden> out = new ArrayList<>();
    for (String name : DESCRIPTORS) {
      ModelFile d = ModelFile.read(dir.resolve(name));
      List<ModelFile> resources = new ArrayList<>();
      for (String key : resourceNames(d.bytes())) {
        resources.add(ModelFile.read(dir.resolve(key)));
      }
      out.add(new FeatureGolden(d.ref().baseName(), d, resources, null));
    }
    out.add(new FeatureGolden("default-pos-context", null, List.of(), null));
    ModelFile names = ModelFile.read(dir.resolve(NAMES_DICTIONARY));
    try (InputStream in = names.stream()) {
      out.add(new FeatureGolden("default-pos-context+names-dictionary", null, List.of(names), new Dictionary(in)));
    }
    return out;
  }

  /** The resources a descriptor names, sorted. */
  static List<String> resourceNames(byte[] descriptor) throws IOException {
    return new ArrayList<>(new TreeMap<>(
        GeneratorFactory.extractArtifactSerializerMappings(new ByteArrayInputStream(descriptor))).keySet());
  }

  /**
   * The resources a descriptor names, read from {@code dir} by the serializers it gives them, as a
   * model loader would read them.
   */
  static Map<String, Object> resources(Path dir, byte[] descriptor) throws IOException {
    Map<String, Object> out = new HashMap<>();
    Map<String, ArtifactSerializer<?>> mapping =
        GeneratorFactory.extractArtifactSerializerMappings(new ByteArrayInputStream(descriptor));
    for (Map.Entry<String, ArtifactSerializer<?>> e : mapping.entrySet()) {
      try (InputStream in = Files.newInputStream(dir.resolve(e.getKey()))) {
        out.put(e.getKey(), e.getValue().create(in));
      }
    }
    return out;
  }

  @Override
  public String annotator() {
    return "features";
  }

  @Override
  public List<ModelRef> models() {
    List<ModelRef> out = new ArrayList<>();
    if (descriptor != null) {
      out.add(descriptor.ref());
    }
    for (ModelFile r : resources) {
      out.add(r.ref());
    }
    return out;
  }

  @Override
  public String modelDir() {
    return dir;
  }

  @Override
  public String annotatorClass() {
    return descriptor != null ? GeneratorFactory.class.getName() : DefaultPOSContextGenerator.class.getName();
  }

  @Override
  public String method() {
    return descriptor != null ? "create" : "getContext";
  }

  @Override
  public Map<String, String> config() {
    return descriptor != null ? Map.of("adaptiveData", "updated after each sentence") : Map.of();
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(Runs.SEQUENCE);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(Runs.SEQUENCE) && input.id().equals("sentences");
  }

  @Override
  public List<Layer> layers(Input input) {
    List<String[]> sentences = Sentences.of(input);
    Table features = new Table("turbo:features", "sentence:u32", "token:u32", "feature:str");
    try {
      if (descriptor != null) {
        Map<String, ArtifactSerializer<?>> serializers =
            GeneratorFactory.extractArtifactSerializerMappings(new ByteArrayInputStream(descriptor.bytes()));
        Map<String, Object> resources = new HashMap<>();
        for (ModelFile r : this.resources) {
          try (InputStream in = r.stream()) {
            resources.put(r.ref().fileName(), serializers.get(r.ref().fileName()).create(in));
          }
        }
        AdaptiveFeatureGenerator g = GeneratorFactory.create(new ByteArrayInputStream(descriptor.bytes()),
            resources::get);
        g.clearAdaptiveData();
        for (int k = 0; k < sentences.size(); k++) {
          String[] tokens = sentences.get(k);
          String[] outcomes = Sentences.syntheticOutcomes(k, tokens.length);
          for (int i = 0; i < tokens.length; i++) {
            List<String> f = new ArrayList<>();
            g.createFeatures(f, tokens, i, outcomes);
            add(features, k, i, f.toArray(new String[0]));
          }
          g.updateAdaptiveData(tokens, outcomes);
        }
      } else {
        DefaultPOSContextGenerator cg = new DefaultPOSContextGenerator(ngram);
        for (int k = 0; k < sentences.size(); k++) {
          String[] tokens = sentences.get(k);
          String[] outcomes = Sentences.syntheticOutcomes(k, tokens.length);
          for (int i = 0; i < tokens.length; i++) {
            add(features, k, i, cg.getContext(i, tokens, outcomes, null));
          }
        }
      }
    } catch (IOException e) {
      throw new IllegalStateException(dir + ": the descriptor", e);
    }
    return List.of(Sentences.tokens(sentences, true).layer(), features.layer());
  }

  private static void add(Table features, int sentence, int token, String[] f) {
    for (String feature : f) {
      features.row(sentence, token, TrainGolden.lossless(feature));
    }
  }
}
