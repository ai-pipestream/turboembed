import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.postag.DefaultPOSContextGenerator;
import opennlp.tools.util.featuregen.AdaptiveFeatureGenerator;
import opennlp.tools.util.featuregen.GeneratorFactory;
import opennlp.tools.util.model.ArtifactSerializer;

/**
 * Writes the features each generator of each descriptor in {@code descriptors/} gives every token
 * of {@code inputs/sentences.txt}, and those of {@code DefaultPOSContextGenerator} with the names
 * dictionary, and the exceptions {@code GeneratorFactory} throws for malformed descriptors.
 *
 * <p>The sentences are one document: the generators' adaptive data is cleared once, and after each
 * sentence {@code updateAdaptiveData} is called with the outcomes that were passed as the previous
 * outcomes ({@link Common#syntheticOutcomes}).
 */
public final class FeatureGolden {

  private FeatureGolden() {
  }

  /** Reads the resources a descriptor names from {@code resources/}, as a model loader would. */
  static Map<String, Object> resources(Path golden, byte[] descriptor) throws IOException {
    Map<String, Object> out = new HashMap<>();
    Map<String, ArtifactSerializer<?>> mapping =
        GeneratorFactory.extractArtifactSerializerMappings(new ByteArrayInputStream(descriptor));
    for (Map.Entry<String, ArtifactSerializer<?>> e : mapping.entrySet()) {
      try (InputStream in = Files.newInputStream(golden.resolve("resources").resolve(e.getKey()))) {
        out.put(e.getKey(), e.getValue().create(in));
      }
    }
    return out;
  }

  public static void main(String[] args) throws IOException {
    Path golden = Path.of(args[0]);
    Path out = Path.of(args[1]);
    String commit = args[2];
    List<String[]> sentences = Common.sentences(golden.resolve("inputs/sentences.txt"));

    List<Path> descriptors;
    try (Stream<Path> s = Files.list(golden.resolve("descriptors"))) {
      descriptors = s.filter(p -> p.toString().endsWith(".xml")).sorted().toList();
    }
    for (Path descriptor : descriptors) {
      byte[] bytes = Files.readAllBytes(descriptor);
      Map<String, Object> resources = resources(golden, bytes);
      AdaptiveFeatureGenerator g =
          GeneratorFactory.create(new ByteArrayInputStream(bytes), resources::get);
      String name = descriptor.getFileName().toString().replace(".xml", "");
      try (Writer w = Common.writer(out.resolve("features-" + name + ".txt"))) {
        Common.header(w, "features", commit);
        Common.line(w, "descriptor: " + descriptor.getFileName());
        g.clearAdaptiveData();
        for (int k = 0; k < sentences.size(); k++) {
          // A new array per sentence: Java's generators tell sentences apart by identity.
          String[] tokens = sentences.get(k).clone();
          String[] outcomes = Common.syntheticOutcomes(k, tokens.length);
          Common.line(w, "sentence: " + k);
          Common.line(w, "tokens: " + Common.join(tokens));
          Common.line(w, "outcomes: " + Common.join(outcomes));
          for (int i = 0; i < tokens.length; i++) {
            List<String> features = new ArrayList<>();
            g.createFeatures(features, tokens, i, outcomes);
            Common.line(w, "at: " + i);
            Common.line(w, "features: " + Common.join(features));
          }
          g.updateAdaptiveData(tokens, outcomes);
          Common.line(w, "update");
        }
      }
    }

    Dictionary names;
    try (InputStream in = Files.newInputStream(golden.resolve("resources/names.dictionary"))) {
      names = new Dictionary(in);
    }
    for (boolean withDictionary : new boolean[] {false, true}) {
      DefaultPOSContextGenerator cg = new DefaultPOSContextGenerator(withDictionary ? names : null);
      String file = withDictionary ? "features-default-pos-dictionary.txt" : "features-default-pos.txt";
      try (Writer w = Common.writer(out.resolve(file))) {
        Common.header(w, "features", commit);
        Common.line(w, "descriptor: DefaultPOSContextGenerator"
            + (withDictionary ? " names.dictionary" : ""));
        for (int k = 0; k < sentences.size(); k++) {
          String[] tokens = sentences.get(k).clone();
          String[] outcomes = Common.syntheticOutcomes(k, tokens.length);
          Common.line(w, "sentence: " + k);
          Common.line(w, "tokens: " + Common.join(tokens));
          Common.line(w, "outcomes: " + Common.join(outcomes));
          for (int i = 0; i < tokens.length; i++) {
            Common.line(w, "at: " + i);
            Common.line(w, "features: " + Common.join(cg.getContext(i, tokens, outcomes, null)));
          }
        }
      }
    }

    writeErrors(out.resolve("descriptor-errors.txt"), commit);
  }

  private static final String P = "opennlp.tools.util.featuregen.";

  /** Malformed descriptors and what GeneratorFactory.create throws for each. */
  private static void writeErrors(Path file, String commit) throws IOException {
    String token = "<generator class=\"" + P + "TokenFeatureGeneratorFactory\"/>";
    String[][] cases = {
        {"classic-root", "<generators>" + token + "</generators>"},
        {"unexpected-child", "<featureGenerators><int name=\"a\">1</int></featureGenerators>"},
        {"no-generators", "<featureGenerators/>"},
        {"blank-class", "<featureGenerators><generator class=\" \"/></featureGenerators>"},
        {"window-without-generator", "<featureGenerators><generator class=\"" + P
            + "WindowFeatureGeneratorFactory\"><int name=\"prevLength\">1</int>"
            + "<int name=\"nextLength\">1</int></generator></featureGenerators>"},
        {"window-without-length", "<featureGenerators><generator class=\"" + P
            + "WindowFeatureGeneratorFactory\"><int name=\"prevLength\">1</int>" + token
            + "</generator></featureGenerators>"},
        {"length-of-wrong-type", "<featureGenerators><generator class=\"" + P
            + "PrefixFeatureGeneratorFactory\"><str name=\"length\">2</str></generator>"
            + "</featureGenerators>"},
        {"bool-of-wrong-type", "<featureGenerators><generator class=\"" + P
            + "TokenFeatureGeneratorFactory\"><int name=\"lowercase\">1</int></generator>"
            + "</featureGenerators>"},
        {"str-of-wrong-type", "<featureGenerators><generator class=\"" + P
            + "BrownClusterTokenFeatureGeneratorFactory\"><int name=\"dict\">1</int></generator>"
            + "</featureGenerators>"},
        {"unknown-parameter-type", "<featureGenerators><generator class=\"" + P
            + "TokenFeatureGeneratorFactory\"><char name=\"x\">a</char></generator>"
            + "</featureGenerators>"},
        {"cached-alone", "<featureGenerators><generator class=\"" + P
            + "CachedFeatureGeneratorFactory\">" + token + "</generator></featureGenerators>"},
        {"cached-without-generator", "<featureGenerators><generator class=\"" + P
            + "CachedFeatureGeneratorFactory\"/>" + token + "</featureGenerators>"},
        {"missing-brown-resource", "<featureGenerators><generator class=\"" + P
            + "BrownClusterTokenFeatureGeneratorFactory\"><str name=\"dict\">nothing</str>"
            + "</generator></featureGenerators>"},
        {"missing-dictionary", "<featureGenerators><generator class=\"" + P
            + "DictionaryFeatureGeneratorFactory\"><str name=\"dict\">nothing</str>"
            + "</generator></featureGenerators>"},
        {"missing-word-cluster-case", "<featureGenerators><generator class=\"" + P
            + "WordClusterFeatureGeneratorFactory\"><str name=\"dict\">nothing</str>"
            + "</generator></featureGenerators>"},
        {"ngram-without-max", "<featureGenerators><generator class=\"" + P
            + "CharacterNgramFeatureGeneratorFactory\"><int name=\"min\">2</int>"
            + "</generator></featureGenerators>"},
    };
    try (Writer w = Common.writer(file)) {
      Common.header(w, "descriptor-errors", commit);
      for (String[] c : cases) {
        Common.line(w, "case: " + c[0]);
        Common.line(w, "xml: " + Common.escape(c[1]));
        String result;
        try {
          AdaptiveFeatureGenerator g = GeneratorFactory.create(
              new ByteArrayInputStream(c[1].getBytes(StandardCharsets.UTF_8)),
              key -> null);
          result = "ok " + g.getClass().getSimpleName();
        } catch (Exception e) {
          result = e.getClass().getSimpleName() + ": " + e.getMessage();
        }
        Common.line(w, "result: " + Common.escape(result));
      }
    }
  }
}
