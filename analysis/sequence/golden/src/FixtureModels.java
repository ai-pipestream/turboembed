import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import golden.StoredClusters;
import opennlp.tools.chunker.ChunkSampleStream;
import opennlp.tools.chunker.ChunkerFactory;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.namefind.BilouCodec;
import opennlp.tools.namefind.BioCodec;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSampleDataStream;
import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.postag.POSDictionary;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerFactory;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.postag.WordTagSampleStream;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.model.BaseModel;

/**
 * Trains the small models in {@code models/} from {@code inputs/train-*.txt}, so the beam search
 * tests have models that are always present. They are trained on a dozen sentences and are good
 * for nothing but tests.
 *
 * <ul>
 *   <li>{@code pos-fixture.bin}: GIS maxent, the default POS descriptor stored in the model, and
 *       {@code resources/tags.tagdict} as the tag dictionary.</li>
 *   <li>{@code chunker-fixture.bin}: perceptron.</li>
 *   <li>{@code ner-bio-fixture.bin}: perceptron, {@code descriptors/everything.xml} with its
 *       resources, BIO codec.</li>
 *   <li>{@code ner-bilou-fixture.bin}: GIS maxent, {@code descriptors/nesting.xml}, BILOU
 *       codec.</li>
 *   <li>{@code ner-qn-fixture.bin}: QN maxent, no descriptor given (OpenNLP stores its default
 *       one in the model), BIO codec. Inside a beam search a QN model's sums start from the
 *       previous evaluation's probabilities (see the crate README).</li>
 * </ul>
 *
 * <p>The cluster resources of the NER models are stored through {@link StoredClusters}, as
 * OpenNLP's own serializers cannot write them into a model at commit cdf33c6.
 *
 * <p>Training runs on one thread, so the models are the same on every run of the same JDK and
 * OpenNLP; the tests read them as they are and do not depend on that.
 */
public final class FixtureModels {

  private FixtureModels() {
  }

  private static TrainingParameters params(String algorithm, int iterations) {
    TrainingParameters p = new TrainingParameters();
    p.put("Algorithm", algorithm);
    p.put("Iterations", iterations);
    p.put("Cutoff", 1);
    p.put("Threads", 1);
    return p;
  }

  private static ObjectStream<String> lines(Path file) throws IOException {
    return new PlainTextByLineStream(() -> Files.newInputStream(file), StandardCharsets.UTF_8);
  }

  private static void write(BaseModel model, Path file) throws IOException {
    Files.createDirectories(file.getParent());
    try (OutputStream out = Files.newOutputStream(file)) {
      model.serialize(out);
    }
  }

  public static void main(String[] args) throws IOException {
    Path golden = Path.of(args[0]);
    StoredClusters.register();
    Path models = golden.resolve("models");
    Path inputs = golden.resolve("inputs");

    byte[] posDescriptor = Files.readAllBytes(golden.resolve("descriptors/pos-default.xml"));
    POSDictionary tags;
    try (InputStream in = Files.newInputStream(golden.resolve("resources/tags.tagdict"))) {
      tags = POSDictionary.create(in);
    }
    POSModel pos = POSTaggerME.train("en", new WordTagSampleStream(lines(inputs.resolve("train-pos.txt"))),
        params("MAXENT", 50), new POSTaggerFactory(posDescriptor, Map.of(), tags));
    write(pos, models.resolve("pos-fixture.bin"));

    ChunkerModel chunker = ChunkerME.train("en",
        new ChunkSampleStream(lines(inputs.resolve("train-chunk.txt"))),
        params("PERCEPTRON", 50), new ChunkerFactory());
    write(chunker, models.resolve("chunker-fixture.bin"));

    byte[] everything = Files.readAllBytes(golden.resolve("descriptors/everything.xml"));
    TokenNameFinderModel bio = NameFinderME.train("en", null,
        new NameSampleDataStream(lines(inputs.resolve("train-names.txt"))),
        params("PERCEPTRON", 50),
        new TokenNameFinderFactory(everything,
            StoredClusters.storable(golden, FeatureGolden.resources(golden, everything)),
            new BioCodec()));
    write(bio, models.resolve("ner-bio-fixture.bin"));

    byte[] nesting = Files.readAllBytes(golden.resolve("descriptors/nesting.xml"));
    TokenNameFinderModel bilou = NameFinderME.train("en", null,
        new NameSampleDataStream(lines(inputs.resolve("train-names.txt"))),
        params("MAXENT", 50),
        new TokenNameFinderFactory(nesting,
            StoredClusters.storable(golden, FeatureGolden.resources(golden, nesting)),
            new BilouCodec()));
    write(bilou, models.resolve("ner-bilou-fixture.bin"));

    TokenNameFinderModel qn = NameFinderME.train("en", null,
        new NameSampleDataStream(lines(inputs.resolve("train-names.txt"))),
        params("MAXENT_QN", 50),
        new TokenNameFinderFactory(null, Map.of(), new BioCodec()));
    write(qn, models.resolve("ner-qn-fixture.bin"));
  }
}
