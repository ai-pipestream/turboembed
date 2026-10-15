package golden;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import opennlp.tools.util.featuregen.BrownCluster;
import opennlp.tools.util.featuregen.WordClusterDictionary;
import opennlp.tools.util.model.ArtifactSerializer;

/**
 * Cluster resources that can be stored in a model.
 *
 * <p>At OpenNLP commit cdf33c6 {@code BrownCluster.serialize} and
 * {@code WordClusterDictionary.serialize} close the stream they are given, which is the model's
 * zip stream, so {@code BaseModel.serialize} fails on the next entry and no model with either
 * resource can be written. These subclasses keep the bytes they were read from and write them
 * back unchanged without closing the stream. A model that names them in its manifest needs these
 * classes on the class path to be loaded by OpenNLP; turbo-sequence reads the entries by the kind
 * the descriptor gives them and ignores the serializer names. The package must be registered
 * with {@code ExtensionLoader.registerAllowedPackage} (see {@link #register}) before such a model
 * is written or read.
 */
public final class StoredClusters {

  private StoredClusters() {
  }

  /** Allows OpenNLP to load the serializers of this package. */
  public static void register() {
    opennlp.tools.util.ext.ExtensionLoader.registerAllowedPackage("golden");
  }

  static byte[] readAll(InputStream in) throws IOException {
    return in.readAllBytes();
  }

  /** A {@link BrownCluster} that is written back as the bytes it was read from. */
  public static final class Brown extends BrownCluster {
    private final byte[] bytes;

    public Brown(byte[] bytes) throws IOException {
      super(new ByteArrayInputStream(bytes));
      this.bytes = bytes;
    }

    @Override
    public void serialize(OutputStream out) throws IOException {
      out.write(bytes);
    }

    @Override
    public Class<?> getArtifactSerializerClass() {
      return BrownSerializer.class;
    }
  }

  public static final class BrownSerializer implements ArtifactSerializer<Brown> {
    @Override
    public Brown create(InputStream in) throws IOException {
      return new Brown(readAll(in));
    }

    @Override
    public void serialize(Brown artifact, OutputStream out) throws IOException {
      artifact.serialize(out);
    }
  }

  /** A {@link WordClusterDictionary} that is written back as the bytes it was read from. */
  public static final class WordClusters extends WordClusterDictionary {
    private final byte[] bytes;

    public WordClusters(byte[] bytes) throws IOException {
      super(new ByteArrayInputStream(bytes));
      this.bytes = bytes;
    }

    @Override
    public void serialize(OutputStream out) throws IOException {
      out.write(bytes);
    }

    @Override
    public Class<?> getArtifactSerializerClass() {
      return WordClustersSerializer.class;
    }
  }

  public static final class WordClustersSerializer implements ArtifactSerializer<WordClusters> {
    @Override
    public WordClusters create(InputStream in) throws IOException {
      return new WordClusters(readAll(in));
    }

    @Override
    public void serialize(WordClusters artifact, OutputStream out) throws IOException {
      artifact.serialize(out);
    }
  }

  /**
   * Replaces the cluster resources of a resource map with storable ones read from the same bytes,
   * the files of {@code dir} the map's keys name.
   */
  public static Map<String, Object> storable(Path dir, Map<String, Object> resources)
      throws IOException {
    Map<String, Object> out = new HashMap<>(resources);
    for (Map.Entry<String, Object> e : resources.entrySet()) {
      byte[] bytes = Files.readAllBytes(dir.resolve(e.getKey()));
      if (e.getValue() instanceof BrownCluster) {
        out.put(e.getKey(), new Brown(bytes));
      } else if (e.getValue() instanceof WordClusterDictionary) {
        out.put(e.getKey(), new WordClusters(bytes));
      }
    }
    return out;
  }
}
