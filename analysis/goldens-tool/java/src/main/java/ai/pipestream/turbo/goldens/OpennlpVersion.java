package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.Properties;
import java.util.jar.JarFile;
import opennlp.tools.tokenize.TokenizerME;

/**
 * The Maven version of the OpenNLP runtime on the class path, read from the
 * {@code pom.properties} Maven puts into the runtime jar itself, so the version recorded is the
 * one of the jar that produced the golden.
 */
final class OpennlpVersion {

  private static final String POM_PROPERTIES = "META-INF/maven/org.apache.opennlp/opennlp-runtime/pom.properties";

  private OpennlpVersion() {}

  static String read() throws IOException {
    Path jar;
    try {
      jar = Path.of(TokenizerME.class.getProtectionDomain().getCodeSource().getLocation().toURI());
    } catch (URISyntaxException e) {
      throw new IOException("cannot locate the OpenNLP runtime jar", e);
    }
    try (JarFile jf = new JarFile(jar.toFile())) {
      var entry = jf.getEntry(POM_PROPERTIES);
      if (entry == null) {
        throw new IOException(jar + " has no " + POM_PROPERTIES);
      }
      Properties p = new Properties();
      try (InputStream in = jf.getInputStream(entry)) {
        p.load(in);
      }
      String version = p.getProperty("version");
      if (version == null || version.isEmpty()) {
        throw new IOException(jar + ": " + POM_PROPERTIES + " has no version");
      }
      return version;
    }
  }
}
