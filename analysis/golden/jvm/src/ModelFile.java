import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Properties;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/** A model file in the models directory: its name, SHA-256 and the manifest values golden files record. */
record ModelFile(Path path, String sha256, String component, String language) {

  static ModelFile open(Path path) throws IOException {
    Properties manifest = new Properties();
    try (ZipFile zip = new ZipFile(path.toFile())) {
      ZipEntry entry = zip.getEntry("manifest.properties");
      if (entry == null) {
        throw new IOException(path.getFileName() + " has no manifest.properties");
      }
      try (var in = zip.getInputStream(entry)) {
        manifest.load(in);
      }
    }
    return new ModelFile(path, sha256(path), manifest.getProperty("Component-Name", ""),
        manifest.getProperty("Language", ""));
  }

  String fileName() {
    return path.getFileName().toString();
  }

  private static String sha256(Path path) throws IOException {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      return HexFormat.of().formatHex(digest.digest(Files.readAllBytes(path)));
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException(e);
    }
  }
}
