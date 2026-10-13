package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A file an annotator loads, a model or a dictionary, read once: the reference a golden's
 * {@code model} record writes (its name and the SHA-512 of the bytes) and the bytes themselves, so
 * the hash is of exactly what was loaded.
 */
record ModelFile(ModelRef ref, byte[] bytes) {

  static ModelFile read(Path file) throws IOException {
    byte[] bytes = Files.readAllBytes(file);
    return new ModelFile(new ModelRef(file.getFileName().toString(), Sha.sha512(bytes)), bytes);
  }

  /** A stream over the bytes, for the OpenNLP constructors that read one. */
  InputStream stream() {
    return new ByteArrayInputStream(bytes);
  }
}
