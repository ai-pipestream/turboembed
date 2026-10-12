package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * One input text: its {@code <set>/<id>}, the file's bytes, the text they decode to and the
 * table from UTF-16 indices to byte offsets. The bytes must be valid UTF-8; the file's bytes are
 * the text, with no byte order mark or line end handling.
 */
record Input(String set, String id, byte[] bytes, String text, Utf8Offsets offsets) {

  /** Reads {@code <inputs>/<set>/<id>.txt}. */
  static Input read(Path file, String set, String id) throws IOException {
    byte[] bytes = Files.readAllBytes(file);
    String text;
    try {
      text = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
          .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes)).toString();
    } catch (CharacterCodingException e) {
      throw new IOException(file + ": not valid UTF-8", e);
    }
    String inputId = set + "/" + id;
    Utf8Offsets offsets = new Utf8Offsets(inputId, text);
    if (offsets.utf8Length() != bytes.length) {
      throw new IllegalStateException(inputId + ": decoded text is not the file's bytes");
    }
    return new Input(set, id, bytes, text, offsets);
  }

  /** The input's id as a golden's {@code input} line writes it. */
  String inputId() {
    return set + "/" + id;
  }
}
