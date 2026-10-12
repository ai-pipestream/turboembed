import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/** Reads the input texts: every {@code *.txt} file of a directory, decoded as UTF-8, in name order. */
final class Inputs {

  /** One input text and the file name (without extension) its golden files are named after. */
  record Input(String name, String text) {
  }

  private Inputs() {
  }

  static List<Input> load(Path dir) throws IOException {
    List<Path> files;
    try (Stream<Path> s = Files.list(dir)) {
      files = s.filter(p -> p.getFileName().toString().endsWith(".txt")).sorted().toList();
    }
    List<Input> inputs = new ArrayList<>();
    for (Path p : files) {
      String name = p.getFileName().toString();
      String text = new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
      inputs.add(new Input(name.substring(0, name.length() - ".txt".length()), text));
    }
    return inputs;
  }
}
