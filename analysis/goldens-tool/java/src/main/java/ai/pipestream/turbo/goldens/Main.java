package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Stream;

/**
 * Writes every golden for every input.
 *
 * <p>Usage: {@code <inputs dir> <models dir> <out dir> <opennlp commit>}. The inputs directory
 * holds one directory per input set, each holding {@code <id>.txt} files. The models directory
 * holds the model files named below. The output directory must be empty or absent; it receives
 * {@code <annotator>/<model>/<set>/<id>.golden} for each annotator and input, written in sorted
 * order of their paths. The commit is the 40 hex digit commit the OpenNLP runtime was built from,
 * which the jar does not record itself.</p>
 */
public final class Main {

  /** The model files the generator loads from the models directory. */
  static final String SENTENCE_MODEL = "opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin";
  static final String TOKEN_MODEL = "opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin";
  static final String LANGUAGE_MODEL = "langdetect-183.bin";

  private Main() {}

  public static void main(String[] args) throws IOException {
    if (args.length != 4) {
      System.err.println("usage: turbo-goldens-java <inputs dir> <models dir> <out dir> <opennlp commit>");
      System.exit(2);
    }
    Path inputs = Path.of(args[0]);
    Path models = Path.of(args[1]);
    Path out = Path.of(args[2]);
    String commit = args[3];
    if (!commit.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("the OpenNLP commit must be 40 lowercase hex digits: " + commit);
    }
    Jdk jdk = Jdk.current();
    String version = OpennlpVersion.read();
    List<AnnotatorGolden> annotators = new ArrayList<>(List.of(
        SentenceGolden.load(models.resolve(SENTENCE_MODEL)),
        TokenGolden.load(models.resolve(TOKEN_MODEL)),
        LanguageGolden.load(models.resolve(LANGUAGE_MODEL))));
    annotators.addAll(NormalizerGolden.all());

    Map<Path, byte[]> files = new TreeMap<>();
    for (Input input : readInputs(inputs)) {
      for (AnnotatorGolden a : annotators) {
        if (!a.accepts(input)) {
          continue;
        }
        Header h = new Header(commit, version, jdk, input.inputId(), input.bytes().length,
            input.text().length(), Sha.sha256(input.bytes()), a.models(), a.annotatorClass(), a.method(),
            a.config());
        Path rel = Path.of(a.annotator(), a.modelDir(), input.set(), input.id() + ".golden");
        files.put(rel, GoldenWriter.write(h, a.layers(input)));
      }
    }
    write(out, files);
    System.out.println("wrote " + files.size() + " goldens to " + out);
  }

  /** Every {@code <set>/<id>.txt} under {@code dir}, sets and ids in sorted order. */
  static List<Input> readInputs(Path dir) throws IOException {
    List<Input> inputs = new ArrayList<>();
    for (Path set : sorted(dir)) {
      if (!Files.isDirectory(set)) {
        throw new IOException(set + ": the inputs directory holds only set directories");
      }
      for (Path file : sorted(set)) {
        String name = file.getFileName().toString();
        if (!Files.isRegularFile(file) || !name.endsWith(".txt")) {
          throw new IOException(file + ": an input set holds only <id>.txt files");
        }
        inputs.add(Input.read(file, set.getFileName().toString(), name.substring(0, name.length() - 4)));
      }
    }
    if (inputs.isEmpty()) {
      throw new IOException(dir + ": no inputs");
    }
    return inputs;
  }

  private static List<Path> sorted(Path dir) throws IOException {
    try (Stream<Path> s = Files.list(dir)) {
      return s.sorted().toList();
    }
  }

  private static void write(Path out, Map<Path, byte[]> files) throws IOException {
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.list(out)) {
        if (s.findAny().isPresent()) {
          throw new IOException(out + " is not empty");
        }
      }
    }
    for (Map.Entry<Path, byte[]> e : files.entrySet()) {
      Path target = out.resolve(e.getKey());
      Files.createDirectories(target.getParent());
      Files.write(target, e.getValue());
    }
  }
}
