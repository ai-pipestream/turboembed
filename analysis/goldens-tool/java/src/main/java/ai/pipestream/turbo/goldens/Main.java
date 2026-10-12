package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * Writes every golden for every input, and with {@code --dumps} every decision point dump too.
 *
 * <p>Usage: {@code <inputs dir> <models dir> <fixtures dir> <out dir> <opennlp commit> [--dumps]}.
 * The inputs directory holds one directory per input set, each holding {@code <id>.txt} files.
 * The models directory holds the published model files and the fixtures directory the checked in
 * ones, named in {@link Runs}. The output directory must be empty or absent; it receives
 * {@code <annotator>/<model>/<set>/<id>.golden} for each annotator run and input it runs on, and
 * with {@code --dumps} {@code <annotator>/<model>/<set>/<id>.dump} beside the golden of each run
 * that asks a maxent model, written in sorted order of their paths. The commit is the 40 hex
 * digit commit the OpenNLP runtime was built from, which the jar does not record itself. Every
 * input must be one that at least one run takes.</p>
 *
 * <p>When the OpenNLP annotator throws on an input, its golden holds the header, the
 * {@code config} records and an {@code error} record with the exception's class and message, and
 * no layers; that input has no dump. Any other exception stops the generator.</p>
 *
 * <p>A dump comes from a second instance of the annotator built over a recording model
 * ({@link AnnotatorDump}). The layers that instance produces must equal the golden byte for byte,
 * so a dump is only written when recording changed nothing the annotator computed.</p>
 */
public final class Main {

  /** The option that adds the decision point dumps. */
  static final String DUMPS_OPTION = "--dumps";

  private Main() {}

  public static void main(String[] args) throws IOException {
    boolean dumps = args.length == 6 && args[5].equals(DUMPS_OPTION);
    if (args.length != 5 && !dumps) {
      System.err.println("usage: turbo-goldens-java <inputs dir> <models dir> <fixtures dir> <out dir> <opennlp commit> ["
          + DUMPS_OPTION + "]");
      System.exit(2);
    }
    Path inputs = Path.of(args[0]);
    Path models = Path.of(args[1]);
    Path fixtures = Path.of(args[2]);
    Path out = Path.of(args[3]);
    String commit = args[4];
    if (!commit.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("the OpenNLP commit must be 40 lowercase hex digits: " + commit);
    }
    Jdk jdk = Jdk.current();
    String version = OpennlpVersion.read();
    Path snapshot =
        inputs.toAbsolutePath().resolveSibling(MoneyGolden.SNAPSHOT_DIR).resolve(MoneyGolden.SNAPSHOT_FILE);
    Runs.All runs = Runs.load(models, fixtures, snapshot);
    List<AnnotatorDump> recorders = dumps ? runs.recorders() : List.of();

    Map<Path, byte[]> files = new TreeMap<>();
    for (Input input : readInputs(inputs)) {
      if (runs.annotators().stream().noneMatch(a -> a.accepts(input))) {
        throw new IllegalStateException("no annotator runs on the input " + input.inputId());
      }
      generate(input, runs.annotators(), recorders, a -> header(commit, version, jdk, input, a), files);
    }
    write(out, files);
    System.out.println("wrote " + files.size() + " files to " + out);
  }

  /**
   * Adds to {@code files} the golden of every annotator that accepts the input and the dump of
   * every recorder whose annotator does. An annotator that throws on the input gets a golden with
   * an {@code error} record and no dump; the recording annotator must throw the same exception.
   * Any exception other than the annotator's own ({@link AnnotatorException}) is not caught.
   */
  static void generate(Input input, List<AnnotatorGolden> annotators, List<AnnotatorDump> recorders,
      Function<AnnotatorGolden, Header> headers, Map<Path, byte[]> files) {
    for (AnnotatorGolden a : annotators) {
      if (a.accepts(input)) {
        files.put(path(a, input, ".golden"), golden(headers.apply(a), a, input));
      }
    }
    for (AnnotatorDump d : recorders) {
      AnnotatorGolden a = d.golden();
      if (!a.accepts(input)) {
        continue;
      }
      Header h = headers.apply(a);
      Path golden = path(a, input, ".golden");
      AnnotatorDump.Run run;
      try {
        run = d.run(input);
      } catch (AnnotatorException e) {
        if (!Arrays.equals(files.get(golden), GoldenWriter.writeError(h, e))) {
          throw new IllegalStateException(golden + ": the annotator over the recording model threw "
              + e.exceptionClass() + " where the plain one did not throw the same", e);
        }
        continue;
      }
      if (!Arrays.equals(files.get(golden), GoldenWriter.write(h, run.layers()))) {
        throw new IllegalStateException(golden + ": the annotator over the recording model produced other layers"
            + " than the plain one");
      }
      files.put(path(a, input, ".dump"), DumpWriter.write(h, run.outcomeLabels(), run.points()));
    }
  }

  /** The golden of one annotator on one input: its layers, or the error record when it threw. */
  static byte[] golden(Header h, AnnotatorGolden a, Input input) {
    try {
      return GoldenWriter.write(h, a.layers(input));
    } catch (AnnotatorException e) {
      return GoldenWriter.writeError(h, e);
    }
  }

  private static Header header(String commit, String version, Jdk jdk, Input input, AnnotatorGolden a) {
    return new Header(commit, version, jdk, input.inputId(), input.bytes().length, input.text().length(),
        Sha.sha256(input.bytes()), a.models(), a.annotatorClass(), a.method(), a.config());
  }

  /** {@code <annotator>/<model>/<set>/<id><extension>}. */
  static Path path(AnnotatorGolden a, Input input, String extension) {
    return Path.of(a.annotator(), a.modelDir(), input.set(), input.id() + extension);
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
