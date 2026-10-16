package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * Writes every golden for every input, and with {@code --dumps} every decision point dump too.
 *
 * <p>Usage: {@code <inputs dir> <models dir> <fixtures dir> <out dir> <opennlp commit> [--dumps]
 * [--skip-annotators <a>,<b>,...] [--only-runs <annotator>/<model>,...]}.
 * The inputs directory holds one directory per input set, each holding {@code <id>.txt} files.
 * The models directory holds the published model files and the fixtures directory the checked in
 * ones, named in {@link Runs}. The output directory must be empty or absent; it receives
 * {@code <annotator>/<model>/<set>/<id>.golden} for each annotator run and input it runs on, and
 * with {@code --dumps} {@code <annotator>/<model>/<set>/<id>.dump} beside the golden of each run
 * that asks a maxent model, written in sorted order of their paths. The commit is the 40 hex
 * digit commit the OpenNLP runtime was built from, which the jar does not record itself. Every
 * input must be one that at least one run takes.</p>
 *
 * <p>{@code --skip-annotators} leaves out every run whose annotator directory is named, and
 * {@code --only-runs} keeps only the runs named by annotator and model directory; the differential
 * fuzzing driver uses both to run a subset of the runs over its generated inputs. The run
 * definitions themselves are not changed, and a name that matches no run is an error.</p>
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

  /** The option that leaves out every run of the named annotator directories. */
  static final String SKIP_OPTION = "--skip-annotators";

  /** The option that keeps only the runs named as {@code <annotator>/<model>}. */
  static final String ONLY_OPTION = "--only-runs";

  private Main() {}

  public static void main(String[] args) throws IOException {
    if (args.length < 5) {
      System.err.println(USAGE);
      System.exit(2);
    }
    Options options;
    try {
      options = Options.parse(Arrays.asList(args).subList(5, args.length));
    } catch (IllegalArgumentException e) {
      System.err.println(e.getMessage());
      System.err.println(USAGE);
      System.exit(2);
      return;
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
    Runs.All runs = options.select(Runs.load(inputs, models, fixtures, snapshot));
    List<AnnotatorDump> recorders = options.dumps ? runs.recorders() : List.of();

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

  private static final String USAGE = "usage: turbo-goldens-java <inputs dir> <models dir> <fixtures dir> <out dir>"
      + " <opennlp commit> [" + DUMPS_OPTION + "] [" + SKIP_OPTION + " <annotator>,...] [" + ONLY_OPTION
      + " <annotator>/<model>,...]";

  /** The options after the five positional arguments. */
  record Options(boolean dumps, Set<String> skipAnnotators, Set<String> onlyRuns) {

    static Options parse(List<String> args) {
      boolean dumps = false;
      Set<String> skip = new TreeSet<>();
      Set<String> only = new TreeSet<>();
      for (int i = 0; i < args.size(); i++) {
        String a = args.get(i);
        switch (a) {
          case DUMPS_OPTION -> dumps = true;
          case SKIP_OPTION, ONLY_OPTION -> {
            if (i + 1 >= args.size()) {
              throw new IllegalArgumentException(a + " needs a comma separated list");
            }
            Set<String> target = a.equals(SKIP_OPTION) ? skip : only;
            for (String name : args.get(++i).split(",")) {
              if (name.isEmpty() || (a.equals(ONLY_OPTION) && name.chars().filter(c -> c == '/').count() != 1)) {
                throw new IllegalArgumentException(a + ": " + name + " is not a name it takes");
              }
              target.add(name);
            }
          }
          default -> throw new IllegalArgumentException("unknown option " + a);
        }
      }
      return new Options(dumps, skip, only);
    }

    /**
     * The runs that pass both filters, with their recorders. A skipped annotator or a selected
     * run that no run matches is an error, so a misspelt name does not silently run nothing.
     */
    Runs.All select(Runs.All all) {
      Set<String> annotators = new TreeSet<>();
      Set<String> runs = new TreeSet<>();
      for (AnnotatorGolden a : all.annotators()) {
        annotators.add(a.annotator());
        runs.add(a.annotator() + "/" + a.modelDir());
      }
      for (String s : skipAnnotators) {
        if (!annotators.contains(s)) {
          throw new IllegalArgumentException(SKIP_OPTION + ": no run has the annotator directory " + s);
        }
      }
      for (String r : onlyRuns) {
        if (!runs.contains(r)) {
          throw new IllegalArgumentException(ONLY_OPTION + ": there is no run " + r);
        }
      }
      List<AnnotatorGolden> selected = all.annotators().stream().filter(this::keeps).toList();
      List<AnnotatorDump> recorders = all.recorders().stream().filter(d -> keeps(d.golden())).toList();
      return new Runs.All(selected, recorders);
    }

    private boolean keeps(AnnotatorGolden a) {
      return !skipAnnotators.contains(a.annotator())
          && (onlyRuns.isEmpty() || onlyRuns.contains(a.annotator() + "/" + a.modelDir()));
    }
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
