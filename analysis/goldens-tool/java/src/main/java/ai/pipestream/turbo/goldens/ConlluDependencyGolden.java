package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import opennlp.tools.formats.conllu.ConlluTagset;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.WhitespaceMode;

/**
 * The golden of {@code ConlluDependencySampleStream} over a CoNLL-U file: every dependency sample
 * it read and every exception it threw.
 *
 * <p>The stream, {@code DependencySample}, {@code DependencyTree} and {@code DependencyArc} are
 * those of the fork's {@code ready/agents-md} branch at {@link #BRANCH_COMMIT}, which is later than
 * the commit the image is built from (that commit's stream builds a {@code DependencyGraph}, reads
 * heads with {@code Integer.parseInt} and keeps a {@code _} tag). The container compiles them, with
 * the branch's {@code ParamChecks}, into a jar of their own, which {@link BranchJar} loads before
 * the generator's OpenNLP jars.</p>
 *
 * <p>A run is one tag set, and for the universal tags also the legacy whitespace mode, in which the
 * stream tests separator lines and relations for whitespace. It reads the {@code conllu} set's
 * files whose id starts with {@code dependency-}, until {@code read} returns null; an exception is
 * recorded with the number of samples before it, and reading goes on. The layers, all anchored to
 * the document: {@code turbo:samples}, what each sample's {@code toString} returned; {@code
 * turbo:read-errors}, {@code after}, {@code class} and {@code message} per exception; {@code
 * turbo:sample-tokens}, the sample's index and the token per token of every sample; and aligned to
 * it {@code turbo:sample-tags}, {@code turbo:sample-heads} (the head's index plus one, 0 for the
 * root) and {@code turbo:sample-relations}.</p>
 */
final class ConlluDependencyGolden implements AnnotatorGolden {

  /** The input set the stream reads. */
  static final String SET = Runs.CONLLU;

  /** The prefix of the input ids the stream reads. */
  static final String PREFIX = "dependency-";

  /** The system property naming the jar of the branch's classes, which the image's entrypoint sets. */
  static final String JAR_PROPERTY = "turbo.goldens.conlluDependencyJar";

  /** The commit of the fork's ready/agents-md branch the jar must be built from. */
  static final String BRANCH_COMMIT = "2da0ee764cd7340b968a207b925cce852485cccb";

  static final String STREAM = "opennlp.tools.formats.conllu.ConlluDependencySampleStream";

  /** The stopping point for a stream that throws without moving on, which would otherwise loop. */
  private static final int MAX_READS = 100_000;

  private final ConlluTagset tagset;
  private final WhitespaceMode whitespace;
  private final Branch branch;

  private ConlluDependencyGolden(ConlluTagset tagset, WhitespaceMode whitespace, Branch branch) {
    this.tagset = tagset;
    this.whitespace = whitespace;
    this.branch = branch;
  }

  /** The runs: each tag set, and the universal tags in the legacy whitespace mode. */
  static List<ConlluDependencyGolden> all() throws IOException {
    Branch branch = Branch.open();
    return List.of(new ConlluDependencyGolden(ConlluTagset.U, WhitespaceMode.UNICODE, branch),
        new ConlluDependencyGolden(ConlluTagset.X, WhitespaceMode.UNICODE, branch),
        new ConlluDependencyGolden(ConlluTagset.U, WhitespaceMode.LEGACY, branch));
  }

  /** The stream's constructor and the sample's and tree's accessors, from the branch's jar. */
  private record Branch(Constructor<?> stream, Method tokens, Method tags, Method tree, Method headOf,
      Method relationOf) {

    static Branch open() throws IOException {
      ClassLoader loader = BranchJar.open(JAR_PROPERTY, BRANCH_COMMIT, List.of(STREAM,
          "opennlp.tools.depparse.DependencySample", "opennlp.tools.depparse.DependencyTree",
          "opennlp.tools.depparse.DependencyArc"));
      try {
        Class<?> stream = Class.forName(STREAM, true, loader);
        Class<?> sample = Class.forName("opennlp.tools.depparse.DependencySample", true, loader);
        Class<?> tree = Class.forName("opennlp.tools.depparse.DependencyTree", true, loader);
        if (stream.getClassLoader() != loader || sample.getClassLoader() != loader) {
          throw new IllegalStateException("the dependency classes were not loaded from the branch's jar");
        }
        return new Branch(stream.getConstructor(InputStreamFactory.class, ConlluTagset.class),
            sample.getMethod("getTokens"), sample.getMethod("getTags"), sample.getMethod("getTree"),
            tree.getMethod("headOf", int.class), tree.getMethod("relationOf", int.class));
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException(STREAM, e);
      }
    }

    ObjectStream<?> open(InputStreamFactory in, ConlluTagset tagset) throws IOException {
      try {
        return (ObjectStream<?>) stream.newInstance(in, tagset);
      } catch (InvocationTargetException e) {
        if (e.getCause() instanceof IOException io) {
          throw io;
        }
        throw new IllegalStateException(STREAM, e.getCause());
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException(STREAM, e);
      }
    }

    Object call(Method m, Object target, Object... args) {
      try {
        return m.invoke(target, args);
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException(m.getName(), e);
      }
    }
  }

  @Override
  public String annotator() {
    return "conllu";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return "dependency-" + tagset.name().toLowerCase(Locale.ROOT) + Whitespace.dirSuffix(whitespace);
  }

  @Override
  public String annotatorClass() {
    return STREAM;
  }

  @Override
  public String method() {
    return "read";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> config = new HashMap<>(Whitespace.config(whitespace));
    config.put("tagset", tagset.name().toLowerCase(Locale.ROOT));
    config.put("branchCommit", BRANCH_COMMIT);
    return config;
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(SET) && input.id().startsWith(PREFIX);
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    return Whitespace.in(whitespace, () -> read(input));
  }

  private List<Layer> read(Input input) {
    List<Object> samples = new ArrayList<>();
    Table errors = new Table("turbo:read-errors", "after:u32", "class:str", "message:str");
    InputStreamFactory in = () -> new ByteArrayInputStream(input.bytes());
    try (ObjectStream<?> stream = branch.open(in, tagset)) {
      for (int reads = 0; ; reads++) {
        if (reads > MAX_READS) {
          throw new IllegalStateException(input.inputId() + ": the stream does not end");
        }
        Object sample;
        try {
          sample = stream.read();
        } catch (IOException | RuntimeException e) {
          errors.row(samples.size(), e.getClass().getName(), FormatGolden.message(e));
          continue;
        }
        if (sample == null) {
          break;
        }
        samples.add(sample);
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the stream failed", e);
    }

    Table written = new Table("turbo:samples", "written:str");
    Table tokens = new Table("turbo:sample-tokens", "sample:u32", "text:str");
    Table tags = Table.aligned("turbo:sample-tags", "turbo:sample-tokens", "tag:str");
    Table heads = Table.aligned("turbo:sample-heads", "turbo:sample-tokens", "head:u32");
    Table relations = Table.aligned("turbo:sample-relations", "turbo:sample-tokens", "relation:str");
    for (int s = 0; s < samples.size(); s++) {
      Object sample = samples.get(s);
      written.row(sample.toString());
      String[] t = (String[]) branch.call(branch.tokens(), sample);
      String[] g = (String[]) branch.call(branch.tags(), sample);
      Object tree = branch.call(branch.tree(), sample);
      for (int i = 0; i < t.length; i++) {
        tokens.row(s, t[i]);
        tags.row(g[i]);
        heads.row((Integer) branch.call(branch.headOf(), tree, i) + 1);
        relations.row((String) branch.call(branch.relationOf(), tree, i));
      }
    }
    return List.of(written.layer(), errors.layer(), tokens.layer(), tags.layer(), heads.layer(),
        relations.layer());
  }
}
