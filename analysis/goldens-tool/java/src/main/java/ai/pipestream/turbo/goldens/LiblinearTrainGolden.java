package ai.pipestream.turbo.goldens;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import opennlp.tools.ml.AbstractEventTrainer;
import opennlp.tools.ml.EventTrainer;
import opennlp.tools.ml.model.Event;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Parameters;
import opennlp.tools.util.TrainingParameters;

/**
 * The liblinear trainer of the OpenNLP addons ({@code liblinear-addon}: {@code LiblinearTrainer}
 * and {@code LiblinearModel}, on liblinear-java 2.44): the events of a {@link TrainGolden.Source},
 * the model the trainer makes of them, and its probabilities for their contexts. The golden
 * directory is {@code train-liblinear/<run>}.
 *
 * <p>The add-on's classes are in the default package, which no named package can import, so they
 * are compiled from the addons commit {@link #ADDON_COMMIT} into a jar of their own in the image
 * (the container's Dockerfile) and loaded here by reflection from the jar the system property
 * {@link #JAR_PROPERTY} names. The trainer is made as the add-on allows: {@code TrainerFactory}
 * cannot make it (it has no constructor without arguments), and its constructor reads the
 * parameters without handing them to {@code AbstractTrainer}, so {@code init} is called with the
 * same parameters before {@code train(events)}.</p>
 *
 * <p>liblinear refuses an instance whose features do not ascend, and the add-on passes each
 * event's predicate ids in the event's order, so a run with {@code events=sorted} gives the
 * trainer each event with its predicates sorted by {@code String.compareTo} and each kept once
 * (the first, with its value). A run with {@code events=raw} gives the source's events as they
 * are.</p>
 *
 * <p>The layers, all anchored on the document:</p>
 * <ul>
 *   <li>{@code turbo:events}: the events given to the trainer, as in {@link TrainGolden}.</li>
 *   <li>{@code turbo:liblinear-error}: when {@code train} throws, its {@code class} and
 *   {@code message}; the layers below are left out.</li>
 *   <li>{@code turbo:liblinear-text}: the liblinear text model inside the bytes
 *   {@code LiblinearModel.serialize} writes ({@code Linear.saveModel}), one {@code line} per
 *   row.</li>
 *   <li>{@code turbo:liblinear-model}: the SHA-256 and the {@code length} of those bytes.</li>
 *   <li>{@code turbo:liblinear-scores}: {@code eval} on the context of each of the first
 *   {@link #SCORED} events, as given and reversed: the {@code event} (its row in
 *   {@code turbo:events}), the {@code order} ({@code given} or {@code reversed}), and per position
 *   of the returned array its {@code outcome} position and {@code value}.</li>
 *   <li>{@code turbo:liblinear-best}: {@code getBestOutcome} of each of those arrays: the
 *   {@code event}, the {@code order} and the {@code outcome} label.</li>
 * </ul>
 */
final class LiblinearTrainGolden implements AnnotatorGolden {

  static final String ANNOTATOR = "train-liblinear";

  /** The system property naming the add-on's jar, which the image's entrypoint sets. */
  static final String JAR_PROPERTY = "turbo.goldens.liblinearJar";

  /** The addons commit, on its main branch, the jar must be built from; the image writes it beside the jar. */
  static final String ADDON_COMMIT = "84631d0111629110ebf094eadb6e938bbf7d738f";

  /** The add-on's trainer class, in the default package. */
  static final String TRAINER_CLASS = "LiblinearTrainer";

  /** The events of an input whose contexts are scored, at most. */
  static final int SCORED = 40;

  /** One run: where its events come from and what the trainer is given. */
  record Run(String name, TrainGolden.Source source, String indexer, int cutoff, String solver, String c,
      String eps, String p, String bias, boolean sorted, Set<String> inputSets) {

    TrainingParameters parameters() {
      TrainingParameters params = new TrainingParameters();
      params.put("solverType", solver);
      params.put("c", c);
      params.put("eps", eps);
      params.put("p", p);
      params.put("bias", bias);
      params.put(Parameters.CUTOFF_PARAM, Integer.toString(cutoff));
      params.put(AbstractEventTrainer.DATA_INDEXER_PARAM, indexer);
      return params;
    }

    Map<String, String> config() {
      Map<String, String> c = new TreeMap<>();
      c.put("source", source.name);
      c.put(AbstractEventTrainer.DATA_INDEXER_PARAM, indexer);
      c.put(Parameters.CUTOFF_PARAM, Integer.toString(cutoff));
      c.put("solverType", solver);
      c.put("c", this.c);
      c.put("eps", eps);
      c.put("p", p);
      c.put("bias", bias);
      c.put("events", sorted ? "sorted" : "raw");
      return c;
    }
  }

  private final Run run;
  private final Constructor<?> trainer;
  private final Method serialize;

  private LiblinearTrainGolden(Run run, Constructor<?> trainer, Method serialize) {
    this.run = run;
    this.trainer = trainer;
    this.serialize = serialize;
  }

  /**
   * Every run: the add-on's sample parameters, each logistic regression solver on the sentence
   * detector's, part-of-speech shaped and real valued events, and the source's events unsorted.
   */
  static List<LiblinearTrainGolden> all(Set<String> texts, Set<String> sentences) throws IOException {
    Set<String> both = new HashSet<>(texts);
    both.addAll(sentences);
    List<Run> runs = List.of(
        // LiblinearParams.txt, the add-on's sample parameters.
        new Run("sentdetect-sample", TrainGolden.Source.SENTDETECT, "TwoPass", 1, "L1R_LR", "1", "0.001", "0.1",
            "0", true, both),
        new Run("postag-l2", TrainGolden.Source.POSTAG, "TwoPass", 1, "L2R_LR", "1", "0.01", "0", "1", true, both),
        new Run("postag-dual", TrainGolden.Source.POSTAG, "OnePass", 0, "L2R_LR_DUAL", "0.5", "0.1", "0", "-1", true,
            texts),
        // A cost high and a tolerance tight enough that the dual solver runs its 300 iterations
        // and the Newton method continues from its solution.
        new Run("sentdetect-dual-newton", TrainGolden.Source.SENTDETECT, "TwoPass", 1, "L2R_LR_DUAL", "10000",
            "1.0E-10", "0", "1", true, sentences),
        new Run("realvalue-l1", TrainGolden.Source.REALVALUE, "OnePassRealValue", 1, "L1R_LR", "2", "0.01", "0", "1",
            true, texts),
        new Run("sentdetect-l2", TrainGolden.Source.SENTDETECT, "TwoPass", 2, "L2R_LR", "0.1", "0.001", "0", "-1",
            true, sentences),
        new Run("sentdetect-raw", TrainGolden.Source.SENTDETECT, "TwoPass", 1, "L2R_LR", "1", "0.01", "0", "0",
            false, both));
    ClassLoader loader = loader();
    List<LiblinearTrainGolden> out = new ArrayList<>();
    try {
      Class<?> trainerClass = Class.forName(TRAINER_CLASS, true, loader);
      Class<?> modelClass = Class.forName("LiblinearModel", true, loader);
      Constructor<?> constructor = trainerClass.getConstructor(TrainingParameters.class);
      Method serialize = modelClass.getMethod("serialize", OutputStream.class);
      for (Run run : runs) {
        out.add(new LiblinearTrainGolden(run, constructor, serialize));
      }
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("the liblinear add-on's classes", e);
    }
    return out;
  }

  /** A class loader over the add-on's jar, checked to be built from {@link #ADDON_COMMIT}. */
  private static ClassLoader loader() throws IOException {
    String jar = System.getProperty(JAR_PROPERTY);
    if (jar == null) {
      throw new IllegalStateException("-D" + JAR_PROPERTY + " must name the jar of the liblinear add-on");
    }
    Path path = Path.of(jar);
    String commit = Files.readString(path.resolveSibling("commit"), StandardCharsets.UTF_8).strip();
    if (!commit.equals(ADDON_COMMIT)) {
      throw new IllegalStateException(path + " was built from " + commit + ", not " + ADDON_COMMIT);
    }
    return new URLClassLoader(new URL[] {path.toUri().toURL()}, LiblinearTrainGolden.class.getClassLoader());
  }

  @Override
  public String annotator() {
    return ANNOTATOR;
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return run.name();
  }

  @Override
  public String annotatorClass() {
    return TRAINER_CLASS;
  }

  @Override
  public String method() {
    return "train";
  }

  @Override
  public Map<String, String> config() {
    return run.config();
  }

  @Override
  public Set<String> inputSets() {
    return run.inputSets();
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    List<Event> events;
    try {
      events = TrainGolden.events(run.source(), input);
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the event stream failed", e);
    }
    if (run.sorted()) {
      events = sorted(events);
    }
    List<Layer> layers = new ArrayList<>();
    layers.add(TrainGolden.eventsLayer(events));

    TrainingParameters params = run.parameters();
    MaxentModel model;
    byte[] bytes;
    try {
      @SuppressWarnings("unchecked")
      EventTrainer<TrainingParameters> t = (EventTrainer<TrainingParameters>) trainer.newInstance(params);
      t.init(params, new HashMap<>());
      model = t.train(ObjectStreamUtils.createObjectStream(events));
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      serialize.invoke(model, out);
      bytes = out.toByteArray();
    } catch (InvocationTargetException e) {
      throw new IllegalStateException(input.inputId() + ": the add-on threw", e.getCause());
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(e);
    } catch (IOException | RuntimeException e) {
      layers.add(new Layer("turbo:liblinear-error", "document",
          new Column.Str("class", new String[] {e.getClass().getName()}),
          new Column.Str("message", new String[] {String.valueOf(e.getMessage())})));
      return layers;
    }

    int textLength = ByteBuffer.wrap(bytes, 0, 4).getInt();
    String text = new String(bytes, 4, textLength, StandardCharsets.UTF_8);
    if (!text.endsWith("\n")) {
      throw new IllegalStateException(input.inputId() + ": the text model does not end its last line");
    }
    layers.add(new Layer("turbo:liblinear-text", "document",
        new Column.Str("line", text.substring(0, text.length() - 1).split("\n", -1))));
    layers.add(new Layer("turbo:liblinear-model", "document",
        new Column.Str("sha256", new String[] {Sha.sha256(bytes)}),
        new Column.U32("length", new long[] {bytes.length})));
    layers.addAll(scores(model, events));
    return layers;
  }

  /** The scores and best outcome layers: see the class comment. */
  private static List<Layer> scores(MaxentModel model, List<Event> events) {
    List<Long> event = new ArrayList<>();
    List<String> order = new ArrayList<>();
    List<Long> outcome = new ArrayList<>();
    List<Double> value = new ArrayList<>();
    List<Long> bestEvent = new ArrayList<>();
    List<String> bestOrder = new ArrayList<>();
    List<String> best = new ArrayList<>();
    int n = Math.min(SCORED, events.size());
    for (int e = 0; e < n; e++) {
      String[] given = events.get(e).getContext();
      String[] reversed = new String[given.length];
      for (int i = 0; i < given.length; i++) {
        reversed[i] = given[given.length - 1 - i];
      }
      for (String o : List.of("given", "reversed")) {
        double[] probs = model.eval(o.equals("given") ? given : reversed);
        for (int i = 0; i < probs.length; i++) {
          event.add((long) e);
          order.add(o);
          outcome.add((long) i);
          value.add(probs[i]);
        }
        bestEvent.add((long) e);
        bestOrder.add(o);
        best.add(model.getBestOutcome(probs));
      }
    }
    return List.of(
        new Layer("turbo:liblinear-scores", "document", new Column.U32("event", longs(event)),
            new Column.Str("order", order.toArray(new String[0])), new Column.U32("outcome", longs(outcome)),
            new Column.F64("value", value.stream().mapToDouble(Double::doubleValue).toArray())),
        new Layer("turbo:liblinear-best", "document", new Column.U32("event", longs(bestEvent)),
            new Column.Str("order", bestOrder.toArray(new String[0])),
            new Column.Str("outcome", best.toArray(new String[0]))));
  }

  private static long[] longs(List<Long> values) {
    return values.stream().mapToLong(Long::longValue).toArray();
  }

  /**
   * Each event with its predicates sorted by {@code String.compareTo}, stably, and each predicate
   * kept once, the first time it comes, with its value.
   */
  static List<Event> sorted(List<Event> events) {
    List<Event> out = new ArrayList<>(events.size());
    for (Event e : events) {
      String[] context = e.getContext();
      float[] values = e.getValues();
      Integer[] at = new Integer[context.length];
      for (int i = 0; i < at.length; i++) {
        at[i] = i;
      }
      Arrays.sort(at, (a, b) -> context[a].compareTo(context[b]));
      List<String> c = new ArrayList<>();
      List<Float> v = new ArrayList<>();
      for (int i : at) {
        if (!c.isEmpty() && c.get(c.size() - 1).equals(context[i])) {
          continue;
        }
        c.add(context[i]);
        if (values != null) {
          v.add(values[i]);
        }
      }
      String[] sortedContext = c.toArray(new String[0]);
      if (values == null) {
        out.add(new Event(e.getOutcome(), sortedContext));
      } else {
        float[] sortedValues = new float[v.size()];
        for (int i = 0; i < sortedValues.length; i++) {
          sortedValues[i] = v.get(i);
        }
        out.add(new Event(e.getOutcome(), sortedContext, sortedValues));
      }
    }
    return out;
  }
}
