package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import opennlp.geo.PlaceProfiles;

/**
 * The goldens of {@code opennlp.geo.PlaceProfiles} of the OpenNLP addons {@code geo} module, over
 * the {@code place-profiles} input set. Each input is a whole profile table.
 *
 * <p>The class is not in the addons commit the rest of the module is built from: the container
 * compiles its source from {@link #PROFILES_COMMIT} into the module (container/Dockerfile).</p>
 *
 * <p>The layers, anchored to the document: {@code turbo:profiles-load} (whether the table loaded,
 * or the exception), {@code turbo:profiles-metrics} (the metric names in column order),
 * {@code turbo:profiles-places} (the place ids sorted by {@code String.compareTo}, since the
 * class keeps them in a map with no order), {@code turbo:profiles-values} (each place's
 * standardized profile, read from the class's private map, as indices into the place and metric
 * layers), {@code turbo:profiles-similarity} ({@code similarity} for every ordered pair of
 * places, the place with itself included) and {@code turbo:profiles-neighbors}
 * ({@code mostSimilar} of every place with the counts 1, 2 and the number of places, each
 * neighbor with its rank). A table that does not load has empty layers after the first.</p>
 */
final class PlaceProfilesGolden implements AnnotatorGolden {

  /** The input set the place profile goldens are made from. */
  static final String SET = Runs.PLACE_PROFILES;

  /** The addons commit (the ready/geo-place-profiles branch) PlaceProfiles.java is taken from. */
  static final String PROFILES_COMMIT = "07e1f670818917d74e9a94367080f51ae6365bfa";

  @Override
  public String annotator() {
    return "place-profiles";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return "place-profiles";
  }

  @Override
  public String annotatorClass() {
    return PlaceProfiles.class.getName();
  }

  @Override
  public String method() {
    return "load";
  }

  @Override
  public Map<String, String> config() {
    return Map.of("profilesCommit", PROFILES_COMMIT);
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public List<Layer> layers(Input input) {
    PlaceProfiles profiles = null;
    long loaded = 1;
    String errorClass = "";
    String errorMessage = "";
    try (InputStream in = new ByteArrayInputStream(input.bytes())) {
      profiles = PlaceProfiles.load(in);
    } catch (IOException | IllegalArgumentException e) {
      loaded = 0;
      errorClass = e.getClass().getName();
      errorMessage = e.getMessage() == null ? "" : e.getMessage();
    }
    List<String> metrics = profiles == null ? List.of() : profiles.metrics();
    Map<String, double[]> table = profiles == null ? Map.of() : new TreeMap<>(standardized(profiles));
    List<String> ids = new ArrayList<>(table.keySet());

    List<Long> valuePlace = new ArrayList<>();
    List<Long> valueMetric = new ArrayList<>();
    List<Double> values = new ArrayList<>();
    for (int p = 0; p < ids.size(); p++) {
      double[] profile = table.get(ids.get(p));
      for (int m = 0; m < profile.length; m++) {
        valuePlace.add((long) p);
        valueMetric.add((long) m);
        values.add(profile[m]);
      }
    }

    List<Long> pairA = new ArrayList<>();
    List<Long> pairB = new ArrayList<>();
    List<Double> pairSimilarity = new ArrayList<>();
    List<Long> neighborQuery = new ArrayList<>();
    List<Long> neighborCount = new ArrayList<>();
    List<Long> neighborRank = new ArrayList<>();
    List<Long> neighborPlace = new ArrayList<>();
    List<Double> neighborSimilarity = new ArrayList<>();
    for (int a = 0; a < ids.size(); a++) {
      for (int b = 0; b < ids.size(); b++) {
        pairA.add((long) a);
        pairB.add((long) b);
        pairSimilarity.add(profiles.similarity(ids.get(a), ids.get(b)));
      }
      for (int count : new int[] {1, 2, ids.size()}) {
        List<PlaceProfiles.Neighbor> found = profiles.mostSimilar(ids.get(a), count);
        for (int r = 0; r < found.size(); r++) {
          neighborQuery.add((long) a);
          neighborCount.add((long) count);
          neighborRank.add((long) r);
          neighborPlace.add((long) ids.indexOf(found.get(r).id()));
          neighborSimilarity.add(found.get(r).similarity());
        }
      }
    }

    return List.of(
        new Layer("turbo:profiles-load", "document", new Column.U32("loaded", new long[] {loaded}),
            new Column.Str("error-class", new String[] {errorClass}),
            new Column.Str("error-message", new String[] {errorMessage})),
        new Layer("turbo:profiles-metrics", "document", new Column.Str("name", metrics.toArray(String[]::new))),
        new Layer("turbo:profiles-places", "document", new Column.Str("id", ids.toArray(String[]::new))),
        new Layer("turbo:profiles-values", "document", new Column.U32("place", longs(valuePlace)),
            new Column.U32("metric", longs(valueMetric)), new Column.F64("value", doubles(values))),
        new Layer("turbo:profiles-similarity", "document", new Column.U32("a", longs(pairA)),
            new Column.U32("b", longs(pairB)), new Column.F64("similarity", doubles(pairSimilarity))),
        new Layer("turbo:profiles-neighbors", "document", new Column.U32("query", longs(neighborQuery)),
            new Column.U32("count", longs(neighborCount)), new Column.U32("rank", longs(neighborRank)),
            new Column.U32("place", longs(neighborPlace)),
            new Column.F64("similarity", doubles(neighborSimilarity))));
  }

  /** The class's private map from place id to standardized profile. */
  @SuppressWarnings("unchecked")
  private static Map<String, double[]> standardized(PlaceProfiles profiles) {
    try {
      Field field = PlaceProfiles.class.getDeclaredField("profiles");
      field.setAccessible(true);
      return (Map<String, double[]>) field.get(profiles);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("PlaceProfiles has no profiles map", e);
    }
  }

  private static long[] longs(List<Long> values) {
    return values.stream().mapToLong(Long::longValue).toArray();
  }

  private static double[] doubles(List<Double> values) {
    return values.stream().mapToDouble(Double::doubleValue).toArray();
  }
}
