package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import opennlp.geo.BundledGazetteer;
import opennlp.geo.GeoNamesGazetteer;
import opennlp.geo.OverlayGazetteer;
import opennlp.geo.OvertureGazetteer;
import opennlp.geo.Suppression;
import opennlp.geo.UserGazetteer;
import opennlp.tools.geo.AttributeValue;
import opennlp.tools.geo.Gazetteer;
import opennlp.tools.geo.GazetteerEntry;

/**
 * The goldens of the gazetteers of the OpenNLP addons {@code geo} module, over the
 * {@code gazetteer} input set.
 *
 * <p>The query runs load one gazetteer and ask it what each line of an input names: an input
 * whose id starts with {@code names} is looked up line by line with {@code lookup}, one starting
 * with {@code ids} holds a source and a record id per line, separated by a tab, for
 * {@code byId}, and one starting with {@code regions} holds a region code per line for
 * {@code byRegion}. The lines are the input split at line feeds, without a last empty line when
 * the input ends with one. The runs: {@code naturalearth-populated-places} (the bundled table,
 * {@code BundledGazetteer.getInstance()}), {@code gazetteer-geonames}, {@code gazetteer-overture}
 * and {@code gazetteer-user} (the fixture tables, the user table with the source {@code sites}),
 * and {@code overlay} (the bundled gazetteer under the user table as additions and the
 * suppression rules of {@code gazetteer-suppressions.txt}).</p>
 *
 * <p>The load runs load each input whose id starts with {@code table-<kind>-} as a table of that
 * kind and record whether it loaded or the exception: {@code load-geonames},
 * {@code load-overture}, {@code load-user} (with the source {@code sites}) and
 * {@code load-suppressions}.</p>
 *
 * <p>The layers of a query run, anchored to the document: {@code turbo:gazetteer-sources} (the
 * sources, sorted, since {@code Set.copyOf} has no order), {@code turbo:gazetteer-queries} (per
 * line, the number of entries found, or the exception thrown), {@code turbo:gazetteer-candidates}
 * (per line, the entries found in order, as indices into the entry layer),
 * {@code turbo:gazetteer-entries} (each distinct entry once, in the order first found, with every
 * field), {@code turbo:gazetteer-alternates}, {@code turbo:gazetteer-containment} and
 * {@code turbo:gazetteer-attributes} (the attributes sorted by key, since {@code Map.copyOf} has
 * no order). The population is written as a decimal string, since it is a {@code long}. A load
 * run writes {@code turbo:gazetteer-load} and {@code turbo:gazetteer-suppressions} (the rules of
 * a suppression table that loads; empty for the other kinds).</p>
 */
final class GazetteerGolden implements AnnotatorGolden {

  /** The input set the gazetteer goldens are made from. */
  static final String SET = Runs.GAZETTEER;

  /** The source the user table is loaded with. */
  static final String USER_SOURCE = "sites";

  /** The bundled table, as {@code BundledGazetteer} names its resource. */
  static final String BUNDLED_TABLE = "naturalearth-populated-places.txt";

  /** The kinds of table the load runs load, as the input ids and run names write them. */
  enum TableKind {
    GEONAMES("geonames", GeoNamesGazetteer.class, "load"),
    OVERTURE("overture", OvertureGazetteer.class, "load"),
    USER("user", UserGazetteer.class, "load"),
    SUPPRESSIONS("suppressions", UserGazetteer.class, "loadSuppressions");

    final String id;
    final Class<?> loader;
    final String method;

    TableKind(String id, Class<?> loader, String method) {
      this.id = id;
      this.loader = loader;
      this.method = method;
    }
  }

  private static final String TABLE_PREFIX = "table-";

  private final String dir;
  private final List<ModelRef> models;
  private final Map<String, String> config;
  private final String annotatorClass;
  /** The gazetteer a query run asks; null for a load run. */
  private final Gazetteer gazetteer;
  /** The kind of table a load run loads; null for a query run. */
  private final TableKind table;

  private GazetteerGolden(String dir, List<ModelRef> models, Map<String, String> config,
      String annotatorClass, Gazetteer gazetteer, TableKind table) {
    this.dir = dir;
    this.models = List.copyOf(models);
    this.config = Map.copyOf(config);
    this.annotatorClass = annotatorClass;
    this.gazetteer = gazetteer;
    this.table = table;
  }

  private static GazetteerGolden query(String dir, List<ModelRef> models, Map<String, String> config,
      Class<?> annotatorClass, Gazetteer gazetteer) {
    return new GazetteerGolden(dir, models, config, annotatorClass.getName(), gazetteer, null);
  }

  /** Every run, with the fixture tables of {@code fixtures}. */
  static List<GazetteerGolden> all(Path fixtures) throws IOException {
    ModelFile bundledFile = bundledTable();
    ModelFile geonamesFile = ModelFile.read(fixtures.resolve(Runs.GAZETTEER_GEONAMES));
    ModelFile overtureFile = ModelFile.read(fixtures.resolve(Runs.GAZETTEER_OVERTURE));
    ModelFile userFile = ModelFile.read(fixtures.resolve(Runs.GAZETTEER_USER));
    ModelFile suppressionsFile = ModelFile.read(fixtures.resolve(Runs.GAZETTEER_SUPPRESSIONS));
    BundledGazetteer bundled = BundledGazetteer.getInstance();
    UserGazetteer user = UserGazetteer.load(userFile.stream(), USER_SOURCE);
    List<Suppression> rules = UserGazetteer.loadSuppressions(suppressionsFile.stream());
    List<GazetteerGolden> all = new ArrayList<>(List.of(
        query(bundledFile.ref().baseName(), List.of(bundledFile.ref()), Map.of(), BundledGazetteer.class, bundled),
        query(geonamesFile.ref().baseName(), List.of(geonamesFile.ref()), Map.of(), GeoNamesGazetteer.class,
            GeoNamesGazetteer.load(geonamesFile.stream())),
        query(overtureFile.ref().baseName(), List.of(overtureFile.ref()), Map.of(), OvertureGazetteer.class,
            OvertureGazetteer.load(overtureFile.stream())),
        query(userFile.ref().baseName(), List.of(userFile.ref()), Map.of("source", USER_SOURCE),
            UserGazetteer.class, user),
        query("overlay", List.of(bundledFile.ref(), userFile.ref(), suppressionsFile.ref()),
            Map.of("source", USER_SOURCE), OverlayGazetteer.class, new OverlayGazetteer(bundled, user, rules))));
    for (TableKind kind : TableKind.values()) {
      Map<String, String> config = kind == TableKind.USER ? Map.of("source", USER_SOURCE) : Map.of();
      all.add(new GazetteerGolden("load-" + kind.id, List.of(), config, kind.loader.getName(), null, kind));
    }
    return all;
  }

  /** The bundled table as the jar holds it, for its hash. */
  private static ModelFile bundledTable() throws IOException {
    try (InputStream in = BundledGazetteer.class.getResourceAsStream(BUNDLED_TABLE)) {
      if (in == null) {
        throw new IOException("the geo module has no resource " + BUNDLED_TABLE);
      }
      byte[] bytes = in.readAllBytes();
      return new ModelFile(new ModelRef(BUNDLED_TABLE, Sha.sha512(bytes)), bytes);
    }
  }

  @Override
  public String annotator() {
    return "gazetteer";
  }

  @Override
  public List<ModelRef> models() {
    return models;
  }

  @Override
  public String modelDir() {
    return dir;
  }

  @Override
  public String annotatorClass() {
    return annotatorClass;
  }

  @Override
  public String method() {
    return table == null ? "lookup" : table.method;
  }

  @Override
  public Map<String, String> config() {
    return config;
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    if (!input.set().equals(SET)) {
      return false;
    }
    return table == null ? !input.id().startsWith(TABLE_PREFIX)
        : input.id().startsWith(TABLE_PREFIX + table.id + "-");
  }

  @Override
  public List<Layer> layers(Input input) {
    return table == null ? queries(input) : load(input);
  }

  /** The lines of the input: split at line feeds, without the empty line after a last one. */
  static List<String> lines(String text) {
    List<String> lines = new ArrayList<>(List.of(text.split("\n", -1)));
    if (lines.get(lines.size() - 1).isEmpty()) {
      lines.remove(lines.size() - 1);
    }
    return lines;
  }

  /** What one question returned: the entries found, or the exception thrown. */
  private record Answer(List<GazetteerEntry> found, RuntimeException error) {}

  private Answer ask(String id, String line) {
    try {
      if (id.startsWith("names")) {
        return new Answer(gazetteer.lookup(line), null);
      }
      if (id.startsWith("ids")) {
        int tab = line.indexOf('\t');
        if (tab < 0) {
          throw new IllegalStateException(id + ": an ids line needs a tab: " + line);
        }
        return new Answer(optional(gazetteer.byId(line.substring(0, tab), line.substring(tab + 1))), null);
      }
      if (id.startsWith("regions")) {
        return new Answer(optional(gazetteer.byRegion(line)), null);
      }
      throw new IllegalStateException("the gazetteer input " + id + " is not names, ids, regions or a table");
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    } catch (IllegalArgumentException e) {
      return new Answer(List.of(), e);
    }
  }

  private static List<GazetteerEntry> optional(Optional<GazetteerEntry> entry) {
    return entry.map(List::of).orElse(List.of());
  }

  private List<Layer> queries(Input input) {
    List<String> sources = new ArrayList<>(new TreeSet<>(gazetteer.sources()));

    List<Long> found = new ArrayList<>();
    List<String> errorClasses = new ArrayList<>();
    List<String> errorMessages = new ArrayList<>();
    List<Long> candidateQuery = new ArrayList<>();
    List<Long> candidateEntry = new ArrayList<>();
    Map<GazetteerEntry, Integer> index = new IdentityHashMap<>();
    List<GazetteerEntry> entries = new ArrayList<>();
    List<String> lines = lines(input.text());
    for (int q = 0; q < lines.size(); q++) {
      Answer a = ask(input.id(), lines.get(q));
      found.add((long) a.found().size());
      errorClasses.add(a.error() == null ? "" : a.error().getClass().getName());
      errorMessages.add(a.error() == null || a.error().getMessage() == null ? "" : a.error().getMessage());
      for (GazetteerEntry e : a.found()) {
        Integer i = index.get(e);
        if (i == null) {
          i = entries.size();
          index.put(e, i);
          entries.add(e);
        }
        candidateQuery.add((long) q);
        candidateEntry.add((long) i);
      }
    }

    int n = entries.size();
    String[] source = new String[n];
    String[] recordId = new String[n];
    String[] name = new String[n];
    double[] latitude = new double[n];
    double[] longitude = new double[n];
    long[] hasBox = new long[n];
    double[] west = new double[n];
    double[] south = new double[n];
    double[] east = new double[n];
    double[] north = new double[n];
    long[] hasCountry = new long[n];
    String[] country = new String[n];
    String[] population = new String[n];
    long[] hasFeatureClass = new long[n];
    String[] featureClass = new String[n];
    List<Long> alternateEntry = new ArrayList<>();
    List<String> alternateName = new ArrayList<>();
    List<Long> containmentEntry = new ArrayList<>();
    List<String> containmentName = new ArrayList<>();
    List<Long> attributeEntry = new ArrayList<>();
    List<String> attributeKey = new ArrayList<>();
    List<String> attributeValue = new ArrayList<>();
    List<String> attributeSource = new ArrayList<>();
    List<String> attributeNotes = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      GazetteerEntry e = entries.get(i);
      source[i] = e.source();
      recordId[i] = e.recordId();
      name[i] = e.name();
      latitude[i] = e.location().latitude();
      longitude[i] = e.location().longitude();
      hasBox[i] = e.boundingBox() == null ? 0 : 1;
      if (e.boundingBox() != null) {
        west[i] = e.boundingBox().west();
        south[i] = e.boundingBox().south();
        east[i] = e.boundingBox().east();
        north[i] = e.boundingBox().north();
      }
      hasCountry[i] = e.countryCode() == null ? 0 : 1;
      country[i] = e.countryCode() == null ? "" : e.countryCode();
      population[i] = Long.toString(e.population());
      hasFeatureClass[i] = e.featureClass() == null ? 0 : 1;
      featureClass[i] = e.featureClass() == null ? "" : e.featureClass();
      for (String alternate : e.alternateNames()) {
        alternateEntry.add((long) i);
        alternateName.add(alternate);
      }
      for (String level : e.containment()) {
        containmentEntry.add((long) i);
        containmentName.add(level);
      }
      for (Map.Entry<String, AttributeValue> a : new TreeMap<>(e.attributes()).entrySet()) {
        attributeEntry.add((long) i);
        attributeKey.add(a.getKey());
        attributeValue.add(a.getValue().value());
        attributeSource.add(a.getValue().source());
        attributeNotes.add(a.getValue().notes());
      }
    }

    return List.of(
        new Layer("turbo:gazetteer-sources", "document", new Column.Str("source", sources.toArray(String[]::new))),
        new Layer("turbo:gazetteer-queries", "document",
            new Column.U32("found", longs(found)),
            new Column.Str("error-class", errorClasses.toArray(String[]::new)),
            new Column.Str("error-message", errorMessages.toArray(String[]::new))),
        new Layer("turbo:gazetteer-candidates", "document",
            new Column.U32("query", longs(candidateQuery)), new Column.U32("entry", longs(candidateEntry))),
        new Layer("turbo:gazetteer-entries", "document",
            new Column.Str("source", source), new Column.Str("record-id", recordId), new Column.Str("name", name),
            new Column.F64("latitude", latitude), new Column.F64("longitude", longitude),
            new Column.U32("has-box", hasBox), new Column.F64("west", west), new Column.F64("south", south),
            new Column.F64("east", east), new Column.F64("north", north),
            new Column.U32("has-country", hasCountry), new Column.Str("country", country),
            new Column.Str("population", population),
            new Column.U32("has-feature-class", hasFeatureClass), new Column.Str("feature-class", featureClass)),
        new Layer("turbo:gazetteer-alternates", "document",
            new Column.U32("entry", longs(alternateEntry)), new Column.Str("name", alternateName.toArray(String[]::new))),
        new Layer("turbo:gazetteer-containment", "document",
            new Column.U32("entry", longs(containmentEntry)),
            new Column.Str("name", containmentName.toArray(String[]::new))),
        new Layer("turbo:gazetteer-attributes", "document",
            new Column.U32("entry", longs(attributeEntry)), new Column.Str("key", attributeKey.toArray(String[]::new)),
            new Column.Str("value", attributeValue.toArray(String[]::new)),
            new Column.Str("source", attributeSource.toArray(String[]::new)),
            new Column.Str("notes", attributeNotes.toArray(String[]::new))));
  }

  private List<Layer> load(Input input) {
    long loaded = 1;
    String errorClass = "";
    String errorMessage = "";
    List<Suppression> rules = List.of();
    try (InputStream in = new ByteArrayInputStream(input.bytes())) {
      switch (table) {
        case GEONAMES -> GeoNamesGazetteer.load(in);
        case OVERTURE -> OvertureGazetteer.load(in);
        case USER -> UserGazetteer.load(in, USER_SOURCE);
        case SUPPRESSIONS -> rules = UserGazetteer.loadSuppressions(in);
      }
    } catch (IOException | IllegalArgumentException e) {
      loaded = 0;
      errorClass = e.getClass().getName();
      errorMessage = e.getMessage() == null ? "" : e.getMessage();
    }
    int n = rules.size();
    String[] ruleName = new String[n];
    long[] hasCountry = new long[n];
    String[] country = new String[n];
    long[] hasFeatureClass = new long[n];
    String[] featureClass = new String[n];
    for (int i = 0; i < n; i++) {
      Suppression r = rules.get(i);
      ruleName[i] = r.name();
      hasCountry[i] = r.countryCode() == null ? 0 : 1;
      country[i] = r.countryCode() == null ? "" : r.countryCode();
      hasFeatureClass[i] = r.featureClass() == null ? 0 : 1;
      featureClass[i] = r.featureClass() == null ? "" : r.featureClass();
    }
    return List.of(
        new Layer("turbo:gazetteer-load", "document", new Column.U32("loaded", new long[] {loaded}),
            new Column.Str("error-class", new String[] {errorClass}),
            new Column.Str("error-message", new String[] {errorMessage})),
        new Layer("turbo:gazetteer-suppressions", "document", new Column.Str("name", ruleName),
            new Column.U32("has-country", hasCountry), new Column.Str("country", country),
            new Column.U32("has-feature-class", hasFeatureClass), new Column.Str("feature-class", featureClass)));
  }

  private static long[] longs(List<Long> values) {
    return values.stream().mapToLong(Long::longValue).toArray();
  }
}
