// Writes the geocode, region ballot, containment hierarchy, and region-aware
// money pipeline test vectors of the turbo-geo crate from the pinned JDK 21
// image and OpenNLP commit cdf33c653d5d61d5af7d37ffee189a83bd8392ea.
//
// Run from the repository root with the JDK 21 source launcher:
//
//     docker run --rm --user "$(id -u):$(id -g)" \
//         -v "$PWD:/work" -v /work/worktrees/opennlp/uber:/opennlp:ro \
//         eclipse-temurin@sha256:3e3c176ffed168beb42c607be9bc1639b466cf00261a0fb04425562c9d0c5c2b \
//         java -cp "/opennlp/opennlp-api/target/classes:/opennlp/opennlp-core/opennlp-runtime/target/classes:/opennlp/opennlp-extensions/opennlp-geo/target/classes:/opennlp/opennlp-extensions/opennlp-uima/target/dependency/*" \
//         /work/analysis/geo/java/GeoPipelineVectors.java \
//         /work/analysis/geo/tests/vectors

import java.io.ByteArrayInputStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import opennlp.geo.ContainmentSpine;
import opennlp.geo.HierarchyAnnotator;
import opennlp.geo.PopulationPriorGeocoder;
import opennlp.geo.SpatialCoherenceGeocoder;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.geo.AttributeValue;
import opennlp.tools.geo.ContainmentChain;
import opennlp.tools.geo.DocumentRegionAnnotator;
import opennlp.tools.geo.Gazetteer;
import opennlp.tools.geo.GazetteerEntry;
import opennlp.tools.geo.GeoPoint;
import opennlp.tools.geo.GeoResolution;
import opennlp.tools.geo.GeocodeAnnotator;
import opennlp.tools.geo.Geocoder;
import opennlp.tools.geo.PlaceAncestor;
import opennlp.tools.geo.RegionVote;
import opennlp.tools.money.MoneyAmount;
import opennlp.tools.money.MoneyAnnotator;
import opennlp.tools.money.RegionAwareMoneyAnnotator;
import opennlp.tools.util.Span;

public final class GeoPipelineVectors {

  private static final String OPENNLP_COMMIT = "cdf33c653d5d61d5af7d37ffee189a83bd8392ea";

  private static final String SPINE_TSV =
      String.join(
          "\n",
          "# id\tparent_id\tname\tplace_type",
          "100\t10\tSpringfield\tlocality",
          "101\t11\tSpringfield\tlocality",
          "102\t12\tChicago\tlocality",
          "103\t20\tParis\tlocality",
          "104\t13\tParis\tlocality",
          "105\t21\tLyon\tlocality",
          "106\t30\tTurin\tlocality",
          "107\t40\tPorto\tlocality",
          "108\t50\tSydney\tlocality",
          "109\t60\tGuadalajara\tlocality",
          "110\t10\tBoston\tlocality",
          "10\t1\tMassachusetts\tregion",
          "11\t1\tIllinois\tregion",
          "12\t1\tIllinois\tregion",
          "13\t1\tTexas\tregion",
          "20\t2\tIle-de-France\tregion",
          "21\t2\tAuvergne-Rhone-Alpes\tregion",
          "30\t3\tPiedmont\tregion",
          "40\t4\tPorto District\tregion",
          "50\t5\tNew South Wales\tregion",
          "60\t6\tJalisco\tregion",
          "1\t\tUnited States\tcountry",
          "2\t\tFrance\tcountry",
          "3\t\tItaly\tcountry",
          "4\t\tPortugal\tcountry",
          "5\t\tAustralia\tcountry",
          "6\t\tMexico\tcountry",
          "");

  private static final String SPINE_WOF_CSV =
      String.join(
          "\n",
          "id,parent_id,name,placetype",
          "200,20,\"Saint-Germain, Paris\",neighbourhood",
          "20,-1,Ile-de-France,region",
          "");

  private GeoPipelineVectors() {}

  private static GazetteerEntry entry(
      String id,
      String name,
      double lat,
      double lon,
      String country,
      long population,
      String wofId) {
    Map<String, AttributeValue> attrs =
        wofId == null
            ? Map.of()
            : Map.of(
                GazetteerEntry.ATTRIBUTE_KEY_WHOSONFIRST,
                new AttributeValue(wofId, "wof", ""));
    return new GazetteerEntry(
        "fixture",
        id,
        name,
        List.of(),
        new GeoPoint(lat, lon),
        country,
        List.of(),
        population,
        GazetteerEntry.FEATURE_CLASS_CITY,
        attrs);
  }

  private static Gazetteer fixtureGazetteer() {
    Map<String, List<GazetteerEntry>> map = new LinkedHashMap<>();
    map.put(
        "Springfield",
        List.of(
            entry("us-il-springfield", "Springfield", 39.7817, -89.6501, "US", 114394, "101"),
            entry("us-ma-springfield", "Springfield", 42.1015, -72.5898, "US", 155929, "100")));
    map.put(
        "Chicago",
        List.of(entry("us-il-chicago", "Chicago", 41.8781, -87.6298, "US", 2746388, "102")));
    map.put(
        "Boston",
        List.of(entry("us-ma-boston", "Boston", 42.3601, -71.0589, "US", 675647, "110")));
    map.put(
        "Paris",
        List.of(
            entry("fr-paris", "Paris", 48.8566, 2.3522, "FR", 2161000, "103"),
            entry("us-tx-paris", "Paris", 33.6609, -95.5555, "US", 24476, "104")));
    map.put(
        "Lyon",
        List.of(entry("fr-lyon", "Lyon", 45.7640, 4.8357, "FR", 522250, "105")));
    map.put(
        "Turin",
        List.of(entry("it-turin", "Turin", 45.0703, 7.6869, "IT", 848748, "106")));
    map.put(
        "Porto",
        List.of(entry("pt-porto", "Porto", 41.1579, -8.6291, "PT", 231800, "107")));
    map.put(
        "Sydney",
        List.of(entry("au-sydney", "Sydney", -33.8688, 151.2093, "AU", 5312163, "108")));
    map.put(
        "Guadalajara",
        List.of(entry("mx-guadalajara", "Guadalajara", 20.6597, -103.3496, "MX", 1385629, "109")));
    map.put(
        "Saint-Germain",
        List.of(entry("fr-st-germain", "Saint-Germain", 48.8540, 2.3327, "FR", 50000, "200")));

    return new Gazetteer() {
      @Override
      public List<GazetteerEntry> lookup(CharSequence name) {
        return map.getOrDefault(name.toString(), List.of());
      }

      @Override
      public Optional<GazetteerEntry> byId(String source, String recordId) {
        return map.values().stream()
            .flatMap(Collection::stream)
            .filter(e -> e.source().equals(source) && e.recordId().equals(recordId))
            .findFirst();
      }

      @Override
      public Optional<GazetteerEntry> byRegion(String countryCode) {
        return map.values().stream()
            .flatMap(Collection::stream)
            .filter(e -> countryCode.equals(e.countryCode()))
            .findFirst();
      }

      @Override
      public Set<String> sources() {
        return Set.of("fixture");
      }
    };
  }

  private record Scenario(String id, String text, List<String> mentions) {}

  private static String hex(double x) {
    return String.format("%016x", Double.doubleToRawLongBits(x));
  }

  private static int utf8ByteOffset(String text, int charIndex) {
    return text.substring(0, charIndex).getBytes(StandardCharsets.UTF_8).length;
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 1) {
      System.err.println("usage: java GeoPipelineVectors.java <out-dir>");
      System.exit(2);
    }
    Path out = Path.of(args[0]);
    Files.createDirectories(out);

    Gazetteer gaz = fixtureGazetteer();
    Path tsvTmp = Files.createTempFile("spine-", ".tsv");
    Path csvTmp = Files.createTempFile("wof-", ".csv");
    Files.writeString(tsvTmp, SPINE_TSV, StandardCharsets.UTF_8);
    Files.writeString(csvTmp, SPINE_WOF_CSV, StandardCharsets.UTF_8);
    ContainmentSpine spine =
        ContainmentSpine.builder().addTable(tsvTmp).addWofMeta(csvTmp).build();
    Files.deleteIfExists(tsvTmp);
    Files.deleteIfExists(csvTmp);

    List<Scenario> scenarios =
        List.of(
            new Scenario(
                "multi-country-ballot",
                "Lyon, Turin and Porto met in C\u00F4te d\u2019Ivoire and Saint Barth\u00E9lemy "
                    + "\uD83C\uDDE9\uD83C\uDDEA \uD83C\uDDE6\uD83C\uDDFA before paying \u20AC1.234,56 and $500.",
                List.of("Lyon", "Turin", "Porto", "C\u00F4te d\u2019Ivoire", "Saint Barth\u00E9lemy")),
            new Scenario(
                "spatial-springfield-il",
                "Springfield and Chicago discussed a $1,250.50 contract.",
                List.of("Springfield", "Chicago")),
            new Scenario(
                "spatial-springfield-ma",
                "Springfield and Boston discussed a $1,250.50 contract.",
                List.of("Springfield", "Boston")),
            new Scenario(
                "guadalajara-boston-mxn",
                "Guadalajara outsells Boston this quarter, and lunch there costs $8.",
                List.of("Guadalajara", "Boston")),
            new Scenario(
                "paris-lyon-wof",
                "Paris, Lyon and Saint-Germain raised \u20AC500 and $200 in France \uD83C\uDDEB\uD83C\uDDF7.",
                List.of("Paris", "Lyon", "Saint-Germain", "France")),
            new Scenario(
                "sydney-australia-aud",
                "Sydney and Australia \uD83C\uDDE6\uD83C\uDDFA closed a $3 million round.",
                List.of("Sydney", "Australia")));

    Map<String, Geocoder> geocoders = new LinkedHashMap<>();
    geocoders.put("prior", new PopulationPriorGeocoder(gaz));
    geocoders.put("spatial", new SpatialCoherenceGeocoder(gaz));

    Path file = out.resolve("geo_pipeline.tsv");
    try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.US_ASCII))) {
      w.println(
          "# generator=GeoPipelineVectors.java\topennlp.commit="
              + OPENNLP_COMMIT
              + "\tjava.version="
              + System.getProperty("java.version")
              + "\tjava.vm.version="
              + System.getProperty("java.vm.version")
              + "\tcolumns=scenario,geocoder,layer,records");

      for (Scenario sc : scenarios) {
        List<Annotation<String>> entityList = new ArrayList<>();
        for (String m : sc.mentions()) {
          int start = sc.text().indexOf(m);
          if (start < 0) {
            throw new IllegalStateException("missing mention: " + m);
          }
          entityList.add(new Annotation<>(new Span(start, start + m.length()), "location"));
        }

        for (Map.Entry<String, Geocoder> gEntry : geocoders.entrySet()) {
          String gName = gEntry.getKey();
          Geocoder geocoder = gEntry.getValue();

          Document doc = Document.of(sc.text()).with(Layers.ENTITIES, entityList);
          doc = new GeocodeAnnotator(geocoder).annotate(doc);
          doc = new DocumentRegionAnnotator().annotate(doc);
          doc = new HierarchyAnnotator(spine).annotate(doc);
          doc = new RegionAwareMoneyAnnotator().annotate(doc);

          // 1. Locations
          List<Annotation<GeoResolution>> locs = doc.get(GeocodeAnnotator.LOCATIONS);
          StringBuilder locStr = new StringBuilder();
          for (int i = 0; i < locs.size(); i++) {
            if (i > 0) {
              locStr.append(';');
            }
            GeoResolution r = locs.get(i).value();
            int sByte = utf8ByteOffset(sc.text(), r.mention().getStart());
            int eByte = utf8ByteOffset(sc.text(), r.mention().getEnd());
            GazetteerEntry e = r.entry();
            locStr
                .append(sByte)
                .append("..")
                .append(eByte)
                .append(':')
                .append(e.source())
                .append(':')
                .append(e.recordId())
                .append(':')
                .append(e.countryCode() == null ? "-" : e.countryCode())
                .append(':')
                .append(e.population())
                .append(':')
                .append(hex(e.location().latitude()))
                .append(':')
                .append(hex(e.location().longitude()))
                .append(':')
                .append(hex(r.confidence()));
          }
          w.println(sc.id() + "\t" + gName + "\tlocations\t" + locStr);

          // 2. Regions
          List<Annotation<RegionVote>> votes = doc.get(DocumentRegionAnnotator.REGIONS);
          StringBuilder regStr = new StringBuilder();
          for (int i = 0; i < votes.size(); i++) {
            if (i > 0) {
              regStr.append(';');
            }
            RegionVote v = votes.get(i).value();
            regStr
                .append(v.countryCode())
                .append(':')
                .append(hex(v.share()))
                .append(':')
                .append(Double.toString(v.share()));
          }
          w.println(sc.id() + "\t" + gName + "\tregions\t" + regStr);

          // 3. Winners at margins 0.0, 0.05, 0.25, 0.5
          StringBuilder winStr = new StringBuilder();
          double[] margins = {0.0, 0.05, 0.25, 0.5};
          for (int i = 0; i < margins.length; i++) {
            if (i > 0) {
              winStr.append(';');
            }
            Optional<RegionVote> win = DocumentRegionAnnotator.winner(votes, margins[i]);
            winStr
                .append(Double.toString(margins[i]))
                .append('=')
                .append(win.map(RegionVote::countryCode).orElse("-"));
          }
          w.println(sc.id() + "\t" + gName + "\twinners\t" + winStr);

          // 4. Containment
          List<Annotation<ContainmentChain>> chains = doc.get(HierarchyAnnotator.CONTAINMENT);
          StringBuilder contStr = new StringBuilder();
          for (int i = 0; i < chains.size(); i++) {
            if (i > 0) {
              contStr.append(';');
            }
            Annotation<ContainmentChain> ann = chains.get(i);
            int sByte = utf8ByteOffset(sc.text(), ann.span().getStart());
            int eByte = utf8ByteOffset(sc.text(), ann.span().getEnd());
            contStr.append(sByte).append("..").append(eByte).append(':');
            List<PlaceAncestor> ancestors = ann.value().ancestors();
            for (int j = 0; j < ancestors.size(); j++) {
              if (j > 0) {
                contStr.append('>');
              }
              PlaceAncestor a = ancestors.get(j);
              contStr.append(a.id()).append('|').append(a.name()).append('|').append(a.type());
            }
          }
          w.println(sc.id() + "\t" + gName + "\tcontainment\t" + contStr);

          // 5. Money
          List<Annotation<MoneyAmount>> money = doc.get(MoneyAnnotator.MONEY);
          StringBuilder moneyStr = new StringBuilder();
          for (int i = 0; i < money.size(); i++) {
            if (i > 0) {
              moneyStr.append(';');
            }
            MoneyAmount m = money.get(i).value();
            int sByte = utf8ByteOffset(sc.text(), m.span().getStart());
            int eByte = utf8ByteOffset(sc.text(), m.span().getEnd());
            moneyStr
                .append(sByte)
                .append("..")
                .append(eByte)
                .append(':')
                .append(m.amount().toPlainString())
                .append(':')
                .append(m.currency());
          }
          w.println(sc.id() + "\t" + gName + "\tmoney\t" + moneyStr);
        }
      }
    }
  }
}
