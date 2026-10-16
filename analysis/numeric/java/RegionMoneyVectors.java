// Writes the region-aware money test vectors of the turbo-numeric crate from
// the pinned JDK 21 image and OpenNLP commit cdf33c653d5d61d5af7d37ffee189a83bd8392ea.
//
// Run from the repository root with the JDK 21 source launcher:
//
//     docker run --rm --user "$(id -u):$(id -g)" \
//         -v "$PWD:/work" -v /work/worktrees/opennlp/uber:/opennlp:ro \
//         eclipse-temurin@sha256:3e3c176ffed168beb42c607be9bc1639b466cf00261a0fb04425562c9d0c5c2b \
//         java -cp "/opennlp/opennlp-api/target/classes:/opennlp/opennlp-core/opennlp-runtime/target/classes" \
//         /work/analysis/numeric/java/RegionMoneyVectors.java \
//         /work/analysis/numeric/tests/vectors

import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.extraction.NumberNotation;
import opennlp.tools.geo.DocumentRegionAnnotator;
import opennlp.tools.geo.RegionVote;
import opennlp.tools.money.CursorMoneyExtractor;
import opennlp.tools.money.MoneyAmount;
import opennlp.tools.money.MoneyAnnotator;
import opennlp.tools.money.RegionAwareMoneyAnnotator;

public final class RegionMoneyVectors {

  private static final String OPENNLP_COMMIT = "cdf33c653d5d61d5af7d37ffee189a83bd8392ea";

  private static final String PROBE_TEXT =
      "$1,234.56 $1.234,56 \u20AC1,234.56 \u20AC1.234,56 \u00A350 \u00A5100 "
          + "\u060B20 \u058F30 \u20BC40 \u09F350 \u20A160 \u20BE70 \u20AA80 "
          + "\u20B990 \u17DB10 \u20A920 \u20B830 \u20AD40 \u20AE50 \u20A660 "
          + "\u20B170 \u20BD80 \u0E3F90 \u20BA10 \u20B420 \u20AB30";

  private RegionMoneyVectors() {}

  @SuppressWarnings("unchecked")
  public static void main(String[] args) throws Exception {
    if (args.length != 1) {
      System.err.println("usage: java RegionMoneyVectors.java <out-dir>");
      System.exit(2);
    }
    Path out = Path.of(args[0]);
    Files.createDirectories(out);

    RegionAwareMoneyAnnotator annotator = new RegionAwareMoneyAnnotator();
    Method localeOf = RegionAwareMoneyAnnotator.class.getDeclaredMethod("localeOf", String.class);
    localeOf.setAccessible(true);
    Method extractorFor =
        RegionAwareMoneyAnnotator.class.getDeclaredMethod("extractorFor", String.class);
    extractorFor.setAccessible(true);
    Field notationField = CursorMoneyExtractor.class.getDeclaredField("notation");
    notationField.setAccessible(true);
    Field cpsField = CursorMoneyExtractor.class.getDeclaredField("symbolCodePoints");
    cpsField.setAccessible(true);
    Field codesField = CursorMoneyExtractor.class.getDeclaredField("currencyCodes");
    codesField.setAccessible(true);

    List<String> countries = new ArrayList<>(List.of(Locale.getISOCountries()));
    countries.add("ZZ");

    Path file = out.resolve("region_for_region.tsv");
    try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.US_ASCII))) {
      w.println(
          "# generator=RegionMoneyVectors.java\topennlp.commit="
              + OPENNLP_COMMIT
              + "\tjava.version="
              + System.getProperty("java.version")
              + "\tjava.vm.version="
              + System.getProperty("java.vm.version")
              + "\tcolumns=country,locale,notation,symbols,mentions");
      for (String cc : countries) {
        Optional<Locale> loc = (Optional<Locale>) localeOf.invoke(null, cc);
        CursorMoneyExtractor ext = (CursorMoneyExtractor) extractorFor.invoke(annotator, cc);
        NumberNotation notation = (NumberNotation) notationField.get(ext);
        int[] cps = (int[]) cpsField.get(ext);
        String[] codes = (String[]) codesField.get(ext);
        StringBuilder syms = new StringBuilder();
        for (int i = 0; i < cps.length; i++) {
          if (i > 0) {
            syms.append(',');
          }
          syms.append(String.format("U+%04X=%s", cps[i], codes[i]));
        }

        Document doc =
            Document.of(PROBE_TEXT)
                .with(
                    DocumentRegionAnnotator.REGIONS,
                    List.of(Annotation.of(new RegionVote(cc, 1.0))));
        Document annotated = annotator.annotate(doc);
        List<Annotation<MoneyAmount>> amounts = annotated.get(MoneyAnnotator.MONEY);
        StringBuilder mentions = new StringBuilder();
        for (int i = 0; i < amounts.size(); i++) {
          if (i > 0) {
            mentions.append(';');
          }
          MoneyAmount m = amounts.get(i).value();
          int startByte =
              PROBE_TEXT
                  .substring(0, m.span().getStart())
                  .getBytes(StandardCharsets.UTF_8)
                  .length;
          int endByte =
              PROBE_TEXT
                  .substring(0, m.span().getEnd())
                  .getBytes(StandardCharsets.UTF_8)
                  .length;
          mentions
              .append(startByte)
              .append("..")
              .append(endByte)
              .append(':')
              .append(m.amount().toPlainString())
              .append(':')
              .append(m.currency());
        }

        w.println(
            cc
                + "\t"
                + loc.map(Locale::toLanguageTag).orElse("-")
                + "\t"
                + notation.name()
                + "\t"
                + syms
                + "\t"
                + mentions);
      }
    }
  }
}
