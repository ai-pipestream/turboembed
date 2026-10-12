package ai.pipestream.turbo.goldens;

import java.util.List;

/**
 * Writes a decision point dump (docs/analysis.md, "Decision point dumps"): the first line
 * {@code dump 1}, the header records a golden has, then one block per decision point with its
 * contexts, its outcome probabilities and, for a sentence detector's split, whether the break was
 * acceptable. The cells use the golden format's forms ({@code u32},
 * {@code f64}, escaped {@code str}), so writing the same values twice gives the same bytes.
 */
final class DumpWriter {

  /** The format version on the first line. */
  static final int FORMAT_VERSION = 1;

  private DumpWriter() {}

  /** Returns the whole file for one header, the model's outcome labels and the points, in UTF-8. */
  static byte[] write(Header h, String[] outcomeLabels, List<DecisionPoint> points) {
    StringBuilder out = new StringBuilder();
    out.append("dump ").append(FORMAT_VERSION).append('\n');
    GoldenWriter.header(out, h);
    for (int i = 0; i < points.size(); i++) {
      point(out, i, outcomeLabels, points.get(i));
    }
    return GoldenWriter.utf8(out);
  }

  private static void point(StringBuilder out, int index, String[] labels, DecisionPoint p) {
    if (p.outcomes().length != labels.length) {
      throw new IllegalArgumentException("point " + index + " has " + p.outcomes().length + " outcomes, the model "
          + labels.length);
    }
    GoldenWriter.line(out, "point", GoldenWriter.u32(index), GoldenWriter.u32(p.position()),
        GoldenWriter.u32(p.contexts().length), GoldenWriter.u32(labels.length), GoldenWriter.u32(p.unit()),
        p.values() == null ? "none" : "given");
    for (int c = 0; c < p.contexts().length; c++) {
      String context = GoldenWriter.escape(p.contexts()[c]);
      if (p.values() == null) {
        GoldenWriter.line(out, "context", context);
      } else {
        GoldenWriter.line(out, "context", context, GoldenWriter.f64(p.values()[c]));
      }
    }
    for (int o = 0; o < labels.length; o++) {
      GoldenWriter.line(out, "outcome", GoldenWriter.escape(labels[o]), GoldenWriter.f64(p.outcomes()[o]));
    }
    if (p.acceptable() != null) {
      GoldenWriter.line(out, "acceptable", GoldenWriter.u32(p.acceptable() ? 1 : 0));
    }
    GoldenWriter.line(out, "end", GoldenWriter.u32(index));
  }
}
