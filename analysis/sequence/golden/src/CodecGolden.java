import java.io.IOException;
import java.io.Writer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import opennlp.tools.namefind.BilouCodec;
import opennlp.tools.namefind.BioCodec;
import opennlp.tools.util.SequenceCodec;
import opennlp.tools.util.SequenceValidator;
import opennlp.tools.util.Span;

/**
 * Writes what {@code BioCodec} and {@code BilouCodec} and their validators give for every outcome
 * sequence of up to three labels from a small alphabet, for some spans to encode and for every
 * subset of a set of outcomes.
 *
 * <p>Lines, all fields separated by tabs, strings escaped as in {@link Common#escape}, a span
 * written as start, end and type (or {@code null}) separated by spaces:
 * <ul>
 *   <li>{@code decode} then the outcomes, then {@code =>} and the spans;</li>
 *   <li>{@code valid} then the prior outcomes, then {@code =>}, the outcome and 0 or 1;</li>
 *   <li>{@code encode} then the length and the spans, then {@code =>} and the outcomes;</li>
 *   <li>{@code compatible} then the outcomes, then {@code =>} and 0 or 1.</li>
 * </ul>
 */
public final class CodecGolden {

  private CodecGolden() {
  }

  static final String[] BIO_LABELS = {
      "other", "a-start", "a-cont", "b-start", "b-cont", "start", "cont", "x-other", "a-b-start",
      "a b-cont"
  };

  static final String[] BILOU_LABELS = {
      "other", "a-start", "a-cont", "a-last", "a-unit", "b-start", "b-last", "b-unit", "start",
      "last", "unit", "cont"
  };

  static final String[] BIO_COMPATIBLE = {"other", "a-start", "a-cont", "b-start", "b-cont", "c"};

  static final String[] BILOU_COMPATIBLE = {
      "other", "a-start", "a-cont", "a-last", "a-unit", "b-unit", "x"
  };

  static String spans(Span[] spans) {
    List<String> out = new ArrayList<>();
    for (Span s : spans) {
      out.add(s.getStart() + " " + s.getEnd() + " " + (s.getType() == null ? "null" : s.getType()));
    }
    return Common.join(out);
  }

  /** Every sequence of length 0 to {@code max} over {@code labels}, shortest first. */
  static List<String[]> sequences(String[] labels, int max) {
    List<String[]> out = new ArrayList<>();
    out.add(new String[0]);
    List<String[]> last = out;
    for (int n = 1; n <= max; n++) {
      List<String[]> next = new ArrayList<>();
      for (String[] prefix : last) {
        for (String l : labels) {
          String[] s = Arrays.copyOf(prefix, n);
          s[n - 1] = l;
          next.add(s);
        }
      }
      out.addAll(next);
      last = next;
    }
    return out;
  }

  static String tab(String s) {
    return s.isEmpty() ? "" : "\t" + s;
  }

  static void write(Writer w, SequenceCodec<String> codec, String[] labels, String[] compatible,
      Span[][] encode, int[] lengths) throws IOException {
    SequenceValidator<String> validator = codec.createSequenceValidator();
    for (String[] s : sequences(labels, 3)) {
      Common.line(w, "decode" + tab(Common.join(s)) + "\t=>"
          + tab(spans(codec.decode(Arrays.asList(s)))));
    }
    for (String[] prior : sequences(labels, 2)) {
      String[] tokens = new String[prior.length + 1];
      Arrays.fill(tokens, "w");
      for (String outcome : labels) {
        boolean ok = validator.validSequence(prior.length, tokens, prior, outcome);
        Common.line(w, "valid" + tab(Common.join(prior)) + "\t=>\t" + Common.escape(outcome) + "\t"
            + (ok ? 1 : 0));
      }
    }
    for (int i = 0; i < encode.length; i++) {
      Common.line(w, "encode\t" + lengths[i] + tab(spans(encode[i])) + "\t=>"
          + tab(Common.join(codec.encode(encode[i], lengths[i]))));
    }
    for (int mask = 0; mask < 1 << compatible.length; mask++) {
      List<String> subset = new ArrayList<>();
      for (int i = 0; i < compatible.length; i++) {
        if ((mask & (1 << i)) != 0) {
          subset.add(compatible[i]);
        }
      }
      boolean ok = codec.areOutcomesCompatible(subset.toArray(new String[0]));
      Common.line(w, "compatible" + tab(Common.join(subset)) + "\t=>\t" + (ok ? 1 : 0));
    }
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]);
    String commit = args[1];
    Span[][] encode = {
        {},
        {new Span(0, 1, "person")},
        {new Span(0, 3, "person"), new Span(4, 5, null)},
        {new Span(1, 2, null), new Span(2, 4, "location")},
        // Overlapping spans: the later one writes over the earlier one.
        {new Span(0, 3, "a"), new Span(1, 2, "b")},
        {new Span(0, 2, "a"), new Span(1, 3, "b")},
        {new Span(3, 4, "x-y")},
    };
    int[] lengths = {3, 2, 6, 5, 4, 4, 5};
    try (Writer w = Common.writer(out.resolve("codec-bio.txt"))) {
      Common.header(w, "codec-bio", commit);
      write(w, new BioCodec(), BIO_LABELS, BIO_COMPATIBLE, encode, lengths);
    }
    try (Writer w = Common.writer(out.resolve("codec-bilou.txt"))) {
      Common.header(w, "codec-bilou", commit);
      write(w, new BilouCodec(), BILOU_LABELS, BILOU_COMPATIBLE, encode, lengths);
    }
  }
}
