package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

import opennlp.tools.namefind.BilouCodec;
import opennlp.tools.namefind.BioCodec;
import opennlp.tools.util.SequenceCodec;
import opennlp.tools.util.SequenceValidator;
import opennlp.tools.util.Span;

/**
 * What {@code BioCodec} and {@code BilouCodec} and their sequence validators give, run on the
 * inputs {@code codec/bio} and {@code codec/bilou}, each the codec's label alphabet one label per
 * line: decoding every outcome sequence of zero to three labels, the validator on every prior
 * sequence of up to two labels followed by every label, encoding seven span lists (overlapping
 * ones among them), and {@code areOutcomesCompatible} on every subset of six or seven outcomes.
 *
 * <p>The layers, anchored to the document, each case numbered from 0 within its kind:
 * {@code turbo:decode} (each case and its number of outcomes), {@code turbo:decode-outcomes},
 * {@code turbo:decode-spans} (a span's start, end and type, with {@code typed} 0 for a span
 * without a type); {@code turbo:valid} (each case's number of prior outcomes, the outcome asked
 * about and the answer) and {@code turbo:valid-priors}; {@code turbo:encode} (each case's sequence
 * length), {@code turbo:encode-spans} and {@code turbo:encode-outcomes}; {@code turbo:compatible}
 * (each case's answer) and {@code turbo:compatible-outcomes}.</p>
 */
final class CodecGolden implements AnnotatorGolden {

  static final String SET = Runs.CODEC;

  private static final String[] BIO_COMPATIBLE = {"other", "a-start", "a-cont", "b-start", "b-cont", "c"};

  private static final String[] BILOU_COMPATIBLE = {"other", "a-start", "a-cont", "a-last", "a-unit", "b-unit", "x"};

  /** The span lists encoded, the later of two overlapping spans writing over the earlier one. */
  private static final Span[][] ENCODE = {
      {},
      {new Span(0, 1, "person")},
      {new Span(0, 3, "person"), new Span(4, 5, null)},
      {new Span(1, 2, null), new Span(2, 4, "location")},
      {new Span(0, 3, "a"), new Span(1, 2, "b")},
      {new Span(0, 2, "a"), new Span(1, 3, "b")},
      {new Span(3, 4, "x-y")},
  };

  /** The sequence length each span list is encoded into. */
  private static final int[] LENGTHS = {3, 2, 6, 5, 4, 4, 5};

  private final String name;
  private final SequenceCodec<String> codec;
  private final String[] compatible;

  private CodecGolden(String name, SequenceCodec<String> codec, String[] compatible) {
    this.name = name;
    this.codec = codec;
    this.compatible = compatible;
  }

  static List<CodecGolden> all() {
    return List.of(new CodecGolden("bio", new BioCodec(), BIO_COMPATIBLE),
        new CodecGolden("bilou", new BilouCodec(), BILOU_COMPATIBLE));
  }

  @Override
  public String annotator() {
    return "codec";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return name;
  }

  @Override
  public String annotatorClass() {
    return codec.getClass().getName();
  }

  @Override
  public String method() {
    return "decode";
  }

  @Override
  public Map<String, String> config() {
    return Map.of("compatibleOutcomes", String.join(" ", compatible));
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(SET) && input.id().equals(name);
  }

  @Override
  public List<Layer> layers(Input input) {
    String[] labels = input.lines().toArray(new String[0]);
    SequenceValidator<String> validator = codec.createSequenceValidator();

    Table decode = new Table("turbo:decode", "case:u32", "outcomes:u32");
    Table decodeOutcomes = new Table("turbo:decode-outcomes", "case:u32", "outcome:str");
    Table decodeSpans = new Table("turbo:decode-spans", "case:u32", "start:u32", "end:u32", "type:str", "typed:u32");
    int c = 0;
    for (String[] s : sequences(labels, 3)) {
      decode.row(c, s.length);
      for (String o : s) {
        decodeOutcomes.row(c, o);
      }
      spans(decodeSpans, c, codec.decode(Arrays.asList(s)));
      c++;
    }

    Table valid = new Table("turbo:valid", "case:u32", "priors:u32", "outcome:str", "valid:u32");
    Table validPriors = new Table("turbo:valid-priors", "case:u32", "outcome:str");
    c = 0;
    for (String[] prior : sequences(labels, 2)) {
      String[] tokens = new String[prior.length + 1];
      Arrays.fill(tokens, "w");
      for (String outcome : labels) {
        valid.row(c, prior.length, outcome, Table.flag(validator.validSequence(prior.length, tokens, prior, outcome)));
        for (String p : prior) {
          validPriors.row(c, p);
        }
        c++;
      }
    }

    Table encode = new Table("turbo:encode", "case:u32", "length:u32");
    Table encodeSpans = new Table("turbo:encode-spans", "case:u32", "start:u32", "end:u32", "type:str", "typed:u32");
    Table encodeOutcomes = new Table("turbo:encode-outcomes", "case:u32", "outcome:str");
    for (c = 0; c < ENCODE.length; c++) {
      encode.row(c, LENGTHS[c]);
      spans(encodeSpans, c, ENCODE[c]);
      for (String o : codec.encode(ENCODE[c], LENGTHS[c])) {
        encodeOutcomes.row(c, o);
      }
    }

    Table compatibleCases = new Table("turbo:compatible", "case:u32", "compatible:u32");
    Table compatibleOutcomes = new Table("turbo:compatible-outcomes", "case:u32", "outcome:str");
    for (int mask = 0; mask < 1 << compatible.length; mask++) {
      List<String> subset = new ArrayList<>();
      for (int i = 0; i < compatible.length; i++) {
        if ((mask & (1 << i)) != 0) {
          subset.add(compatible[i]);
          compatibleOutcomes.row(mask, compatible[i]);
        }
      }
      compatibleCases.row(mask, Table.flag(codec.areOutcomesCompatible(subset.toArray(new String[0]))));
    }
    return List.of(decode.layer(), decodeOutcomes.layer(), decodeSpans.layer(), valid.layer(), validPriors.layer(),
        encode.layer(), encodeSpans.layer(), encodeOutcomes.layer(), compatibleCases.layer(),
        compatibleOutcomes.layer());
  }

  private static void spans(Table t, int c, Span[] spans) {
    for (Span s : spans) {
      t.row(c, s.getStart(), s.getEnd(), s.getType() == null ? "" : s.getType(), Table.flag(s.getType() != null));
    }
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
}
