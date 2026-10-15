package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTagFormat;
import opennlp.tools.postag.POSTagFormatMapper;
import opennlp.tools.postag.POSTaggerME;

/**
 * {@code POSTagFormatMapper} for a tagger model's outcomes, run on the {@code sequence/sentences}
 * input, and the format it guesses for lists of outcomes, run on {@code tag-format/guesses}.
 *
 * <p>A model run's layers, anchored to the document: {@code turbo:outcomes} (the model's
 * outcomes, in order), {@code turbo:guessed-format} (one row), {@code turbo:converted-tags}
 * ({@code convertTag} of every outcome, every key of both conversion tables and a few other
 * strings) and {@code turbo:requested-tags}: for each format asked of {@code POSTaggerME} in turn
 * ({@code UD}, {@code PENN}, {@code CUSTOM}, {@code UNKNOWN}), each token's tag, beside the tag the
 * tagger returns with {@code CUSTOM}, which is the outcome unchanged.</p>
 *
 * <p>The guess run's input holds one list of outcomes per line, separated by spaces (an empty
 * line is the empty list); its layer {@code turbo:guessed-formats} has the format guessed for each
 * line.</p>
 */
final class TagFormatGolden implements AnnotatorGolden {

  static final String SET = Runs.TAG_FORMAT;

  private static final POSTagFormat[] REQUESTED = {
      POSTagFormat.UD, POSTagFormat.PENN, POSTagFormat.CUSTOM, POSTagFormat.UNKNOWN};

  /** Every key of both conversion tables of POSTagFormatMapper, and some strings in neither. */
  private static final String[] PROBES = {
      "#", "$", "''", ",", "-LRB-", "-RRB-", ".", ":", "AFX", "CC", "CD", "DT", "EX", "FW",
      "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NIL", "NN", "NNP", "NNPS", "NNS", "PDT",
      "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG",
      "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB",
      "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
      "PROPN", "PUNCT", "SCONJ", "VERB", "X",
      "``", "-LCB-", "-RCB-", "-NONE-", "?", "", "nn", "noun", "NN ", "Noun"};

  /** The model, or null for the guess run. */
  private final ModelFile file;

  private TagFormatGolden(ModelFile file) {
    this.file = file;
  }

  static List<TagFormatGolden> all(Path models, Path fixtures) throws IOException {
    List<TagFormatGolden> out = new ArrayList<>();
    out.add(new TagFormatGolden(ModelFile.read(fixtures.resolve(Runs.POS_FIXTURE))));
    for (String name : List.of(Runs.POS_MODEL_EN, Runs.POS_MODEL_EN_15_MAXENT, Runs.POS_MODEL_EN_15_PERCEPTRON)) {
      out.add(new TagFormatGolden(ModelFile.read(models.resolve(name))));
    }
    out.add(new TagFormatGolden(null));
    return out;
  }

  /** A mapper for a list of outcomes; the constructor is protected. */
  private static POSTagFormatMapper mapper(String[] outcomes) {
    return new POSTagFormatMapper(outcomes) {
    };
  }

  @Override
  public String annotator() {
    return "tag-format";
  }

  @Override
  public List<ModelRef> models() {
    return file == null ? List.of() : List.of(file.ref());
  }

  @Override
  public String modelDir() {
    return file == null ? "guesses" : file.ref().baseName();
  }

  @Override
  public String annotatorClass() {
    return POSTagFormatMapper.class.getName();
  }

  @Override
  public String method() {
    return file == null ? "getGuessedFormat" : "convertTag";
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(file == null ? SET : Runs.SEQUENCE);
  }

  @Override
  public boolean accepts(Input input) {
    return file == null ? input.set().equals(SET)
        : input.set().equals(Runs.SEQUENCE) && input.id().equals("sentences");
  }

  @Override
  public List<Layer> layers(Input input) {
    if (file == null) {
      Table t = new Table("turbo:guessed-formats", "line:u32", "format:str");
      List<String> lines = input.lines();
      for (int i = 0; i < lines.size(); i++) {
        String[] list = lines.get(i).isEmpty() ? new String[0] : lines.get(i).split(" ");
        t.row(i, mapper(list).getGuessedFormat().toString());
      }
      return List.of(t.layer());
    }
    POSModel model;
    try (InputStream in = file.stream()) {
      model = new POSModel(in);
    } catch (IOException e) {
      throw new IllegalStateException(file.ref().fileName(), e);
    }
    String[] outcomes = model.getPosSequenceModel().getOutcomes();
    POSTagFormatMapper mapper = mapper(outcomes);
    if (mapper.getGuessedFormat() != POSTagFormatMapper.guessFormat(model)) {
      throw new IllegalStateException(file.ref().fileName() + ": the guess from the outcomes is not the model's");
    }
    Table outcomeTable = new Table("turbo:outcomes", "outcome:str");
    for (String o : outcomes) {
      outcomeTable.row(o);
    }
    Table guessed = new Table("turbo:guessed-format", "format:str").row(mapper.getGuessedFormat().toString());
    Table converted = new Table("turbo:converted-tags", "tag:str", "converted:str");
    Set<String> probes = new LinkedHashSet<>(Arrays.asList(outcomes));
    probes.addAll(Arrays.asList(PROBES));
    for (String probe : probes) {
      converted.row(probe, mapper.convertTag(probe));
    }
    List<String[]> sentences = Sentences.of(input);
    POSTaggerME custom = new POSTaggerME(model, POSTagFormat.CUSTOM);
    List<String[]> raw = new ArrayList<>();
    for (String[] sentence : sentences) {
      raw.add(custom.tag(sentence));
    }
    Table requested = new Table("turbo:requested-tags", "format:str", "sentence:u32", "raw:str", "tag:str");
    for (POSTagFormat format : REQUESTED) {
      POSTaggerME tagger = new POSTaggerME(model, format);
      for (int k = 0; k < sentences.size(); k++) {
        String[] tags = tagger.tag(sentences.get(k));
        if (tags.length != raw.get(k).length) {
          throw new IllegalStateException(file.ref().fileName() + ": " + format + " gave another number of tags");
        }
        for (int i = 0; i < tags.length; i++) {
          requested.row(format.toString(), k, raw.get(k)[i], tags[i]);
        }
      }
    }
    return List.of(outcomeTable.layer(), guessed.layer(), converted.layer(), requested.layer());
  }
}
