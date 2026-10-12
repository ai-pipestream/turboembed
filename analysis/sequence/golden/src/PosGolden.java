import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import opennlp.tools.postag.POSDictionary;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTagFormat;
import opennlp.tools.postag.POSTagFormatMapper;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.postag.TagDictionary;
import opennlp.tools.util.StringUtil;

/**
 * Writes the part-of-speech tag dictionary and tag format goldens.
 *
 * <p>{@code tagdict-<name>.txt}: a tag dictionary as {@code POSDictionary} reads it, from a
 * dictionary file or from a model's {@code tags.tagdict}. The file gives whether the dictionary
 * is case sensitive, its number of words, and the SHA-256 of its words and tags as lines
 * {@code word TAB count TAB tag TAB tag ...}, sorted by word in UTF-16 order, each ending with a line
 * feed, with the strings escaped. For a small dictionary every such line follows as an
 * {@code entry:}. Then {@code getTags} for probes: every token of the sentences, its lower and
 * upper case, and every word of the dictionary file in the same three forms, each with the
 * count and tags {@code getTags} returns or {@code null}.
 *
 * <p>{@code tagformat-<name>.txt}: {@code POSTagFormatMapper} for a model's outcomes: the
 * model's SHA-256, its outcomes, the guessed format, {@code convertTag} of every outcome, every key of both conversion tables and
 * a few other strings, and what {@code POSTaggerME.tag} returns for each sentence with each
 * requested format, next to the tags it returns with {@code CUSTOM} (which are the outcomes
 * unchanged).
 *
 * <p>{@code tagformat-guess.txt}: the format guessed for some lists of outcomes.
 *
 * <p>Arguments: the golden directory, the output directory, the commit, and optionally the
 * directory of the public models.
 */
public final class PosGolden {

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

  private PosGolden() {
  }

  /** A mapper for a list of outcomes; the constructor is protected. */
  private static POSTagFormatMapper mapper(String[] outcomes) {
    return new POSTagFormatMapper(outcomes) {
    };
  }

  private static String sha256(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException(e);
    }
  }

  /** The number of tags, then the tags, joined by tabs; {@code null} for no array. */
  private static String tags(String[] tags) {
    if (tags == null) {
      return "null";
    }
    return tags.length == 0 ? "0" : tags.length + "\t" + Common.join(tags);
  }

  /** The word, its lower case as OpenNLP lowercases, and its upper case. */
  private static void variants(Set<String> out, String word) {
    out.add(word);
    out.add(StringUtil.toLowerCase(word));
    out.add(word.toUpperCase(Locale.ROOT));
  }

  private static void tagDictionary(Path out, String name, String commit, POSDictionary dict,
      boolean listEntries, List<String[]> sentences, List<String> words) throws IOException {
    try (Writer w = Common.writer(out.resolve("tagdict-" + name + ".txt"))) {
      Common.header(w, "POSDictionary", commit);
      Common.line(w, "present: " + (dict != null));
      if (dict == null) {
        return;
      }
      List<String> sorted = new ArrayList<>();
      for (String word : dict) {
        sorted.add(word);
      }
      sorted.sort(null);
      StringBuilder dump = new StringBuilder();
      List<String> lines = new ArrayList<>();
      for (String word : sorted) {
        String line = Common.escape(word) + "\t" + tags(dict.getTags(word));
        lines.add(line);
        dump.append(line).append('\n');
      }
      Common.line(w, "case_sensitive: " + dict.isCaseSensitive());
      Common.line(w, "words: " + sorted.size());
      Common.line(w, "sha256: " + sha256(dump.toString().getBytes(StandardCharsets.UTF_8)));
      if (listEntries) {
        for (String line : lines) {
          Common.line(w, "entry: " + line);
        }
      }
      Set<String> probes = new LinkedHashSet<>();
      for (String[] sentence : sentences) {
        for (String token : sentence) {
          variants(probes, token);
        }
      }
      for (String word : words) {
        variants(probes, word);
      }
      TagDictionary td = dict;
      for (String probe : probes) {
        Common.line(w, "lookup: " + Common.escape(probe) + "\t" + tags(td.getTags(probe)));
      }
    }
  }

  private static POSDictionary readDictionary(Path file) throws IOException {
    try (InputStream in = Files.newInputStream(file)) {
      return POSDictionary.create(in);
    }
  }

  /** The tokens of every entry of a dictionary file, in file order, read with a plain scan. */
  private static List<String> fileWords(Path file) throws IOException {
    List<String> words = new ArrayList<>();
    String text = Files.readString(file, StandardCharsets.UTF_8);
    int at = 0;
    while ((at = text.indexOf("<token>", at)) >= 0) {
      int end = text.indexOf("</token>", at);
      words.add(text.substring(at + "<token>".length(), end));
      at = end;
    }
    return words;
  }

  private static void model(Path out, String name, String commit, Path file,
      List<String[]> sentences, boolean listEntries) throws IOException {
    POSModel model = new POSModel(file);
    POSDictionary dict = (POSDictionary) model.getFactory().getTagDictionary();
    List<String> words = new ArrayList<>();
    if (dict != null && listEntries) {
      for (String word : dict) {
        words.add(word);
      }
      words.sort(null);
    }
    tagDictionary(out, name, commit, dict, listEntries, sentences, words);

    String[] outcomes = model.getPosSequenceModel().getOutcomes();
    POSTagFormatMapper mapper = mapper(outcomes);
    check(mapper.getGuessedFormat() == POSTagFormatMapper.guessFormat(model), name);
    try (Writer w = Common.writer(out.resolve("tagformat-" + name + ".txt"))) {
      Common.header(w, "POSTagFormatMapper", commit);
      Common.line(w, "model: " + Common.sha256(file));
      Common.line(w, "outcomes: " + Common.join(outcomes));
      Common.line(w, "guessed: " + mapper.getGuessedFormat());
      Set<String> probes = new LinkedHashSet<>(Arrays.asList(outcomes));
      probes.addAll(Arrays.asList(PROBES));
      for (String probe : probes) {
        Common.line(w, "convert: " + Common.escape(probe) + "\t" + Common.escape(mapper.convertTag(probe)));
      }
      POSTaggerME custom = new POSTaggerME(model, POSTagFormat.CUSTOM);
      List<String[]> raw = new ArrayList<>();
      for (String[] sentence : sentences) {
        raw.add(custom.tag(sentence));
      }
      for (POSTagFormat requested : REQUESTED) {
        POSTaggerME tagger = new POSTaggerME(model, requested);
        Common.line(w, "requested: " + requested);
        for (int k = 0; k < sentences.size(); k++) {
          Common.line(w, "raw: " + Common.join(raw.get(k)));
          Common.line(w, "tags: " + Common.join(tagger.tag(sentences.get(k))));
        }
      }
    }
  }

  private static void guesses(Path out, String commit) throws IOException {
    String[][] lists = {
        {},
        {"SYM"},
        {"NN", "NOUN"},
        {"NN", "NOUN", "VERB"},
        {"NN", "NOUN", "VB"},
        {"PERSON", "LOCATION"},
        {"X", "FW", "SYM", "NIL"},
        {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
            "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"},
        {"``", "''", "-LCB-", "-RCB-"},
        {"noun", "nn"},
    };
    try (Writer w = Common.writer(out.resolve("tagformat-guess.txt"))) {
      Common.header(w, "POSTagFormatMapper", commit);
      for (String[] list : lists) {
        Common.line(w, "outcomes: " + Common.join(list));
        Common.line(w, "guessed: " + mapper(list).getGuessedFormat());
      }
    }
  }

  private static void check(boolean ok, String what) {
    if (!ok) {
      throw new IllegalStateException("check failed: " + what);
    }
  }

  public static void main(String[] args) throws IOException {
    Path golden = Path.of(args[0]);
    Path out = Path.of(args[1]);
    String commit = args[2];
    List<String[]> sentences = Common.sentences(golden.resolve("inputs/sentences.txt"));

    for (String name : new String[] {"tags", "tags-cases", "tags-exact"}) {
      Path file = golden.resolve("resources/" + name + ".tagdict");
      tagDictionary(out, name, commit, readDictionary(file), true, sentences, fileWords(file));
    }
    model(out, "pos-fixture", commit, golden.resolve("models/pos-fixture.bin"), sentences, true);
    guesses(out, commit);

    if (args.length > 3) {
      Path real = Path.of(args[3]);
      model(out, "en-ud-ewt-pos", commit, real.resolve("opennlp-en-ud-ewt-pos-1.3-2.5.4.bin"),
          sentences, false);
      model(out, "en-pos-maxent", commit, real.resolve("en-pos-maxent.bin"), sentences, false);
      model(out, "en-pos-perceptron", commit, real.resolve("en-pos-perceptron.bin"), sentences,
          false);
    }
  }
}
