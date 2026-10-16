package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import opennlp.tools.postag.POSDictionary;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.TagDictionary;
import opennlp.tools.util.StringUtil;

/**
 * A part-of-speech tag dictionary as {@code POSDictionary} reads it, from a dictionary file of the
 * fixtures or from a tagger model's {@code tags.tagdict}, run on the {@code sequence/sentences}
 * input.
 *
 * <p>The layers, anchored to the document: {@code turbo:tagdict}, one row: whether there is a
 * dictionary, whether it is case sensitive, its number of words, and the SHA-256 of its words and
 * tags written as lines {@code word TAB count TAB tag TAB tag ...}, sorted by word in UTF-16 order,
 * each ending with a line feed, the strings escaped as the sequence crate's {@code escape} writes
 * them (every UTF-16 unit outside U+0020 to U+007E and the backslash as a backslash, {@code u} and
 * four upper case hex digits, a tab, line feed and carriage return as a backslash and {@code t},
 * {@code n} or {@code r}); for a dictionary file and the fixture model, whose dictionaries are
 * small, {@code turbo:tagdict-entries} (each word and its number of tags) and
 * {@code turbo:tagdict-entry-tags}; and {@code turbo:tagdict-lookups}, {@code getTags} for every
 * token of the sentences and every word of the dictionary (the words of a file in file order, read
 * with a plain scan for {@code <token>}, and of the fixture model sorted), each as it is, in lower
 * case as OpenNLP lowercases and in upper case, with whether it returned an array and the array's
 * length, and {@code turbo:tagdict-lookup-tags}.</p>
 */
final class TagDictionaryGolden implements AnnotatorGolden {

  private final ModelFile file;
  private final boolean isModel;
  private final boolean listEntries;

  private TagDictionaryGolden(ModelFile file, boolean isModel, boolean listEntries) {
    this.file = file;
    this.isModel = isModel;
    this.listEntries = listEntries;
  }

  static List<TagDictionaryGolden> all(Path models, Path fixtures) throws IOException {
    List<TagDictionaryGolden> out = new ArrayList<>();
    for (String name : List.of(Runs.TAG_DICTIONARY, Runs.TAG_DICTIONARY_CASES, Runs.TAG_DICTIONARY_EXACT)) {
      out.add(new TagDictionaryGolden(ModelFile.read(fixtures.resolve(name)), false, true));
    }
    out.add(new TagDictionaryGolden(ModelFile.read(fixtures.resolve(Runs.POS_FIXTURE)), true, true));
    for (String name : List.of(Runs.POS_MODEL_EN, Runs.POS_MODEL_EN_15_MAXENT, Runs.POS_MODEL_EN_15_PERCEPTRON)) {
      out.add(new TagDictionaryGolden(ModelFile.read(models.resolve(name)), true, false));
    }
    return out;
  }

  @Override
  public String annotator() {
    return "tag-dictionary";
  }

  @Override
  public List<ModelRef> models() {
    return List.of(file.ref());
  }

  @Override
  public String modelDir() {
    // A dictionary file keeps its extension out of the path as a model does.
    String name = file.ref().fileName();
    return name.endsWith(".tagdict") ? name.substring(0, name.length() - ".tagdict".length()) : file.ref().baseName();
  }

  @Override
  public String annotatorClass() {
    return POSDictionary.class.getName();
  }

  @Override
  public String method() {
    return "getTags";
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(Runs.SEQUENCE);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(Runs.SEQUENCE) && input.id().equals("sentences");
  }

  @Override
  public List<Layer> layers(Input input) {
    POSDictionary dict;
    List<String> words = new ArrayList<>();
    try {
      if (isModel) {
        POSModel model;
        try (InputStream in = file.stream()) {
          model = new POSModel(in);
        }
        dict = (POSDictionary) model.getFactory().getTagDictionary();
        if (dict != null && listEntries) {
          for (String word : dict) {
            words.add(word);
          }
          words.sort(null);
        }
      } else {
        try (InputStream in = file.stream()) {
          dict = POSDictionary.create(in);
        }
        words = fileWords(new String(file.bytes(), StandardCharsets.UTF_8));
      }
    } catch (IOException e) {
      throw new IllegalStateException(file.ref().fileName(), e);
    }
    Table summary = new Table("turbo:tagdict", "present:u32", "case-sensitive:u32", "words:u32", "sha256:str");
    Table entries = new Table("turbo:tagdict-entries", "entry:u32", "word:str", "tags:u32");
    Table entryTags = new Table("turbo:tagdict-entry-tags", "entry:u32", "tag:str");
    Table lookups = new Table("turbo:tagdict-lookups", "lookup:u32", "word:str", "found:u32", "tags:u32");
    Table lookupTags = new Table("turbo:tagdict-lookup-tags", "lookup:u32", "tag:str");
    if (dict == null) {
      summary.row(0, 0, 0, "");
      return List.of(summary.layer());
    }
    List<String> sorted = new ArrayList<>();
    for (String word : dict) {
      sorted.add(word);
    }
    sorted.sort(null);
    StringBuilder dump = new StringBuilder();
    for (String word : sorted) {
      String[] tags = dict.getTags(word);
      dump.append(escape(word)).append('\t').append(tags.length);
      for (String t : tags) {
        dump.append('\t').append(escape(t));
      }
      dump.append('\n');
      if (listEntries) {
        int e = entries.size();
        entries.row(e, word, tags.length);
        for (String t : tags) {
          entryTags.row(e, t);
        }
      }
    }
    summary.row(1, Table.flag(dict.isCaseSensitive()), sorted.size(),
        Sha.sha256(dump.toString().getBytes(StandardCharsets.UTF_8)));
    Set<String> probes = new LinkedHashSet<>();
    for (String[] sentence : Sentences.of(input)) {
      for (String token : sentence) {
        variants(probes, token);
      }
    }
    for (String word : words) {
      variants(probes, word);
    }
    TagDictionary td = dict;
    for (String probe : probes) {
      String[] tags = td.getTags(probe);
      int l = lookups.size();
      lookups.row(l, probe, Table.flag(tags != null), tags == null ? 0 : tags.length);
      if (tags != null) {
        for (String t : tags) {
          lookupTags.row(l, t);
        }
      }
    }
    List<Layer> layers = new ArrayList<>();
    layers.add(summary.layer());
    if (listEntries) {
      layers.add(entries.layer());
      layers.add(entryTags.layer());
    }
    layers.add(lookups.layer());
    layers.add(lookupTags.layer());
    return layers;
  }

  /** The word, its lower case as OpenNLP lowercases, and its upper case. */
  private static void variants(Set<String> out, String word) {
    out.add(word);
    out.add(StringUtil.toLowerCase(word));
    out.add(word.toUpperCase(Locale.ROOT));
  }

  /** The tokens of every entry of a dictionary file, in file order, read with a plain scan. */
  private static List<String> fileWords(String text) {
    List<String> words = new ArrayList<>();
    int at = 0;
    while ((at = text.indexOf("<token>", at)) >= 0) {
      int end = text.indexOf("</token>", at);
      words.add(text.substring(at + "<token>".length(), end));
      at = end;
    }
    return words;
  }

  /**
   * Every UTF-16 unit outside U+0020 to U+007E, and the backslash, as a backslash, {@code u} and
   * four upper case hex digits, except the tab, line feed and carriage return: the escaping the
   * dictionary's hash is taken over.
   */
  static String escape(String s) {
    StringBuilder b = new StringBuilder(s.length());
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      switch (c) {
        case '\\' -> b.append("\\\\");
        case '\t' -> b.append("\\t");
        case '\n' -> b.append("\\n");
        case '\r' -> b.append("\\r");
        default -> {
          if (c >= 0x20 && c <= 0x7E) {
            b.append(c);
          } else {
            b.append(String.format(Locale.ROOT, "\\u%04X", (int) c));
          }
        }
      }
    }
    return b.toString();
  }
}
