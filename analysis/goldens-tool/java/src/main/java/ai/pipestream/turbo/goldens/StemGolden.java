package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;

import opennlp.tools.stemmer.PorterStemmer;
import opennlp.tools.stemmer.Stemmer;
import opennlp.tools.stemmer.light.EnglishMinimalStemmer;
import opennlp.tools.stemmer.light.FinnishLightStemmer;
import opennlp.tools.stemmer.light.FrenchLightStemmer;
import opennlp.tools.stemmer.light.FrenchMinimalStemmer;
import opennlp.tools.stemmer.light.GermanLightStemmer;
import opennlp.tools.stemmer.light.GermanMinimalStemmer;
import opennlp.tools.stemmer.light.HungarianLightStemmer;
import opennlp.tools.stemmer.light.ItalianLightStemmer;
import opennlp.tools.stemmer.light.NorwegianLightStemmer;
import opennlp.tools.stemmer.light.NorwegianMinimalStemmer;
import opennlp.tools.stemmer.light.NorwegianVariety;
import opennlp.tools.stemmer.light.PortugueseLightStemmer;
import opennlp.tools.stemmer.light.RussianLightStemmer;
import opennlp.tools.stemmer.light.SpanishLightStemmer;
import opennlp.tools.stemmer.light.SpanishMinimalStemmer;
import opennlp.tools.stemmer.light.SwedishLightStemmer;
import opennlp.tools.stemmer.light.SwedishMinimalStemmer;
import opennlp.tools.stemmer.snowball.SnowballStemmer;
import opennlp.tools.stemmer.snowball.SnowballStemmer.ALGORITHM;

/**
 * What each OpenNLP stemmer returns for the words of the {@code stem} input set: 46 runs,
 * {@code porter} ({@code PorterStemmer}), {@code snowball-<language>} for every
 * {@code SnowballStemmer.ALGORITHM}, {@code snowball-<language>-repeat2} for English, German,
 * Russian and Turkish, and the light and minimal stemmers, the Norwegian ones for each variety and
 * both. A run reads the inputs of its language: {@code <language>} (a sample of Snowball's
 * reference vocabulary, where its license allows one), {@code generated-<language>} (words made
 * from the language's Snowball suffix tables and its light stemmers' endings) and {@code edge},
 * which every run reads.
 *
 * <p>An input holds one word per line; a line starting with {@code #} is a comment, and a word is
 * escaped as {@link EscapedLines} reads it, since some are half a surrogate pair. The layer
 * {@code turbo:stems}, anchored to the document, has one row per word: the word and its stem
 * ({@link TrainGolden#lossless}), or, where {@code stem} threw, an empty stem and the exception's
 * class in {@code thrown}.</p>
 */
final class StemGolden implements AnnotatorGolden {

  static final String SET = Runs.STEM;

  /** The input every stemmer reads. */
  static final String EDGE = "edge";

  private final String name;
  private final String language;
  private final String description;
  private final Supplier<Stemmer> stemmer;

  private StemGolden(String name, String language, String description, Supplier<Stemmer> stemmer) {
    this.name = name;
    this.language = language;
    this.description = description;
    this.stemmer = stemmer;
  }

  static List<StemGolden> all() {
    List<StemGolden> out = new ArrayList<>();
    out.add(new StemGolden("porter", "porter", "PorterStemmer", PorterStemmer::new));
    for (ALGORITHM a : ALGORITHM.values()) {
      String lang = a.name().toLowerCase(java.util.Locale.ROOT);
      out.add(new StemGolden("snowball-" + lang, lang, "SnowballStemmer(" + a + ")", () -> new SnowballStemmer(a)));
    }
    for (ALGORITHM a : List.of(ALGORITHM.ENGLISH, ALGORITHM.GERMAN, ALGORITHM.RUSSIAN, ALGORITHM.TURKISH)) {
      String lang = a.name().toLowerCase(java.util.Locale.ROOT);
      out.add(new StemGolden("snowball-" + lang + "-repeat2", lang, "SnowballStemmer(" + a + ", 2)",
          () -> new SnowballStemmer(a, 2)));
    }
    out.add(new StemGolden("english-minimal", "english", "EnglishMinimalStemmer", EnglishMinimalStemmer::new));
    out.add(new StemGolden("finnish-light", "finnish", "FinnishLightStemmer", FinnishLightStemmer::new));
    out.add(new StemGolden("french-light", "french", "FrenchLightStemmer", FrenchLightStemmer::new));
    out.add(new StemGolden("french-minimal", "french", "FrenchMinimalStemmer", FrenchMinimalStemmer::new));
    out.add(new StemGolden("german-light", "german", "GermanLightStemmer", GermanLightStemmer::new));
    out.add(new StemGolden("german-minimal", "german", "GermanMinimalStemmer", GermanMinimalStemmer::new));
    out.add(new StemGolden("hungarian-light", "hungarian", "HungarianLightStemmer", HungarianLightStemmer::new));
    out.add(new StemGolden("italian-light", "italian", "ItalianLightStemmer", ItalianLightStemmer::new));
    NorwegianVariety b = NorwegianVariety.BOKMAAL;
    NorwegianVariety n = NorwegianVariety.NYNORSK;
    out.add(new StemGolden("norwegian-light-bokmaal", "norwegian", "NorwegianLightStemmer(BOKMAAL)",
        () -> new NorwegianLightStemmer(b)));
    out.add(new StemGolden("norwegian-light-nynorsk", "norwegian", "NorwegianLightStemmer(NYNORSK)",
        () -> new NorwegianLightStemmer(n)));
    out.add(new StemGolden("norwegian-light-both", "norwegian", "NorwegianLightStemmer(BOKMAAL, NYNORSK)",
        () -> new NorwegianLightStemmer(b, n)));
    out.add(new StemGolden("norwegian-minimal-bokmaal", "norwegian", "NorwegianMinimalStemmer(BOKMAAL)",
        () -> new NorwegianMinimalStemmer(b)));
    out.add(new StemGolden("norwegian-minimal-nynorsk", "norwegian", "NorwegianMinimalStemmer(NYNORSK)",
        () -> new NorwegianMinimalStemmer(n)));
    out.add(new StemGolden("norwegian-minimal-both", "norwegian", "NorwegianMinimalStemmer(BOKMAAL, NYNORSK)",
        () -> new NorwegianMinimalStemmer(b, n)));
    out.add(new StemGolden("portuguese-light", "portuguese", "PortugueseLightStemmer", PortugueseLightStemmer::new));
    out.add(new StemGolden("russian-light", "russian", "RussianLightStemmer", RussianLightStemmer::new));
    out.add(new StemGolden("spanish-light", "spanish", "SpanishLightStemmer", SpanishLightStemmer::new));
    out.add(new StemGolden("spanish-minimal", "spanish", "SpanishMinimalStemmer", SpanishMinimalStemmer::new));
    out.add(new StemGolden("swedish-light", "swedish", "SwedishLightStemmer", SwedishLightStemmer::new));
    out.add(new StemGolden("swedish-minimal", "swedish", "SwedishMinimalStemmer", SwedishMinimalStemmer::new));
    return out;
  }

  @Override
  public String annotator() {
    return "stem";
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
    return stemmer.get().getClass().getName();
  }

  @Override
  public String method() {
    return "stem";
  }

  @Override
  public Map<String, String> config() {
    return Map.of("stemmer", description);
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(SET)
        && (input.id().equals(EDGE) || input.id().equals(language) || input.id().equals("generated-" + language));
  }

  @Override
  public List<Layer> layers(Input input) {
    Stemmer s = stemmer.get();
    Table t = new Table("turbo:stems", "word:str", "stem:str", "thrown:str");
    for (String line : input.lines()) {
      if (line.startsWith("#")) {
        continue;
      }
      String word = EscapedLines.unescape(line);
      String stem;
      String thrown = "";
      try {
        stem = s.stem(word).toString();
      } catch (RuntimeException e) {
        stem = "";
        thrown = e.getClass().getName();
      }
      t.row(TrainGolden.lossless(word), TrainGolden.lossless(stem), thrown);
    }
    return List.of(t.layer());
  }
}
