import java.io.IOException;
import java.io.Writer;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
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
 * Writes the golden files of turbo-stem: the word lists made from each language's suffixes
 * (inputs/generated/) and what each OpenNLP stemmer returns for its words (expected/).
 *
 * <p>Usage: StemGolden golden-directory opennlp-commit
 */
public final class StemGolden {

  /** Supplementary characters the generated words mix in: a letter, an emoji, a Han ideograph. */
  private static final String[] SUPPLEMENTARY = {"𝐚", "😀", "𠀀"};

  private static final int GENERATED_WORDS = 1000;

  /**
   * One language of the word lists: its Snowball algorithm (whose suffix tables seed the
   * generated words), the letters its bases are made of, and the endings of its light and
   * minimal stemmers.
   */
  private record Language(String name, ALGORITHM algorithm, String letters, String[] light) {
  }

  private static final String LATIN = "abcdefghijklmnopqrstuvwxyz";

  private static final Language[] LANGUAGES = {
      new Language("arabic", ALGORITHM.ARABIC,
          "ابتثجحخدذرزسش"
              + "صضطظعغفقكلمنه"
              + "ويةىءآأإؤئًَِ"
              + "٠٥ﻻ", new String[0]),
      new Language("catalan", ALGORITHM.CATALAN, LATIN + "àèéíï"
          + "òóúüç·", new String[0]),
      new Language("danish", ALGORITHM.DANISH, LATIN + "æøå", new String[0]),
      new Language("dutch", ALGORITHM.DUTCH, LATIN + "äëïöüá"
          + "éíóúè", new String[0]),
      new Language("english", ALGORITHM.ENGLISH, LATIN + "'", new String[] {
          "ies", "es", "s", "us", "ss", "eis", "ais", "ois", "ees"}),
      new Language("finnish", ALGORITHM.FINNISH, LATIN + "äöå", new String[] {
          "kin", "ko", "dellinen", "dellisuus", "lla", "tse", "sti", "ni", "aa", "nnen", "ntena",
          "tten", "eiden", "neen", "niin", "seen", "teen", "inen", "han", "hen", "den", "ksen",
          "ssa", "sta", "lta", "tta", "ksi", "lle", "na", "ne", "nei", "ja", "ta", "an", "hde",
          "ei", "at", "kk", "pp", "tt"}),
      new Language("french", ALGORITHM.FRENCH, LATIN + "àâçéèê"
          + "ëîïôûùüÿœæ'", new String[] {
          "aux", "eux", "x", "s", "issement", "issant", "ement", "ivement", "ficatrice",
          "ficateur", "catrice", "cateur", "atrice", "ateur", "trice", "ième", "teuse",
          "teur", "euse", "ère", "ive", "folle", "molle", "nnelle", "nnel", "ète",
          "ique", "esse", "inage", "isation", "ualisation", "isateur", "ation", "ition", "ie",
          "er", "é", "ee", "ll", "ss"}),
      new Language("german", ALGORITHM.GERMAN, LATIN + "äöüßàá"
          + "âòóôïìíîùúû", new String[] {
          "ern", "em", "en", "er", "es", "e", "s", "est", "st", "nen", "bst", "dst", "mst"}),
      new Language("greek", ALGORITHM.GREEK,
          "αβγδεζηθικλμνξ"
              + "οπρσςτυφχψωάέ"
              + "ήίόύώϊϋΐΰΑΕΣ",
          new String[0]),
      new Language("hungarian", ALGORITHM.HUNGARIAN, LATIN + "áéíóö"
          + "őúüűëõũû", new String[] {
          "kent", "nak", "nek", "val", "vel", "ert", "rol", "ban", "ben", "bol", "nal", "nel",
          "hoz", "hez", "tol", "al", "el", "at", "et", "ot", "va", "ve", "ra", "re", "ba", "be",
          "ul", "ig", "on", "en", "atok", "otok", "etek", "itek", "itok", "unk", "tok", "tek",
          "juk", "ink", "am", "em", "om", "ad", "ed", "od", "uk", "nk", "ja", "je", "im", "id",
          "ik", "ak", "ok", "ek", "k"}),
      new Language("indonesian", ALGORITHM.INDONESIAN, LATIN, new String[0]),
      new Language("irish", ALGORITHM.IRISH, LATIN + "áéíóú'",
          new String[0]),
      new Language("italian", ALGORITHM.ITALIAN, LATIN + "àèéìí"
          + "îòóùúáâäôöêëû"
          + "üï'", new String[] {"ie", "he", "hi", "ii", "ia", "io", "e", "i", "a", "o"}),
      new Language("norwegian", ALGORITHM.NORWEGIAN, LATIN + "æøåéô",
          new String[] {
          "s", "heter", "heten", "heita", "heiter", "leiken", "leikar", "dom", "het", "heit",
          "semd", "leik", "elser", "elsen", "ende", "ande", "else", "este", "aste", "eren",
          "aren", "ere", "are", "est", "ast", "ene", "ane", "er", "en", "et", "ar", "st", "te",
          "a", "e", "n"}),
      new Language("porter", ALGORITHM.PORTER, LATIN, new String[] {
          "sses", "ies", "eed", "ed", "ing", "at", "bl", "iz", "y", "ational", "tional", "enci",
          "anci", "izer", "bli", "alli", "entli", "eli", "ousli", "ization", "ation", "ator",
          "alism", "iveness", "fulness", "ousness", "aliti", "iviti", "biliti", "logi", "icate",
          "ative", "alize", "iciti", "ical", "ful", "ness", "al", "ance", "ence", "er", "ic",
          "able", "ible", "ant", "ement", "ment", "ent", "sion", "tion", "ou", "ism", "ate",
          "iti", "ous", "ive", "ize", "e", "ll"}),
      new Language("portuguese", ALGORITHM.PORTUGUESE, LATIN + "àáâã"
          + "äçéêíóôõúü", new String[] {
          "res", "ses", "les", "zes", "ns", "eis", "éis", "ais", "óis", "is",
          "ões", "ães", "mente", "s", "inha", "iaca", "eira", "osa", "ica", "ida",
          "ada", "iva", "ama", "ona", "ora", "esa", "na", "e", "a", "o"}),
      new Language("romanian", ALGORITHM.ROMANIAN, LATIN + "ăâîșț"
          + "şţ", new String[0]),
      new Language("russian", ALGORITHM.RUSSIAN,
          "абвгдеёжзийклм"
              + "нопрстуфхцчшщ"
              + "ъыьэюя", new String[] {
          "иями", "оями", "иям",
          "иях", "оях", "ями", "оям",
          "оьв", "ами", "его", "ему",
          "ери", "ими", "ого", "ому",
          "ыми", "оев", "ая", "яя",
          "ях", "юю", "ах", "ею", "их",
          "ия", "ию", "ьв", "ою", "ую",
          "ям", "ых", "ея", "ам", "ем",
          "ей", "ём", "ев", "ий", "им",
          "ое", "ой", "ом", "ов", "ые",
          "ый", "ым", "ми", "нн", "ь",
          "и"}),
      new Language("spanish", ALGORITHM.SPANISH, LATIN + "áéíóú"
          + "üñàèìòùâêîôûä"
          + "ëïö", new String[] {
          "eses", "ces", "os", "as", "es", "o", "a", "e", "s", "res", "des", "les", "nes",
          "xes", "yes", "ues", "ies", "ses", "ais", "dis", "quis", "guis", "gues", "lunes",
          "crisis", "relojes", "clubes", "noes"}),
      new Language("swedish", ALGORITHM.SWEDISH, LATIN + "äöåé",
          new String[] {
          "s", "elser", "heten", "arne", "erna", "arna", "ande", "else", "aste", "orna", "aren",
          "are", "ast", "het", "ar", "er", "or", "en", "at", "te", "et", "t", "a", "e", "n"}),
      new Language("turkish", ALGORITHM.TURKISH, LATIN + "çğıöşü"
          + "âîûİ", new String[0]),
  };

  /** One stemmer to record: its golden name, the language whose words it gets, and a maker. */
  private record Config(String name, String language, String description,
                        Supplier<Stemmer> stemmer) {
  }

  private static List<Config> configs() {
    List<Config> configs = new ArrayList<>();
    configs.add(new Config("porter", "porter", "opennlp.tools.stemmer.PorterStemmer",
        PorterStemmer::new));
    for (Language language : LANGUAGES) {
      ALGORITHM algorithm = language.algorithm();
      configs.add(new Config("snowball-" + language.name(), language.name(),
          "SnowballStemmer(" + algorithm + ")", () -> new SnowballStemmer(algorithm)));
    }
    for (String name : new String[] {"english", "german", "russian", "turkish"}) {
      ALGORITHM algorithm = ALGORITHM.valueOf(name.toUpperCase());
      configs.add(new Config("snowball-" + name + "-repeat2", name,
          "SnowballStemmer(" + algorithm + ", 2)", () -> new SnowballStemmer(algorithm, 2)));
    }
    configs.add(new Config("english-minimal", "english", "EnglishMinimalStemmer",
        EnglishMinimalStemmer::new));
    configs.add(new Config("finnish-light", "finnish", "FinnishLightStemmer",
        FinnishLightStemmer::new));
    configs.add(new Config("french-light", "french", "FrenchLightStemmer",
        FrenchLightStemmer::new));
    configs.add(new Config("french-minimal", "french", "FrenchMinimalStemmer",
        FrenchMinimalStemmer::new));
    configs.add(new Config("german-light", "german", "GermanLightStemmer",
        GermanLightStemmer::new));
    configs.add(new Config("german-minimal", "german", "GermanMinimalStemmer",
        GermanMinimalStemmer::new));
    configs.add(new Config("hungarian-light", "hungarian", "HungarianLightStemmer",
        HungarianLightStemmer::new));
    configs.add(new Config("italian-light", "italian", "ItalianLightStemmer",
        ItalianLightStemmer::new));
    NorwegianVariety b = NorwegianVariety.BOKMAAL;
    NorwegianVariety n = NorwegianVariety.NYNORSK;
    configs.add(new Config("norwegian-light-bokmaal", "norwegian",
        "NorwegianLightStemmer(BOKMAAL)", () -> new NorwegianLightStemmer(b)));
    configs.add(new Config("norwegian-light-nynorsk", "norwegian",
        "NorwegianLightStemmer(NYNORSK)", () -> new NorwegianLightStemmer(n)));
    configs.add(new Config("norwegian-light-both", "norwegian",
        "NorwegianLightStemmer(BOKMAAL, NYNORSK)", () -> new NorwegianLightStemmer(b, n)));
    configs.add(new Config("norwegian-minimal-bokmaal", "norwegian",
        "NorwegianMinimalStemmer(BOKMAAL)", () -> new NorwegianMinimalStemmer(b)));
    configs.add(new Config("norwegian-minimal-nynorsk", "norwegian",
        "NorwegianMinimalStemmer(NYNORSK)", () -> new NorwegianMinimalStemmer(n)));
    configs.add(new Config("norwegian-minimal-both", "norwegian",
        "NorwegianMinimalStemmer(BOKMAAL, NYNORSK)", () -> new NorwegianMinimalStemmer(b, n)));
    configs.add(new Config("portuguese-light", "portuguese", "PortugueseLightStemmer",
        PortugueseLightStemmer::new));
    configs.add(new Config("russian-light", "russian", "RussianLightStemmer",
        RussianLightStemmer::new));
    configs.add(new Config("spanish-light", "spanish", "SpanishLightStemmer",
        SpanishLightStemmer::new));
    configs.add(new Config("spanish-minimal", "spanish", "SpanishMinimalStemmer",
        SpanishMinimalStemmer::new));
    configs.add(new Config("swedish-light", "swedish", "SwedishLightStemmer",
        SwedishLightStemmer::new));
    configs.add(new Config("swedish-minimal", "swedish", "SwedishMinimalStemmer",
        SwedishMinimalStemmer::new));
    return configs;
  }

  public static void main(String[] args) throws Exception {
    Path golden = Path.of(args[0]);
    String pin = args[1];
    for (Language language : LANGUAGES) {
      List<String> words = generate(language);
      write(golden.resolve("inputs/generated/" + language.name() + ".txt"),
          words.stream().map(StemGolden::escape).toList());
    }
    for (Config config : configs()) {
      List<String> inputs = new ArrayList<>();
      Path sample = golden.resolve("inputs/" + config.language() + ".txt");
      if (Files.exists(sample)) {
        inputs.add(config.language() + ".txt");
      }
      inputs.add("edge.txt");
      inputs.add("generated/" + config.language() + ".txt");
      Stemmer stemmer = config.stemmer().get();
      List<String> lines = new ArrayList<>();
      lines.add("# " + config.description() + " at OpenNLP " + pin + " over "
          + String.join(" ", inputs));
      for (String input : inputs) {
        for (String word : read(golden.resolve("inputs/" + input))) {
          String stem;
          try {
            stem = escape(stemmer.stem(word).toString());
          } catch (RuntimeException e) {
            stem = "!" + e.getClass().getName();
          }
          lines.add(stem);
        }
      }
      write(golden.resolve("expected/" + config.name() + ".txt"), lines);
    }
  }

  /**
   * Words made of a random base from the language's letters and one or two endings from its
   * Snowball suffix tables and its light stemmers, some with an upper-case letter, a digit, an
   * apostrophe, a hyphen, a supplementary character or a lone surrogate mixed in.
   */
  private static List<String> generate(Language language) throws ReflectiveOperationException {
    TreeSet<String> suffixSet = new TreeSet<>(snowballSuffixes(language.algorithm()));
    suffixSet.addAll(List.of(language.light()));
    suffixSet.remove("");
    String[] suffixes = suffixSet.toArray(new String[0]);
    String letters = language.letters();
    Random random = new Random(language.name().hashCode());
    List<String> words = new ArrayList<>(GENERATED_WORDS);
    while (words.size() < GENERATED_WORDS) {
      StringBuilder word = new StringBuilder();
      int shape = random.nextInt(10);
      if (shape < 5) {
        base(word, letters, random, 2 + random.nextInt(7));
        word.append(suffixes[random.nextInt(suffixes.length)]);
      } else if (shape < 7) {
        base(word, letters, random, 1 + random.nextInt(6));
        word.append(suffixes[random.nextInt(suffixes.length)]);
        word.append(suffixes[random.nextInt(suffixes.length)]);
      } else if (shape < 8) {
        word.append(suffixes[random.nextInt(suffixes.length)]);
        if (random.nextBoolean()) {
          word.append(suffixes[random.nextInt(suffixes.length)]);
        }
      } else if (shape < 9) {
        base(word, letters, random, 1 + random.nextInt(12));
      } else {
        base(word, letters, random, 2 + random.nextInt(7));
        word.append(suffixes[random.nextInt(suffixes.length)]);
        mutate(word, random);
      }
      words.add(word.toString());
    }
    return words;
  }

  private static void base(StringBuilder word, String letters, Random random, int length) {
    for (int i = 0; i < length; i++) {
      word.append(letters.charAt(random.nextInt(letters.length())));
    }
  }

  private static void mutate(StringBuilder word, Random random) {
    int at = random.nextInt(word.length() + 1);
    switch (random.nextInt(7)) {
      case 0 -> {
        int i = random.nextInt(word.length());
        word.setCharAt(i, Character.toUpperCase(word.charAt(i)));
      }
      case 1 -> word.insert(at, (char) ('0' + random.nextInt(10)));
      case 2 -> word.insert(at, random.nextBoolean() ? '\'' : '-');
      case 3 -> word.insert(at, SUPPLEMENTARY[random.nextInt(SUPPLEMENTARY.length)]);
      case 4 -> word.append(SUPPLEMENTARY[random.nextInt(SUPPLEMENTARY.length)]);
      case 5 -> word.insert(at, random.nextBoolean() ? '\uD83D' : '\uDE00');
      default -> word.insert(at, "́");
    }
  }

  /** Every string of every Among table of the language's generated Snowball class. */
  private static List<String> snowballSuffixes(ALGORITHM algorithm)
      throws ReflectiveOperationException {
    Class<?> engine = Class.forName("opennlp.tools.stemmer.snowball."
        + algorithm.name().toLowerCase() + "Stemmer");
    Class<?> among = Class.forName("opennlp.tools.stemmer.snowball.Among");
    Field s = among.getDeclaredField("s");
    s.setAccessible(true);
    List<String> out = new ArrayList<>();
    for (Field field : engine.getDeclaredFields()) {
      if (Modifier.isStatic(field.getModifiers()) && field.getType().isArray()
          && field.getType().getComponentType() == among) {
        field.setAccessible(true);
        for (Object entry : (Object[]) field.get(null)) {
          out.add(new String((char[]) s.get(entry)));
        }
      }
    }
    return out;
  }

  /**
   * A word or stem as one line: the backslash, tab, line feed, carriage return, other C0
   * controls, lone surrogates and a leading {@code #} (which starts a comment line) are
   * escaped; everything else is written as UTF-8.
   */
  static String escape(String s) {
    StringBuilder out = new StringBuilder(s.length());
    for (int i = 0; i < s.length(); i++) {
      if (i == 0 && s.charAt(0) == '#') {
        out.append("\\u0023");
        continue;
      }
      char c = s.charAt(i);
      boolean pair = Character.isHighSurrogate(c) && i + 1 < s.length()
          && Character.isLowSurrogate(s.charAt(i + 1));
      if (pair) {
        out.append(c).append(s.charAt(++i));
      } else if (c == '\\') {
        out.append("\\\\");
      } else if (c == '\t') {
        out.append("\\t");
      } else if (c == '\n') {
        out.append("\\n");
      } else if (c == '\r') {
        out.append("\\r");
      } else if (c < 0x20 || Character.isSurrogate(c)) {
        out.append(String.format("\\u%04X", (int) c));
      } else {
        out.append(c);
      }
    }
    return out.toString();
  }

  static String unescape(String s) {
    StringBuilder out = new StringBuilder(s.length());
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c != '\\') {
        out.append(c);
        continue;
      }
      char next = s.charAt(++i);
      switch (next) {
        case '\\' -> out.append('\\');
        case 't' -> out.append('\t');
        case 'n' -> out.append('\n');
        case 'r' -> out.append('\r');
        case 'u' -> {
          out.append((char) Integer.parseInt(s.substring(i + 1, i + 5), 16));
          i += 4;
        }
        default -> throw new IllegalArgumentException("bad escape in " + s);
      }
    }
    return out.toString();
  }

  /** The words of an input file, one per line; a line starting with # is a comment. */
  private static List<String> read(Path path) throws IOException {
    List<String> words = new ArrayList<>();
    for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
      if (!line.startsWith("#")) {
        words.add(unescape(line));
      }
    }
    return words;
  }

  private static void write(Path path, List<String> lines) throws IOException {
    Files.createDirectories(path.getParent());
    try (Writer out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
      for (String line : lines) {
        out.write(line);
        out.write('\n');
      }
    }
  }
}
