package opennlp.tools.stemmer.hunspell;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * The calls the Hunspell goldens make, compiled into the jar of the Hunspell classes of the
 * fork's {@code OPENNLP-1927-hunspell-compatibility} branch (the container's Dockerfile), so it
 * can call {@code HunspellStemmer.analyze}, which is package private there. Every method takes
 * and returns JDK types only, because the generator calls them through reflection from another
 * class loader.
 */
public final class HunspellBridge {

  private HunspellBridge() {
  }

  /**
   * Loads a dictionary from the bytes of its affix and word list files through the stream API,
   * so a diagnostic names the {@code affix stream} rather than a path.
   */
  public static Object load(byte[] affix, byte[] words, boolean partial) throws IOException {
    return HunspellDictionary.load(new ByteArrayInputStream(affix), new ByteArrayInputStream(words),
        partial ? HunspellDictionary.LoadMode.ALLOW_PARTIAL : HunspellDictionary.LoadMode.STRICT);
  }

  /** The directives a partial load skipped: the name, a tab and the line, in encounter order. */
  public static List<String> unsupported(Object dictionary) {
    List<String> out = new ArrayList<>();
    for (HunspellDictionary.UnsupportedDirective d : ((HunspellDictionary) dictionary).getUnsupportedDirectives()) {
      out.add(d.directive() + "\t" + d.lineNumber());
    }
    return out;
  }

  /** {@code new HunspellStemmer(dictionary).stemAll(word)}, as strings. */
  public static List<String> stemAll(Object dictionary, String word) {
    List<String> out = new ArrayList<>();
    for (CharSequence s : new HunspellStemmer((HunspellDictionary) dictionary).stemAll(word)) {
      out.add(s.toString());
    }
    return out;
  }

  /** {@code new HunspellStemmer(dictionary).stem(word)}, as a string. */
  public static String stem(Object dictionary, String word) {
    return new HunspellStemmer((HunspellDictionary) dictionary).stem(word).toString();
  }

  /** {@code new HunspellStemmer(dictionary).analyze(word)}. */
  public static List<String> analyze(Object dictionary, String word) {
    return new HunspellStemmer((HunspellDictionary) dictionary).analyze(word);
  }
}
