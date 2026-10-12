import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import opennlp.tools.util.featuregen.FeatureGeneratorUtil;
import opennlp.tools.util.featuregen.StringPattern;

/**
 * Writes what {@code StringPattern.recognize} and {@code FeatureGeneratorUtil.tokenFeature}
 * return for a list of tokens.
 *
 * <p>Arguments: the golden directory, the output file, the OpenNLP commit and the mode label
 * written in the header. Run with the OpenNLP jars of commit cdf33c6 it records the UTF-16 unit version;
 * with the {@code ready/string-pattern-unicode} classes first on the class path it records the
 * code point version (see the crate README).
 */
public final class PatternGolden {

  private PatternGolden() {
  }

  public static void main(String[] args) throws IOException {
    Path golden = Path.of(args[0]);
    Path out = Path.of(args[1]);
    String commit = args[2];
    String mode = args[3];

    Set<String> tokens = new LinkedHashSet<>();
    for (String line : Files.readAllLines(golden.resolve("inputs/tokens.txt"), StandardCharsets.UTF_8)) {
      tokens.add(Common.unescape(line));
    }
    for (String[] sentence : Common.sentences(golden.resolve("inputs/sentences.txt"))) {
      tokens.addAll(List.of(sentence));
    }
    int[][] ranges = {
        {0x20, 0x2FF}, {0x370, 0x3FF}, {0x400, 0x45F}, {0x590, 0x5FF}, {0x600, 0x6FF},
        {0x900, 0x97F}, {0x1E00, 0x1EFF}, {0x2000, 0x206F}, {0x2150, 0x218F}, {0x2460, 0x24FF},
        {0x3000, 0x30FF}, {0x4E00, 0x4E0F}, {0xAC00, 0xAC0F}, {0xFF00, 0xFF65},
        {0x10400, 0x1044F}, {0x1D400, 0x1D433}, {0x1F600, 0x1F60F}, {0x20000, 0x20007}
    };
    for (int[] range : ranges) {
      for (int cp = range[0]; cp <= range[1]; cp++) {
        tokens.add(new String(Character.toChars(cp)));
      }
    }
    for (int cp : new int[] {0xD800, 0xDBFF, 0xDC00, 0xDFFF}) {
      tokens.add(String.valueOf((char) cp));
    }
    // A character and a period, for the cp class.
    List<String> capitals = new ArrayList<>();
    for (int cp = 0x41; cp <= 0x5A; cp++) {
      capitals.add(new String(Character.toChars(cp)));
    }
    for (int cp = 0xC0; cp <= 0xDE; cp++) {
      capitals.add(new String(Character.toChars(cp)));
    }
    for (int cp = 0x391; cp <= 0x3A9; cp++) {
      capitals.add(new String(Character.toChars(cp)));
    }
    for (int cp = 0x1D400; cp <= 0x1D433; cp++) {
      capitals.add(new String(Character.toChars(cp)));
    }
    for (String c : capitals) {
      tokens.add(c + ".");
    }

    try (Writer w = Common.writer(out)) {
      Common.header(w, "string-pattern", commit);
      Common.line(w, "mode: " + mode);
      Common.line(w, "columns: token, flags (allLetter initialCapital allCapital allLowerCase allDigit"
          + " allHiragana allKatakana period comma slash digit hyphen letters), digits, class");
      for (String token : tokens) {
        StringPattern p = StringPattern.recognize(token);
        boolean[] flags = {
            p.isAllLetter(), p.isInitialCapitalLetter(), p.isAllCapitalLetter(),
            p.isAllLowerCaseLetter(), p.isAllDigit(), p.isAllHiragana(), p.isAllKatakana(),
            p.containsPeriod(), p.containsComma(), p.containsSlash(), p.containsDigit(),
            p.containsHyphen(), p.containsLetters()
        };
        StringBuilder f = new StringBuilder();
        for (boolean b : flags) {
          f.append(b ? '1' : '0');
        }
        Common.line(w, Common.escape(token) + "\t" + f + "\t" + p.digits() + "\t"
            + FeatureGeneratorUtil.tokenFeature(token));
      }
    }
  }
}
