package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.List;

/**
 * The tokenized sentences of an input of the {@code sequence} and {@code namefind} sets: one
 * sentence per line, its tokens separated by single spaces.
 */
final class Sentences {

  private Sentences() {}

  /** Every sentence of the input, each a new array, since Java's feature generators tell sentences apart by identity. */
  static List<String[]> of(Input input) {
    List<String[]> out = new ArrayList<>();
    for (String line : input.lines()) {
      out.add(line.split(" "));
    }
    return out;
  }

  /**
   * The outcomes the feature goldens pass as the previous outcomes of a sentence: five strings, a
   * name finder's, a tagger's and one with an accent, in turn from the sentence's number on.
   */
  static String[] syntheticOutcomes(int sentence, int length) {
    String[] choices = {"other", "person-start", "person-cont", "NN", "té"};
    String[] out = new String[length];
    for (int i = 0; i < length; i++) {
      out[i] = choices[(i + sentence) % choices.length];
    }
    return out;
  }

  /** The layer of the sentences' tokens: sentence number, token and, for the feature goldens, the outcome passed. */
  static Table tokens(List<String[]> sentences, boolean outcomes) {
    Table t = outcomes ? new Table("turbo:sentence-tokens", "sentence:u32", "token:str", "outcome:str")
        : new Table("turbo:sentence-tokens", "sentence:u32", "token:str");
    for (int k = 0; k < sentences.size(); k++) {
      String[] tokens = sentences.get(k);
      String[] o = syntheticOutcomes(k, tokens.length);
      for (int i = 0; i < tokens.length; i++) {
        if (outcomes) {
          t.row(k, tokens[i], o[i]);
        } else {
          t.row(k, tokens[i]);
        }
      }
    }
    return t;
  }
}
