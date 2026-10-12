package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import opennlp.tools.tokenize.TokenizerFactory;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.Span;

/**
 * The tokenizer's decision points: every candidate split inside a whitespace separated token
 * that {@link TokenizerME#tokenizePos(String)} asks its model about.
 *
 * <p>The tokenizer jumps over the dots of an abbreviation depending on a decision, so the
 * candidates cannot be listed from outside. The context generator is wrapped instead
 * ({@link RecordingTokenizerFactory}): it reports each {@code getContext(token, index)} call,
 * whose array {@code tokenizePos} passes straight to {@code eval}. The generator sees only the
 * whitespace token (a fresh substring per token) and a UTF-16 index into it, so the token's
 * start in the input comes from the same whitespace split {@code tokenizePos} makes
 * ({@code WhitespaceTokenizer} without kept line breaks, the tokenizer's default, which this
 * class never changes): the requests of one token are those with the same token object, and
 * each token is paired with the next whitespace span whose text equals it. Whether
 * {@code tokenizePos} evaluates a whitespace token depends on its text alone (its length and the
 * alphanumeric pattern), so a span skipped over can never have the text of an evaluated token;
 * this is checked. A point's position is the token's start plus the index, the UTF-16 index the
 * candidate split is before, converted to its UTF-8 byte offset. The tokenizer walks UTF-16
 * units, so a candidate can fall between the halves of a surrogate pair; such a point has unit 1
 * (see {@link DecisionPoint}).</p>
 */
final class TokenDump implements AnnotatorDump {

  private static final WhitespaceTokenizer WHITESPACE = WhitespaceTokenizer.newInstance(false);

  private final TokenGolden golden;
  private final MaxentRecorder recorder;
  private final TokenizerME tokenizer;

  private TokenDump(TokenGolden golden, MaxentRecorder recorder, TokenizerME tokenizer) {
    this.golden = golden;
    this.recorder = recorder;
    this.tokenizer = tokenizer;
  }

  /**
   * Builds a tokenizer model with the loaded model's manifest entries and factory settings over a
   * recorder of its maxent model, through
   * {@link TokenizerModel#TokenizerModel(opennlp.tools.ml.model.MaxentModel, java.util.Map, TokenizerFactory)}.
   */
  static TokenDump of(TokenGolden golden) {
    RecordingModels.allowFactories();
    TokenizerModel loaded = golden.loaded();
    RecordingModels.checkFactory(loaded.getFactory(), TokenizerFactory.class);
    MaxentRecorder recorder = MaxentRecorder.wrap(loaded.getMaxentModel());
    TokenizerModel recording = new TokenizerModel(recorder, RecordingModels.entriesFor(loaded),
        new RecordingTokenizerFactory(loaded.getFactory()));
    RecordingModels.checkManifest(loaded, recording, RecordingTokenizerFactory.class);
    return new TokenDump(golden, recorder, new TokenizerME(recording));
  }

  @Override
  public AnnotatorGolden golden() {
    return golden;
  }

  @Override
  public Run run(Input input) {
    List<Layer> layers = TokenGolden.layers(tokenizer, input);
    String text = input.text();
    Span[] spans = WHITESPACE.tokenizePos(text);
    Set<String> evaluated = new HashSet<>();
    Set<String> skipped = new HashSet<>();
    List<DecisionPoint> points = new ArrayList<>();
    int next = 0;
    String token = null;
    int tokenStart = -1;
    int lastIndex = 0;
    for (Evaluation e : recorder.take()) {
      ContextRequest r = e.request();
      if (r == null || e.values() != null) {
        throw new IllegalStateException(input.inputId() + ": an eval call that is not a context request without values");
      }
      String requested = r.text().toString();
      // By identity: tokenizePos makes one substring per whitespace token and passes it to every
      // request about that token, while two tokens with the same text are two objects.
      if (requested != token) {
        while (next < spans.length && !spans[next].getCoveredText(text).toString().equals(requested)) {
          skipped.add(spans[next].getCoveredText(text).toString());
          next++;
        }
        if (next == spans.length) {
          throw new IllegalStateException(input.inputId() + ": no whitespace token \"" + requested
              + "\" is left for the tokenizer's request");
        }
        token = requested;
        tokenStart = spans[next].getStart();
        evaluated.add(requested);
        next++;
        lastIndex = 0;
      }
      if (r.index() <= lastIndex || r.index() >= token.length()) {
        throw new IllegalStateException(input.inputId() + ": index " + r.index() + " into \"" + token
            + "\" after index " + lastIndex);
      }
      lastIndex = r.index();
      points.add(DecisionPoint.at(input.offsets(), tokenStart + r.index(), e));
    }
    for (; next < spans.length; next++) {
      skipped.add(spans[next].getCoveredText(text).toString());
    }
    skipped.retainAll(evaluated);
    if (!skipped.isEmpty()) {
      throw new IllegalStateException(input.inputId() + ": whitespace tokens " + skipped
          + " were both evaluated and skipped, so the pairing is ambiguous");
    }
    return new Run(recorder.outcomeLabels(), points, layers);
  }
}
