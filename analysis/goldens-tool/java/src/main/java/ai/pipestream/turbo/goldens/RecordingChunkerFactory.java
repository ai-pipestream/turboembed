package ai.pipestream.turbo.goldens;

import opennlp.tools.chunker.ChunkerContextGenerator;
import opennlp.tools.chunker.ChunkerFactory;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.util.TokenTag;

/**
 * A {@link ChunkerFactory} whose context generator is the one the plain factory makes, wrapped so
 * that each request the beam search makes ({@code getContext(index, sequence, priorDecisions,
 * additionalContext)} over {@link TokenTag} tuples) is reported to the model's
 * {@link MaxentRecorder} together with the array it returned. The request's text is the token of
 * {@code sequence[index]}, the very string object the chunker was given ({@code TokenTag.create}
 * keeps the strings), so the dump can tell by identity which token of which sentence the decision
 * was about. The sequence validator is the plain factory's.
 *
 * <p>{@code ChunkerModel} makes its own instance of this class through {@code ExtensionLoader},
 * as {@link RecordingSentenceDetectorFactory} describes.</p>
 */
public final class RecordingChunkerFactory extends ChunkerFactory {

  /** The constructor {@code ExtensionLoader} calls. */
  public RecordingChunkerFactory() {}

  @Override
  public ChunkerContextGenerator getContextGenerator() {
    ChunkerContextGenerator plain = super.getContextGenerator();
    if (!(artifactProvider instanceof ChunkerModel m
        && m.getArtifact(ChunkGolden.CLASSIFIER_ENTRY) instanceof MaxentRecorder recorder)) {
      throw new IllegalStateException("a recording chunker factory needs a chunker model whose classifier"
          + " is a MaxentRecorder");
    }
    return new ChunkerContextGenerator() {
      @Override
      public String[] getContext(int idx, String[] toks, String[] tags, String[] preds) {
        String[] context = plain.getContext(idx, toks, tags, preds);
        recorder.contextRequested(new ContextRequest(toks[idx], idx), context);
        return context;
      }

      @Override
      public String[] getContext(int index, TokenTag[] sequence, String[] priorDecisions,
          Object[] additionalContext) {
        String[] context = plain.getContext(index, sequence, priorDecisions, additionalContext);
        recorder.contextRequested(new ContextRequest(sequence[index].token(), index), context);
        return context;
      }
    };
  }
}
