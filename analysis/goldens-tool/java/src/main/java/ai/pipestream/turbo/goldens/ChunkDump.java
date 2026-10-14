package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import opennlp.tools.chunker.ChunkerFactory;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;

/**
 * The chunker's decision points: every context {@code BeamSearch} asks the model about while
 * {@link ChunkerME#chunk(String[], String[])} chunks a sentence, one evaluation per token per
 * path of the beam, in the order the search made them. The context generator is wrapped
 * ({@link RecordingChunkerFactory}) as {@link PosDump} describes; the point's position is the
 * start of the token the request was about.
 */
final class ChunkDump implements AnnotatorDump {

  private final ChunkGolden golden;
  private final MaxentRecorder recorder;
  private final ChunkerME chunker;

  private ChunkDump(ChunkGolden golden, MaxentRecorder recorder, ChunkerME chunker) {
    this.golden = golden;
    this.recorder = recorder;
    this.chunker = chunker;
  }

  /**
   * Builds a chunker model with the loaded model's beam size and manifest entries over a
   * recorder of its classifier, through
   * {@link ChunkerModel#ChunkerModel(String, opennlp.tools.ml.model.MaxentModel, int, Map, ChunkerFactory)},
   * and the chunker over it.
   */
  static ChunkDump of(ChunkGolden golden) {
    RecordingModels.allowFactories();
    ChunkerModel loaded = golden.loaded();
    RecordingModels.checkFactory(loaded.getFactory(), ChunkerFactory.class);
    MaxentRecorder recorder = MaxentRecorder.wrap(ChunkGolden.classifier(loaded));
    ChunkerModel recording = new ChunkerModel(loaded.getLanguage(), recorder, ChunkGolden.beamSize(loaded),
        ChunkGolden.entriesFor(loaded), new RecordingChunkerFactory());
    RecordingModels.checkManifest(loaded, recording, RecordingChunkerFactory.class);
    return new ChunkDump(golden, recorder, new ChunkerME(recording));
  }

  @Override
  public AnnotatorGolden golden() {
    return golden;
  }

  @Override
  public Run run(Input input) throws AnnotatorException {
    ChunkGolden.Run run;
    try {
      run = ChunkGolden.run(chunker, golden.pipeline(), input);
    } catch (AnnotatorException e) {
      recorder.discard();
      throw e;
    }
    Map<String, int[]> tokens = new IdentityHashMap<>();
    for (int s = 0; s < run.sentenceTokens().size(); s++) {
      String[] words = run.sentenceTokens().get(s);
      int[] starts = run.tokenStarts().get(s);
      for (int i = 0; i < words.length; i++) {
        if (tokens.put(words[i], new int[] {i, starts[i]}) != null) {
          throw new IllegalStateException(input.inputId() + ": two tokens are one string object");
        }
      }
    }
    List<DecisionPoint> points = new ArrayList<>();
    for (Evaluation e : recorder.take()) {
      ContextRequest r = e.request();
      if (r == null || e.values() != null) {
        throw new IllegalStateException(input.inputId()
            + ": an eval call that is not a context request without values");
      }
      int[] token = tokens.get(r.text());
      if (token == null || token[0] != r.index()) {
        throw new IllegalStateException(input.inputId() + ": a context request about index " + r.index()
            + " is not about a token of its sentence");
      }
      points.add(DecisionPoint.at(input.offsets(), token[1], e));
    }
    return new Run(recorder.outcomeLabels(), points, run.layers());
  }
}
