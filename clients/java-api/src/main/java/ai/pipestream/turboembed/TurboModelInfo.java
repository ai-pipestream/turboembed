package ai.pipestream.turboembed;

import java.util.Arrays;
import java.util.List;

/**
 * What {@code turbo_model_get_info} says of a session's model, and the device it runs on.
 *
 * @param device             the device index the session's context is on
 * @param task               the model's task
 * @param dim                values per output row
 * @param pooling            embed: the bundle's pooling
 * @param normalize          embed: the bundle's normalization
 * @param maxSeq             tokens a row may hold
 * @param maxBatch           rows a session may take
 * @param dtype              the artifact's {@code TURBO_DTYPE_*}
 * @param modelId            the bundle's model id
 * @param revision           the bundle's model revision
 * @param manifestSha256     hex SHA-256 of {@code manifest.json}
 * @param artifactSha256     hex SHA-256 of the artifact file loaded
 * @param tokenizerSha256    hex SHA-256 of the tokenizer file
 * @param prefixQuery        embed: the query prefix, empty for none
 * @param prefixDocument     embed: the document prefix, empty for none
 * @param outputDims         embed: the widths besides {@code dim} an output dim may name, ascending
 * @param rerankQueryTokens  rerank: the default query budget
 * @param rerankPairTemplate rerank: {@code TURBO_PAIR_BERT} or {@code TURBO_PAIR_ROBERTA}, 0 otherwise
 */
public record TurboModelInfo(
        int device,
        TurboSessionOptions.Task task,
        int dim,
        TurboEmbedOptions.Pooling pooling,
        TurboEmbedOptions.Normalize normalize,
        int maxSeq,
        int maxBatch,
        int dtype,
        String modelId,
        String revision,
        String manifestSha256,
        String artifactSha256,
        String tokenizerSha256,
        String prefixQuery,
        String prefixDocument,
        List<Integer> outputDims,
        int rerankQueryTokens,
        int rerankPairTemplate) {

    /** Copies the output dims. */
    public TurboModelInfo {
        outputDims = List.copyOf(outputDims);
    }

    /**
     * The info a native bridge read: {@code numbers} as {@code [device, task, dim, pooling,
     * normalize, maxSeq, maxBatch, dtype, rerankQueryTokens, rerankPairTemplate, outputDims...]}
     * and {@code text} as {@code [modelId, revision, manifestSha256, artifactSha256,
     * tokenizerSha256, prefixQuery, prefixDocument]}.
     */
    public static TurboModelInfo fromNative(int[] numbers, String[] text) {
        if (numbers.length < 10 || text.length != 7) {
            throw new IllegalStateException(
                    "native model info has " + numbers.length + " numbers and " + text.length + " strings");
        }
        List<Integer> dims = Arrays.stream(numbers, 10, numbers.length).boxed().toList();
        return new TurboModelInfo(
                numbers[0],
                TurboSessionOptions.Task.fromNative(numbers[1]),
                numbers[2],
                TurboEmbedOptions.Pooling.fromNative(numbers[3]),
                TurboEmbedOptions.Normalize.fromNative(numbers[4]),
                numbers[5],
                numbers[6],
                numbers[7],
                text[0],
                text[1],
                text[2],
                text[3],
                text[4],
                text[5],
                text[6],
                dims,
                numbers[8],
                numbers[9]);
    }
}
