// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

/** Immutable model metadata. */
public record ModelInfo(
        int dimension,
        int vocabularySize,
        int maxSequenceLength,
        int maxBatchSize,
        boolean normalized,
        String modelId,
        String revision,
        String tokenizerSha256,
        String pooling) {}
