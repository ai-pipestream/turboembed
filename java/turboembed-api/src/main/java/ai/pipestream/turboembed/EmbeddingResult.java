// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

import java.nio.FloatBuffer;

/**
 * A result lease confined to the thread that created its execution slot. Close
 * it deterministically before reusing or closing that slot.
 */
public interface EmbeddingResult extends AutoCloseable {
    /** Returns the fixed result batch size. */
    int batch();

    /** Returns the embedding dimension. */
    int dimension();

    /**
     * Explicitly copies {@code batch() * dimension()} row-major floats into the
     * buffer starting at its current position, then advances that position by
     * the copied count. The limit defines available capacity; other elements
     * and aliases outside the destination range are unchanged.
     */
    void readInto(FloatBuffer output);

    /** Returns a borrowed, read-only OpenCL view of this result when supported. */
    OpenClView openCl();

    /** Releases the result lease. Repeated close calls have no effect. */
    @Override
    void close();
}
