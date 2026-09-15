// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

/**
 * Reusable, fixed-shape execution storage confined to its creating thread.
 * Close it deterministically. Its current result must be closed before any
 * write, upload, execution, or slot close.
 */
public interface ExecutionSlot extends AutoCloseable {
    /**
     * Returns provider-owned editable staging for IDs, attention masks, and
     * token types. Each buffer contains {@code batch * sequenceLength}
     * row-major elements and remains valid on the owner thread until slot close.
     */
    TokenBuffers inputs();

    /**
     * Copies every element of all three fixed staging buffers to the execution
     * inputs, regardless of each buffer's position or limit.
     */
    void upload();

    /**
     * Encodes to UTF-8 and tokenizes exactly one Java string per batch row,
     * including empty strings and embedded NULs. Unpaired UTF-16 surrogates
     * are rejected. This path does not populate the buffers from
     * {@link #inputs()}.
     */
    void writeText(String... texts);

    /** Executes synchronously and returns the slot's sole active result lease. */
    EmbeddingResult execute();

    /** Returns the current counters for this slot. */
    SlotStats stats();

    /**
     * Releases the slot. Closing with a live result fails; repeated successful
     * close calls have no effect.
     */
    @Override
    void close();
}
