// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

/**
 * A loaded model. Close it deterministically; slots created from it retain the
 * resources they need and remain usable after the model is closed.
 * Methods may be called from multiple threads; close is serialized with use.
 */
public interface Model extends AutoCloseable {
    /** Returns the bundle's validated model contract. */
    ModelInfo info();

    /** Creates reusable execution storage with a fixed batch and sequence length. */
    ExecutionSlot slot(int batch, int sequenceLength);

    /** Releases this model handle. Repeated close calls have no effect. */
    @Override
    void close();
}
