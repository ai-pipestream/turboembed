// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

import java.nio.file.Path;

/**
 * A device context. Close it deterministically; models created from it retain
 * the resources they need and remain usable after the context is closed.
 * Methods may be called from multiple threads; close is serialized with use.
 */
public interface DeviceContext extends AutoCloseable {
    /** Returns metadata for this context. */
    DeviceInfo info();

    /** Loads and verifies a prepared model bundle without downloading files. */
    Model loadModel(Path bundle);

    /** Releases this context handle. Repeated close calls have no effect. */
    @Override
    void close();
}
