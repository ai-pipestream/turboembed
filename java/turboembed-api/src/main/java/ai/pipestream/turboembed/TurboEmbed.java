// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

/**
 * Entry point for creating device contexts. Close instances deterministically;
 * existing contexts remain usable because they retain their native resources.
 */
public interface TurboEmbed extends AutoCloseable {
    /** Creates a context for the selected device and nonnegative ordinal. */
    DeviceContext context(Device device, int ordinal);

    /** Releases this provider handle. Repeated close calls have no effect. */
    @Override
    void close();
}
