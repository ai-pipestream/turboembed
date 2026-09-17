// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

import java.util.List;

/**
 * Entry point for creating device contexts. Close instances deterministically;
 * existing contexts remain usable because they retain their native resources.
 */
public interface TurboEmbed extends AutoCloseable {
    /**
     * Lists the devices this provider can select on the running host as the
     * runtime reports them at call time: GPUs in ascending ordinal order, then
     * CPU. Discovery informs selection and never creates a context; selecting
     * an absent device still fails and CPU is never an automatic fallback.
     */
    List<DeviceInfo> devices();

    /** Creates a context for the selected device and nonnegative ordinal. */
    DeviceContext context(Device device, int ordinal);

    /** Releases this provider handle. Repeated close calls have no effect. */
    @Override
    void close();
}
