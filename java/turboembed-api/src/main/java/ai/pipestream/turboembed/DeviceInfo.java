// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

/** Immutable device metadata. */
public record DeviceInfo(
        int device,
        int ordinal,
        long capabilities,
        String name,
        String runtimeVersion,
        String driverVersion) {}
