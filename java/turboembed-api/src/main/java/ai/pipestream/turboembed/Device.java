// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

/**
 * Device selection policy. {@link #AUTO} selects the host GPU and does not
 * fall back to CPU; CPU execution must be requested explicitly.
 */
public enum Device {
    AUTO,
    OPENVINO_GPU,
    OPENVINO_CPU
}
