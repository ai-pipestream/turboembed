// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

/**
 * Borrowed OpenCL resource handles valid on the owner thread while the result
 * lease remains open. The buffer is read-only. Use only the supplied in-order
 * queue; closing the result waits for that queue to finish. Do not release these
 * handles or dereference their numeric values as host-memory addresses.
 */
public interface OpenClView {
    /** Returns the borrowed {@code cl_context} identifier. */
    long context();

    /** Returns the borrowed provider-owned {@code cl_command_queue} identifier. */
    long queue();

    /** Returns the borrowed read-only {@code cl_mem} identifier. */
    long buffer();

    /** Returns the output buffer size in bytes. */
    long byteSize();
}
