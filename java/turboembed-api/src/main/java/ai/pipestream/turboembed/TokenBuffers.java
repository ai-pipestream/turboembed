// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

import java.nio.IntBuffer;

/**
 * Provider-owned, fixed-size row-major token staging buffers. The three
 * buffers have equal capacity; IDs must be within the model vocabulary and
 * attention masks and token types contain only zero or one.
 */
public record TokenBuffers(IntBuffer ids, IntBuffer attentionMask, IntBuffer tokenTypes) {}
