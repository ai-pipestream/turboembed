// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed;

/**
 * Counters for one slot. Write and read bytes count explicit adapter GPU
 * transfers. Owned bytes describe bound input and output tensors. The counters
 * exclude GPU host staging, tokenizer scratch, provider allocations, and other
 * process allocations.
 */
public record SlotStats(
        long executions,
        long inputWriteBytes,
        long outputReadBytes,
        long ownedInputBytes,
        long ownedOutputBytes) {}
