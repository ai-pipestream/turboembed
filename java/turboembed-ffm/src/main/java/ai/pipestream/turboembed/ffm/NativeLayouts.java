// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed.ffm;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.util.List;

/** Layouts checked against the canonical C header by NativeLayoutTest. */
final class NativeLayouts {
    private NativeLayouts() {}
    private static MemoryLayout u32(String name) { return ValueLayout.JAVA_INT.withName(name); }
    private static MemoryLayout u64(String name) { return ValueLayout.JAVA_LONG.withName(name); }
    private static MemoryLayout chars(String name, int count) {
        return MemoryLayout.sequenceLayout(count, ValueLayout.JAVA_BYTE).withName(name);
    }
    private static StructLayout struct(String name, MemoryLayout... fields) {
        return MemoryLayout.structLayout(fields).withName(name);
    }
    static final StructLayout ERROR = struct("te_error", u32("code"), chars("message", 508));
    static final StructLayout CONTEXT_OPTIONS = struct("te_context_options",
        u32("struct_size"), u32("version"), u32("device"), u32("ordinal"));
    static final StructLayout CONTEXT_INFO = struct("te_context_info",
        u32("struct_size"), u32("version"), u32("device"), u32("ordinal"), u64("capabilities"),
        chars("device_name", 128), chars("runtime_version", 128), chars("driver_version", 128));
    static final StructLayout MODEL_INFO = struct("te_model_info",
        u32("struct_size"), u32("version"), u32("dimension"), u32("vocab_size"),
        u32("max_sequence_length"), u32("max_batch_size"), u32("normalized"), u32("reserved"),
        chars("model_id", 128), chars("revision", 64), chars("tokenizer_sha256", 65), chars("pooling", 15));
    static final StructLayout SLOT_OPTIONS = struct("te_slot_options",
        u32("struct_size"), u32("version"), u32("batch"), u32("sequence_length"));
    static final StructLayout RESULT_INFO = struct("te_result_info",
        u32("struct_size"), u32("version"), u32("batch"), u32("dimension"), u64("byte_size"));
    static final StructLayout OPENCL = struct("te_opencl_view",
        u32("struct_size"), u32("version"), ValueLayout.ADDRESS.withName("context"),
        ValueLayout.ADDRESS.withName("queue"), ValueLayout.ADDRESS.withName("buffer"),
        u64("byte_size"), u32("batch"), u32("dimension"));
    static final StructLayout STATS = struct("te_slot_stats",
        u32("struct_size"), u32("version"), u64("executions"), u64("input_write_bytes"),
        u64("output_read_bytes"), u64("owned_input_bytes"), u64("owned_output_bytes"));
    static final StructLayout TEXT = struct("te_text", ValueLayout.ADDRESS.withName("ptr"), u64("byte_length"));
    static final long TEXT_PTR = offset(TEXT, "ptr");
    static final long TEXT_BYTE_LENGTH = offset(TEXT, "byte_length");
    static final long STATS_EXECUTIONS = offset(STATS, "executions");
    static final long STATS_INPUT_WRITE_BYTES = offset(STATS, "input_write_bytes");
    static final long STATS_OUTPUT_READ_BYTES = offset(STATS, "output_read_bytes");
    static final long STATS_OWNED_INPUT_BYTES = offset(STATS, "owned_input_bytes");
    static final long STATS_OWNED_OUTPUT_BYTES = offset(STATS, "owned_output_bytes");
    static final List<StructLayout> ALL = List.of(ERROR, CONTEXT_OPTIONS, CONTEXT_INFO, MODEL_INFO,
        SLOT_OPTIONS, RESULT_INFO, OPENCL, STATS, TEXT);
    static long offset(StructLayout layout, String field) {
        return layout.byteOffset(MemoryLayout.PathElement.groupElement(field));
    }
}
