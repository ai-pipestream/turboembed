// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed.ffm;

import ai.pipestream.turboembed.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.MemoryLayout;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ReadOnlyBufferException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Objects;
import static ai.pipestream.turboembed.ffm.NativeBindings.*;
import static ai.pipestream.turboembed.ffm.NativeLayouts.*;

/** JDK 25+ in-process adapter for the installed Intel native SDK. */
public final class FfmTurboEmbed implements TurboEmbed {
    private final NativeBindings api;

    private FfmTurboEmbed(Path library) { api = new NativeBindings(library); }

    /** Selects this adapter explicitly. No service, model download or JNI is used. */
    public static FfmTurboEmbed open(Path sdkPrefix) {
        return new FfmTurboEmbed(Objects.requireNonNull(sdkPrefix).resolve("lib/libturboembed_prepared.so.1"));
    }

    @Override public DeviceContext context(Device device, int ordinal) {
        Objects.requireNonNull(device);
        if (ordinal < 0) { throw new IllegalArgumentException("ordinal must be nonnegative"); }
        api.retainRoot();
        MemorySegment handle = MemorySegment.NULL;
        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment options = descriptor(scratch, CONTEXT_OPTIONS);
            options.set(I, offset(CONTEXT_OPTIONS, "device"), switch (device) { case AUTO -> 0; case OPENVINO_GPU -> 1; case OPENVINO_CPU -> 2; });
            options.set(I, offset(CONTEXT_OPTIONS, "ordinal"), ordinal);
            MemorySegment out = scratch.allocate(P), error = scratch.allocate(ERROR);
            check(api.contextCreate(options, out, error), error);
            handle = handle(out);
            return new ContextHandle(api, handle);
        } catch (Throwable failure) {
            if (handle.address() != 0) { api.contextRelease(handle); }
            api.releaseDependent(); throw failure;
        }
    }

    /** Existing context/model/slot handles independently retain the loaded library. */
    @Override public void close() { api.close(); }

    private static byte[] utf8(String value) {
        Objects.requireNonNull(value, "text");
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (Character.isHighSurrogate(c)) {
                if (++i >= value.length() || !Character.isLowSurrogate(value.charAt(i))) {
                    throw new IllegalArgumentException("text contains an unpaired UTF-16 surrogate");
                }
            } else if (Character.isLowSurrogate(c)) {
                throw new IllegalArgumentException("text contains an unpaired UTF-16 surrogate");
            }
        }
        return value.getBytes(StandardCharsets.UTF_8);
    }

    private static final class ContextHandle implements DeviceContext {
        private final NativeBindings api;
        private MemorySegment handle;
        ContextHandle(NativeBindings api, MemorySegment handle) { this.api = api; this.handle = handle; }
        private void open() { if (handle.address() == 0) { throw new IllegalStateException("context is closed"); } }

        @Override public synchronized DeviceInfo info() {
            open();
            try (Arena scratch = Arena.ofConfined()) {
                MemorySegment out = descriptor(scratch, CONTEXT_INFO), error = scratch.allocate(ERROR);
                check(api.contextInfo(handle, out, error), error);
                return new DeviceInfo(out.get(I, offset(CONTEXT_INFO, "device")), out.get(I, offset(CONTEXT_INFO, "ordinal")), out.get(L, offset(CONTEXT_INFO, "capabilities")),
                    out.asSlice(offset(CONTEXT_INFO, "device_name"), 128).getString(0), out.asSlice(offset(CONTEXT_INFO, "runtime_version"), 128).getString(0), out.asSlice(offset(CONTEXT_INFO, "driver_version"), 128).getString(0));
            }
        }

        @Override public synchronized Model loadModel(Path bundle) {
            open(); byte[] bytes = utf8(Objects.requireNonNull(bundle).toAbsolutePath().toString());
            api.retainDependent(); MemorySegment child = MemorySegment.NULL;
            try (Arena scratch = Arena.ofConfined()) {
                MemorySegment path = scratch.allocateFrom(java.lang.foreign.ValueLayout.JAVA_BYTE, bytes);
                MemorySegment out = scratch.allocate(P), error = scratch.allocate(ERROR);
                check(api.modelLoad(handle, path, bytes.length, out, error), error);
                child = handle(out);
                return new ModelHandle(api, child);
            } catch (Throwable failure) {
                if (child.address() != 0) { api.modelRelease(child); }
                api.releaseDependent(); throw failure;
            }
        }

        @Override public synchronized void close() {
            if (handle.address() != 0) {
                MemorySegment old = handle; handle = MemorySegment.NULL;
                try { api.contextRelease(old); } finally { api.releaseDependent(); }
            }
        }
    }

    private static final class ModelHandle implements Model {
        private final NativeBindings api;
        private final ModelInfo info;
        private MemorySegment handle;
        ModelHandle(NativeBindings api, MemorySegment handle) {
            this.api = api; this.handle = handle;
            try (Arena scratch = Arena.ofConfined()) {
                MemorySegment out = descriptor(scratch, MODEL_INFO), error = scratch.allocate(ERROR);
                check(api.modelInfo(handle, out, error), error);
                info = new ModelInfo(out.get(I, offset(MODEL_INFO, "dimension")), out.get(I, offset(MODEL_INFO, "vocab_size")), out.get(I, offset(MODEL_INFO, "max_sequence_length")), out.get(I, offset(MODEL_INFO, "max_batch_size")), out.get(I, offset(MODEL_INFO, "normalized")) != 0,
                    out.asSlice(offset(MODEL_INFO, "model_id"), 128).getString(0), out.asSlice(offset(MODEL_INFO, "revision"), 64).getString(0),
                    out.asSlice(offset(MODEL_INFO, "tokenizer_sha256"), 65).getString(0), out.asSlice(offset(MODEL_INFO, "pooling"), 15).getString(0));
            }
        }
        private void open() { if (handle.address() == 0) { throw new IllegalStateException("model is closed"); } }
        @Override public synchronized ModelInfo info() { open(); return info; }
        @Override public synchronized ExecutionSlot slot(int batch, int sequenceLength) {
            open();
            if (batch < 1 || batch > info.maxBatchSize() || sequenceLength < 2 || sequenceLength > info.maxSequenceLength()) {
                throw new IllegalArgumentException("slot shape exceeds model limits");
            }
            api.retainDependent(); MemorySegment child = MemorySegment.NULL;
            try (Arena scratch = Arena.ofConfined()) {
                MemorySegment options = descriptor(scratch, SLOT_OPTIONS);
                options.set(I, offset(SLOT_OPTIONS, "batch"), batch);
                options.set(I, offset(SLOT_OPTIONS, "sequence_length"), sequenceLength);
                MemorySegment out = scratch.allocate(P), error = scratch.allocate(ERROR);
                check(api.slotCreate(handle, options, out, error), error);
                child = handle(out);
                return new SlotHandle(api, child, batch, sequenceLength, info.dimension());
            } catch (Throwable failure) {
                if (child.address() != 0) { api.slotRelease(child); }
                api.releaseDependent(); throw failure;
            }
        }
        @Override public synchronized void close() {
            if (handle.address() != 0) {
                MemorySegment old = handle; handle = MemorySegment.NULL;
                try { api.modelRelease(old); } finally { api.releaseDependent(); }
            }
        }
    }

    private static final class SlotHandle implements ExecutionSlot {
        private final NativeBindings api;
        private final Thread owner = Thread.currentThread();
        private final Arena arena = Arena.ofConfined();
        private final int batch, dimension;
        private final long tokenCount;
        private final MemorySegment ids, masks, types, textDescriptors, outputPointer, error;
        private final MemorySegment hostOutput, openClInfo, statsInfo;
        private final FloatBuffer hostFloats;
        private final TokenBuffers inputs;
        private MemorySegment handle;
        private ResultHandle active;

        SlotHandle(NativeBindings api, MemorySegment handle, int batch, int sequence, int dimension) {
            this.api = api; this.handle = handle; this.batch = batch; this.dimension = dimension;
            tokenCount = Math.multiplyExact((long) batch, sequence);
            try {
                ids = arena.allocate(tokenCount * 4, 4); masks = arena.allocate(tokenCount * 4, 4);
                types = arena.allocate(tokenCount * 4, 4);
                inputs = new TokenBuffers(ints(ids), ints(masks), ints(types));
                textDescriptors = arena.allocate(MemoryLayout.sequenceLayout(batch, TEXT));
                outputPointer = arena.allocate(P); error = arena.allocate(ERROR);
                hostOutput = arena.allocate((long) batch * dimension * 4, 4);
                hostFloats = hostOutput.asByteBuffer().order(ByteOrder.nativeOrder()).asFloatBuffer();
                openClInfo = descriptor(arena, OPENCL); statsInfo = descriptor(arena, STATS);
            } catch (Throwable failure) { arena.close(); throw failure; }
        }
        private static IntBuffer ints(MemorySegment segment) {
            return segment.asByteBuffer().order(ByteOrder.nativeOrder()).asIntBuffer();
        }
        private void owner() {
            if (Thread.currentThread() != owner) { throw new IllegalStateException("slot/result belongs to another thread"); }
        }
        private void open() {
            owner(); if (handle.address() == 0) { throw new IllegalStateException("slot is closed"); }
        }
        private void idle() {
            open(); if (active != null) { throw new IllegalStateException("release the result before reusing the slot"); }
        }
        @Override public TokenBuffers inputs() { open(); return inputs; }
        /** Uploads every element of the fixed buffers; buffer positions are ignored. */
        @Override public void upload() {
            idle(); check(api.writeTokens(handle, ids, masks, types, tokenCount, error), error);
        }
        private void invalidate() {
            // This intentionally fails and invalidates prior prepared inputs.
            check(api.writeTokens(handle, MemorySegment.NULL, MemorySegment.NULL, MemorySegment.NULL, 0, error), error);
        }
        @Override public void writeText(String... texts) {
            idle();
            if (texts == null || texts.length != batch) { invalidate(); return; }
            byte[][] encoded = new byte[batch][];
            try {
                for (int i = 0; i < batch; i++) { encoded[i] = utf8(texts[i]); }
            } catch (IllegalArgumentException | NullPointerException failure) {
                // Preserve native failed-write semantics even for invalid UTF-16.
                try { invalidate(); } catch (NativeException ignored) { /* expected */ }
                throw failure;
            }
            try (Arena scratch = Arena.ofConfined()) {
                for (int i = 0; i < batch; i++) {
                    byte[] bytes = encoded[i];
                    MemorySegment text = bytes.length == 0 ? MemorySegment.NULL
                        : scratch.allocateFrom(java.lang.foreign.ValueLayout.JAVA_BYTE, bytes);
                    textDescriptors.set(P, i * TEXT.byteSize() + TEXT_PTR, text);
                    textDescriptors.set(L, i * TEXT.byteSize() + TEXT_BYTE_LENGTH, bytes.length);
                }
                check(api.writeText(handle, textDescriptors, batch, error), error);
            }
        }
        @Override public EmbeddingResult execute() {
            idle(); check(api.execute(handle, outputPointer, error), error);
            MemorySegment result = handle(outputPointer);
            try { active = new ResultHandle(this, result); return active; }
            catch (Throwable failure) { api.resultRelease(result, error); throw failure; }
        }
        @Override public SlotStats stats() {
            open(); check(api.stats(handle, statsInfo, error), error);
            return new SlotStats(statsInfo.get(L, STATS_EXECUTIONS), statsInfo.get(L, STATS_INPUT_WRITE_BYTES), statsInfo.get(L, STATS_OUTPUT_READ_BYTES),
                statsInfo.get(L, STATS_OWNED_INPUT_BYTES), statsInfo.get(L, STATS_OWNED_OUTPUT_BYTES));
        }
        @Override public void close() {
            owner();
            if (handle.address() == 0) { return; }
            if (active != null) { throw new IllegalStateException("release the result before closing its slot"); }
            MemorySegment old = handle; handle = MemorySegment.NULL;
            try { api.slotRelease(old); }
            finally { arena.close(); api.releaseDependent(); }
        }
    }

    private static final class ResultHandle implements EmbeddingResult {
        private final SlotHandle slot;
        private final MemorySegment handle;
        private boolean closed;
        ResultHandle(SlotHandle slot, MemorySegment handle) { this.slot = slot; this.handle = handle; }
        private void open() {
            slot.open();
            if (closed || slot.active != this) { throw new IllegalStateException("result is closed"); }
        }
        @Override public int batch() { open(); return slot.batch; }
        @Override public int dimension() { open(); return slot.dimension; }
        @Override public void readInto(FloatBuffer output) {
            open(); Objects.requireNonNull(output);
            if (output.isReadOnly()) { throw new ReadOnlyBufferException(); }
            int count = Math.multiplyExact(slot.batch, slot.dimension);
            if (output.remaining() < count) { throw new IllegalArgumentException("output buffer is too small"); }
            if (output.isDirect() && output.order().equals(ByteOrder.nativeOrder())) {
                MemorySegment destination = MemorySegment.ofBuffer(output);
                // ByteBuffer slices can expose a FloatBuffer at an unaligned address.
                if ((destination.address() & 3) == 0) {
                    check(slot.api.resultRead(handle, destination, count, slot.error), slot.error);
                    output.position(output.position() + count);
                    return;
                }
            }
            check(slot.api.resultRead(handle, slot.hostOutput, count, slot.error), slot.error);
            slot.hostFloats.clear(); output.put(slot.hostFloats);
        }
        @Override public OpenClView openCl() {
            open(); check(slot.api.resultOpenCl(handle, slot.openClInfo, slot.error), slot.error);
            long context = slot.openClInfo.get(L, offset(OPENCL, "context")), queue = slot.openClInfo.get(L, offset(OPENCL, "queue"));
            long buffer = slot.openClInfo.get(L, offset(OPENCL, "buffer")), bytes = slot.openClInfo.get(L, offset(OPENCL, "byte_size"));
            return new OpenClView() {
                public long context() { open(); return context; }
                public long queue() { open(); return queue; }
                public long buffer() { open(); return buffer; }
                public long byteSize() { open(); return bytes; }
            };
        }
        @Override public void close() {
            slot.owner(); if (closed) { return; }
            closed = true;
            try { check(slot.api.resultRelease(handle, slot.error), slot.error); }
            finally { slot.active = null; }
        }
    }
}
