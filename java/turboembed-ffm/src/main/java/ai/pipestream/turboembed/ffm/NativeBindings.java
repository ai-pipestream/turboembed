// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed.ffm;

import ai.pipestream.turboembed.NativeException;
import java.lang.foreign.Arena;
import java.lang.foreign.AddressLayout;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import static ai.pipestream.turboembed.ffm.NativeLayouts.*;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/** Exact pointer-based declarations for turboembed_prepared.h, Linux x86_64. */
@SuppressWarnings("restricted") // The isolated native ABI boundary requires FFM downcalls.
final class NativeBindings implements AutoCloseable {
    static final ValueLayout.OfInt I = ValueLayout.JAVA_INT;
    static final ValueLayout.OfLong L = ValueLayout.JAVA_LONG;
    static final AddressLayout P = ValueLayout.ADDRESS;
    private final Arena libraryArena = Arena.ofShared();
    private final MethodHandle version, contextCreate, contextInfo, contextRelease;
    private final MethodHandle modelLoad, modelInfo, modelRelease, slotCreate, slotRelease;
    private final MethodHandle writeTokens, writeText, execute, stats, resultRead, resultOpenCl, resultRelease;
    private int references = 1;
    private boolean rootClosed;

    NativeBindings(Path library) {
        try {
            if (!System.getProperty("os.name").equals("Linux") || !System.getProperty("os.arch").equals("amd64") || P.byteSize() != 8) {
                throw new UnsupportedOperationException("The FFM provider currently requires Linux x86_64 and JDK 25+");
            }
            Linker linker = Linker.nativeLinker();
            SymbolLookup symbols = SymbolLookup.libraryLookup(library.toAbsolutePath(), libraryArena);
            version = function(linker, symbols, "version", FunctionDescriptor.of(I));
            contextCreate = function(linker, symbols, "context_create", FunctionDescriptor.of(I, P, P, P));
            contextInfo = function(linker, symbols, "context_info", FunctionDescriptor.of(I, P, P, P));
            contextRelease = function(linker, symbols, "context_release", FunctionDescriptor.ofVoid(P));
            modelLoad = function(linker, symbols, "model_load", FunctionDescriptor.of(I, P, P, L, P, P));
            modelInfo = function(linker, symbols, "model_info", FunctionDescriptor.of(I, P, P, P));
            modelRelease = function(linker, symbols, "model_release", FunctionDescriptor.ofVoid(P));
            slotCreate = function(linker, symbols, "slot_create", FunctionDescriptor.of(I, P, P, P, P));
            slotRelease = function(linker, symbols, "slot_release", FunctionDescriptor.ofVoid(P));
            writeTokens = function(linker, symbols, "slot_write_tokens", FunctionDescriptor.of(I, P, P, P, P, L, P));
            writeText = function(linker, symbols, "slot_write_text", FunctionDescriptor.of(I, P, P, L, P));
            execute = function(linker, symbols, "slot_execute", FunctionDescriptor.of(I, P, P, P));
            stats = function(linker, symbols, "slot_stats", FunctionDescriptor.of(I, P, P, P));
            resultRead = function(linker, symbols, "result_read", FunctionDescriptor.of(I, P, P, L, P));
            resultOpenCl = function(linker, symbols, "result_opencl", FunctionDescriptor.of(I, P, P, P));
            resultRelease = function(linker, symbols, "result_release", FunctionDescriptor.of(I, P, P));
            if (version() != 1) { throw new NativeException(8, "unsupported prepared ABI version"); }
        } catch (Throwable failure) {
            libraryArena.close();
            throw failure;
        }
    }
    private static MethodHandle function(Linker linker, SymbolLookup lookup, String suffix, FunctionDescriptor descriptor) {
        return linker.downcallHandle(lookup.findOrThrow("turboembed_prepared_v1_" + suffix), descriptor);
    }
    synchronized void retainRoot() {
        if (rootClosed) { throw new IllegalStateException("provider is closed"); }
        references++;
    }
    synchronized void retainDependent() {
        if (references == 0) { throw new IllegalStateException("native library is closed"); }
        references++;
    }
    synchronized void releaseDependent() {
        if (--references == 0) { libraryArena.close(); }
    }
    @Override public synchronized void close() {
        if (!rootClosed) { rootClosed = true; releaseDependent(); }
    }
    static MemorySegment descriptor(Arena arena, StructLayout layout) {
        MemorySegment result = arena.allocate(layout);
        result.set(I, offset(layout, "struct_size"), Math.toIntExact(layout.byteSize()));
        result.set(I, offset(layout, "version"), 1); return result;
    }
    static void check(int code, MemorySegment error) {
        if (code != 0) { throw new NativeException(code, error.asSlice(offset(ERROR, "message"), 508).getString(0)); }
    }
    static MemorySegment handle(MemorySegment pointer) {
        MemorySegment value = pointer.get(P, 0);
        if (value.address() == 0) { throw new NativeException(5, "native success returned a null handle"); }
        return value;
    }
    private static AssertionError invocation(Throwable failure) {
        if (failure instanceof RuntimeException runtime) { throw runtime; }
        if (failure instanceof Error error) { throw error; }
        return new AssertionError("native invocation or ABI declaration failed", failure);
    }
    int version() {
        try { return (int) version.invokeExact(); } catch (Throwable t) { throw invocation(t); }
    }
    int contextCreate(MemorySegment options, MemorySegment out, MemorySegment error) {
        try { return (int) contextCreate.invokeExact(options, out, error); } catch (Throwable t) { throw invocation(t); }
    }
    int contextInfo(MemorySegment handle, MemorySegment out, MemorySegment error) {
        try { return (int) contextInfo.invokeExact(handle, out, error); } catch (Throwable t) { throw invocation(t); }
    }
    void contextRelease(MemorySegment handle) {
        try { contextRelease.invokeExact(handle); } catch (Throwable t) { throw invocation(t); }
    }
    int modelLoad(MemorySegment context, MemorySegment path, long length, MemorySegment out, MemorySegment error) {
        try { return (int) modelLoad.invokeExact(context, path, length, out, error); } catch (Throwable t) { throw invocation(t); }
    }
    int modelInfo(MemorySegment handle, MemorySegment out, MemorySegment error) {
        try { return (int) modelInfo.invokeExact(handle, out, error); } catch (Throwable t) { throw invocation(t); }
    }
    void modelRelease(MemorySegment handle) {
        try { modelRelease.invokeExact(handle); } catch (Throwable t) { throw invocation(t); }
    }
    int slotCreate(MemorySegment model, MemorySegment options, MemorySegment out, MemorySegment error) {
        try { return (int) slotCreate.invokeExact(model, options, out, error); } catch (Throwable t) { throw invocation(t); }
    }
    void slotRelease(MemorySegment handle) {
        try { slotRelease.invokeExact(handle); } catch (Throwable t) { throw invocation(t); }
    }
    int writeTokens(MemorySegment slot, MemorySegment ids, MemorySegment mask, MemorySegment types, long count, MemorySegment error) {
        try { return (int) writeTokens.invokeExact(slot, ids, mask, types, count, error); } catch (Throwable t) { throw invocation(t); }
    }
    int writeText(MemorySegment slot, MemorySegment texts, long count, MemorySegment error) {
        try { return (int) writeText.invokeExact(slot, texts, count, error); } catch (Throwable t) { throw invocation(t); }
    }
    int execute(MemorySegment slot, MemorySegment out, MemorySegment error) {
        try { return (int) execute.invokeExact(slot, out, error); } catch (Throwable t) { throw invocation(t); }
    }
    int stats(MemorySegment slot, MemorySegment out, MemorySegment error) {
        try { return (int) stats.invokeExact(slot, out, error); } catch (Throwable t) { throw invocation(t); }
    }
    int resultRead(MemorySegment result, MemorySegment out, long count, MemorySegment error) {
        try { return (int) resultRead.invokeExact(result, out, count, error); } catch (Throwable t) { throw invocation(t); }
    }
    int resultOpenCl(MemorySegment result, MemorySegment out, MemorySegment error) {
        try { return (int) resultOpenCl.invokeExact(result, out, error); } catch (Throwable t) { throw invocation(t); }
    }
    int resultRelease(MemorySegment result, MemorySegment error) {
        try { return (int) resultRelease.invokeExact(result, error); } catch (Throwable t) { throw invocation(t); }
    }
}
