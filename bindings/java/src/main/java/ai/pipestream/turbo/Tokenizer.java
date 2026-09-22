package ai.pipestream.turbo;

import static ai.pipestream.turbo.ffi.TurboNative.*;

import ai.pipestream.turbo.ffi.turbo_tokenizer_info;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * The tokenizer a bundle declares ({@code turbo_tokenizer_*}). Thread-safe
 * and independent of any device; it keeps the runtime alive.
 */
public final class Tokenizer implements AutoCloseable {
    private final MemorySegment t;
    private final TokenizerInfo info;
    private boolean closed;

    Tokenizer(MemorySegment t) {
        this.t = t;
        this.info = readInfo();
    }

    private TokenizerInfo readInfo() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment s = turbo_tokenizer_info.allocate(arena);
            turbo_tokenizer_info.struct_size(s, (int) turbo_tokenizer_info.sizeof());
            MemorySegment err = Native.error(arena);
            Native.check(turbo_tokenizer_get_info(t, s, err), err);
            return TokenizerInfo.from(s);
        }
    }

    MemorySegment handle() {
        if (closed) {
            throw new IllegalStateException("tokenizer is closed");
        }
        return t;
    }

    public TokenizerInfo info() {
        return info;
    }

    /**
     * Encode {@code texts} into rows of {@code rowStride} ids, padded with
     * the pad id and mask 0 up to {@code opts.padTo()} (or the stride).
     */
    public Encoding encode(List<String> texts, int rowStride, EncodeOptions opts) {
        int n = texts.size();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment ids = arena.allocate(ValueLayout.JAVA_INT, (long) n * rowStride);
            MemorySegment mask = arena.allocate(ValueLayout.JAVA_INT, (long) n * rowStride);
            MemorySegment lengths = arena.allocate(ValueLayout.JAVA_INT, n);
            MemorySegment err = Native.error(arena);
            Native.check(
                    turbo_tokenizer_encode(handle(), Native.texts(arena, texts), n, opts.toNative(arena), ids, mask, MemorySegment.NULL, rowStride, lengths, err),
                    err);
            return new Encoding(n, rowStride, ids.toArray(ValueLayout.JAVA_INT), mask.toArray(ValueLayout.JAVA_INT), lengths.toArray(ValueLayout.JAVA_INT));
        }
    }

    /** Decode ids to text. */
    public String decode(int[] ids, boolean skipSpecialTokens) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment src = arena.allocateFrom(ValueLayout.JAVA_INT, ids);
            MemorySegment written = arena.allocate(ValueLayout.JAVA_LONG);
            MemorySegment err = Native.error(arena);
            long capacity = Math.max(64L, ids.length * 8L);
            MemorySegment dst = arena.allocate(capacity);
            int rc = turbo_tokenizer_decode(handle(), src, ids.length, skipSpecialTokens ? 1 : 0, dst, capacity, written, err);
            if (rc == TURBO_E_CAPACITY()) {
                capacity = written.get(ValueLayout.JAVA_LONG, 0);
                dst = arena.allocate(capacity);
                err = Native.error(arena);
                rc = turbo_tokenizer_decode(handle(), src, ids.length, skipSpecialTokens ? 1 : 0, dst, capacity, written, err);
            }
            Native.check(rc, err);
            long n = written.get(ValueLayout.JAVA_LONG, 0);
            byte[] bytes = new byte[(int) n];
            MemorySegment.copy(dst, ValueLayout.JAVA_BYTE, 0, bytes, 0, (int) n);
            return new String(bytes, StandardCharsets.UTF_8);
        }
    }

    /** Number of tokens {@code text} produces, without truncation or prefix. */
    public int count(String text, boolean addSpecialTokens) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_tokenizer_count(handle(), Native.text(arena, text), addSpecialTokens ? 1 : 0, out, err), err);
            return out.get(ValueLayout.JAVA_INT, 0);
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            turbo_tokenizer_release(t);
        }
    }
}
