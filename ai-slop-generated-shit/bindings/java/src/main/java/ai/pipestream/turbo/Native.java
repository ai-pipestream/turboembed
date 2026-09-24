package ai.pipestream.turbo;

import static ai.pipestream.turbo.ffi.TurboNative.*;

import ai.pipestream.turbo.ffi.turbo_error;
import ai.pipestream.turbo.ffi.turbo_kv;
import ai.pipestream.turbo.ffi.turbo_text;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

/**
 * Helpers shared by the safe API: caller-owned error structs, text views,
 * fixed-size C strings, and the status check that turns a code into a
 * {@link TurboException}. Package-private on purpose.
 */
final class Native {
    private Native() {}

    /** A zeroed {@code turbo_error} with its {@code struct_size} set. */
    static MemorySegment error(Arena arena) {
        MemorySegment e = turbo_error.allocate(arena);
        turbo_error.struct_size(e, (int) turbo_error.sizeof());
        return e;
    }

    /** Throw when {@code rc} is not {@code TURBO_OK}. */
    static void check(int rc, MemorySegment err) {
        if (rc == TURBO_OK()) {
            return;
        }
        String message = turbo_error.message(err).getString(0, StandardCharsets.UTF_8);
        throw new TurboException(rc, turbo_error.field(err), message);
    }

    /** Symbolic status name from the library. */
    static String statusName(int code) {
        return turbo_status_name(code).getString(0, StandardCharsets.UTF_8);
    }

    /** A {@code turbo_text} over a UTF-8 copy of {@code s} (no terminator needed). */
    static MemorySegment text(Arena arena, String s) {
        MemorySegment t = turbo_text.allocate(arena);
        fillText(arena, t, s);
        return t;
    }

    static void fillText(Arena arena, MemorySegment t, String s) {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        MemorySegment buf = arena.allocate(Math.max(bytes.length, 1));
        MemorySegment.copy(bytes, 0, buf, ValueLayout.JAVA_BYTE, 0, bytes.length);
        turbo_text.ptr(t, buf);
        turbo_text.len(t, bytes.length);
    }

    /** An array of {@code turbo_text} for {@code texts}. */
    static MemorySegment texts(Arena arena, List<String> texts) {
        MemorySegment arr = turbo_text.allocateArray(texts.size(), arena);
        for (int i = 0; i < texts.size(); i++) {
            fillText(arena, turbo_text.asSlice(arr, i), texts.get(i));
        }
        return arr;
    }

    /** An array of {@code turbo_kv} for {@code options}, or NULL when empty. */
    static MemorySegment kvs(Arena arena, Map<String, String> options) {
        if (options.isEmpty()) {
            return MemorySegment.NULL;
        }
        MemorySegment arr = turbo_kv.allocateArray(options.size(), arena);
        int i = 0;
        for (Map.Entry<String, String> e : options.entrySet()) {
            MemorySegment kv = turbo_kv.asSlice(arr, i++);
            fillText(arena, turbo_kv.key(kv), e.getKey());
            fillText(arena, turbo_kv.value(kv), e.getValue());
        }
        return arr;
    }

    /** Read a NUL-terminated fixed-size {@code char[]} field. */
    static String fixed(MemorySegment field) {
        return field.getString(0, StandardCharsets.UTF_8);
    }

    /** Read a {@code turbo_text} view the library returned. */
    static String textOf(MemorySegment t) {
        long len = turbo_text.len(t);
        if (len == 0) {
            return "";
        }
        byte[] bytes = new byte[(int) len];
        MemorySegment.copy(turbo_text.ptr(t).reinterpret(len), ValueLayout.JAVA_BYTE, 0, bytes, 0, (int) len);
        return new String(bytes, StandardCharsets.UTF_8);
    }
}
