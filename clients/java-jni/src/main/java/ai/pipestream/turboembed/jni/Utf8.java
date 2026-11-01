package ai.pipestream.turboembed.jni;

import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * UTF-8 encoding of the strings that cross to the native library: one string as its bytes, and
 * a sequence of strings packed end to end with a byte length each, so a call crosses the
 * boundary with two arrays however many strings it carries.
 */
final class Utf8 {
    private Utf8() {
    }

    /** The UTF-8 bytes of {@code text}; the empty array for {@code null}. */
    static byte[] bytes(String text) {
        return text == null ? new byte[0] : text.getBytes(StandardCharsets.UTF_8);
    }

    /** Strings packed end to end. */
    static final class Packed {
        final byte[] bytes;
        final int[] lengths;

        private Packed(byte[] bytes, int[] lengths) {
            this.bytes = bytes;
            this.lengths = lengths;
        }
    }

    /** Packs {@code items}, none of them null, into one byte array and their lengths. */
    static Packed pack(String[] items, String name) {
        Objects.requireNonNull(items, name);
        byte[][] encoded = new byte[items.length][];
        long total = 0L;
        for (int i = 0; i < items.length; i++) {
            Objects.requireNonNull(items[i], name + "[" + i + "]");
            encoded[i] = items[i].getBytes(StandardCharsets.UTF_8);
            total += encoded[i].length;
        }
        if (total > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(name + " hold more than Integer.MAX_VALUE UTF-8 bytes together");
        }
        byte[] bytes = new byte[(int) total];
        int[] lengths = new int[items.length];
        int offset = 0;
        for (int i = 0; i < items.length; i++) {
            System.arraycopy(encoded[i], 0, bytes, offset, encoded[i].length);
            lengths[i] = encoded[i].length;
            offset += encoded[i].length;
        }
        return new Packed(bytes, lengths);
    }

    /** The slice {@code [offset, offset + length)} of a UTF-8 pool as a string. */
    static String decode(byte[] pool, int offset, int length) {
        return new String(pool, offset, length, StandardCharsets.UTF_8);
    }
}
