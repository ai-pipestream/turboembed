package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.turbo_encode_options;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Tokenizer encode options ({@code turbo_encode_options}): whether to add
 * the special tokens, the truncation policy and budget, the padded row
 * length ({@code 0}: the row stride), and the prompt prefix role.
 */
public record EncodeOptions(boolean addSpecialTokens, Truncate truncate, int maxTokens, int padTo, PromptRole promptRole) {

    public static EncodeOptions defaults() {
        return new EncodeOptions(true, Truncate.MODEL, 0, 0, PromptRole.NONE);
    }

    public EncodeOptions withTruncate(Truncate t) {
        return new EncodeOptions(addSpecialTokens, t, maxTokens, padTo, promptRole);
    }

    public EncodeOptions withMaxTokens(int n) {
        return new EncodeOptions(addSpecialTokens, truncate, n, padTo, promptRole);
    }

    public EncodeOptions withPadTo(int n) {
        return new EncodeOptions(addSpecialTokens, truncate, maxTokens, n, promptRole);
    }

    public EncodeOptions withSpecialTokens(boolean add) {
        return new EncodeOptions(add, truncate, maxTokens, padTo, promptRole);
    }

    MemorySegment toNative(Arena arena) {
        MemorySegment o = turbo_encode_options.allocate(arena);
        turbo_encode_options.struct_size(o, (int) turbo_encode_options.sizeof());
        turbo_encode_options.add_special_tokens(o, addSpecialTokens ? 1 : 0);
        turbo_encode_options.truncate(o, truncate.value());
        turbo_encode_options.max_tokens(o, maxTokens);
        turbo_encode_options.pad_to(o, padTo);
        turbo_encode_options.prompt_role(o, promptRole.value());
        return o;
    }
}
