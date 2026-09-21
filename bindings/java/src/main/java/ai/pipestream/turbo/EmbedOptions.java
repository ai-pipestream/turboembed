package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.turbo_embed_options;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Per-call embedding options ({@code turbo_embed_options}). Every field is
 * honored exactly or the write fails with {@code TURBO_E_UNSUPPORTED_OPTION}
 * naming the field; {@code MODEL} values and zeros mean the bundle contract.
 */
public record EmbedOptions(
        Truncate truncate,
        int maxTokens,
        PromptRole promptRole,
        Normalize normalize,
        Pooling pooling,
        int outputDim,
        OutputDType outputDtype) {

    /** The bundle contract for everything. */
    public static EmbedOptions defaults() {
        return new EmbedOptions(Truncate.MODEL, 0, PromptRole.NONE, Normalize.MODEL, Pooling.MODEL, 0, OutputDType.MODEL);
    }

    public EmbedOptions withTruncate(Truncate t) {
        return new EmbedOptions(t, maxTokens, promptRole, normalize, pooling, outputDim, outputDtype);
    }

    public EmbedOptions withMaxTokens(int n) {
        return new EmbedOptions(truncate, n, promptRole, normalize, pooling, outputDim, outputDtype);
    }

    public EmbedOptions withPromptRole(PromptRole r) {
        return new EmbedOptions(truncate, maxTokens, r, normalize, pooling, outputDim, outputDtype);
    }

    public EmbedOptions withNormalize(Normalize n) {
        return new EmbedOptions(truncate, maxTokens, promptRole, n, pooling, outputDim, outputDtype);
    }

    public EmbedOptions withPooling(Pooling p) {
        return new EmbedOptions(truncate, maxTokens, promptRole, normalize, p, outputDim, outputDtype);
    }

    public EmbedOptions withOutputDim(int d) {
        return new EmbedOptions(truncate, maxTokens, promptRole, normalize, pooling, d, outputDtype);
    }

    MemorySegment toNative(Arena arena) {
        MemorySegment o = turbo_embed_options.allocate(arena);
        turbo_embed_options.struct_size(o, (int) turbo_embed_options.sizeof());
        turbo_embed_options.truncate(o, truncate.value());
        turbo_embed_options.max_tokens(o, maxTokens);
        turbo_embed_options.prompt_role(o, promptRole.value());
        turbo_embed_options.normalize(o, normalize.value());
        turbo_embed_options.pooling(o, pooling.value());
        turbo_embed_options.output_dim(o, outputDim);
        turbo_embed_options.output_dtype(o, outputDtype.value());
        return o;
    }
}
