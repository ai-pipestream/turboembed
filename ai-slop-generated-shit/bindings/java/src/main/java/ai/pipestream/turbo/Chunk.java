package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.turbo_generation_chunk;
import ai.pipestream.turbo.ffi.turbo_text;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

/**
 * One step of a generation ({@code turbo_generation_chunk}), copied out of
 * native memory so it stays valid after the next step.
 *
 * @param sequence which sequence this chunk belongs to
 * @param tokens the new token ids
 * @param text their decoded text (may lag the tokens on partial UTF-8)
 * @param logprobs one value per token when requested, else empty
 * @param done true on the final chunk
 * @param finishReason why the generation stopped; {@link FinishReason#NONE} until done
 * @param promptTokens tokens the prompt occupied
 * @param generatedTokens tokens produced so far in this sequence
 */
public record Chunk(
        int sequence,
        int[] tokens,
        String text,
        float[] logprobs,
        boolean done,
        FinishReason finishReason,
        int promptTokens,
        int generatedTokens) {

    static Chunk from(MemorySegment c) {
        int n = turbo_generation_chunk.n_tokens(c);
        int[] tokens = new int[n];
        if (n > 0) {
            MemorySegment p = turbo_generation_chunk.tokens(c).reinterpret((long) n * Integer.BYTES);
            MemorySegment.copy(p, ValueLayout.JAVA_INT, 0, tokens, 0, n);
        }
        int nl = turbo_generation_chunk.n_logprobs(c);
        float[] logprobs = new float[nl];
        if (nl > 0) {
            MemorySegment p = turbo_generation_chunk.logprobs(c).reinterpret((long) nl * Float.BYTES);
            MemorySegment.copy(p, ValueLayout.JAVA_FLOAT, 0, logprobs, 0, nl);
        }
        MemorySegment t = turbo_generation_chunk.text(c);
        long len = turbo_text.len(t);
        String text = "";
        if (len > 0) {
            byte[] bytes = new byte[(int) len];
            MemorySegment.copy(turbo_text.ptr(t).reinterpret(len), ValueLayout.JAVA_BYTE, 0, bytes, 0, (int) len);
            text = new String(bytes, StandardCharsets.UTF_8);
        }
        return new Chunk(
                turbo_generation_chunk.sequence(c),
                tokens,
                text,
                logprobs,
                turbo_generation_chunk.done(c) != 0,
                FinishReason.of(turbo_generation_chunk.finish_reason(c)),
                turbo_generation_chunk.prompt_tokens(c),
                turbo_generation_chunk.generated_tokens(c));
    }
}
