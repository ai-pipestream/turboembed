package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.turbo_generate_desc;
import ai.pipestream.turbo.ffi.turbo_logit_bias;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;

/**
 * Generation parameters ({@code turbo_generate_desc}). Zero and empty values
 * mean the model's defaults; every non-default value is honored exactly or
 * the generation fails with {@code TURBO_E_UNSUPPORTED_OPTION} naming the
 * field, following the device's {@code TURBO_CAP_OPT_GEN_*} bits.
 */
public record GenerateDesc(
        int maxNewTokens,
        int minNewTokens,
        int nSequences,
        float temperature,
        int topK,
        float topP,
        float minP,
        float repeatPenalty,
        float presencePenalty,
        float frequencyPenalty,
        Long seed,
        List<String> stop,
        int[] stopTokens,
        Map<Integer, Float> logitBias,
        int logprobs,
        boolean echo) {

    /** The model's defaults: greedy, unbounded by anything but the context. */
    public static GenerateDesc defaults() {
        return new GenerateDesc(0, 0, 0, 0f, 0, 0f, 0f, 0f, 0f, 0f, null, List.of(), new int[0], Map.of(), 0, false);
    }

    public GenerateDesc withMaxNewTokens(int n) {
        return new GenerateDesc(n, minNewTokens, nSequences, temperature, topK, topP, minP, repeatPenalty, presencePenalty,
                frequencyPenalty, seed, stop, stopTokens, logitBias, logprobs, echo);
    }

    public GenerateDesc withMinNewTokens(int n) {
        return new GenerateDesc(maxNewTokens, n, nSequences, temperature, topK, topP, minP, repeatPenalty, presencePenalty,
                frequencyPenalty, seed, stop, stopTokens, logitBias, logprobs, echo);
    }

    public GenerateDesc withSampling(float temperature, int topK, float topP, float minP) {
        return new GenerateDesc(maxNewTokens, minNewTokens, nSequences, temperature, topK, topP, minP, repeatPenalty,
                presencePenalty, frequencyPenalty, seed, stop, stopTokens, logitBias, logprobs, echo);
    }

    public GenerateDesc withSeed(long seed) {
        return new GenerateDesc(maxNewTokens, minNewTokens, nSequences, temperature, topK, topP, minP, repeatPenalty,
                presencePenalty, frequencyPenalty, seed, stop, stopTokens, logitBias, logprobs, echo);
    }

    public GenerateDesc withStop(List<String> stop) {
        return new GenerateDesc(maxNewTokens, minNewTokens, nSequences, temperature, topK, topP, minP, repeatPenalty,
                presencePenalty, frequencyPenalty, seed, List.copyOf(stop), stopTokens, logitBias, logprobs, echo);
    }

    public GenerateDesc withStopTokens(int[] stopTokens) {
        return new GenerateDesc(maxNewTokens, minNewTokens, nSequences, temperature, topK, topP, minP, repeatPenalty,
                presencePenalty, frequencyPenalty, seed, stop, stopTokens.clone(), logitBias, logprobs, echo);
    }

    public GenerateDesc withLogprobs(int n) {
        return new GenerateDesc(maxNewTokens, minNewTokens, nSequences, temperature, topK, topP, minP, repeatPenalty,
                presencePenalty, frequencyPenalty, seed, stop, stopTokens, logitBias, n, echo);
    }

    public GenerateDesc withEcho(boolean echo) {
        return new GenerateDesc(maxNewTokens, minNewTokens, nSequences, temperature, topK, topP, minP, repeatPenalty,
                presencePenalty, frequencyPenalty, seed, stop, stopTokens, logitBias, logprobs, echo);
    }

    /** The native descriptor; every pointer it holds lives in {@code arena}. */
    MemorySegment toNative(Arena arena) {
        MemorySegment d = turbo_generate_desc.allocate(arena);
        turbo_generate_desc.struct_size(d, (int) turbo_generate_desc.sizeof());
        turbo_generate_desc.max_new_tokens(d, maxNewTokens);
        turbo_generate_desc.min_new_tokens(d, minNewTokens);
        turbo_generate_desc.n_sequences(d, nSequences);
        turbo_generate_desc.temperature(d, temperature);
        turbo_generate_desc.top_k(d, topK);
        turbo_generate_desc.top_p(d, topP);
        turbo_generate_desc.min_p(d, minP);
        turbo_generate_desc.repeat_penalty(d, repeatPenalty);
        turbo_generate_desc.presence_penalty(d, presencePenalty);
        turbo_generate_desc.frequency_penalty(d, frequencyPenalty);
        turbo_generate_desc.has_seed(d, seed == null ? 0 : 1);
        turbo_generate_desc.seed(d, seed == null ? 0L : seed);
        turbo_generate_desc.n_stop(d, stop.size());
        turbo_generate_desc.stop(d, stop.isEmpty() ? MemorySegment.NULL : Native.texts(arena, stop));
        turbo_generate_desc.n_stop_tokens(d, stopTokens.length);
        turbo_generate_desc.stop_tokens(d, stopTokens.length == 0 ? MemorySegment.NULL : arena.allocateFrom(ValueLayout.JAVA_INT, stopTokens));
        turbo_generate_desc.n_logit_bias(d, logitBias.size());
        if (logitBias.isEmpty()) {
            turbo_generate_desc.logit_bias(d, MemorySegment.NULL);
        } else {
            MemorySegment arr = turbo_logit_bias.allocateArray(logitBias.size(), arena);
            int i = 0;
            for (Map.Entry<Integer, Float> e : logitBias.entrySet()) {
                MemorySegment b = turbo_logit_bias.asSlice(arr, i++);
                turbo_logit_bias.token(b, e.getKey());
                turbo_logit_bias.bias(b, e.getValue());
            }
            turbo_generate_desc.logit_bias(d, arr);
        }
        turbo_generate_desc.logprobs(d, logprobs);
        turbo_generate_desc.structured_kind(d, 0);
        turbo_generate_desc.echo(d, echo ? 1 : 0);
        return d;
    }
}
