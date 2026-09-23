package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.turbo_tokenizer_info;
import java.lang.foreign.MemorySegment;

/** Static facts about a tokenizer ({@code turbo_tokenizer_info}). */
public record TokenizerInfo(
        int vocabSize,
        int maxSeq,
        int specialsPerSequence,
        int padId,
        int bosId,
        int eosId,
        int unkId,
        String kind,
        String sha256) {

    static TokenizerInfo from(MemorySegment s) {
        return new TokenizerInfo(
                turbo_tokenizer_info.vocab_size(s),
                turbo_tokenizer_info.max_seq(s),
                turbo_tokenizer_info.specials_per_sequence(s),
                turbo_tokenizer_info.pad_id(s),
                turbo_tokenizer_info.bos_id(s),
                turbo_tokenizer_info.eos_id(s),
                turbo_tokenizer_info.unk_id(s),
                Native.fixed(turbo_tokenizer_info.kind(s)),
                Native.fixed(turbo_tokenizer_info.sha256(s)));
    }
}
