package ai.pipestream.turbo;

import ai.pipestream.turbo.ffi.turbo_model_info;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

/** What actually loaded ({@code turbo_model_info}). */
public record ModelInfo(
        Task task,
        ModelKind kind,
        Modality modality,
        int dim,
        List<String> labels,
        Pooling pooling,
        Normalize normalize,
        int maxSeq,
        int maxBatch,
        DType dtypeUsed,
        boolean fullyAccelerated,
        int[] stagePlacement,
        int vocabSize,
        String modelId,
        String revision,
        String tokenizerSha256,
        String providerId,
        String prefixQuery,
        String prefixDocument) {

    static ModelInfo from(MemorySegment s, List<String> labels) {
        MemorySegment placements = turbo_model_info.stage_placement(s);
        int[] stages = new int[(int) turbo_model_info.stage_placement$dimensions()[0]];
        for (int i = 0; i < stages.length; i++) {
            stages[i] = placements.getAtIndex(ValueLayout.JAVA_INT, i);
        }
        int pooling = turbo_model_info.pooling(s);
        int normalize = turbo_model_info.normalize(s);
        int dtype = turbo_model_info.dtype_used(s);
        return new ModelInfo(
                Task.of(turbo_model_info.task(s)),
                ModelKind.of(turbo_model_info.kind(s)),
                Modality.of(turbo_model_info.modality(s)),
                turbo_model_info.dim(s),
                labels,
                pooling == 0 ? null : Pooling.of(pooling),
                normalize == 0 ? null : Normalize.of(normalize),
                turbo_model_info.max_seq(s),
                turbo_model_info.max_batch(s),
                dtype == 0 ? null : DType.of(dtype),
                turbo_model_info.fully_accelerated(s) != 0,
                stages,
                turbo_model_info.vocab_size(s),
                Native.fixed(turbo_model_info.model_id(s)),
                Native.fixed(turbo_model_info.revision(s)),
                Native.fixed(turbo_model_info.tokenizer_sha256(s)),
                Native.fixed(turbo_model_info.provider_id(s)),
                Native.fixed(turbo_model_info.prefix_query(s)),
                Native.fixed(turbo_model_info.prefix_document(s)));
    }
}
