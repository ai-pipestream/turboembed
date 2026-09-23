// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import ai.pipestream.turbo.Capability;
import ai.pipestream.turbo.Chunk;
import ai.pipestream.turbo.ClassifyOptions;
import ai.pipestream.turbo.Context;
import ai.pipestream.turbo.DeviceInfo;
import ai.pipestream.turbo.EmbedOptions;
import ai.pipestream.turbo.EncodeOptions;
import ai.pipestream.turbo.Encoding;
import ai.pipestream.turbo.GenerateDesc;
import ai.pipestream.turbo.Generation;
import ai.pipestream.turbo.Message;
import ai.pipestream.turbo.Modality;
import ai.pipestream.turbo.Model;
import ai.pipestream.turbo.ModelInfo;
import ai.pipestream.turbo.ModelKind;
import ai.pipestream.turbo.RerankOptions;
import ai.pipestream.turbo.Result;
import ai.pipestream.turbo.SelectPolicy;
import ai.pipestream.turbo.Span;
import ai.pipestream.turbo.Task;
import ai.pipestream.turbo.Tokenizer;
import ai.pipestream.turbo.TokenizerInfo;
import ai.pipestream.turbo.Turbo;
import jakarta.annotation.PreDestroy;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import org.springframework.stereotype.Service;

/**
 * The models this server serves and the operations over them.
 *
 * <p>One runtime holds every provider library named in the configuration; one
 * context per device holds the models loaded onto it. Each model keeps its own
 * session pool, so an embedder and a reranker never contend. Nothing falls
 * back: an unknown model name is {@link ModelNotFound}, a task no loaded model
 * serves is {@link TaskNotServed}, a full pool is {@link Overloaded}, and every
 * libturbo refusal propagates as a {@link ai.pipestream.turbo.TurboException}
 * carrying its status code and the 1-based field index.
 */
@Service
public class TurboService implements AutoCloseable {
    /** Raised when every session or generation slot of a model is in use. */
    public static final class Overloaded extends RuntimeException {
        public Overloaded(String message) {
            super(message);
        }
    }

    /** Raised when a request names a model this server did not load. */
    public static final class ModelNotFound extends RuntimeException {
        public ModelNotFound(String message) {
            super(message);
        }
    }

    /** Raised when no loaded model performs the requested task. */
    public static final class TaskNotServed extends RuntimeException {
        public TaskNotServed(String message) {
            super(message);
        }
    }

    /** The device a call ran on, as every response reports it. */
    public record DeviceRef(int index, String name, String kind, String providerId, int ordinal, String runtimeVersion) {
        static DeviceRef of(DeviceInfo d) {
            return new DeviceRef(d.index(), d.name(), d.kind().name(), d.providerId(), d.ordinal(), d.runtimeVersion());
        }
    }

    /** Wall-clock milliseconds spent in each phase of a session call. */
    public record Timings(double writeMs, double runMs, double readMs, double totalMs) {}

    /** Wall-clock milliseconds and the token rate of a generation. */
    public record GenTimings(double firstTokenMs, double totalMs, double tokensPerSecond) {}

    /** Embedding vectors and where they were produced. */
    public record EmbedResult(String model, DeviceRef device, int dim, float[][] vectors, String placement, Timings timings) {}

    /** Rerank scores, and the ranking when one was asked for. */
    public record RerankResult(String model, DeviceRef device, float[] scores, int[] sorted, String placement, Timings timings) {}

    /** One score row per input text, with the bundle's labels. */
    public record ClassifyResult(String model, DeviceRef device, List<String> labels, float[][] scores, String placement, Timings timings) {}

    /** One aggregated span: byte offsets into its input text, the bundle's label and the score. */
    public record SpanResult(int row, long byteStart, long byteEnd, int labelIndex, String label, float score, String text) {}

    /** Spans plus the raw per-token score tensor the model produced. */
    public record TokenClassifyResult(String model, DeviceRef device, List<String> labels, List<SpanResult> spans,
            long[] scoreShape, float[] scores, String placement, Timings timings) {}

    /** One tokenized text. */
    public record TokenizeRow(int[] ids, int[] mask, List<String> tokens, int count) {}

    /** Tokenizer output for a batch, with the tokenizer's own identity. */
    public record TokenizeResult(String model, String tokenizerBundle, TokenizerInfo tokenizer, List<TokenizeRow> rows, Timings timings) {}

    /** Decoded texts. */
    public record DetokenizeResult(String model, String tokenizerBundle, List<String> texts, Timings timings) {}

    /** A finished generation. */
    public record GenerateResult(String model, DeviceRef device, String text, String finishReason, int promptTokens,
            int generatedTokens, List<Float> logprobs, GenTimings timings) {}

    /** One cell of a device's task-by-modality capability matrix. */
    public record CapabilityCell(String task, String modality, String status, String dtype, String referenceDtype,
            float cosineFloor, float maxAbsError, boolean deterministic, String notes) {}

    /** One device of one provider, with its feature bits and its capability matrix. */
    public record DeviceReport(int index, String name, String kind, int ordinal, String vendor, int vendorId,
            String providerId, String providerVersion, String runtimeVersion, String driverVersion,
            long memoryTotal, long memoryFree, long caps, List<String> features, List<CapabilityCell> capabilities,
            List<String> models) {}

    /** One loaded model's contract, exactly as {@code turbo_model_info} reports it. */
    public record ModelReport(String name, String bundle, String task, String kind, String modality, int dim,
            List<String> labels, String pooling, String normalize, int maxSeq, int maxBatch, String dtype,
            boolean fullyAccelerated, Map<String, String> stagePlacement, int vocabSize, String modelId,
            String revision, String tokenizerSha256, String prefixQuery, String prefixDocument, String tokenizerBundle,
            DeviceRef device) {}

    /** Liveness and the shape of the server. */
    public record HealthReport(String status, int abiVersion, int deviceCount, List<String> models) {}

    private static final String[] STAGE_NAMES = {"tokenize", "encode", "pool", "normalize", "postprocess"};
    private static final String[] STAGE_PLACEMENTS = {"unused", "host", "device", "fused"};

    private final Turbo runtime;
    private final Map<Integer, Context> contexts = new LinkedHashMap<>();
    private final Map<String, LoadedModel> models = new LinkedHashMap<>();
    private final Map<Task, LoadedModel> defaults = new LinkedHashMap<>();
    private final List<DeviceInfo> devices;
    private boolean closed;

    public TurboService(TurboProperties props) {
        List<TurboProperties.ModelConfig> configs = props.resolved();
        if (configs.isEmpty()) {
            throw new IllegalStateException(
                    "no model is configured: pass --turbo.bundle=<bundle dir>, --turbo.generate-bundle=<bundle dir>, "
                            + "or turbo.models[0].bundle=<bundle dir>");
        }
        List<String> libs = new ArrayList<>();
        for (TurboProperties.ModelConfig c : configs) {
            if (!c.getProviderLib().isBlank() && !libs.contains(c.getProviderLib())) {
                libs.add(c.getProviderLib());
            }
        }
        runtime = libs.isEmpty() ? Turbo.create() : Turbo.create(libs);
        try {
            devices = List.copyOf(runtime.devices());
            for (TurboProperties.ModelConfig c : configs) {
                load(c);
            }
        } catch (RuntimeException e) {
            close();
            throw e;
        }
    }

    private void load(TurboProperties.ModelConfig c) {
        if (c.getBundle().isBlank()) {
            throw new IllegalStateException("a turbo.models entry has no bundle; every entry needs bundle=<bundle dir>");
        }
        int index = c.getProvider().isBlank()
                ? runtime.selectDevice()
                : runtime.selectDevice(SelectPolicy.EXPLICIT, c.getProvider(), c.getOrdinal());
        DeviceInfo device = runtime.device(index);
        Context context = contexts.computeIfAbsent(index, runtime::createContext);
        Model model = context.loadModel(c.getBundle());
        ModelInfo mi;
        try {
            mi = model.info();
            if (mi.kind() == ModelKind.GENERIC) {
                throw new IllegalStateException(c.getBundle()
                        + " is a GENERIC (RUN) bundle; this server serves embedding, reranker, classifier, "
                        + "token classifier and generative bundles");
            }
        } catch (RuntimeException e) {
            model.close();
            throw e;
        }
        String name = c.getName().isBlank() ? deriveName(mi.modelId()) : c.getName();
        if (models.containsKey(name)) {
            model.close();
            throw new IllegalStateException("two models would be served as `" + name
                    + "`; give one of them turbo.models[i].name=<unique name>");
        }
        int maxBatch = c.getMaxBatch() > 0 ? c.getMaxBatch() : mi.maxBatch();
        LoadedModel loaded;
        try {
            loaded = new LoadedModel(name, c.getBundle(), c.getTokenizerBundle(), runtime, model, device, maxBatch,
                    c.getSessions(), c.getGenerations());
        } catch (RuntimeException e) {
            model.close();
            throw e;
        }
        models.put(name, loaded);
        defaults.putIfAbsent(mi.task(), loaded);
    }

    /** The name a bundle is served under when the configuration does not give one. */
    static String deriveName(String modelId) {
        int slash = modelId.lastIndexOf('/');
        String tail = slash >= 0 ? modelId.substring(slash + 1) : modelId;
        String cleaned = tail.replaceAll("[^A-Za-z0-9._-]", "-");
        return cleaned.isBlank() ? "model" : cleaned;
    }

    /** Every model, in load order. */
    public List<LoadedModel> loaded() {
        return List.copyOf(models.values());
    }

    /** The model served as {@code name}. */
    public LoadedModel model(String name) {
        LoadedModel m = models.get(name);
        if (m == null) {
            throw new ModelNotFound("no model named `" + name + "`; this server serves " + models.keySet());
        }
        return m;
    }

    /**
     * The model a request means: the one it names, or the first loaded model
     * whose bundle declares {@code task} when it names none.
     */
    public LoadedModel modelFor(String name, Task task) {
        if (name != null && !name.isBlank()) {
            LoadedModel m = model(name);
            if (m.info().task() != task) {
                throw new TaskNotServed("model `" + m.name() + "` performs " + m.info().task()
                        + ", not " + task + "; this server serves " + names(task) + " for " + task);
            }
            return m;
        }
        LoadedModel m = defaults.get(task);
        if (m == null) {
            throw new TaskNotServed("no loaded model performs " + task
                    + "; start the server with a " + task + " bundle (see GET /api/v1/models)");
        }
        return m;
    }

    /** True when at least one loaded model performs {@code task}. */
    public boolean serves(Task task) {
        return defaults.containsKey(task);
    }

    private List<String> names(Task task) {
        return models.values().stream().filter(m -> m.info().task() == task).map(LoadedModel::name).toList();
    }

    /** Liveness plus the shape of this server. */
    public HealthReport health() {
        return new HealthReport("ok", Turbo.abiVersion(), devices.size(), List.copyOf(models.keySet()));
    }

    /** Every device of every loaded provider, its feature bits and its capability matrix. */
    public List<DeviceReport> devices() {
        List<DeviceReport> out = new ArrayList<>(devices.size());
        for (DeviceInfo d : devices) {
            List<CapabilityCell> cells = new ArrayList<>();
            for (Task task : Task.values()) {
                for (Modality modality : Modality.values()) {
                    Capability c = runtime.capability(d.index(), task, modality);
                    cells.add(new CapabilityCell(task.name(), modality.name(), c.status().name(),
                            c.dtype() == null ? null : c.dtype().name(),
                            c.referenceDtype() == null ? null : c.referenceDtype().name(),
                            c.cosineFloor(), c.maxAbsError(), c.deterministic(), c.notes()));
                }
            }
            List<String> here = models.values().stream()
                    .filter(m -> m.device().index() == d.index())
                    .map(LoadedModel::name)
                    .toList();
            out.add(new DeviceReport(d.index(), d.name(), d.kind().name(), d.ordinal(), d.vendor(), d.vendorId(),
                    d.providerId(), d.providerVersion(), d.runtimeVersion(), d.driverVersion(), d.memoryTotal(),
                    d.memoryFree(), d.caps(), CapabilityBits.names(d.caps()), cells, here));
        }
        return out;
    }

    /** Every loaded model's contract. */
    public List<ModelReport> models() {
        return models.values().stream().map(TurboService::report).toList();
    }

    /** One loaded model's contract. */
    public static ModelReport report(LoadedModel m) {
        ModelInfo i = m.info();
        Map<String, String> stages = new LinkedHashMap<>();
        int[] placements = i.stagePlacement();
        for (int s = 0; s < STAGE_NAMES.length && s < placements.length; s++) {
            int p = placements[s];
            stages.put(STAGE_NAMES[s], p >= 0 && p < STAGE_PLACEMENTS.length ? STAGE_PLACEMENTS[p] : "unknown(" + p + ")");
        }
        return new ModelReport(m.name(), m.bundle(), i.task().name(), i.kind().name(), i.modality().name(), i.dim(),
                i.labels(), i.pooling() == null ? null : i.pooling().name(),
                i.normalize() == null ? null : i.normalize().name(), i.maxSeq(), m.maxBatch(),
                i.dtypeUsed() == null ? null : i.dtypeUsed().name(), i.fullyAccelerated(), stages, i.vocabSize(),
                i.modelId(), i.revision(), i.tokenizerSha256(), i.prefixQuery(), i.prefixDocument(),
                m.tokenizerBundle(), DeviceRef.of(m.device()));
    }

    private static void requireBatch(LoadedModel m, int n, String what) {
        if (n == 0) {
            throw new IllegalArgumentException(what + " is empty");
        }
        if (n > m.maxBatch()) {
            throw new IllegalArgumentException("request has " + n + " " + what + " but model `" + m.name()
                    + "` has a batch of " + m.maxBatch());
        }
    }

    /** Embed {@code texts} on {@code m} with {@code opts}. */
    public EmbedResult embed(LoadedModel m, List<String> texts, EmbedOptions opts) {
        requireBatch(m, texts.size(), "texts");
        return m.withSession(session -> {
            long t0 = System.nanoTime();
            session.writeText(texts, opts);
            long t1 = System.nanoTime();
            try (Result r = session.run()) {
                long t2 = System.nanoTime();
                long[] shape = r.output(0).shape();
                int dim = (int) shape[shape.length - 1];
                float[] flat = r.readFloats(0);
                String placement = r.placement().name();
                long t3 = System.nanoTime();
                float[][] vectors = new float[texts.size()][];
                for (int i = 0; i < texts.size(); i++) {
                    vectors[i] = new float[dim];
                    System.arraycopy(flat, i * dim, vectors[i], 0, dim);
                }
                return new EmbedResult(m.name(), DeviceRef.of(m.device()), dim, vectors, placement,
                        new Timings(ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t0, t3)));
            }
        });
    }

    /** Rerank {@code documents} against {@code query} on {@code m}. */
    public RerankResult rerank(LoadedModel m, String query, List<String> documents, RerankOptions opts) {
        if (query == null || query.isBlank()) {
            throw new IllegalArgumentException("query is empty");
        }
        requireBatch(m, documents.size(), "documents");
        if (opts.topN() > documents.size()) {
            throw new IllegalArgumentException(
                    "top_n is " + opts.topN() + " but the request carries " + documents.size() + " documents");
        }
        return m.withSession(session -> {
            long t0 = System.nanoTime();
            session.writePairs(query, documents, opts);
            long t1 = System.nanoTime();
            try (Result r = session.run()) {
                long t2 = System.nanoTime();
                float[] scores = r.readFloats(0);
                int[] sorted = r.outputCount() > 1 ? r.readInts(1) : null;
                String placement = r.placement().name();
                long t3 = System.nanoTime();
                return new RerankResult(m.name(), DeviceRef.of(m.device()), scores, sorted, placement,
                        new Timings(ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t0, t3)));
            }
        });
    }

    /** Classify {@code texts} on {@code m}, one score row per text. */
    public ClassifyResult classify(LoadedModel m, List<String> texts, ClassifyOptions opts) {
        requireBatch(m, texts.size(), "texts");
        List<String> labels = m.info().labels();
        return m.withSession(session -> {
            long t0 = System.nanoTime();
            session.writeTextClassify(texts, opts);
            long t1 = System.nanoTime();
            try (Result r = session.run()) {
                long t2 = System.nanoTime();
                long[] shape = r.output(0).shape();
                int width = (int) shape[shape.length - 1];
                float[] flat = r.readFloats(0);
                String placement = r.placement().name();
                long t3 = System.nanoTime();
                float[][] scores = new float[texts.size()][];
                for (int i = 0; i < texts.size(); i++) {
                    scores[i] = new float[width];
                    System.arraycopy(flat, i * width, scores[i], 0, width);
                }
                return new ClassifyResult(m.name(), DeviceRef.of(m.device()), labels, scores, placement,
                        new Timings(ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t0, t3)));
            }
        });
    }

    /** Token-classify {@code texts} on {@code m}, returning the aggregated spans. */
    public TokenClassifyResult tokenClassify(LoadedModel m, List<String> texts, ClassifyOptions opts) {
        requireBatch(m, texts.size(), "texts");
        List<String> labels = m.info().labels();
        return m.withSession(session -> {
            long t0 = System.nanoTime();
            session.writeTextClassify(texts, opts);
            long t1 = System.nanoTime();
            try (Result r = session.run()) {
                long t2 = System.nanoTime();
                long[] shape = r.output(0).shape();
                float[] scores = r.readFloats(0);
                List<Span> spans = r.spans();
                String placement = r.placement().name();
                long t3 = System.nanoTime();
                List<SpanResult> out = new ArrayList<>(spans.size());
                for (Span s : spans) {
                    String text = slice(texts.get(s.row()), s.byteStart(), s.byteEnd());
                    String label = s.label() >= 0 && s.label() < labels.size() ? labels.get(s.label()) : null;
                    out.add(new SpanResult(s.row(), s.byteStart(), s.byteEnd(), s.label(), label, s.score(), text));
                }
                return new TokenClassifyResult(m.name(), DeviceRef.of(m.device()), labels, out, shape, scores,
                        placement, new Timings(ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t0, t3)));
            }
        });
    }

    /** The substring of {@code text} between two byte offsets of its UTF-8 encoding. */
    static String slice(String text, long byteStart, long byteEnd) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int from = (int) Math.max(0, Math.min(byteStart, bytes.length));
        int to = (int) Math.max(from, Math.min(byteEnd, bytes.length));
        return new String(bytes, from, to - from, StandardCharsets.UTF_8);
    }

    /**
     * Encode {@code texts} with the tokenizer of {@code m}'s tokenizer bundle.
     * Rows are padded to the tokenizer's {@code max_seq} and each row's live
     * token count is reported separately.
     */
    public TokenizeResult tokenize(LoadedModel m, List<String> texts, boolean addSpecialTokens) {
        if (texts.isEmpty()) {
            throw new IllegalArgumentException("texts is empty");
        }
        long t0 = System.nanoTime();
        Tokenizer tokenizer = m.tokenizer();
        TokenizerInfo info = tokenizer.info();
        int stride = info.maxSeq();
        if (stride <= 0) {
            throw new IllegalStateException(
                    "the tokenizer of bundle " + m.tokenizerBundle() + " reports max_seq " + stride);
        }
        EncodeOptions opts = EncodeOptions.defaults().withSpecialTokens(addSpecialTokens);
        long t1 = System.nanoTime();
        Encoding e = tokenizer.encode(texts, stride, opts);
        long t2 = System.nanoTime();
        List<TokenizeRow> rows = new ArrayList<>(texts.size());
        for (int i = 0; i < texts.size(); i++) {
            int count = e.lengths()[i];
            int[] ids = new int[count];
            int[] mask = new int[count];
            System.arraycopy(e.ids(), i * e.rowStride(), ids, 0, count);
            System.arraycopy(e.mask(), i * e.rowStride(), mask, 0, count);
            List<String> tokens = new ArrayList<>(count);
            for (int id : ids) {
                tokens.add(tokenizer.decode(new int[] {id}, false));
            }
            rows.add(new TokenizeRow(ids, mask, tokens, count));
        }
        long t3 = System.nanoTime();
        return new TokenizeResult(m.name(), m.tokenizerBundle(), info, rows,
                new Timings(ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t0, t3)));
    }

    /** Decode id rows with the tokenizer of {@code m}'s tokenizer bundle. */
    public DetokenizeResult detokenize(LoadedModel m, List<int[]> rows, boolean skipSpecialTokens) {
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("ids is empty");
        }
        long t0 = System.nanoTime();
        Tokenizer tokenizer = m.tokenizer();
        long t1 = System.nanoTime();
        List<String> texts = new ArrayList<>(rows.size());
        for (int[] ids : rows) {
            texts.add(tokenizer.decode(ids, skipSpecialTokens));
        }
        long t2 = System.nanoTime();
        return new DetokenizeResult(m.name(), m.tokenizerBundle(), texts,
                new Timings(ms(t0, t1), ms(t1, t2), 0, ms(t0, t2)));
    }

    /**
     * Generate on {@code m}, handing every chunk to {@code sink}; a sink that
     * returns false cancels the generation and the final chunk reports
     * {@code CANCELLED}. The permit is released whatever happens.
     */
    public GenerateResult generate(LoadedModel m, List<Message> messages, GenerateDesc desc, Predicate<Chunk> sink) {
        if (messages.isEmpty()) {
            throw new IllegalArgumentException("messages is empty");
        }
        m.acquireGeneration();
        try (Generation g = m.handle().createGeneration(desc)) {
            long t0 = System.nanoTime();
            g.prompt(messages);
            StringBuilder text = new StringBuilder();
            List<Float> logprobs = new ArrayList<>();
            long[] first = {0L};
            Chunk last = g.drain(chunk -> {
                if (first[0] == 0L && chunk.tokens().length > 0) {
                    first[0] = System.nanoTime();
                }
                text.append(chunk.text());
                for (float p : chunk.logprobs()) {
                    logprobs.add(p);
                }
                return sink.test(chunk);
            });
            long t1 = System.nanoTime();
            double total = ms(t0, t1);
            double firstMs = first[0] == 0L ? total : ms(t0, first[0]);
            double rate = total > 0 ? last.generatedTokens() * 1000.0 / total : 0;
            return new GenerateResult(m.name(), DeviceRef.of(m.device()), text.toString(), last.finishReason().name(),
                    last.promptTokens(), last.generatedTokens(), List.copyOf(logprobs),
                    new GenTimings(firstMs, total, rate));
        } finally {
            m.releaseGeneration();
        }
    }

    /** Cosine similarity of every pair of rows. */
    public static double[][] similarity(float[][] vectors) {
        double[][] sim = new double[vectors.length][vectors.length];
        for (int a = 0; a < vectors.length; a++) {
            for (int b = 0; b < vectors.length; b++) {
                sim[a][b] = cosine(vectors[a], vectors[b]);
            }
        }
        return sim;
    }

    static double cosine(float[] a, float[] b) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
        }
        double denominator = Math.sqrt(na) * Math.sqrt(nb);
        if (denominator == 0) {
            throw new IllegalStateException("a zero vector has no cosine similarity");
        }
        return dot / denominator;
    }

    private static double ms(long from, long to) {
        return (to - from) / 1_000_000.0;
    }

    @PreDestroy
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        models.values().forEach(LoadedModel::close);
        models.clear();
        contexts.values().forEach(Context::close);
        contexts.clear();
        runtime.close();
    }
}
