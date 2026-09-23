// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.swagger.v3.oas.annotations.media.Schema;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.springframework.stereotype.Service;

/**
 * The committed benchmark receipts, read from {@code turbo.receipts}.
 *
 * <p>Three kinds live in that directory and all three are served: a {@code
 * benchmark} receipt is one workload measured through libturbo, a {@code
 * native} receipt is the same workload driven against the runtime alone by
 * the reference program of that pair, and a {@code compare} receipt is the
 * matched pair with a ratio per cell and the verdict that follows from them.
 *
 * <p>Nothing here computes a figure. Every number is copied out of a receipt,
 * and a figure a receipt does not carry stays null rather than being filled in
 * with a default. A directory that is not there, or that holds no receipt, is
 * a refusal naming the path, never an empty answer.
 */
@Service
public class BenchmarkReceipts {
    /** The receipts directory does not exist, is not a directory, or holds no receipt. */
    public static class ReceiptsNotFound extends RuntimeException {
        private static final long serialVersionUID = 1L;

        public ReceiptsNotFound(String message) {
            super(message);
        }
    }

    private final ObjectMapper json;
    private final TurboProperties props;

    public BenchmarkReceipts(ObjectMapper json, TurboProperties props) {
        this.json = json;
        this.props = props;
    }

    /** The configured directory, resolved against the working directory. */
    public Path directory() {
        return Path.of(props.getReceipts()).toAbsolutePath().normalize();
    }

    /** Every receipt in the directory, grouped by kind. */
    public BenchmarkReport report() {
        Path dir = directory();
        if (!Files.isDirectory(dir)) {
            throw new ReceiptsNotFound("no benchmark receipts: " + dir
                    + " is not a directory; point turbo.receipts at the tree's testdata/receipts/turbo/bench");
        }
        List<Path> files;
        try (Stream<Path> listing = Files.list(dir)) {
            files = listing.filter(p -> Files.isRegularFile(p) && p.getFileName().toString().endsWith(".json"))
                    .sorted()
                    .toList();
        } catch (IOException e) {
            throw new IllegalStateException("cannot list the benchmark receipts in " + dir + ": " + e.getMessage(), e);
        }
        if (files.isEmpty()) {
            throw new ReceiptsNotFound("no benchmark receipts: " + dir + " holds no .json file");
        }

        List<Comparison> comparisons = new ArrayList<>();
        List<Run> turbo = new ArrayList<>();
        List<Run> natives = new ArrayList<>();
        for (Path file : files) {
            String name = file.getFileName().toString();
            JsonNode node = read(file);
            String kind = node.path("kind").asText("");
            switch (kind) {
                case "compare" -> comparisons.add(comparison(name, convert(file, node, RawComparison.class)));
                case "benchmark" -> turbo.add(run(name, convert(file, node, RawReceipt.class)));
                case "native" -> natives.add(run(name, convert(file, node, RawReceipt.class)));
                default -> throw new IllegalStateException(file
                        + ": kind is `" + kind + "`, not one of benchmark, native or compare");
            }
        }
        comparisons.sort(Comparator.comparing(Comparison::device).thenComparing(Comparison::task)
                .thenComparing(Comparison::file));
        Comparator<Run> byRun = Comparator.comparing(Run::device).thenComparing(Run::task)
                .thenComparing(Comparator.comparing(Run::date).reversed()).thenComparing(Run::file);
        turbo.sort(byRun);
        natives.sort(byRun);
        return new BenchmarkReport(dir.toString(), files.size(), comparisons, turbo, natives);
    }

    private JsonNode read(Path file) {
        try {
            return json.readTree(Files.readString(file));
        } catch (IOException e) {
            throw new IllegalStateException(file + " is not a readable benchmark receipt: " + e.getMessage(), e);
        }
    }

    private <T> T convert(Path file, JsonNode node, Class<T> type) {
        try {
            return json.treeToValue(node, type);
        } catch (IOException e) {
            throw new IllegalStateException(file + " does not match the receipt schema: " + e.getMessage(), e);
        }
    }

    /** The task a cell name or a receipt's contents names: {@code embed}, {@code rerank} or {@code generate}. */
    private static String taskOf(RawComparison c) {
        if (c.cells() == null || c.cells().isEmpty()) {
            return "";
        }
        String cell = c.cells().get(0).cell();
        int space = cell.indexOf(' ');
        return space < 0 ? cell : cell.substring(0, space);
    }

    private static String taskOf(RawReceipt r) {
        List<String> tasks = new ArrayList<>();
        if (r.embed() != null && !r.embed().isEmpty()) {
            tasks.add("embed");
        }
        if (r.rerank() != null) {
            tasks.add("rerank");
        }
        if (r.generate() != null) {
            tasks.add("generate");
        }
        return String.join(", ", tasks);
    }

    private static Comparison comparison(String file, RawComparison c) {
        List<Cell> cells = new ArrayList<>();
        for (RawCompareCell cell : c.cells()) {
            cells.add(new Cell(cell.cell(), cell.measure(), cell.turbo(), cell.nativeValue(), cell.ratio(),
                    cell.withinFloor()));
        }
        return new Comparison(file, c.verdict(), c.floor(), taskOf(c),
                c.turbo().device().name(), c.turbo().device().kind(), c.turbo().provider().id(),
                c.turbo().provider().version(), c.turbo().provider().runtimeVersion(),
                blankToNull(c.turbo().provider().driverVersion()),
                c.nativeSide().provider().id(), c.nativeSide().provider().runtimeVersion(),
                blankToNull(c.nativeSide().provider().driverVersion()),
                c.bundle().modelId(), c.bundle().manifestSha256(),
                c.turbo().date(), c.turbo().commit(), c.nativeSide().date(), c.nativeSide().commit(),
                cells, c.unmatched() == null ? List.of() : c.unmatched());
    }

    private static Run run(String file, RawReceipt r) {
        List<EmbedCell> embed = new ArrayList<>();
        if (r.embed() != null) {
            for (RawEmbedCell cell : r.embed()) {
                embed.add(new EmbedCell(cell.batch(), cell.seq(), cell.liveTokensPerRow(), cell.tokenCountSource(),
                        latency(cell.textPath()), latency(cell.preparedTokensPath()),
                        blankToNull(cell.preparedTokensNote())));
            }
        }
        RerankCell rerank = r.rerank() == null ? null
                : new RerankCell(r.rerank().docs(), r.rerank().seq(), latency(r.rerank().textPath()));
        RawGenerateCell g = r.generate();
        GenerateCell generate = g == null ? null
                : new GenerateCell(g.newTokensRequested(), g.generatedTokensMean(), g.promptTokens(),
                        g.timeToFirstTokenMsP50(), g.decodeTokensPerSP50(), g.totalMsP50(),
                        g.finishReasons() == null ? List.of() : g.finishReasons(), g.iters());
        return new Run(file, r.kind(), taskOf(r), r.date(), r.commit(),
                r.machine() == null ? null : r.machine().hostname(),
                r.provider().id(), r.provider().version(), r.provider().runtimeVersion(),
                blankToNull(r.provider().driverVersion()),
                r.device().name(), r.device().kind(), r.device().ordinal(),
                r.bundle().modelId(), r.bundle().manifestSha256(), embed, rerank, generate);
    }

    private static Latency latency(RawLatency l) {
        return l == null ? null
                : new Latency(l.p50Ms(), l.p99Ms(), l.meanMs(), l.rowsPerS(), l.tokensPerS(), l.iters());
    }

    /** An empty string in a receipt means "not recorded", which the API says as null. */
    private static String blankToNull(String value) {
        return value == null || value.isBlank() ? null : value;
    }

    // -----------------------------------------------------------------------
    // What the endpoint answers with
    // -----------------------------------------------------------------------

    /** Every receipt in the directory, by kind. */
    @Schema(name = "BenchmarkReport",
            description = "The committed benchmark receipts: the matched comparisons, the libturbo runs and the "
                    + "direct-native runs they were matched against")
    public record BenchmarkReport(
            @Schema(description = "The directory the receipts were read from",
                    example = "/src/turbo/testdata/receipts/turbo/bench")
            String directory,
            @Schema(description = "Receipt files read", example = "31")
            int receiptCount,
            @Schema(description = "One entry per compare receipt")
            List<Comparison> comparisons,
            @Schema(description = "One entry per libturbo (kind benchmark) receipt")
            List<Run> turbo,
            @JsonProperty("native")
            @Schema(description = "One entry per direct-native (kind native) receipt")
            List<Run> nativeRuns) {}

    /** One compare receipt: libturbo against the runtime alone on the same bundle and device. */
    @Schema(name = "BenchmarkComparison", description = "libturbo against the same runtime driven directly")
    public record Comparison(
            @Schema(description = "The receipt file this entry was read from",
                    example = "compare-cuda-rtx4080-embed-2026-09-22.json")
            String file,
            @Schema(description = "SUPPORTED when every cell is at the floor or better and nothing is unmatched, "
                    + "else EXPERIMENTAL", example = "SUPPORTED")
            String verdict,
            @Schema(description = "The fraction of native every cell must reach for SUPPORTED", example = "0.95")
            double floor,
            @Schema(description = "embed, rerank or generate, from the cell names", example = "embed")
            String task,
            @Schema(description = "The device both sides ran on, as libturbo names it",
                    example = "NVIDIA GeForce RTX 4080 SUPER (sm_89)")
            String device,
            @Schema(description = "Device kind", example = "Gpu")
            String deviceKind,
            @Schema(description = "The libturbo provider", example = "cuda")
            String provider,
            @Schema(description = "The provider's own version", example = "2.0.0-alpha.0")
            String providerVersion,
            @Schema(description = "The runtime under that provider")
            String runtime,
            @Schema(description = "Driver version, when the provider reported one", example = "13.2")
            String driver,
            @Schema(description = "The runtime driven directly, as the reference program names it",
                    example = "onnxruntime-cuda")
            String nativeProvider,
            @Schema(description = "The native side's runtime version")
            String nativeRuntime,
            @Schema(description = "The native side's driver version, when it reported one", example = "595.84")
            String nativeDriver,
            @Schema(description = "The bundle both sides ran", example = "sentence-transformers/all-MiniLM-L6-v2")
            String modelId,
            @Schema(description = "SHA-256 of the bundle manifest both sides measured")
            String bundleSha256,
            @Schema(description = "UTC date of the libturbo run", example = "2026-09-22")
            String date,
            @Schema(description = "Commit the libturbo side was built from")
            String commit,
            @Schema(description = "UTC date of the native run", example = "2026-09-22")
            String nativeDate,
            @Schema(description = "Commit the native side was built from")
            String nativeCommit,
            @Schema(description = "Every matched cell")
            List<Cell> cells,
            @Schema(description = "Cells one side has and the other does not; any of these makes the verdict "
                    + "EXPERIMENTAL")
            List<String> unmatched) {}

    /** One matched cell of a comparison. */
    @Schema(name = "BenchmarkCell", description = "One workload shape measured on both sides")
    public record Cell(
            @Schema(description = "The workload shape", example = "embed 8x128")
            String cell,
            @Schema(description = "What was compared", example = "prepared tokens p50 ms")
            String measure,
            @Schema(description = "The libturbo figure in the unit the measure names", example = "1.452284")
            double turbo,
            @JsonProperty("native")
            @Schema(description = "The same figure from the runtime alone", example = "3.08509")
            double nativeValue,
            @Schema(description = "libturbo throughput as a fraction of native; above 1.0 is faster than the raw "
                    + "runtime", example = "2.1243021337424364")
            double ratio,
            @Schema(description = "Whether the ratio reaches the floor", example = "true")
            boolean withinFloor) {}

    /** One benchmark or native receipt: one workload on one device. */
    @Schema(name = "BenchmarkRun", description = "One measured run, through libturbo or against the runtime alone")
    public record Run(
            @Schema(description = "The receipt file this entry was read from",
                    example = "cuda-rtx4080-embed-2026-09-22.json")
            String file,
            @Schema(description = "benchmark (through libturbo) or native (the runtime alone)", example = "benchmark")
            String kind,
            @Schema(description = "What the receipt measured", example = "embed")
            String task,
            @Schema(description = "UTC date of the run", example = "2026-09-22")
            String date,
            @Schema(description = "Commit the tool was built from")
            String commit,
            @Schema(description = "The machine, by uname -n, or by TURBO_BENCH_MACHINE when the run set it", example = "rtx4080")
            String hostname,
            @Schema(description = "Provider id, or the runtime's name on a native receipt", example = "cuda")
            String provider,
            @Schema(description = "Provider version", example = "2.0.0-alpha.0")
            String providerVersion,
            @Schema(description = "Runtime version")
            String runtime,
            @Schema(description = "Driver version, when one was reported", example = "13.2")
            String driver,
            @Schema(description = "Device name", example = "NVIDIA GeForce RTX 4080 SUPER (sm_89)")
            String device,
            @Schema(description = "Device kind", example = "Gpu")
            String deviceKind,
            @Schema(description = "Device ordinal within the provider", example = "0")
            int ordinal,
            @Schema(description = "The bundle that was measured", example = "sentence-transformers/all-MiniLM-L6-v2")
            String modelId,
            @Schema(description = "SHA-256 of the bundle manifest")
            String bundleSha256,
            @Schema(description = "One entry per batch by sequence-length cell")
            List<EmbedCell> embed,
            @Schema(description = "The rerank cell, when the receipt carries one")
            RerankCell rerank,
            @Schema(description = "The generation cell, when the receipt carries one")
            GenerateCell generate) {}

    /** One embedding cell of a receipt. */
    @Schema(name = "BenchmarkEmbedCell", description = "One batch by sequence-length embedding shape")
    public record EmbedCell(
            @Schema(description = "Rows per run", example = "32")
            int batch,
            @Schema(description = "Session sequence length", example = "128")
            int seq,
            @Schema(description = "Live tokens per row", example = "21.0")
            double liveTokensPerRow,
            @Schema(description = "tokenizer, word-estimate or unrecorded", example = "tokenizer")
            String tokenCountSource,
            @Schema(description = "Text in, vectors out: tokenization included")
            Latency text,
            @Schema(description = "Prepared token ids in, vectors out; absent when the bundle carries no tokenizer "
                    + "the core can encode with")
            Latency preparedTokens,
            @Schema(description = "Why the prepared-tokens path is absent")
            String preparedTokensNote) {}

    /** The rerank cell of a receipt. */
    @Schema(name = "BenchmarkRerankCell", description = "One query against a fixed number of documents")
    public record RerankCell(
            @Schema(description = "Documents per query", example = "16")
            int docs,
            @Schema(description = "Session sequence length", example = "128")
            int seq,
            @Schema(description = "Query and documents in, scores out")
            Latency text) {}

    /** The generation cell of a receipt. */
    @Schema(name = "BenchmarkGenerateCell", description = "A fixed number of new tokens from a fixed prompt")
    public record GenerateCell(
            @Schema(description = "Tokens asked for", example = "128")
            int newTokensRequested,
            @Schema(description = "Tokens produced per iteration, on average", example = "128.0")
            double generatedTokensMean,
            @Schema(description = "Prompt tokens after the chat template", example = "19")
            int promptTokens,
            @Schema(description = "Prompt submission to the first token, median milliseconds", example = "5.111717")
            Double timeToFirstTokenMsP50,
            @JsonProperty("decode_tokens_per_s_p50")
            @Schema(description = "Tokens after the first over the time after it; absent when no iteration produced "
                    + "a second chunk", example = "626.0293006162336")
            Double decodeTokensPerSP50,
            @Schema(description = "Prompt submission to the final token, median milliseconds", example = "208.761781")
            Double totalMsP50,
            @Schema(description = "Why each timed iteration ended")
            List<String> finishReasons,
            @Schema(description = "Timed iterations", example = "10")
            int iters) {}

    /** The latency figures of one cell. */
    @Schema(name = "BenchmarkLatency", description = "Latency and throughput over a cell's timed iterations")
    public record Latency(
            @Schema(description = "Nearest-rank median, milliseconds", example = "1.452284")
            Double p50Ms,
            @Schema(description = "Nearest-rank p99; absent below 100 samples", example = "1.9988380000000001")
            Double p99Ms,
            @Schema(description = "Arithmetic mean, milliseconds", example = "1.4680001")
            Double meanMs,
            @Schema(description = "Rows per second at the mean latency", example = "5449.0")
            Double rowsPerS,
            @Schema(description = "Tokens per second at the mean latency; absent when no tokenizer counted them",
                    example = "114429.0")
            Double tokensPerS,
            @Schema(description = "Timed iterations the figures summarize", example = "30")
            Integer iters) {}

    // -----------------------------------------------------------------------
    // The receipt files as they are on disk (crates/turbo-bench/src/receipt.rs)
    // -----------------------------------------------------------------------

    record RawLatency(Double p50Ms, Double p99Ms, Double meanMs, Double minMs, Double maxMs, Double rowsPerS,
            Double tokensPerS, Integer iters) {}

    record RawEmbedCell(int batch, int seq, double liveTokensPerRow, String tokenCountSource, RawLatency textPath,
            RawLatency preparedTokensPath, String preparedTokensNote) {}

    record RawRerankCell(int docs, int seq, RawLatency textPath) {}

    // Two capitals in a row ("...PerSP50") would collapse into one snake_case
    // word, so that field names the receipt's key itself.
    record RawGenerateCell(int newTokensRequested, double generatedTokensMean, int promptTokens,
            Double timeToFirstTokenMsP50, @JsonProperty("decode_tokens_per_s_p50") Double decodeTokensPerSP50,
            Double totalMsP50, List<String> finishReasons, int iters) {}

    record RawProvider(String id, String version, String runtimeVersion, String driverVersion) {}

    record RawDevice(String name, String kind, int ordinal, String caps, long memoryTotal) {}

    record RawBundle(String dir, String modelId, String manifestSha256, Map<String, String> artifacts) {}

    record RawMachine(String hostname, String os, String arch) {}

    record RawReceipt(int receiptVersion, String kind, String date, RawMachine machine, String commit,
            RawProvider provider, RawDevice device, RawBundle bundle, List<RawEmbedCell> embed, RawRerankCell rerank,
            RawGenerateCell generate, String nativeReference) {}

    record RawSide(String file, String commit, RawProvider provider, RawDevice device, String date) {}

    record RawCompareCell(String cell, String measure, double turbo, @JsonProperty("native") double nativeValue,
            double ratio, boolean withinFloor) {}

    record RawComparison(int receiptVersion, String kind, RawSide turbo, @JsonProperty("native") RawSide nativeSide,
            RawBundle bundle, double floor, List<RawCompareCell> cells, List<String> unmatched, String verdict) {}
}
