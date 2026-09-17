// Apple matched native-overhead pilot (Machine C): direct mlx-swift Metal
// vs the `libTurboEmbed.dylib` C ABI, following the methodology of
// `docs/library-design.md#native-performance-acceptance` and the NVIDIA
// pilot (`crates/bench-turbo/src/bin/bench-nvidia-overhead.rs`).
//
// Both paths run the identical MiniLM MLX checkpoint (models/mlx/minilm),
// f32 precision, mean+L2 postprocessing, and a fixed `[batch, 256]`
// execution shape. The direct reference is a raw mlx-swift consumer of the
// same `MLXEmbedders` model container the Swift server uses: HF tokenizer,
// fixed-length padding, BERT forward, mean+L2 pooling with MLX ops on
// device, then one host read of `[batch, dim]`. The ABI path is a plain C
// consumer of `turboembed_*` symbols resolved with dlopen/dlsym from the
// actual `libTurboEmbed.dylib` — the same artifact the Apple SDK packages —
// which tokenizes with native WordPiece into a turbo_buffer Metal SHARED
// arena and pools on host from unified memory.
//
// Process isolation: the ABI leg runs in a separate `bench-abi-worker`
// process (one per case) that links no MLX and no swift-transformers, so
// the dylib's embedded copies of those Objective-C classes are the only
// ones in that process. Running both legs in one process duplicated the
// Tokenizers/MLX classes and doubled the Metal resource footprint, which
// distorted ABI timings and hit the Metal resource limit on the full
// 18-case grid. Only one leg executes at a time (the orchestrator blocks
// on the worker), so the legs never contend for the GPU; the direct-path
// MLX buffer cache is cleared between cases.
//
// Gates per repeat (predeclared, same as the NVIDIA/Intel pilots): ABI p50
// within 5% of the direct baseline and ABI throughput at least 95% of it.
// Parity per case: max abs error <= 5e-4 and RMSE <= 1e-4 between the two
// paths. p99 is descriptive only below 1000 samples. The ABI steady-state
// turbo_buffer allocation counter must be zero after warmup.
//
// Run through `make bench-apple-overhead` on the Machine C Metal host; the
// receipt is `testdata/receipts/bench/machine-c-metal-overhead.json`.

import ArgumentParser
import BenchOverheadCore
import Darwin
import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import Metal
import MlxEngine

let kParityMaxAbs: Float = 5e-4
let kParityMaxRmse = 1e-4
let kP50OverheadLimit = 1.05
let kThroughputFloor = 0.95

// MARK: - ABI worker client (one isolated process per case)

/// Client for one `bench-abi-worker` process: newline-delimited commands on
/// its stdin, one JSON object per line on its stdout, carried over the raw
/// POSIX wire layer in BenchOverheadCore (WorkerWire.swift) with a hard
/// deadline on every reply. The worker warms up before replying
/// `{"ready":true}`; its stderr passes through. A worker that misses a
/// deadline is killed and the run fails loudly instead of hanging (the
/// 2026-09-17 Machine C run wedged for hours on the ready handshake).
final class AbiWorker {
    private let client: WorkerClient
    /// Grace added on top of a command's own expected duration.
    private let replyGraceSeconds: Double

    init(
        executable: String, dylib: String, benchCase: BenchCase, warmup: Int,
        readySeconds: Double, replyGraceSeconds: Double
    ) throws {
        self.replyGraceSeconds = replyGraceSeconds
        var args = [
            "--dylib", dylib,
            "--batch", String(benchCase.batch),
            "--tokens", String(benchCase.targetTokens),
            "--warmup", String(warmup),
        ]
        if benchCase.mixed { args.append("--mixed") }
        FileHandle.standardError.write(
            Data("spawning abi worker for \(benchCase.name) (ready deadline \(Int(readySeconds))s)\n".utf8))
        self.client = try WorkerClient(
            executable: executable, arguments: args,
            label: "abi worker \(benchCase.name)", readySeconds: readySeconds)
    }

    /// Parity vectors for the case inputs (`batch * dim` floats).
    func collect() throws -> [Float] {
        let reply = try client.request("collect", timeoutSeconds: replyGraceSeconds)
        guard let values = reply["values"] as? [Any] else {
            throw fail("abi worker collect reply missing values: \(reply)")
        }
        return try values.map { any -> Float in
            guard let number = any as? NSNumber else {
                throw fail("abi worker collect reply has a non-numeric value")
            }
            return Float(number.doubleValue)
        }
    }

    /// Arena allocation count on one post-warmup request.
    func steadyAllocs() throws -> UInt64 {
        let reply = try client.request("steady", timeoutSeconds: replyGraceSeconds)
        guard let allocs = reply["allocs"] as? NSNumber else {
            throw fail("abi worker steady reply missing allocs: \(reply)")
        }
        return allocs.uint64Value
    }

    /// One timed repeat on the worker's engine; returns the stats JSON.
    func time(maxSeconds: Double, maxRequests: Int) throws -> [String: Any] {
        let reply = try client.request(
            "time \(maxSeconds) \(maxRequests)",
            timeoutSeconds: maxSeconds + replyGraceSeconds)
        guard let stats = reply["stats"] as? [String: Any] else {
            throw fail("abi worker time reply missing stats: \(reply)")
        }
        return stats
    }

    /// Two-engine concurrent observation inside the worker process. The
    /// worker creates and warms the second engine before timing, so the
    /// deadline includes the ready budget again.
    func concurrent(maxSeconds: Double, maxRequests: Int, readySeconds: Double) throws
        -> [String: Any]
    {
        try client.request(
            "concurrent \(maxSeconds) \(maxRequests)",
            timeoutSeconds: maxSeconds + readySeconds + replyGraceSeconds)
    }

    /// Ask the worker to destroy its engine and exit, then reap it.
    func shutdown() {
        client.shutdown()
    }
}

/// Run the worker binary's built-in wire-protocol self-test (no dylib,
/// engine, model, or Metal) before touching any of them. Milliseconds of
/// cost; catches a broken pipe protocol before hours of GPU work.
func preflightWorkerIO(workerPath: String) throws {
    let process = Process()
    process.executableURL = URL(filePath: workerPath)
    process.arguments = ["--io-selftest-parent"]
    process.standardError = FileHandle.standardError
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw fail(
            "bench-abi-worker --io-selftest-parent failed (exit \(process.terminationStatus)) — "
                + "the worker wire protocol is broken; not starting GPU work")
    }
}

func statNumber(_ stats: [String: Any], _ key: String) throws -> Double {
    guard let number = stats[key] as? NSNumber else {
        throw fail("abi worker stats missing \(key): \(stats)")
    }
    return number.doubleValue
}

// MARK: - Direct mlx-swift Metal reference (no turboembed API on this path)

/// Direct MLX Metal reference: a raw `MLXEmbedders` consumer with the HF
/// tokenizer, fixed `[batch, 256]` padding, BERT forward, and mean+L2
/// pooling with MLX ops on device. One host read of `[batch, dim]` per
/// request (unified memory; there is no discrete-GPU D2H on this path).
final class DirectMlx {
    private let container: EmbedderModelContainer

    init(modelDir: URL) async throws {
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw fail(
                "MLX model directory not found at \(modelDir.path) — run `make fetch-mlx ALIASES=minilm`"
            )
        }
        self.container = try await EmbedderModelFactory.shared.loadContainer(
            from: modelDir, using: HFTokenizerLoader())
    }

    /// Token count per row (specials included, no padding) for grid checks.
    func rowTokens(_ texts: [String]) async throws -> [Int] {
        try await container.perform { context -> [Int] in
            texts.map { context.tokenizer.encode(text: $0, addSpecialTokens: true).count }
        }
    }

    /// One synchronous text-to-pooled-result request. Returns row-major
    /// `[batch * dim]` host floats.
    func embed(_ texts: [String]) async throws -> [Float] {
        try await container.perform { context -> [Float] in
            let padId =
                context.tokenizer.convertTokenToId("[PAD]")
                ?? context.tokenizer.convertTokenToId("<pad>")
                ?? 0
            let sepId =
                context.tokenizer.convertTokenToId("[SEP]")
                ?? context.tokenizer.convertTokenToId("</s>")
            let encoded = texts.map { text -> [Int] in
                var ids = context.tokenizer.encode(text: text, addSpecialTokens: true)
                if ids.count > kMaxSeq {
                    ids = Array(ids.prefix(kMaxSeq))
                    if let sepId { ids[kMaxSeq - 1] = sepId }
                }
                return ids
            }
            // Fixed [batch, kMaxSeq] execution shape, matching the ABI
            // engine's warmed arena stride (and the NVIDIA pilot).
            let padded = stacked(
                encoded.map { ids in
                    MLXArray(ids + Array(repeating: padId, count: kMaxSeq - ids.count))
                })
            let tokenMask = padded .!= padId
            let tokenTypes = MLXArray.zeros(like: padded)
            let output = context.model(
                padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: tokenMask)
            guard let hidden = output.hiddenStates else {
                throw fail("direct BERT forward returned no hidden states")
            }
            let weights = tokenMask.asType(hidden.dtype)
            let weighted = hidden * weights.expandedDimensions(axes: [-1])
            var pooled = sum(weighted, axis: 1) / sum(weights, axis: -1, keepDims: true)
            let norm = sqrt(sum(pooled * pooled, axis: -1, keepDims: true))
            pooled = pooled / maximum(norm, MLXArray(Float(1e-12)))
            eval(pooled)
            let flat = pooled.asArray(Float.self)
            guard flat.count == texts.count * kDim else {
                throw fail("direct pooled size \(flat.count) != \(texts.count * kDim)")
            }
            return flat
        }
    }
}

// MARK: - Host metadata

func workspaceRoot() -> URL {
    if let env = ProcessInfo.processInfo.environment["INFERSTREAM_ROOT"], !env.isEmpty {
        return URL(filePath: env)
    }
    var dir = URL(filePath: FileManager.default.currentDirectoryPath)
    for _ in 0..<8 {
        if FileManager.default.fileExists(atPath: dir.appending(path: "include/turboembed.h").path) {
            return dir
        }
        let parent = dir.deletingLastPathComponent()
        if parent.path == dir.path { break }
        dir = parent
    }
    return URL(filePath: FileManager.default.currentDirectoryPath)
}

func commandOutput(_ launchPath: String, _ args: [String], cwd: URL? = nil) -> String {
    let process = Process()
    process.executableURL = URL(filePath: launchPath)
    process.arguments = args
    if let cwd { process.currentDirectoryURL = cwd }
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = Pipe()
    do {
        try process.run()
        process.waitUntilExit()
    } catch {
        return ""
    }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    return String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
}

// MARK: - Entry point

@main
struct BenchAppleOverhead: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench-apple-overhead",
        abstract: "Matched direct-MLX-Metal vs libTurboEmbed.dylib ABI MiniLM overhead pilot"
    )

    @Option(help: "Untimed per-path executions before each case's parity and timing.")
    var warmup: Int = 20

    @Option(help: "Timed repeats per case per path (order alternates).")
    var repeats: Int = 3

    @Option(help: "Per-repeat wall-clock cap in seconds.")
    var maxSeconds: Double = 10

    @Option(help: "Per-repeat completed-request cap.")
    var maxRequests: Int = 10_000

    @Flag(help: "Reduced case grid for smoke runs (never a receipt).")
    var quick = false

    @Flag(help: "Skip the two-engine concurrent observation.")
    var skipConcurrent = false

    @Option(help: "Path to libTurboEmbed.dylib (default: next to this binary, then swift/.build/release).")
    var dylib: String?

    @Option(help: "Path to bench-abi-worker (default: next to this binary, then swift/.build/release).")
    var worker: String?

    @Option(
        help:
            "Fail-loud deadline (seconds) for a worker to load the model, warm up, and reply ready.")
    var workerReadySeconds: Double = 600

    @Option(
        help:
            "Fail-loud grace (seconds) on top of a worker command's own expected duration.")
    var workerReplyGraceSeconds: Double = 120

    @Option(help: "Receipt output path.")
    var out: String

    mutating func run() async throws {
        let root = workspaceRoot()
        setenv("INFERSTREAM_ROOT", root.path, 0)

        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            throw fail("no Metal device on this host — refusing to bench (fail loud, no CPU fallback)")
        }
        let gpu = metalDevice.name

        let dylibPath = try resolve(
            overridePath: dylib, name: "libTurboEmbed.dylib", root: root,
            buildHint: "swift build -c release --package-path swift --product TurboEmbed")
        let workerPath = try resolve(
            overridePath: worker, name: "bench-abi-worker", root: root,
            buildHint: "swift build -c release --package-path swift --product bench-abi-worker")
        FileHandle.standardError.write(Data("ABI dylib: \(dylibPath)\nABI worker: \(workerPath)\n".utf8))

        // Prove the worker wire protocol under pipes before loading any
        // model or touching the GPU (fail fast, not after an hour).
        try preflightWorkerIO(workerPath: workerPath)

        let modelDir = root.appending(path: "models/mlx/\(kAlias)")
        FileHandle.standardError.write(Data("direct MLX load: \(modelDir.path)\n".utf8))
        let direct = try await DirectMlx(modelDir: modelDir)

        let cases = caseGrid(quick: quick)
        var caseReports: [[String: Any]] = []
        var allPass = true

        for (caseIndex, c) in cases.enumerated() {
            let texts = caseTexts(c)
            let rowTokens = try await direct.rowTokens(texts)
            if !c.mixed && rowTokens.contains(where: { $0 != c.targetTokens }) {
                throw fail("\(c.name): constructed rows are \(rowTokens), expected \(c.targetTokens)")
            }

            // One isolated ABI worker per case; it warms itself up before
            // reporting ready. Warm the direct path in this process, then
            // check parity before any timing.
            let abiWorker = try AbiWorker(
                executable: workerPath, dylib: dylibPath, benchCase: c, warmup: warmup,
                readySeconds: workerReadySeconds, replyGraceSeconds: workerReplyGraceSeconds)
            for _ in 0..<warmup {
                _ = try await direct.embed(texts)
            }
            let directOut = try await direct.embed(texts)
            let abiOut = try abiWorker.collect()
            guard abiOut.count == directOut.count else {
                throw fail("\(c.name): ABI returned \(abiOut.count) values, direct \(directOut.count)")
            }
            var maxAbs: Float = 0
            var sqSum = 0.0
            for (a, d) in zip(abiOut, directOut) {
                let diff = abs(a - d)
                maxAbs = max(maxAbs, diff)
                sqSum += Double(diff) * Double(diff)
            }
            let rmse = (sqSum / Double(abiOut.count)).squareRoot()
            let parityOk = maxAbs <= kParityMaxAbs && rmse <= kParityMaxRmse
            if !parityOk { allPass = false }

            // Steady-state ABI allocation counter on one post-warmup request.
            let steadyAllocs = try abiWorker.steadyAllocs()
            let countersZero = steadyAllocs == 0
            if !countersZero { allPass = false }

            var repeatReports: [[String: Any]] = []
            for repeatIndex in 0..<repeats {
                let abiFirst = (caseIndex + repeatIndex) % 2 == 1
                var directStats: RepeatStats?
                var abiStats: [String: Any]?
                for leg in 0..<2 {
                    let runAbi = (leg == 0) == abiFirst
                    if runAbi {
                        abiStats = try abiWorker.time(
                            maxSeconds: maxSeconds, maxRequests: maxRequests)
                    } else {
                        directStats = try await timedRepeat(
                            maxSeconds: maxSeconds, maxRequests: maxRequests
                        ) {
                            _ = try await direct.embed(texts)
                        }
                    }
                }
                guard let directStats, let abiStats else {
                    throw fail("\(c.name): repeat \(repeatIndex) missing a leg")
                }
                let abiP50 = try statNumber(abiStats, "p50_us")
                let abiRps = try statNumber(abiStats, "requests_per_second")
                let p50Ratio = abiP50 / Double(directStats.p50Us)
                let throughputRatio = abiRps / directStats.rps
                let pass = p50Ratio <= kP50OverheadLimit && throughputRatio >= kThroughputFloor
                if !pass { allPass = false }
                repeatReports.append([
                    "order": abiFirst ? "abi_first" : "direct_first",
                    "direct": directStats.json,
                    "abi": abiStats,
                    "abi_p50_over_direct_p50": p50Ratio,
                    "abi_throughput_over_direct": throughputRatio,
                    "pass": pass,
                ])
                let ratioText = String(format: "%.4f", p50Ratio)
                let rpsText = String(format: "%.1f/%.1f", directStats.rps, abiRps)
                FileHandle.standardError.write(
                    Data(
                        "\(c.name) repeat \(repeatIndex): direct p50=\(directStats.p50Us)us abi p50=\(UInt64(abiP50))us ratio=\(ratioText) rps \(rpsText) pass=\(pass)\n"
                            .utf8))
            }

            // Exit the worker (releasing its engine and every Metal
            // resource the case accumulated) and drop the direct path's
            // MLX buffer cache before the next case.
            abiWorker.shutdown()
            MLX.Memory.clearCache()

            caseReports.append([
                "case": c.name,
                "batch": c.batch,
                "target_tokens": c.targetTokens,
                "mixed": c.mixed,
                "row_tokens": rowTokens,
                "execution_shape": [c.batch, kMaxSeq],
                "parity": [
                    "max_abs": Double(maxAbs),
                    "rmse": rmse,
                    "max_abs_gate": Double(kParityMaxAbs),
                    "rmse_gate": kParityMaxRmse,
                    "pass": parityOk,
                ],
                "abi_steady_state_counters": ["arena_allocs": steadyAllocs],
                "abi_counters_zero": countersZero,
                "repeats": repeatReports,
            ])
        }

        // Two-engine concurrent observation (arena isolation under load),
        // run inside one isolated ABI worker process.
        var concurrent: Any = NSNull()
        if !skipConcurrent {
            let c = BenchCase(batch: 8, targetTokens: 128, mixed: true)
            let abiWorker = try AbiWorker(
                executable: workerPath, dylib: dylibPath, benchCase: c, warmup: warmup,
                readySeconds: workerReadySeconds, replyGraceSeconds: workerReplyGraceSeconds)
            let stats = try abiWorker.concurrent(
                maxSeconds: maxSeconds, maxRequests: maxRequests,
                readySeconds: workerReadySeconds)
            abiWorker.shutdown()
            concurrent = [
                "case": c.name,
                "engines": 2,
                "engine_a": stats["engine_a"] ?? NSNull(),
                "engine_b": stats["engine_b"] ?? NSNull(),
                "note": "Recorded two-engine observation on one Metal GPU, not a scalability acceptance result.",
            ] as [String: Any]
        }

        let receipt: [String: Any] = [
            "experiment": "apple-metal-native-overhead-pilot",
            "provider": "Apple MLX Metal",
            "alias": kAlias,
            "model": modelDir.path,
            "tokenizer": modelDir.appending(path: "tokenizer.json").path,
            "direct_baseline":
                "raw mlx-swift consumer: MLXEmbedders model container, HF tokenizer, fixed [batch, 256] padding, BERT forward, mean+L2 MLX ops on device, one host read of [batch, dim] (unified memory)",
            "abi_path":
                "turboembed.h text ABI via dlopen(libTurboEmbed.dylib) in an isolated bench-abi-worker process (one per case, no MLX or swift-transformers linked into the worker): native WordPiece into a turbo_buffer Metal SHARED arena, MLX BERT forward, host mean+L2 from unified memory into a rented SHARED result",
            "process_isolation":
                "Each case's ABI leg runs in its own bench-abi-worker process and exits afterwards, so the dylib's Objective-C classes are never duplicated against the direct baseline's and Metal resources are released per case. Legs execute sequentially (the orchestrator blocks on the worker); the direct path clears the MLX buffer cache between cases. Each path warms up in its own process before parity and timing.",
            "execution_shape_note":
                "Both paths execute fixed [batch, 256]. Apple unified memory has no discrete H2D/D2H; the ABI copies the last hidden state into the SHARED activation rent and pools on host, the direct baseline pools on device and reads back [batch, dim]. That difference is recorded, not hidden.",
            "abi_dylib": dylibPath,
            "abi_worker": workerPath,
            "mlx_build": "mlx-swift (pins in swift/Package.resolved at git_sha)",
            "gpu": gpu,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "chip": commandOutput("/usr/sbin/sysctl", ["-n", "machdep.cpu.brand_string"]),
            "host": commandOutput("/usr/sbin/scutil", ["--get", "LocalHostName"]),
            "git_sha": commandOutput("/usr/bin/git", ["rev-parse", "HEAD"], cwd: root),
            "quick": quick,
            "gates": [
                "abi_p50_over_direct_p50_max": kP50OverheadLimit,
                "abi_throughput_over_direct_min": kThroughputFloor,
                "parity_max_abs": Double(kParityMaxAbs),
                "parity_rmse": kParityMaxRmse,
                "p99_min_samples": kP99MinSamples,
            ],
            "config": [
                "warmup": warmup,
                "repeats": repeats,
                "max_seconds": maxSeconds,
                "max_requests": maxRequests,
                "worker_ready_seconds": workerReadySeconds,
                "worker_reply_grace_seconds": workerReplyGraceSeconds,
            ],
            "cases": caseReports,
            "concurrent_two_engines": concurrent,
            "pass": allPass,
        ]

        let outURL = URL(filePath: out)
        try FileManager.default.createDirectory(
            at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try JSONSerialization.data(
            withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
        try (data + Data("\n".utf8)).write(to: outURL)
        FileHandle.standardError.write(
            Data("wrote \(outURL.path)\napple overhead pilot pass=\(allPass)\n".utf8))
        if !allPass {
            throw ExitCode(2)
        }
    }

    private func resolve(overridePath: String?, name: String, root: URL, buildHint: String) throws
        -> String
    {
        if let overridePath {
            guard FileManager.default.fileExists(atPath: overridePath) else {
                throw fail("\(overridePath) does not exist")
            }
            return overridePath
        }
        var candidates: [URL] = []
        if let exe = Bundle.main.executableURL {
            candidates.append(exe.deletingLastPathComponent().appending(path: name))
        }
        candidates.append(root.appending(path: "swift/.build/release/\(name)"))
        for candidate in candidates {
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate.path
            }
        }
        throw fail(
            "\(name) not found (tried \(candidates.map(\.path).joined(separator: ", "))); build it with `\(buildHint)`"
        )
    }
}
