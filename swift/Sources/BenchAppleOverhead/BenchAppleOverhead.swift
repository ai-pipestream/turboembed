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
// Gates per repeat (predeclared, same as the NVIDIA/Intel pilots): ABI p50
// within 5% of the direct baseline and ABI throughput at least 95% of it.
// Parity per case: max abs error <= 5e-4 and RMSE <= 1e-4 between the two
// paths. p99 is descriptive only below 1000 samples. The ABI steady-state
// turbo_buffer allocation counter must be zero after warmup.
//
// Run through `make bench-apple-overhead` on the Machine C Metal host; the
// receipt is `testdata/receipts/bench/machine-c-metal-overhead.json`.

import ArgumentParser
import Darwin
import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import Metal
import MlxEngine

#if canImport(TurboEmbedC)
    import TurboEmbedC
#endif

let kAlias = "minilm"
let kMaxSeq = 256
let kDim = 384
let kParityMaxAbs: Float = 5e-4
let kParityMaxRmse = 1e-4
let kP50OverheadLimit = 1.05
let kThroughputFloor = 0.95
let kP99MinSamples = 1000

enum BenchError: Error, CustomStringConvertible {
    case message(String)

    var description: String {
        switch self {
        case .message(let m): return m
        }
    }
}

func fail(_ message: String) -> BenchError { .message(message) }

// MARK: - dlopen'd C ABI (the packaged dylib, not an in-process module)

typealias AbiVersionFn = @convention(c) () -> UInt32
typealias CreateFn = @convention(c) (
    turboembed_device, UnsafePointer<CChar>?, UnsafeMutablePointer<OpaquePointer?>?
) -> turboembed_status
typealias DestroyFn = @convention(c) (OpaquePointer?) -> Void
typealias LastErrorFn = @convention(c) (OpaquePointer?) -> UnsafePointer<CChar>?
typealias LoadFn = @convention(c) (
    OpaquePointer?, UnsafePointer<CChar>?, Int
) -> turboembed_status
typealias EmbedFn = @convention(c) (
    OpaquePointer?, UnsafePointer<CChar>?, Int,
    UnsafePointer<turboembed_str>?, Int,
    UnsafePointer<turboembed_embed_options>?,
    UnsafeMutablePointer<UnsafeMutablePointer<turboembed_embed_result>?>?
) -> turboembed_status
typealias ResultFreeFn = @convention(c) (
    UnsafeMutablePointer<turboembed_embed_result>?
) -> Void
typealias CounterFn = @convention(c) () -> UInt64
typealias CounterResetFn = @convention(c) () -> Void
typealias MetalOwnsFn = @convention(c) (UnsafeRawPointer?) -> Int32

/// `turboembed_*` (and turbo_buffer counter) symbols resolved from
/// `libTurboEmbed.dylib`. Every ABI request in this bench goes through the
/// dylib's exported C symbols — the same surface a packaged consumer loads.
final class AbiDylib: @unchecked Sendable {
    let path: String
    let abiVersion: AbiVersionFn
    let create: CreateFn
    let destroy: DestroyFn
    let lastError: LastErrorFn
    let load: LoadFn
    let embed: EmbedFn
    let resultFree: ResultFreeFn
    let allocCounter: CounterFn
    let allocCounterReset: CounterResetFn
    let metalOwns: MetalOwnsFn

    init(path: String) throws {
        self.path = path
        guard let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            let why = dlerror().map { String(cString: $0) } ?? "unknown dlopen error"
            throw fail("dlopen(\(path)): \(why)")
        }
        func sym<T>(_ name: String, as type: T.Type) throws -> T {
            guard let raw = dlsym(handle, name) else {
                throw fail("dlsym(\(name)) missing in \(path)")
            }
            return unsafeBitCast(raw, to: T.self)
        }
        self.abiVersion = try sym("turboembed_abi_version", as: AbiVersionFn.self)
        self.create = try sym("turboembed_engine_create", as: CreateFn.self)
        self.destroy = try sym("turboembed_engine_destroy", as: DestroyFn.self)
        self.lastError = try sym("turboembed_last_error", as: LastErrorFn.self)
        self.load = try sym("turboembed_load_model", as: LoadFn.self)
        self.embed = try sym("turboembed_embed", as: EmbedFn.self)
        self.resultFree = try sym("turboembed_embed_result_free", as: ResultFreeFn.self)
        self.allocCounter = try sym("turbo_buffer_alloc_counter", as: CounterFn.self)
        self.allocCounterReset = try sym(
            "turbo_buffer_alloc_counter_reset", as: CounterResetFn.self)
        self.metalOwns = try sym("turbo_buffer_metal_owns", as: MetalOwnsFn.self)
    }

    func errorText(_ engine: OpaquePointer?) -> String {
        guard let ptr = lastError(engine) else { return "" }
        return String(cString: ptr)
    }
}

/// One ABI engine on METAL with `minilm` loaded, plus reusable
/// caller-owned UTF-8 input buffers (the ABI takes pointer+length views).
final class AbiEngine: @unchecked Sendable {
    let dylib: AbiDylib
    let engine: OpaquePointer
    private let aliasBytes: [CChar]
    private var opts: turboembed_embed_options

    init(dylib: AbiDylib) throws {
        self.dylib = dylib
        var out: OpaquePointer?
        let st = dylib.create(TURBOEMBED_DEVICE_METAL, nil, &out)
        guard st == TURBOEMBED_OK, let engine = out else {
            throw fail(
                "turboembed_engine_create(METAL) failed (fail loud, no CPU fallback): "
                    + dylib.errorText(nil))
        }
        self.engine = engine
        self.aliasBytes = Array(kAlias.utf8CString)
        self.opts = turboembed_embed_options(
            pooling: TURBOEMBED_POOLING_MEAN,
            normalize: 1,
            truncate_to: UInt32(kMaxSeq),
            output_format: TURBOEMBED_OUTPUT_TYPED
        )
        let st2 = aliasBytes.withUnsafeBufferPointer { alias in
            dylib.load(engine, alias.baseAddress, kAlias.utf8.count)
        }
        guard st2 == TURBOEMBED_OK else {
            throw fail("turboembed_load_model(\(kAlias)): " + dylib.errorText(engine))
        }
    }

    deinit {
        dylib.destroy(engine)
    }

    /// One synchronous text-to-pooled-result ABI request; result freed.
    func embedOnce(_ texts: CTextViews) throws {
        var out: UnsafeMutablePointer<turboembed_embed_result>?
        let st = aliasBytes.withUnsafeBufferPointer { alias in
            texts.views.withUnsafeBufferPointer { views in
                dylib.embed(
                    engine, alias.baseAddress, kAlias.utf8.count,
                    views.baseAddress, views.count, &opts, &out)
            }
        }
        guard st == TURBOEMBED_OK, let result = out else {
            throw fail("turboembed_embed: " + dylib.errorText(engine))
        }
        dylib.resultFree(result)
    }

    /// One ABI request keeping the result; caller receives copied rows.
    /// Verifies the result values pointer is turbo_buffer Metal SHARED.
    func embedCollect(_ texts: CTextViews) throws -> [Float] {
        var out: UnsafeMutablePointer<turboembed_embed_result>?
        let st = aliasBytes.withUnsafeBufferPointer { alias in
            texts.views.withUnsafeBufferPointer { views in
                dylib.embed(
                    engine, alias.baseAddress, kAlias.utf8.count,
                    views.baseAddress, views.count, &opts, &out)
            }
        }
        guard st == TURBOEMBED_OK, let result = out else {
            throw fail("turboembed_embed: " + dylib.errorText(engine))
        }
        defer { dylib.resultFree(result) }
        let dim = Int(result.pointee.dim)
        let count = Int(result.pointee.count)
        guard dim == kDim, count == texts.views.count else {
            throw fail("ABI returned dim=\(dim) count=\(count), expected \(kDim)x\(texts.views.count)")
        }
        guard dylib.metalOwns(UnsafeRawPointer(result.pointee.values)) == 1 else {
            throw fail("FAKE: result.values is not turbo_buffer Metal SHARED")
        }
        return Array(UnsafeBufferPointer(start: result.pointee.values, count: dim * count))
    }
}

/// Caller-owned UTF-8 buffers for `turboembed_str` views, allocated once
/// per case (outside the timed loop, matching the Rust `&str` views the
/// NVIDIA pilot passes).
final class CTextViews: @unchecked Sendable {
    private var storage: [UnsafeMutablePointer<UInt8>] = []
    private(set) var views: [turboembed_str] = []

    init(_ texts: [String]) {
        views.reserveCapacity(texts.count)
        for text in texts {
            let bytes = Array(text.utf8)
            if bytes.isEmpty {
                views.append(turboembed_str(ptr: nil, len: 0))
                continue
            }
            let copy = UnsafeMutablePointer<UInt8>.allocate(capacity: bytes.count)
            bytes.withUnsafeBufferPointer { src in
                copy.update(from: src.baseAddress!, count: src.count)
            }
            storage.append(copy)
            views.append(
                turboembed_str(
                    ptr: UnsafeRawPointer(copy).assumingMemoryBound(to: CChar.self),
                    len: bytes.count
                ))
        }
    }

    deinit {
        storage.forEach { $0.deallocate() }
    }
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

// MARK: - Case grid (identical to the NVIDIA pilot)

struct Case {
    let batch: Int
    let targetTokens: Int
    let mixed: Bool

    var name: String { "b\(batch)_t\(targetTokens)_\(mixed ? "mixed" : "full")" }
}

func caseGrid(quick: Bool) -> [Case] {
    let batches = quick ? [1, 8] : [1, 8, 32]
    let targets = quick ? [32] : [32, 128, 256]
    var cases: [Case] = []
    for batch in batches {
        for target in targets {
            for mixed in [false, true] {
                cases.append(Case(batch: batch, targetTokens: target, mixed: mixed))
            }
        }
    }
    return cases
}

/// Deterministic texts. Every "the" is one WordPiece token, so a row
/// targeting `t` tokens is `t - 2` words plus [CLS]/[SEP].
func caseTexts(_ c: Case) -> [String] {
    func rowTokens(_ target: Int, _ index: Int, _ batch: Int, _ mixed: Bool) -> Int {
        if !mixed { return target }
        let lo = min(16, target)
        if batch <= 1 { return (lo + target) / 2 }
        return lo + ((target - lo) * index) / (batch - 1)
    }
    return (0..<c.batch).map { i in
        let tokens = max(rowTokens(c.targetTokens, i, c.batch, c.mixed), 3)
        return Array(repeating: "the", count: tokens - 2).joined(separator: " ")
    }
}

// MARK: - Timing

struct RepeatStats {
    var n: Int
    var seconds: Double
    var p50Us: UInt64
    var p90Us: UInt64
    var p99Us: UInt64
    var minUs: UInt64
    var maxUs: UInt64
    var rps: Double

    var json: [String: Any] {
        [
            "n": n,
            "seconds": seconds,
            "p50_us": p50Us,
            "p90_us": p90Us,
            "p99_us": p99Us,
            "p99_sufficient": n >= kP99MinSamples,
            "min_us": minUs,
            "max_us": maxUs,
            "requests_per_second": rps,
        ]
    }
}

/// Nearest-rank percentile in microseconds; `sorted` ascending.
func nearestRankUs(_ sorted: [UInt64], _ percent: Double) -> UInt64 {
    guard !sorted.isEmpty else { return 0 }
    let rank = Int((percent / 100.0 * Double(sorted.count)).rounded(.up))
    return sorted[min(max(rank, 1), sorted.count) - 1]
}

func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

func timedRepeat(
    maxSeconds: Double,
    maxRequests: Int,
    _ call: () async throws -> Void
) async throws -> RepeatStats {
    var samples: [UInt64] = []
    samples.reserveCapacity(min(maxRequests, 16_384))
    let start = nowNs()
    let deadlineNs = start + UInt64(maxSeconds * 1e9)
    while samples.count < maxRequests && nowNs() < deadlineNs {
        let t0 = nowNs()
        try await call()
        let elapsed = nowNs() - t0
        samples.append((elapsed + 999) / 1000)
    }
    let seconds = Double(nowNs() - start) / 1e9
    guard !samples.isEmpty else { throw fail("timed repeat produced no samples") }
    samples.sort()
    return RepeatStats(
        n: samples.count,
        seconds: seconds,
        p50Us: nearestRankUs(samples, 50),
        p90Us: nearestRankUs(samples, 90),
        p99Us: nearestRankUs(samples, 99),
        minUs: samples[0],
        maxUs: samples[samples.count - 1],
        rps: Double(samples.count) / seconds
    )
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

    @Option(help: "Receipt output path.")
    var out: String

    mutating func run() async throws {
        let root = workspaceRoot()
        setenv("INFERSTREAM_ROOT", root.path, 0)

        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            throw fail("no Metal device on this host — refusing to bench (fail loud, no CPU fallback)")
        }
        let gpu = metalDevice.name

        let dylibPath = try resolveDylib(root: root)
        FileHandle.standardError.write(Data("ABI dylib: \(dylibPath)\n".utf8))
        let abi = try AbiDylib(path: dylibPath)
        guard abi.abiVersion() == 1 else {
            throw fail("dylib reports ABI version \(abi.abiVersion()), expected 1")
        }

        let modelDir = root.appending(path: "models/mlx/\(kAlias)")
        FileHandle.standardError.write(Data("direct MLX load: \(modelDir.path)\n".utf8))
        let direct = try await DirectMlx(modelDir: modelDir)

        FileHandle.standardError.write(Data("ABI engine load: alias \(kAlias) on METAL\n".utf8))
        let engine = try AbiEngine(dylib: abi)

        let cases = caseGrid(quick: quick)
        var caseReports: [[String: Any]] = []
        var allPass = true

        for (caseIndex, c) in cases.enumerated() {
            let texts = caseTexts(c)
            let views = CTextViews(texts)
            let rowTokens = try await direct.rowTokens(texts)
            if !c.mixed && rowTokens.contains(where: { $0 != c.targetTokens }) {
                throw fail("\(c.name): constructed rows are \(rowTokens), expected \(c.targetTokens)")
            }

            // Warmup both paths, then check parity before any timing.
            for _ in 0..<warmup {
                _ = try await direct.embed(texts)
                try engine.embedOnce(views)
            }
            let directOut = try await direct.embed(texts)
            let abiOut = try engine.embedCollect(views)
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
            abi.allocCounterReset()
            try engine.embedOnce(views)
            let steadyAllocs = abi.allocCounter()
            let countersZero = steadyAllocs == 0
            if !countersZero { allPass = false }

            var repeatReports: [[String: Any]] = []
            for repeatIndex in 0..<repeats {
                let abiFirst = (caseIndex + repeatIndex) % 2 == 1
                var directStats: RepeatStats?
                var abiStats: RepeatStats?
                for leg in 0..<2 {
                    let runAbi = (leg == 0) == abiFirst
                    if runAbi {
                        abiStats = try await timedRepeat(
                            maxSeconds: maxSeconds, maxRequests: maxRequests
                        ) {
                            try engine.embedOnce(views)
                        }
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
                let p50Ratio = Double(abiStats.p50Us) / Double(directStats.p50Us)
                let throughputRatio = abiStats.rps / directStats.rps
                let pass = p50Ratio <= kP50OverheadLimit && throughputRatio >= kThroughputFloor
                if !pass { allPass = false }
                repeatReports.append([
                    "order": abiFirst ? "abi_first" : "direct_first",
                    "direct": directStats.json,
                    "abi": abiStats.json,
                    "abi_p50_over_direct_p50": p50Ratio,
                    "abi_throughput_over_direct": throughputRatio,
                    "pass": pass,
                ])
                let ratioText = String(format: "%.4f", p50Ratio)
                let rpsText = String(format: "%.1f/%.1f", directStats.rps, abiStats.rps)
                FileHandle.standardError.write(
                    Data(
                        "\(c.name) repeat \(repeatIndex): direct p50=\(directStats.p50Us)us abi p50=\(abiStats.p50Us)us ratio=\(ratioText) rps \(rpsText) pass=\(pass)\n"
                            .utf8))
            }

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

        // Two-engine concurrent observation (arena isolation under load).
        var concurrent: Any = NSNull()
        if !skipConcurrent {
            let c = Case(batch: 8, targetTokens: 128, mixed: true)
            let texts = caseTexts(c)
            let engineB = try AbiEngine(dylib: abi)
            let warmupCount = warmup
            let maxSecondsLocal = maxSeconds
            let maxRequestsLocal = maxRequests
            func runOne(_ eng: AbiEngine) async throws -> RepeatStats {
                let views = CTextViews(texts)
                for _ in 0..<warmupCount {
                    try eng.embedOnce(views)
                }
                return try await timedRepeat(
                    maxSeconds: maxSecondsLocal, maxRequests: maxRequestsLocal
                ) {
                    try eng.embedOnce(views)
                }
            }
            async let statsA = runOne(engine)
            async let statsB = runOne(engineB)
            let (a, b) = try await (statsA, statsB)
            concurrent = [
                "case": c.name,
                "engines": 2,
                "engine_a": a.json,
                "engine_b": b.json,
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
                "turboembed.h text ABI via dlopen(libTurboEmbed.dylib): native WordPiece into a turbo_buffer Metal SHARED arena, MLX BERT forward, host mean+L2 from unified memory into a rented SHARED result",
            "execution_shape_note":
                "Both paths execute fixed [batch, 256]. Apple unified memory has no discrete H2D/D2H; the ABI copies the last hidden state into the SHARED activation rent and pools on host, the direct baseline pools on device and reads back [batch, dim]. That difference is recorded, not hidden.",
            "abi_dylib": dylibPath,
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

    private func resolveDylib(root: URL) throws -> String {
        if let dylib {
            guard FileManager.default.fileExists(atPath: dylib) else {
                throw fail("--dylib \(dylib) does not exist")
            }
            return dylib
        }
        var candidates: [URL] = []
        if let exe = Bundle.main.executableURL {
            candidates.append(
                exe.deletingLastPathComponent().appending(path: "libTurboEmbed.dylib"))
        }
        candidates.append(root.appending(path: "swift/.build/release/libTurboEmbed.dylib"))
        for candidate in candidates {
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate.path
            }
        }
        throw fail(
            "libTurboEmbed.dylib not found (tried \(candidates.map(\.path).joined(separator: ", "))); build it with `swift build -c release --package-path swift --product TurboEmbed`"
        )
    }
}
