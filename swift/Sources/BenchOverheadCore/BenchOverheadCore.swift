// Shared support for the Apple matched native-overhead pilot: the dlopen'd
// `libTurboEmbed.dylib` ABI wrappers, the deterministic case grid, and the
// timing helpers. This module deliberately has **no** MLX or
// swift-transformers dependency so the `bench-abi-worker` executable can
// link it without duplicating the Objective-C classes (Tokenizers, MLX)
// that the dylib already embeds. The direct-baseline MLX code lives only
// in the `bench-apple-overhead` orchestrator.

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif
import Foundation
import TurboEmbedC

public let kAlias = "minilm"
public let kMaxSeq = 256
public let kDim = 384

public enum BenchError: Error, CustomStringConvertible {
    case message(String)

    public var description: String {
        switch self {
        case .message(let m): return m
        }
    }
}

public func fail(_ message: String) -> BenchError { .message(message) }

// MARK: - dlopen'd C ABI (the packaged dylib, not an in-process module)

public typealias AbiVersionFn = @convention(c) () -> UInt32
public typealias CreateFn = @convention(c) (
    turboembed_device, UnsafePointer<CChar>?, UnsafeMutablePointer<OpaquePointer?>?
) -> turboembed_status
public typealias DestroyFn = @convention(c) (OpaquePointer?) -> Void
public typealias LastErrorFn = @convention(c) (OpaquePointer?) -> UnsafePointer<CChar>?
public typealias LoadFn = @convention(c) (
    OpaquePointer?, UnsafePointer<CChar>?, Int
) -> turboembed_status
public typealias EmbedFn = @convention(c) (
    OpaquePointer?, UnsafePointer<CChar>?, Int,
    UnsafePointer<turboembed_str>?, Int,
    UnsafePointer<turboembed_embed_options>?,
    UnsafeMutablePointer<UnsafeMutablePointer<turboembed_embed_result>?>?
) -> turboembed_status
public typealias ResultFreeFn = @convention(c) (
    UnsafeMutablePointer<turboembed_embed_result>?
) -> Void
public typealias CounterFn = @convention(c) () -> UInt64
public typealias CounterResetFn = @convention(c) () -> Void
public typealias MetalOwnsFn = @convention(c) (UnsafeRawPointer?) -> Int32

/// `turboembed_*` (and turbo_buffer counter) symbols resolved from
/// `libTurboEmbed.dylib`. Every ABI request goes through the dylib's
/// exported C symbols — the same surface a packaged consumer loads.
public final class AbiDylib: @unchecked Sendable {
    public let path: String
    public let abiVersion: AbiVersionFn
    public let create: CreateFn
    public let destroy: DestroyFn
    public let lastError: LastErrorFn
    public let load: LoadFn
    public let embed: EmbedFn
    public let resultFree: ResultFreeFn
    public let allocCounter: CounterFn
    public let allocCounterReset: CounterResetFn
    public let metalOwns: MetalOwnsFn

    public init(path: String) throws {
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

    public func errorText(_ engine: OpaquePointer?) -> String {
        guard let ptr = lastError(engine) else { return "" }
        return String(cString: ptr)
    }
}

/// One ABI engine on METAL with `minilm` loaded, plus reusable
/// caller-owned UTF-8 input buffers (the ABI takes pointer+length views).
public final class AbiEngine: @unchecked Sendable {
    public let dylib: AbiDylib
    public let engine: OpaquePointer
    private let aliasBytes: [CChar]
    private var opts: turboembed_embed_options

    public init(dylib: AbiDylib) throws {
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
    public func embedOnce(_ texts: CTextViews) throws {
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
    public func embedCollect(_ texts: CTextViews) throws -> [Float] {
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
public final class CTextViews: @unchecked Sendable {
    private var storage: [UnsafeMutablePointer<UInt8>] = []
    public private(set) var views: [turboembed_str] = []

    public init(_ texts: [String]) {
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

// MARK: - Case grid (identical to the NVIDIA pilot)

public struct BenchCase {
    public let batch: Int
    public let targetTokens: Int
    public let mixed: Bool

    public init(batch: Int, targetTokens: Int, mixed: Bool) {
        self.batch = batch
        self.targetTokens = targetTokens
        self.mixed = mixed
    }

    public var name: String { "b\(batch)_t\(targetTokens)_\(mixed ? "mixed" : "full")" }
}

public func caseGrid(quick: Bool) -> [BenchCase] {
    let batches = quick ? [1, 8] : [1, 8, 32]
    let targets = quick ? [32] : [32, 128, 256]
    var cases: [BenchCase] = []
    for batch in batches {
        for target in targets {
            for mixed in [false, true] {
                cases.append(BenchCase(batch: batch, targetTokens: target, mixed: mixed))
            }
        }
    }
    return cases
}

/// Deterministic texts. Every "the" is one WordPiece token, so a row
/// targeting `t` tokens is `t - 2` words plus [CLS]/[SEP].
public func caseTexts(_ c: BenchCase) -> [String] {
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

public let kP99MinSamples = 1000

public struct RepeatStats {
    public var n: Int
    public var seconds: Double
    public var p50Us: UInt64
    public var p90Us: UInt64
    public var p99Us: UInt64
    public var minUs: UInt64
    public var maxUs: UInt64
    public var rps: Double

    public var json: [String: Any] {
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

public func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

public func timedRepeat(
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
