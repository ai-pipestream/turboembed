#if canImport(TurboEmbedC)
import TurboEmbedC
#endif

import Darwin
import Foundation
import InferstreamCore
import MlxEngine

/// Catalog alias resolved to a local FP MLX directory.
///
/// MiniLM **must** use hidden-state mean + L2. mlx-swift-lm's
/// `Pooling.Strategy.cls` is BERT `tanh(dense(CLS))` (`pooledOutput`) —
/// that is the NSP pooler that produced apple↔nvidia cosine ≈ 0.
/// `MlxEngine.poolHidden` uses last-hidden mean / first-token CLS and
/// never applies an extra LayerNorm.
struct MlxAlias: Sendable {
    var alias: String
    var path: String
    var pooling: String
    var maxSeqLen: Int?
    var dim: UInt32
    var ready: Bool
}

enum MlxProviderError: Error, LocalizedError, Sendable {
    case metalUnavailable(String)
    case missingWeights(String)
    case fakeDim(alias: String, dim: Int)
    case embed(String)

    var errorDescription: String? {
        switch self {
        case .metalUnavailable(let device):
            "Metal/GPU requested but ping device=\(device) metal=false — refusing CPU fallback"
        case .missingWeights(let alias):
            "MLX weights for \(alias) not found under models/mlx/\(alias) — run `make fetch-mlx ALIASES=\(alias)`"
        case .fakeDim(let alias, let dim):
            "FAKE: \(alias) returned dim=\(dim); MiniLM is 384-d mean+L2, mock-embed is 8-d. BERT pooler and the ABI stub are forbidden here."
        case .embed(let message):
            message
        }
    }
}

enum MlxProvider {
    /// Make `Paths` / `Catalog.builtin` see the repo when cargo test cwd is
    /// `crates/turboembed`. Does not overwrite a caller-set `INFERSTREAM_ROOT`.
    static func ensureWorkspaceRoot() {
        if let env = ProcessInfo.processInfo.environment["INFERSTREAM_ROOT"], !env.isEmpty {
            return
        }
        var dir = URL(filePath: FileManager.default.currentDirectoryPath)
        for _ in 0..<8 {
            let marker = dir.appending(path: "include/turboembed.h")
            if FileManager.default.fileExists(atPath: marker.path) {
                setenv("INFERSTREAM_ROOT", dir.path, 0)
                return
            }
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path { return }
            dir = parent
        }
    }

    static func pingOrThrow() throws -> (MlxEngine.Engine, PingInfo) {
        let engine = MlxEngine.Engine()
        let ping = try engine.ping()
        return (engine, ping)
    }

    static func requireMetal(_ ping: PingInfo) throws {
        guard ping.metalAvailable, ping.device.contains("gpu") else {
            throw MlxProviderError.metalUnavailable(ping.device)
        }
    }

    /// Same names as the Rust catalog: generative MLX is not an embedder.
    private static let llmAliases: Set<String> = ["default-llm", "qwen-0.5b", "qwen-7b"]

    static func discover(catalog: Catalog?) -> [String: MlxAlias] {
        var out: [String: MlxAlias] = [:]
        let aliases: [String]
        if let catalog {
            aliases = catalog.aliases.filter { alias in
                !llmAliases.contains(alias)
                    && (try? catalog.resolve(alias: alias, arch: .apple))?.backend == .mlx
            }
        } else {
            aliases = ["minilm", "bge-small"]
        }
        for alias in aliases {
            guard let resolved = resolve(alias: alias, catalog: catalog) else { continue }
            out[alias] = resolved
        }
        return out
    }

    static func resolve(alias: String, catalog: Catalog?) -> MlxAlias? {
        if llmAliases.contains(alias) { return nil }
        var pooling = defaultPooling(alias)
        var maxSeqLen: Int? = alias == "minilm" ? 256 : nil
        var pathHint = "models/mlx/\(alias)"
        if let spec = try? catalog?.resolve(alias: alias, arch: .apple) {
            if spec.backend != .mlx { return nil }
            pooling = spec.pooling ?? pooling
            if let n = spec.maxSeqLen, n > 0 { maxSeqLen = Int(n) }
            if let p = spec.path, !p.isEmpty { pathHint = p }
        }
        let url = Paths.resolveExistingDirectory(pathHint)
        guard Paths.directoryExists(url) else { return nil }
        let dim: UInt32 = alias.hasPrefix("minilm") || alias.contains("small") ? 384 : 0
        return MlxAlias(
            alias: alias,
            path: url.path,
            pooling: pooling,
            maxSeqLen: maxSeqLen,
            dim: dim,
            ready: false
        )
    }

    /// Family default. MiniLM / E5 / GTE / MPNet = mean; BGE = first-token CLS
    /// (hidden state, not the BERT NSP pooler).
    static func defaultPooling(_ alias: String) -> String {
        if alias.hasPrefix("bge") { return "cls" }
        return "mean"
    }

    static func poolingName(_ opts: UnsafePointer<turboembed_embed_options>?, fallback: String)
        -> Result<String, MlxProviderError>
    {
        guard let opts else { return .success(fallback) }
        switch opts.pointee.pooling {
        case TURBOEMBED_POOLING_DEFAULT:
            return .success(fallback)
        case TURBOEMBED_POOLING_MEAN:
            return .success("mean")
        case TURBOEMBED_POOLING_CLS:
            return .success("cls")
        case TURBOEMBED_POOLING_LAST:
            return .failure(.embed("LAST pooling is not wired for MiniLM; use MEAN"))
        default:
            return .failure(.embed("unknown turboembed_pooling"))
        }
    }

    static func normalize(_ opts: UnsafePointer<turboembed_embed_options>?) -> Bool {
        guard let opts else { return true }
        switch opts.pointee.normalize {
        case 0: return false
        default: return true
        }
    }

    static func truncate(_ opts: UnsafePointer<turboembed_embed_options>?, fallback: Int?) -> Int? {
        guard let opts, opts.pointee.truncate_to > 0 else { return fallback }
        return Int(opts.pointee.truncate_to)
    }

    static func embedArena(
        engine: MlxEngine.Engine,
        model: MlxAlias,
        texts: [String],
        pooling: String,
        normalize: Bool,
        maxSeqLen: Int?,
        arena: MetalArena,
        logProvenance: Bool
    ) throws -> (dim: Int, header: UnsafeMutablePointer<turboembed_embed_result>) {
        if texts.count > Int(arena.maxBatch) {
            throw MetalArenaError.batch(
                "batch \(texts.count) exceeds arena max_batch \(arena.maxBatch)"
            )
        }
        var resultView = try arena.rentResult(rows: UInt32(texts.count), cols: arena.dim)
        let slots = ArenaEmbedSlots(
            inputIds: arena.inputIds,
            attentionMask: arena.attentionMask,
            tokenTypes: arena.tokenTypes,
            activations: arena.hiddenStates,
            results: resultView.ptr.assumingMemoryBound(to: Float.self),
            maxBatch: Int(arena.maxBatch),
            maxSeq: Int(arena.maxSeq),
            hiddenCap: Int(arena.hidden)
        )
        let pair: (dim: Int, count: Int)
        do {
            pair = try runBlocking {
                try await engine.embedArena(
                    modelPath: model.path,
                    texts: texts,
                    normalize: normalize,
                    pooling: pooling,
                    maxSeqLen: maxSeqLen,
                    slots: slots
                )
            }
        } catch {
            arena.returnView(&resultView)
            throw error
        }
        if pair.dim == 0 || pair.dim == 8 {
            arena.returnView(&resultView)
            throw MlxProviderError.fakeDim(alias: model.alias, dim: pair.dim)
        }
        if model.alias == "minilm", pair.dim != 384 {
            arena.returnView(&resultView)
            throw MlxProviderError.fakeDim(alias: model.alias, dim: pair.dim)
        }
        try arena.requireMetalShared(resultView.ptr, what: "result")
        try arena.requireMetalShared(arena.tokensIds.ptr, what: "input_ids")
        try arena.requireMetalShared(arena.activations.ptr, what: "activations")
        let values = resultView.ptr.assumingMemoryBound(to: Float.self)
        let rec = UnsafeMutablePointer<EmbedResultRec>.allocate(capacity: 1)
        rec.pointee = EmbedResultRec(
            pub: turboembed_embed_result(
                dim: UInt32(pair.dim),
                count: UInt32(pair.count),
                values: UnsafePointer(values),
                packed: UnsafeRawPointer(values).assumingMemoryBound(to: UInt8.self),
                packed_len: pair.count * pair.dim * MemoryLayout<Float>.size
            ),
            view: resultView,
            arena: arena.raw
        )
        // Provenance is logged once per engine+alias (load/warmup), not per
        // request: this fputs was in the timed hot path of the overhead
        // pilot, and stderr writes can block on the consumer.
        if logProvenance {
            fputs(
                "[turboembed] mlx embed \(model.alias) pooling=\(pooling) normalize=\(normalize ? 1 : 0) dim=\(pair.dim) n=\(pair.count) — arena SHARED tokens/activations/result, hidden-state mean+L2, not BERT pooler, not mock\n",
                stderr)
        }
        return (
            pair.dim,
            UnsafeMutableRawPointer(rec).assumingMemoryBound(to: turboembed_embed_result.self)
        )
    }
}

func runBlocking<T: Sendable>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
    let box = BlockingBox<T>()
    let sem = DispatchSemaphore(value: 0)
    Task.detached {
        do {
            box.set(.success(try await body()))
        } catch {
            box.set(.failure(error))
        }
        sem.signal()
    }
    sem.wait()
    return try box.get()
}

private final class BlockingBox<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Result<T, any Error>?
    func set(_ result: Result<T, any Error>) {
        lock.lock()
        value = result
        lock.unlock()
    }
    func get() throws -> T {
        lock.lock()
        defer { lock.unlock() }
        switch value {
        case .success(let v): return v
        case .failure(let e): throw e
        case .none: throw MlxProviderError.embed("blocking box empty")
        }
    }
}
