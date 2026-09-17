#if canImport(WordPieceC)
import WordPieceC
#endif
import Accelerate
import Foundation
import Metal
import MLX
import MLXEmbedders
import MLXLLM
import MLXLMCommon

public struct PingInfo: Sendable {
    public var mlxVersion: String
    public var device: String
    public var metalAvailable: Bool
    public var matmulOk: Bool
    public var activeMemory: UInt64
    public var peakMemory: UInt64
}

public struct EmbedResult: Sendable {
    public var dimensions: Int
    public var vectors: [[Float]]
}

/// Caller-rented turbo_buffer Metal SHARED slots for one embed.
/// Pointers must stay valid for the duration of `embedArena`.
public struct ArenaEmbedSlots: @unchecked Sendable {
    public var inputIds: UnsafeMutablePointer<Int32>
    public var attentionMask: UnsafeMutablePointer<Int32>
    public var tokenTypes: UnsafeMutablePointer<Int32>
    public var activations: UnsafeMutablePointer<Float>
    public var results: UnsafeMutablePointer<Float>
    public var maxBatch: Int
    public var maxSeq: Int
    public var hiddenCap: Int

    public init(
        inputIds: UnsafeMutablePointer<Int32>,
        attentionMask: UnsafeMutablePointer<Int32>,
        tokenTypes: UnsafeMutablePointer<Int32>,
        activations: UnsafeMutablePointer<Float>,
        results: UnsafeMutablePointer<Float>,
        maxBatch: Int,
        maxSeq: Int,
        hiddenCap: Int
    ) {
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.tokenTypes = tokenTypes
        self.activations = activations
        self.results = results
        self.maxBatch = maxBatch
        self.maxSeq = maxSeq
        self.hiddenCap = hiddenCap
    }
}

public struct GenerateStats: Sendable {
    public var tokens: UInt32
    public var decodeTps: Double
}

public enum GenerateEvent: Sendable {
    case token(String)
    case done(GenerateStats)
}

public enum EngineError: Error, LocalizedError, Sendable {
    case missingModel(String)
    case emptyEmbed
    case internalError(String)

    public var errorDescription: String? {
        switch self {
        case .missingModel(let path):
            "MLX model directory not found at \(path) — run `make fetch-mlx`"
        case .emptyEmbed:
            "engine returned an empty embedding tensor"
        case .internalError(let message):
            message
        }
    }
}

/// Loaded-once native WordPiece vocab for one model directory, or the
/// recorded decision that the directory's tokenizer is not WordPiece
/// (SentencePiece etc. stay on swift-transformers). Cached per engine so
/// the embed hot path never re-reads or re-parses tokenizer files — the
/// 2026-09-17 Machine C overhead run paid a full `tokenizer.json` DOM
/// parse (~tens of ms) on every request because the vocab was loaded and
/// destroyed inside the request.
private enum WordPieceFront {
    case loaded(OpaquePointer)
    case unsupported
}

/// Engine-cached vocab handle passed into the model-container closure.
/// `@unchecked Sendable` like `ArenaEmbedSlots`: the vocab's lifetime is
/// owned by the engine (destroyed in `deinit`), and calls on one engine
/// are serialized by the ABI contract.
private struct WordPieceHandle: @unchecked Sendable {
    var vocab: OpaquePointer?
}

/// Once-validated MLXArray wraps of one arena's token slots. The arena
/// slots live for the engine's lifetime, so the no-copy wrap checks
/// (make_buffer hit, MTLBuffer backing == arena pointer) hold for every
/// later request over the same pointers; re-wrapping and re-`eval`ing
/// three arrays per request only added fixed hot-path cost.
private struct ArenaTokenWraps {
    var basePtr: UnsafeRawPointer
    var rows: Int
    var cols: Int
    var ids: MLXArray
    var mask: MLXArray
    var types: MLXArray
}

/// Process-default Metal device, created once. `MTLCreateSystemDefaultDevice`
/// was being called four times per embed request. `nonisolated(unsafe)`:
/// written once at initialization; MTLDevice itself is documented
/// thread-safe.
nonisolated(unsafe) let sharedMtlDevice: MTLDevice? = MTLCreateSystemDefaultDevice()

/// In-process native MLX engine. Weights stay in unified memory; Metal is
/// the default device. Used directly by the Swift gRPC server — no C ABI.
public final class Engine: @unchecked Sendable {
    private let lock = NSLock()
    private var lms: [String: ModelContainer] = [:]
    private var embeds: [String: EmbedderModelContainer] = [:]
    private var wordpieceFronts: [String: WordPieceFront] = [:]
    private var tokenWrapCache: [UnsafeRawPointer: ArenaTokenWraps] = [:]
    private let tokenizerLoader = HFTokenizerLoader()

    /// `TURBOEMBED_TIMING=1` prints one per-request phase line to stderr
    /// (tokenize / forward+eval / hidden copy / host pool, µs). Off by
    /// default; explicitly a diagnostic, never part of a receipt.
    static let timingEnabled =
        ProcessInfo.processInfo.environment["TURBOEMBED_TIMING"] == "1"

    public init() {
        _ = Device.gpu
    }

    deinit {
        #if canImport(WordPieceC)
        for case .loaded(let vocab) in wordpieceFronts.values {
            wordpiece_vocab_destroy(vocab)
        }
        #endif
    }

    public func ping() throws -> PingInfo {
        let a = MLXRandom.normal([64, 64])
        let b = MLXRandom.normal([64, 64])
        let c = (a.matmul(b)).sum()
        eval(c)
        let value = c.item(Float.self)
        let device = "\(Device.defaultDevice())"
        let metal = device.contains("gpu")
        return PingInfo(
            mlxVersion: "mlx-swift",
            device: device,
            metalAvailable: metal,
            matmulOk: value == value,
            activeMemory: metal ? UInt64(Memory.activeMemory) : 0,
            peakMemory: metal ? UInt64(Memory.peakMemory) : 0
        )
    }

    public func embed(
        modelPath: String,
        texts: [String],
        normalize: Bool,
        pooling: String = "mean",
        maxSeqLen: Int? = nil
    ) async throws -> EmbedResult {
        let kind = try PoolingKind.parse(pooling)
        let container = try await loadEmbed(modelPath)
        return try await container.perform { context -> EmbedResult in
            let padId =
                context.tokenizer.convertTokenToId("[PAD]")
                ?? context.tokenizer.convertTokenToId("<pad>")
                ?? 0
            let sepId =
                context.tokenizer.convertTokenToId("[SEP]")
                ?? context.tokenizer.convertTokenToId("</s>")
            var encoded = texts.map {
                context.tokenizer.encode(text: $0, addSpecialTokens: true)
            }
            if let maxSeqLen {
                encoded = encoded.map { truncateEncoderIds($0, max: maxSeqLen, sepId: sepId) }
            }
            let maxLength = encoded.map(\.count).max() ?? 0
            let padded = stacked(
                encoded.map { ids in
                    MLXArray(ids + Array(repeating: padId, count: maxLength - ids.count))
                })
            // Boolean 1=token / 0=pad. BERT's forward turns this into an additive
            // log-mask; we keep the boolean copy for mean pooling so pad rows
            // do not enter the average.
            let tokenMask = padded .!= padId
            let tokenTypes = MLXArray.zeros(like: padded)
            let output = context.model(
                padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: tokenMask)
            let pooled = poolHidden(output, mask: tokenMask, kind: kind, normalize: normalize)
            eval(pooled)
            let rows = pooled.map { $0.asArray(Float.self) }
            let dim = rows.first?.count ?? 0
            guard dim > 0, rows.count == texts.count else {
                throw EngineError.emptyEmbed
            }
            return EmbedResult(dimensions: dim, vectors: rows)
        }
    }

    /// Tokenize into arena i32 slots, wrap those MTL contents as MLXArray
    /// (fail if MLX copies off the arena), run BERT, copy last hidden into
    /// the activation rent, pool mean/CLS+L2 into the result rent.
    public func embedArena(
        modelPath: String,
        texts: [String],
        normalize: Bool,
        pooling: String = "mean",
        maxSeqLen: Int? = nil,
        slots: ArenaEmbedSlots
    ) async throws -> (dim: Int, count: Int) {
        let kind = try PoolingKind.parse(pooling)
        guard !texts.isEmpty else { throw EngineError.emptyEmbed }
        if texts.count > slots.maxBatch {
            throw EngineError.internalError(
                "batch \(texts.count) exceeds arena max_batch \(slots.maxBatch)"
            )
        }
        let container = try await loadEmbed(modelPath)
        let wordpiece = WordPieceHandle(vocab: wordpieceFront(for: modelPath))
        return try await container.perform { context -> (dim: Int, count: Int) in
            let timing = Engine.timingEnabled
            let t0 = timing ? DispatchTime.now().uptimeNanoseconds : 0
            let n = texts.count
            let stride = slots.maxSeq
            if !encodeWordPieceIntoArena(
                vocab: wordpiece.vocab, texts: texts, slots: slots, maxSeqLen: maxSeqLen)
            {
                let padId =
                    context.tokenizer.convertTokenToId("[PAD]")
                    ?? context.tokenizer.convertTokenToId("<pad>")
                    ?? 0
                let sepId =
                    context.tokenizer.convertTokenToId("[SEP]")
                    ?? context.tokenizer.convertTokenToId("</s>")
                var encoded = texts.map {
                    context.tokenizer.encode(text: $0, addSpecialTokens: true)
                }
                if let maxSeqLen {
                    encoded = encoded.map { truncateEncoderIds($0, max: maxSeqLen, sepId: sepId) }
                }
                let seq = encoded.map(\.count).max() ?? 0
                if seq > slots.maxSeq {
                    throw EngineError.internalError(
                        "seq \(seq) exceeds arena max_seq \(slots.maxSeq)"
                    )
                }
                for i in 0..<n {
                    let ids = encoded[i]
                    for t in 0..<stride {
                        let idx = i * stride + t
                        if t < ids.count {
                            slots.inputIds[idx] = Int32(ids[t])
                            slots.attentionMask[idx] = 1
                        } else {
                            slots.inputIds[idx] = Int32(padId)
                            slots.attentionMask[idx] = 0
                        }
                        slots.tokenTypes[idx] = 0
                    }
                }
            }
            let seq = stride
            let tTok = timing ? DispatchTime.now().uptimeNanoseconds : 0
            let wraps = try self.tokenWraps(for: slots)
            let padded = wraps.ids[0..<n, 0..<seq]
            let tokenMaskI32 = wraps.mask[0..<n, 0..<seq]
            let tokenTypes = wraps.types[0..<n, 0..<seq]
            let tokenMask = tokenMaskI32 .!= Int32(0)
            let output = context.model(
                padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: tokenMask)
            guard let hidden = output.hiddenStates else {
                throw EngineError.internalError("BERT returned no hidden states")
            }
            eval(hidden)
            let tFwd = timing ? DispatchTime.now().uptimeNanoseconds : 0
            let dims = hidden.shape
            guard dims.count == 3, dims[0] == n, dims[2] > 0, dims[2] <= slots.hiddenCap else {
                throw EngineError.internalError("hidden shape \(dims) is not [n, seq, hidden]")
            }
            let hid = dims[2]
            let hiddenSeq = dims[1]
            try copyHiddenToArena(hidden, dest: slots.activations, n: n, seq: hiddenSeq, hidden: hid)
            let tCopy = timing ? DispatchTime.now().uptimeNanoseconds : 0
            poolArena(
                hidden: slots.activations,
                mask: slots.attentionMask,
                n: n,
                seq: hiddenSeq,
                hidden: hid,
                tokenStride: stride,
                kind: kind,
                normalize: normalize,
                out: slots.results
            )
            if timing {
                let tPool = DispatchTime.now().uptimeNanoseconds
                fputs(
                    "[turboembed-timing] n=\(n) tokenize=\((tTok - t0) / 1000)us "
                        + "forward_eval=\((tFwd - tTok) / 1000)us "
                        + "hidden_copy=\((tCopy - tFwd) / 1000)us "
                        + "host_pool=\((tPool - tCopy) / 1000)us "
                        + "total=\((tPool - t0) / 1000)us\n",
                    stderr)
            }
            return (hid, n)
        }
    }

    /// Loaded-once WordPiece vocab (or recorded unsupported) for one model
    /// directory. The load — file read, `tokenizer.json` DOM parse, hash
    /// build, config validation — runs at most once per engine per model,
    /// exactly like the Rust NVIDIA path's load-time `TokenFront`.
    private func wordpieceFront(for modelPath: String) -> OpaquePointer? {
        #if canImport(WordPieceC)
        if let cached = locked({ self.wordpieceFronts[modelPath] }) {
            switch cached {
            case .loaded(let vocab): return vocab
            case .unsupported: return nil
            }
        }
        var vocab: OpaquePointer?
        let st = modelPath.withCString { wordpiece_vocab_load_dir($0, &vocab) }
        if st == WORDPIECE_OK, let vocab {
            locked { self.wordpieceFronts[modelPath] = .loaded(vocab) }
            return vocab
        }
        locked { self.wordpieceFronts[modelPath] = .unsupported }
        return nil
        #else
        return nil
        #endif
    }

    /// Once-validated no-copy MLXArray wraps of the arena token slots
    /// (keyed by the input_ids base pointer, one arena per engine). The
    /// full wrap validation — make_buffer hit, eval, MTLBuffer backing ==
    /// arena pointer — runs on first use; later requests reuse the same
    /// leaf arrays over the same unified-memory buffers the tokenizer
    /// just wrote.
    private func tokenWraps(for slots: ArenaEmbedSlots) throws -> ArenaTokenWraps {
        let key = UnsafeRawPointer(slots.inputIds)
        if let cached = locked({ self.tokenWrapCache[key] }),
            cached.rows == slots.maxBatch, cached.cols == slots.maxSeq
        {
            return cached
        }
        let wraps = ArenaTokenWraps(
            basePtr: key,
            rows: slots.maxBatch,
            cols: slots.maxSeq,
            ids: try wrapArenaI32(
                slots.inputIds, rows: slots.maxBatch, cols: slots.maxSeq, what: "input_ids"),
            mask: try wrapArenaI32(
                slots.attentionMask, rows: slots.maxBatch, cols: slots.maxSeq,
                what: "attention_mask"),
            types: try wrapArenaI32(
                slots.tokenTypes, rows: slots.maxBatch, cols: slots.maxSeq,
                what: "token_type_ids")
        )
        locked { self.tokenWrapCache[key] = wraps }
        return wraps
    }

    public func generate(
        modelPath: String,
        prompt: String,
        maxTokens: Int,
        onToken: @escaping @Sendable (String) -> Void
    ) async throws -> GenerateStats {
        let container = try await loadLM(modelPath)
        return try await container.perform { context -> GenerateStats in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user(prompt)]))
            var params = GenerateParameters(maxTokens: maxTokens, temperature: 0.0)
            params.topP = 1.0
            let stream = try MLXLMCommon.generate(
                input: input, parameters: params, context: context)
            var count: UInt32 = 0
            var tps: Double = 0
            for await part in stream {
                if let chunk = part.chunk, !chunk.isEmpty {
                    count += 1
                    onToken(chunk)
                }
                if let info = part.info {
                    tps = info.tokensPerSecond
                    count = UInt32(info.generationTokenCount)
                }
            }
            return GenerateStats(tokens: count, decodeTps: tps)
        }
    }

    private func loadLM(_ path: String) async throws -> ModelContainer {
        if let hit = locked({ self.lms[path] }) {
            return hit
        }
        let url = URL(filePath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw EngineError.missingModel(path)
        }
        fputs("[mlx-engine] loading LM \(path) on \(Device.defaultDevice())\n", stderr)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: url, using: tokenizerLoader)
        locked { self.lms[path] = container }
        return container
    }

    private func loadEmbed(_ path: String) async throws -> EmbedderModelContainer {
        if let hit = locked({ self.embeds[path] }) {
            return hit
        }
        let url = URL(filePath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw EngineError.missingModel(path)
        }
        fputs("[mlx-engine] loading embedder \(path) on \(Device.defaultDevice())\n", stderr)
        let container = try await EmbedderModelFactory.shared.loadContainer(
            from: url, using: tokenizerLoader)
        locked { self.embeds[path] = container }
        return container
    }

    private func locked<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
}

/// Catalog family pooling. MiniLM / MPNet / E5 / GTE use **mean** of last
/// hidden states; BGE uses the first token (`[CLS]`) hidden state — not the
/// BERT NSP pooler (`tanh(dense(CLS))`). That pooler is why a missing
/// `1_Pooling/config.json` produced apple↔nvidia cosine ≈ 0.
enum PoolingKind {
    case mean
    case cls

    static func parse(_ raw: String) throws -> PoolingKind {
        switch raw.lowercased() {
        case "", "mean":
            return .mean
        case "cls", "first":
            return .cls
        default:
            throw EngineError.internalError(
                "unknown pooling \(raw); expected \"mean\" or \"cls\"")
        }
    }
}

/// HF BERT / MiniLM truncation: keep `[CLS] … [SEP]` inside `max`.
func truncateEncoderIds(_ ids: [Int], max: Int, sepId: Int?) -> [Int] {
    guard max > 0, ids.count > max else { return ids }
    var out = Array(ids.prefix(max))
    if let sepId {
        out[max - 1] = sepId
    }
    return out
}

/// Attention-mask-weighted mean or first-token CLS, then optional L2.
/// Never applies an extra LayerNorm — sentence-transformers MiniLM / BGE
/// do not; `applyLayerNorm: true` was scrambling the 384-d space.
func poolHidden(
    _ output: EmbeddingModelOutput,
    mask: MLXArray,
    kind: PoolingKind,
    normalize: Bool
) -> MLXArray {
    guard let hidden = output.hiddenStates else {
        return output.pooledOutput ?? MLXArray([])
    }
    let weights = mask.asType(hidden.dtype)
    let pooled: MLXArray
    switch kind {
    case .mean:
        let weighted = hidden * weights.expandedDimensions(axes: [-1])
        pooled = sum(weighted, axis: 1) / sum(weights, axis: -1, keepDims: true)
    case .cls:
        pooled = hidden[0..., 0, 0...]
    }
    if normalize {
        let squares = pooled * pooled
        let norm = sqrt(sum(squares, axis: -1, keepDims: true))
        return pooled / maximum(norm, MLXArray(Float(1e-12)))
    }
    return pooled
}

/// Wrap an arena MTL contents pointer as MLXArray. Fail if MLX copies
/// onto a private buffer (`make_buffer` miss → malloc+memcpy).
func wrapArenaI32(
    _ ptr: UnsafeMutablePointer<Int32>,
    rows: Int,
    cols: Int,
    what: String
) throws -> MLXArray {
    let expected = UnsafeRawPointer(ptr)
    let gate = WrapGate()
    let array = MLXArray(rawPointer: UnsafeMutableRawPointer(ptr), [rows, cols], dtype: .int32) {
        if !gate.live {
            gate.stolen = true
        }
    }
    gate.live = true
    if gate.stolen {
        throw EngineError.internalError(
            "FAKE: MLX copied \(what) off the turbo_buffer arena (make_buffer miss) — refusing private Metal alloc"
        )
    }
    eval(array)
    guard let device = sharedMtlDevice else {
        throw EngineError.internalError("MTLCreateSystemDefaultDevice failed")
    }
    guard let buf = array.asMTLBuffer(device: device, noCopy: true) else {
        throw EngineError.internalError(
            "FAKE: \(what) wrap has no no-copy MTLBuffer — arena path bypassed"
        )
    }
    if buf.contents() != expected {
        throw EngineError.internalError(
            "FAKE: \(what) MLX backing is not the arena pointer — refusing private MTL"
        )
    }
    return array
}

/// Copy the (already `eval`ed) last hidden state into the activation rent.
func copyHiddenToArena(
    _ hidden: MLXArray,
    dest: UnsafeMutablePointer<Float>,
    n: Int,
    seq: Int,
    hidden hid: Int
) throws {
    let need = n * seq * hid
    if let device = sharedMtlDevice,
        let buf = hidden.asMTLBuffer(device: device, noCopy: true)
    {
        dest.update(
            from: buf.contents().assumingMemoryBound(to: Float.self), count: need)
        return
    }
    let flat = hidden.asArray(Float.self)
    guard flat.count == need else {
        throw EngineError.internalError("hidden copy size \(flat.count) != \(need)")
    }
    dest.update(from: flat, count: need)
}

func poolArena(
    hidden: UnsafePointer<Float>,
    mask: UnsafePointer<Int32>,
    n: Int,
    seq: Int,
    hidden hid: Int,
    tokenStride: Int,
    kind: PoolingKind,
    normalize: Bool,
    out: UnsafeMutablePointer<Float>
) {
    for i in 0..<n {
        let dst = out.advanced(by: i * hid)
        switch kind {
        case .cls:
            let src = hidden.advanced(by: i * seq * hid)
            dst.update(from: src, count: hid)
        case .mean:
            var count = 0.0
            vDSP_vclr(dst, 1, vDSP_Length(hid))
            for t in 0..<seq {
                if mask[i * tokenStride + t] == 0 { continue }
                count += 1
                let src = hidden.advanced(by: (i * seq + t) * hid)
                vDSP_vadd(dst, 1, src, 1, dst, 1, vDSP_Length(hid))
            }
            if count > 0 {
                var inv = Float(1.0 / count)
                vDSP_vsmul(dst, 1, &inv, dst, 1, vDSP_Length(hid))
            }
        }
        if normalize {
            var sumSq = 0.0
            for h in 0..<hid {
                sumSq += Double(dst[h]) * Double(dst[h])
            }
            let norm = Float(max(sumSq.squareRoot(), 1e-12))
            for h in 0..<hid {
                dst[h] /= norm
            }
        }
    }
}

/// MiniLM-compatible WordPiece into arena i32 slots using an engine-cached
/// vocab (see `Engine.wordpieceFront(for:)`). Returns false when the model
/// dir had no vocab.txt / WordPiece tokenizer.json (SentencePiece etc. stay
/// on swift-transformers). Never loads tokenizer files itself — that is
/// engine-lifetime work, not per-request work.
private func encodeWordPieceIntoArena(
    vocab: OpaquePointer?,
    texts: [String],
    slots: ArenaEmbedSlots,
    maxSeqLen: Int?
) -> Bool {
    #if canImport(WordPieceC)
    guard let vocab else { return false }
    wordpiece_hot_alloc_counter_reset()
    let seq = UInt32(maxSeqLen ?? slots.maxSeq)
    let stride = UInt32(slots.maxSeq)
    if seq > stride { return false }
    for (i, text) in texts.enumerated() {
        let off = i * slots.maxSeq
        let rc = text.utf8.withContiguousStorageIfAvailable { buf -> Int32 in
            wordpiece_encode_sentence(
                vocab,
                buf.baseAddress.map { UnsafeRawPointer($0).assumingMemoryBound(to: CChar.self) },
                buf.count,
                slots.inputIds.advanced(by: off),
                slots.attentionMask.advanced(by: off),
                slots.tokenTypes.advanced(by: off),
                nil,
                seq,
                stride,
                4
            )
        }
        let code: Int32
        if let rc {
            code = rc
        } else {
            var bytes = Array(text.utf8)
            code = bytes.withUnsafeMutableBytes { raw in
                wordpiece_encode_sentence(
                    vocab,
                    raw.baseAddress?.assumingMemoryBound(to: CChar.self),
                    raw.count,
                    slots.inputIds.advanced(by: off),
                    slots.attentionMask.advanced(by: off),
                    slots.tokenTypes.advanced(by: off),
                    nil,
                    seq,
                    stride,
                    4
                )
            }
        }
        if code != WORDPIECE_OK { return false }
    }
    return wordpiece_hot_alloc_counter() == 0
    #else
    return false
    #endif
}

private final class WrapGate: @unchecked Sendable {
    var live = false
    var stolen = false
}
