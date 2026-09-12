import Foundation
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

/// In-process native MLX engine. Weights stay in unified memory; Metal is
/// the default device. Used directly by the Swift gRPC server — no C ABI.
public final class Engine: @unchecked Sendable {
    private let lock = NSLock()
    private var lms: [String: ModelContainer] = [:]
    private var embeds: [String: EmbedderModelContainer] = [:]
    private let tokenizerLoader = HFTokenizerLoader()

    public init() {
        _ = Device.gpu
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
