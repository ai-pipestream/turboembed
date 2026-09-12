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

    public func embed(modelPath: String, texts: [String], normalize: Bool) async throws -> EmbedResult
    {
        let container = try await loadEmbed(modelPath)
        return try await container.perform { context -> EmbedResult in
            let encoded = texts.map {
                context.tokenizer.encode(text: $0, addSpecialTokens: true)
            }
            let maxLength = encoded.map(\.count).max() ?? 0
            let padId = context.tokenizer.eosTokenId ?? 0
            let padded = stacked(
                encoded.map { ids in
                    MLXArray(ids + Array(repeating: padId, count: maxLength - ids.count))
                })
            let mask = padded .!= padId
            let tokenTypes = MLXArray.zeros(like: padded)
            let output = context.model(
                padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask)
            let pooled = context.pooling(output, normalize: normalize, applyLayerNorm: true)
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
