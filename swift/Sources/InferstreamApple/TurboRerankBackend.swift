import Foundation
import InferstreamCore
import TurboRerank
#if canImport(TurboRerankC)
import TurboRerankC
#endif

/// Catalog MiniLM-L6 cross-encoder served through `include/turborerank.h`.
///
/// Scores are sigmoid(CLS logit) in input order. Missing Metal / weights
/// fail at engine create — never the word-overlap mock.
final class TurboRerankBackend: ModelBackend, Sendable {
    let name: String
    let config: ModelConfig
    let engine: TurboRerankEngine
    private let lock = NSLock()
    private let maxDocuments: Int

    var id: String { "turborerank" }
    var hasTokenizer: Bool { false }

    init(name: String, config: ModelConfig) throws {
        if name.caseInsensitiveCompare("ms-marco-minilm-l6") != .orderedSame
            && name.caseInsensitiveCompare("ms-marco-minilm-l6-v2") != .orderedSame
        {
            throw ServeError.invalid("TurboRerankBackend refuses non-CE alias \(name)")
        }
        if let device = config.device, device.lowercased() == "mock" {
            throw ServeError.unavailable(
                "catalog CE alias \(name) refuses device=mock; word-overlap is not MiniLM"
            )
        }
        let abiDevice = Self.device(from: config.device)
        let path = config.path
        do {
            self.engine = try TurboRerankEngine(device: abiDevice, configPath: path)
            try self.engine.load(alias: name)
        } catch {
            throw ServeError.unavailable(
                "TurboRerank C ABI failed to load \(name) on \(config.device ?? "auto"): \(error). Missing Metal or weights never fall back to word-overlap"
            )
        }
        self.name = name
        self.config = config
        self.maxDocuments = Int(config.maxBatchSize ?? 32)
    }

    func modelReady() async -> Bool { true }

    func modelMetadata(name: String) async -> ModelMeta {
        ModelMeta(
            platform: "turborerank",
            versions: ["1"],
            embeddingDim: 0,
            properties: [
                "backend": "turborerank",
                "alias": self.name,
                "engine": "turborerank-metal",
                "activation": "sigmoid",
                "path": config.path ?? "",
            ]
        )
    }

    func infer(_ request: Inference_ModelInferRequest) async throws -> Inference_ModelInferResponse {
        throw ServeError.unavailable(
            "catalog alias \(name) is a TurboRerank cross-encoder; use the inferstream.v1 Rerank RPC"
        )
    }

    func inferStream(
        _ request: Inference_ModelInferRequest,
        write: @Sendable (Inference_ModelInferResponse) async throws -> Void
    ) async throws {
        throw ServeError.unavailable(
            "catalog alias \(name) is a TurboRerank cross-encoder; use the inferstream.v1 Rerank RPC"
        )
    }

    func tokenize(_ texts: [String], options: TokenizeOptions) async throws -> [TokenEncoding] {
        throw ServeError.unavailable("turborerank backend has no tokenizer RPC; configure tokenizer_dir")
    }

    func detokenize(_ sequences: [[UInt32]], skipSpecialTokens: Bool) async throws -> [String] {
        throw ServeError.unavailable("turborerank backend has no tokenizer RPC; configure tokenizer_dir")
    }

    func rerank(query: String, documents: [String], rawScores: Bool) async throws -> [Float] {
        if documents.isEmpty {
            throw ServeError.invalid("documents must not be empty")
        }
        if documents.count > maxDocuments {
            throw ServeError.invalid(
                "rerank batch of \(documents.count) exceeds max_client_batch_size \(maxDocuments)"
            )
        }
        return try lock.withLock {
            do {
                #if canImport(TurboRerankC)
                return try engine.score(
                    query: query,
                    documents: documents,
                    activation: rawScores ? TURBORERANK_ACT_IDENTITY : TURBORERANK_ACT_SIGMOID,
                    maxLength: 0
                )
                #else
                throw ServeError.unavailable("TurboRerankC is not linked")
                #endif
            } catch {
                throw ServeError.unavailable("TurboRerank score \(name): \(error)")
            }
        }
    }

    private static func device(from raw: String?) -> turborerank_device {
        #if canImport(TurboRerankC)
        switch raw?.lowercased() {
        case nil, "", "auto": return TURBORERANK_DEVICE_AUTO
        case "cpu": return TURBORERANK_DEVICE_CPU
        case "metal", "gpu": return TURBORERANK_DEVICE_METAL
        default: return TURBORERANK_DEVICE_AUTO
        }
        #else
        return 0
        #endif
    }
}

func isCatalogCrossEncoder(_ name: String) -> Bool {
    let n = name.lowercased()
    return n == "ms-marco-minilm-l6" || n == "ms-marco-minilm-l6-v2"
}
