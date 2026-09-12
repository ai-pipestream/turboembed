import Foundation
import InferstreamCore
import MlxEngine

enum ServeError: Error, LocalizedError, Sendable {
    case notFound(String)
    case invalid(String)
    case unavailable(String)
    case internalError(String)

    var errorDescription: String? {
        switch self {
        case .notFound(let m), .invalid(let m), .unavailable(let m), .internalError(let m):
            m
        }
    }
}

protocol ModelBackend: Sendable {
    var id: String { get }
    var hasTokenizer: Bool { get }
    func modelReady() async -> Bool
    func modelMetadata(name: String) async -> ModelMeta
    func infer(_ request: Inference_ModelInferRequest) async throws -> Inference_ModelInferResponse
    func inferStream(
        _ request: Inference_ModelInferRequest,
        write: @Sendable (Inference_ModelInferResponse) async throws -> Void
    ) async throws
    func tokenize(_ texts: [String], options: TokenizeOptions) async throws -> [TokenEncoding]
    func detokenize(_ sequences: [[UInt32]], skipSpecialTokens: Bool) async throws -> [String]
    func rerank(query: String, documents: [String]) async throws -> [Float]
}

struct ModelMeta: Sendable {
    var platform: String
    var versions: [String]
    var embeddingDim: Int64
    var properties: [String: String]
}

final class Registry: Sendable {
    let models: [String: any ModelBackend]

    init(models: [String: any ModelBackend]) {
        self.models = models
    }

    var names: [String] { models.keys.sorted() }
    var isEmpty: Bool { models.isEmpty }

    func lookup(_ name: String) -> (any ModelBackend)? { models[name] }

    func require(_ name: String) throws -> any ModelBackend {
        guard let backend = models[name] else {
            throw ServeError.notFound("model \(name) is not configured")
        }
        return backend
    }

    static func build(config: ServerConfig, engine: Engine, tokenizers: TokenizerMap) throws
        -> Registry
    {
        var map: [String: any ModelBackend] = [:]
        let mock = MockBackend()
        for model in config.models {
            switch model.backend {
            case .mock:
                map[model.name] = mock
            case .mlx:
                guard let path = model.path, !path.isEmpty else {
                    throw ServeError.invalid("mlx model \(model.name) requires path")
                }
                let resolved = Paths.resolveExistingDirectory(path, configURL: config.configURL)
                map[model.name] = MlxBackend(
                    name: model.name,
                    config: model,
                    resolved: resolved,
                    engine: engine,
                    tokenizer: tokenizers.get(model.name)
                )
            default:
                throw ServeError.invalid(
                    "backend \(model.backend.rawValue) is not an Apple-arch engine; use inferstream-nvidia or inferstream-intel"
                )
            }
        }
        return Registry(models: map)
    }
}
