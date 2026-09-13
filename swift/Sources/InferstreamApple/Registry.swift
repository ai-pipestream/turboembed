import Foundation
import InferstreamCore
import MlxEngine
#if canImport(TurboEmbedC)
import TurboEmbedC
#endif
#if canImport(TurboRerankC)
import TurboRerankC
#endif

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
        var turboGate: TurboEmbedGate?
        for model in config.models {
            switch model.backend {
            case .mock:
                if isCatalogCrossEncoder(model.name) {
                    throw ServeError.invalid(
                        "catalog CE alias \(model.name) refuses backend=mock; word-overlap is not MiniLM. Use backend = \"turborerank\""
                    )
                }
                map[model.name] = mock
            case .turborerank:
                do {
                    map[model.name] = try TurboRerankBackend(name: model.name, config: model)
                } catch let error as ServeError {
                    throw error
                } catch {
                    throw ServeError.unavailable(
                        "TurboRerank load(\(model.name)) failed: \(error). Catalog CE aliases never use word-overlap"
                    )
                }
            case .mlx:
                guard let path = model.path, !path.isEmpty else {
                    throw ServeError.invalid("mlx model \(model.name) requires path")
                }
                if isCatalogEmbed(model) {
                    if turboGate == nil {
                        do {
                            turboGate = try TurboEmbedGate(device: TURBOEMBED_DEVICE_AUTO)
                        } catch {
                            throw ServeError.unavailable(
                                "TurboEmbed AUTO/Metal create failed for catalog embed \(model.name): \(error). Missing Metal never falls back to mock or CPU"
                            )
                        }
                    }
                    guard let gate = turboGate else {
                        throw ServeError.internalError("TurboEmbed gate missing after create")
                    }
                    do {
                        try gate.load(alias: model.name)
                    } catch {
                        throw ServeError.unavailable(
                            "TurboEmbed load(\(model.name)) failed: \(error). Catalog aliases never use the 8-d mock"
                        )
                    }
                    map[model.name] = TurboEmbedBackend(
                        name: model.name,
                        config: model,
                        gate: gate,
                        tokenizer: tokenizers.get(model.name),
                        dim: model.name == "minilm" || model.name == "minilm-l12"
                            || model.name.hasPrefix("bge-small") || model.name.hasPrefix("e5-small")
                            || model.name.hasPrefix("gte-small") ? 384 : 0
                    )
                } else {
                    let resolved = Paths.resolveExistingDirectory(path, configURL: config.configURL)
                    map[model.name] = MlxBackend(
                        name: model.name,
                        config: model,
                        resolved: resolved,
                        engine: engine,
                        tokenizer: tokenizers.get(model.name)
                    )
                }
            default:
                throw ServeError.invalid(
                    "backend \(model.backend.rawValue) is not an Apple-arch engine; use inferstream-nvidia or inferstream-intel"
                )
            }
        }
        return Registry(models: map)
    }
}
