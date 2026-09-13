import Foundation
import InferstreamCore
import Synchronization
import TurboEmbed
#if canImport(TurboEmbedC)
import TurboEmbedC
#endif

/// Shared TurboEmbed engine. The C ABI is not thread-safe on one engine.
final class TurboEmbedGate: @unchecked Sendable {
    let engine: TurboEmbed.Engine
    private let lock = NSLock()

    init(device: turboembed_device = TURBOEMBED_DEVICE_AUTO) throws {
        self.engine = try TurboEmbed.Engine(device: device)
    }

    func load(alias: String) throws {
        lock.lock()
        defer { lock.unlock() }
        try engine.load(alias: alias)
    }

    func embed(alias: String, texts: [String], options: TurboEmbed.EmbedOptions?) throws
        -> TurboEmbed.Embeddings
    {
        lock.lock()
        defer { lock.unlock() }
        return try engine.embed(alias: alias, texts: texts, options: options)
    }
}

/// Catalog embed alias served through `include/turboembed.h` (MLX Metal).
///
/// Generation / StreamInfer stays on [`MlxBackend`]. Missing Metal fails at
/// engine create — never the 8-d mock.
final class TurboEmbedBackend: ModelBackend, Sendable {
    let name: String
    let config: ModelConfig
    let gate: TurboEmbedGate
    let tokenizer: LocalTokenizer?
    private let cachedDim: Mutex<Int>

    var id: String { "turboembed" }
    var hasTokenizer: Bool { tokenizer != nil }

    init(
        name: String,
        config: ModelConfig,
        gate: TurboEmbedGate,
        tokenizer: LocalTokenizer?,
        dim: Int
    ) {
        self.name = name
        self.config = config
        self.gate = gate
        self.tokenizer = tokenizer
        self.cachedDim = Mutex(dim)
    }

    func modelReady() async -> Bool { true }

    func modelMetadata(name: String) async -> ModelMeta {
        let dim = cachedDim.withLock { $0 }
        return ModelMeta(
            platform: "turboembed",
            versions: ["1"],
            embeddingDim: dim > 0 ? Int64(dim) : 0,
            properties: [
                "backend": "turboembed",
                "alias": self.name,
                "engine": "turboembed-mlx",
                "path": config.path ?? "",
            ]
        )
    }

    func infer(_ request: Inference_ModelInferRequest) async throws -> Inference_ModelInferResponse {
        if isGeneration(request) {
            throw ServeError.invalid(
                "catalog embed alias \(name) is TurboEmbed-only; use an LLM alias for StreamInfer"
            )
        }
        let texts = try utf8Batch(request, name: "text")
        let maxBatch = Int(config.maxBatchSize ?? 32)
        if texts.count > maxBatch {
            throw ServeError.invalid(
                "batch of \(texts.count) exceeds max_batch \(maxBatch); chunk upstream")
        }
        let options = embedOptions(request, config: config)
        let embed: TurboEmbed.Embeddings
        do {
            embed = try gate.embed(alias: name, texts: texts, options: options)
        } catch {
            throw ServeError.unavailable("TurboEmbed embed \(name): \(error)")
        }
        if name == "minilm" && embed.dim != 384 {
            throw ServeError.internalError(
                "FAKE: minilm returned dim=\(embed.dim); MiniLM is 384-d. Mock is forbidden"
            )
        }
        if embed.dim == 8 {
            throw ServeError.internalError(
                "FAKE: catalog alias \(name) returned dim=8 (FNV mock)"
            )
        }
        cachedDim.withLock { $0 = embed.dim }
        let values = Array(embed.values)
        let shape: [Int64] =
            texts.count == 1
            ? [Int64(embed.dim)]
            : [Int64(texts.count), Int64(embed.dim)]
        var response = Inference_ModelInferResponse()
        response.modelName = request.modelName
        response.modelVersion = request.modelVersion
        response.id = request.id
        var output = Inference_ModelInferResponse.InferOutputTensor()
        output.name = "embedding"
        output.datatype = OipDataType.fp32.rawValue
        output.shape = shape
        response.outputs = [output]
        var blob = OutputScratch.shared.rentBytes(minCap: values.count * 4)
        Tensor.packFP32(values, into: &blob)
        response.rawOutputContents = [blob]
        OutputScratch.shared.recycleBytes(blob)
        return response
    }

    func inferStream(
        _ request: Inference_ModelInferRequest,
        write: @Sendable (Inference_ModelInferResponse) async throws -> Void
    ) async throws {
        try await write(try await infer(request))
    }

    func tokenize(_ texts: [String], options: TokenizeOptions) async throws -> [TokenEncoding] {
        guard let tokenizer else {
            throw ServeError.unavailable(
                "no tokenizer_dir for \(name); set tokenizer_dir to a tokenizer.json directory")
        }
        return try tokenizer.tokenize(texts, options: options)
    }

    func detokenize(_ sequences: [[UInt32]], skipSpecialTokens: Bool) async throws -> [String] {
        guard let tokenizer else {
            throw ServeError.unavailable(
                "no tokenizer_dir for \(name); set tokenizer_dir to a tokenizer.json directory")
        }
        return try tokenizer.detokenize(sequences, skipSpecialTokens: skipSpecialTokens)
    }

    func rerank(query: String, documents: [String]) async throws -> [Float] {
        throw ServeError.unavailable("turboembed backend has no reranker")
    }
}

func isCatalogEmbed(_ model: ModelConfig) -> Bool {
    if let pooling = model.pooling, !pooling.isEmpty {
        return true
    }
    return false
}

private func embedOptions(
    _ request: Inference_ModelInferRequest, config: ModelConfig
) -> TurboEmbed.EmbedOptions {
    let poolingName = stringParam(request, "pooling") ?? config.pooling ?? ""
    let pooling: turboembed_pooling
    switch poolingName.lowercased() {
    case "mean": pooling = TURBOEMBED_POOLING_MEAN
    case "cls": pooling = TURBOEMBED_POOLING_CLS
    case "last": pooling = TURBOEMBED_POOLING_LAST
    default: pooling = TURBOEMBED_POOLING_DEFAULT
    }
    let normalize = boolParam(request, "normalize") ?? config.normalize
    let truncate = intParam(request, "truncate").map { UInt32($0) }
        ?? config.maxSeqLen
    return TurboEmbed.EmbedOptions(
        pooling: pooling,
        normalize: normalize,
        truncateTo: truncate
    )
}

private func isGeneration(_ request: Inference_ModelInferRequest) -> Bool {
    request.parameters.keys.contains("max_tokens")
        || request.inputs.contains(where: { $0.name == "prompt" })
}

private func utf8Batch(_ request: Inference_ModelInferRequest, name: String) throws -> [String] {
    guard let index = request.inputs.firstIndex(where: { $0.name == name }) else {
        throw ServeError.invalid("expected an input tensor named \(name)")
    }
    let tensor = request.inputs[index]
    if tensor.datatype != OipDataType.bytes.rawValue {
        throw ServeError.invalid("input \(name) must be BYTES, got \(tensor.datatype)")
    }
    guard request.rawInputContents.indices.contains(index) else {
        throw ServeError.invalid("input \(name) must be sent via raw_input_contents")
    }
    let texts = try Tensor.unpackUTF8(request.rawInputContents[index])
    if texts.isEmpty {
        throw ServeError.invalid("input \(name) contained no elements")
    }
    return texts
}

private func stringParam(_ request: Inference_ModelInferRequest, _ name: String) -> String? {
    if case .stringParam(let v) = request.parameters[name]?.parameterChoice, !v.isEmpty {
        return v
    }
    let raw = request.parameters[name]?.stringParam ?? ""
    return raw.isEmpty ? nil : raw
}

private func boolParam(_ request: Inference_ModelInferRequest, _ name: String) -> Bool? {
    guard request.parameters[name]?.parameterChoice != nil else {
        return request.parameters[name].map(\.boolParam)
    }
    if case .boolParam(let v) = request.parameters[name]?.parameterChoice { return v }
    return nil
}

private func intParam(_ request: Inference_ModelInferRequest, _ name: String) -> Int64? {
    if case .int64Param(let v) = request.parameters[name]?.parameterChoice { return v }
    let p = request.parameters[name]
    if let p, p.int64Param != 0 { return p.int64Param }
    return nil
}
