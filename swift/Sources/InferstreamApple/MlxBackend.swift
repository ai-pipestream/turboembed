import Foundation
import InferstreamCore
import MlxEngine

/// In-process MLX backend. Embeddings via MLXEmbedders; generation via
/// mlx-swift-lm. No FFI, no Python.
final class MlxBackend: ModelBackend, Sendable {
    let name: String
    let config: ModelConfig
    let resolved: URL
    let engine: Engine
    let tokenizer: LocalTokenizer?
    private let dimLock = NSLock()
    private var cachedDim: Int = 0

    var id: String { "mlx" }
    var hasTokenizer: Bool { tokenizer != nil }

    init(
        name: String,
        config: ModelConfig,
        resolved: URL,
        engine: Engine,
        tokenizer: LocalTokenizer?
    ) {
        self.name = name
        self.config = config
        self.resolved = resolved
        self.engine = engine
        self.tokenizer = tokenizer
    }

    func modelReady() async -> Bool {
        (try? engine.ping()) != nil
    }

    func modelMetadata(name: String) async -> ModelMeta {
        dimLock.lock()
        let dim = cachedDim
        dimLock.unlock()
        return ModelMeta(
            platform: "mlx",
            versions: ["1"],
            embeddingDim: dim > 0 ? Int64(dim) : 0,
            properties: [
                "backend": "mlx",
                "model": config.path ?? "",
                "resolved": resolved.path,
                "engine": "mlx-swift",
            ]
        )
    }

    func infer(_ request: Inference_ModelInferRequest) async throws -> Inference_ModelInferResponse {
        if isGeneration(request) {
            var last = Inference_ModelInferResponse()
            try await inferStream(request) { chunk in last = chunk }
            return last
        }
        let texts = try utf8Batch(request, name: "text")
        let maxBatch = Int(config.maxBatchSize ?? 32)
        if texts.count > maxBatch {
            throw ServeError.invalid(
                "batch of \(texts.count) exceeds max_batch \(maxBatch); chunk upstream")
        }
        let normalize = boolParam(request, "normalize") ?? (config.normalize ?? true)
        let embed = try await engine.embed(
            modelPath: resolved.path, texts: texts, normalize: normalize)
        dimLock.lock()
        cachedDim = embed.dimensions
        dimLock.unlock()
        var values: [Float] = []
        values.reserveCapacity(texts.count * embed.dimensions)
        for row in embed.vectors { values.append(contentsOf: row) }
        let shape: [Int64] =
            texts.count == 1
            ? [Int64(embed.dimensions)]
            : [Int64(texts.count), Int64(embed.dimensions)]
        var response = Inference_ModelInferResponse()
        response.modelName = request.modelName
        response.modelVersion = request.modelVersion
        response.id = request.id
        var output = Inference_ModelInferResponse.InferOutputTensor()
        output.name = "embedding"
        output.datatype = OipDataType.fp32.rawValue
        output.shape = shape
        response.outputs = [output]
        response.rawOutputContents = [Tensor.packFP32(values)]
        return response
    }

    func inferStream(
        _ request: Inference_ModelInferRequest,
        write: @Sendable (Inference_ModelInferResponse) async throws -> Void
    ) async throws {
        if !isGeneration(request) {
            try await write(try await infer(request))
            return
        }
        let prompt = try (try? utf8Batch(request, name: "prompt")) ?? utf8Batch(request, name: "text")
        guard let text = prompt.first else {
            throw ServeError.invalid("generation request contained no prompt")
        }
        let maxTokens = Int(intParam(request, "max_tokens") ?? 256)
        let mailbox = TokenMailbox()
        async let generate: GenerateStats = {
            defer { mailbox.close() }
            return try await engine.generate(
                modelPath: resolved.path,
                prompt: text,
                maxTokens: maxTokens
            ) { token in
                mailbox.push(token)
            }
        }()
        while let token = await mailbox.next() {
            try await write(tokenChunk(request, token: token, isFinal: false, tps: nil))
        }
        let stats = try await generate
        try await write(tokenChunk(request, token: "", isFinal: true, tps: stats.decodeTps))
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
        throw ServeError.unavailable("mlx backend has no reranker")
    }

    private func tokenChunk(
        _ request: Inference_ModelInferRequest,
        token: String,
        isFinal: Bool,
        tps: Double?
    ) -> Inference_ModelInferResponse {
        var response = Inference_ModelInferResponse()
        response.modelName = request.modelName
        response.modelVersion = request.modelVersion
        response.id = request.id
        var finalParam = Inference_InferParameter()
        finalParam.boolParam = isFinal
        response.parameters["final"] = finalParam
        if let tps {
            var tpsParam = Inference_InferParameter()
            tpsParam.doubleParam = tps
            response.parameters["decode_tokens_per_second"] = tpsParam
        }
        var output = Inference_ModelInferResponse.InferOutputTensor()
        output.name = "token"
        output.datatype = OipDataType.bytes.rawValue
        output.shape = [1]
        response.outputs = [output]
        response.rawOutputContents = [Tensor.packUTF8([token])]
        return response
    }
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

/// Bridges the sync mlx-swift-lm token callback onto an async consumer.
private final class TokenMailbox: @unchecked Sendable {
    private let lock = NSLock()
    private var queue: [String] = []
    private var closed = false
    private var waiters: [CheckedContinuation<String?, Never>] = []

    func push(_ token: String) {
        lock.lock()
        if let waiter = waiters.first {
            waiters.removeFirst()
            lock.unlock()
            waiter.resume(returning: token)
            return
        }
        queue.append(token)
        lock.unlock()
    }

    func close() {
        lock.lock()
        closed = true
        let pending = waiters
        waiters.removeAll()
        lock.unlock()
        for waiter in pending { waiter.resume(returning: nil) }
    }

    func next() async -> String? {
        lock.lock()
        if !queue.isEmpty {
            let token = queue.removeFirst()
            lock.unlock()
            return token
        }
        if closed {
            lock.unlock()
            return nil
        }
        return await withCheckedContinuation { continuation in
            waiters.append(continuation)
            lock.unlock()
        }
    }
}
