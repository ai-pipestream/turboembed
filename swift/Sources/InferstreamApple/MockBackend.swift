import Foundation
import InferstreamCore

/// Deterministic mock so `mock-embed` in apple.toml still answers.
final class MockBackend: ModelBackend, Sendable {
    var id: String { "mock" }
    var hasTokenizer: Bool { true }

    func modelReady() async -> Bool { true }

    func modelMetadata(name: String) async -> ModelMeta {
        ModelMeta(
            platform: "mock",
            versions: ["1"],
            embeddingDim: 8,
            properties: ["backend": "mock"]
        )
    }

    func infer(_ request: Inference_ModelInferRequest) async throws -> Inference_ModelInferResponse {
        let texts = try utf8Texts(request)
        var values: [Float] = []
        for (i, text) in texts.enumerated() {
            values.append(contentsOf: mockVector(text, salt: i))
        }
        var response = Inference_ModelInferResponse()
        response.modelName = request.modelName
        response.modelVersion = request.modelVersion
        response.id = request.id
        var output = Inference_ModelInferResponse.InferOutputTensor()
        output.name = "embedding"
        output.datatype = OipDataType.fp32.rawValue
        output.shape = texts.count == 1 ? [8] : [Int64(texts.count), 8]
        response.outputs = [output]
        response.rawOutputContents = [Tensor.packFP32(values)]
        return response
    }

    func inferStream(
        _ request: Inference_ModelInferRequest,
        write: @Sendable (Inference_ModelInferResponse) async throws -> Void
    ) async throws {
        try await write(try await infer(request))
    }

    func tokenize(_ texts: [String], options: TokenizeOptions) async throws -> [TokenEncoding] {
        texts.map { text in
            let ids = text.utf8.map { UInt32($0) }
            let tokens = text.map { String($0) }
            return TokenEncoding(
                inputIds: ids,
                attentionMask: [UInt32](repeating: 1, count: ids.count),
                tokens: tokens,
                offsets: [])
        }
    }

    func detokenize(_ sequences: [[UInt32]], skipSpecialTokens: Bool) async throws -> [String] {
        sequences.map { ids in
            String(decoding: ids.map { UInt8(truncatingIfNeeded: $0) }, as: UTF8.self)
        }
    }

    func rerank(query: String, documents: [String], rawScores _: Bool) async throws -> [Float] {
        documents.map { doc in
            Float(doc.contains(query) ? 1.0 : 0.1)
        }
    }

    private func utf8Texts(_ request: Inference_ModelInferRequest) throws -> [String] {
        guard let index = request.inputs.firstIndex(where: { $0.name == "text" }) else {
            throw ServeError.invalid("expected an input tensor named text")
        }
        guard request.rawInputContents.indices.contains(index) else {
            throw ServeError.invalid("input text must be sent via raw_input_contents")
        }
        return try Tensor.unpackUTF8(request.rawInputContents[index])
    }

    private func mockVector(_ text: String, salt: Int) -> [Float] {
        var seed = UInt64(text.utf8.reduce(0) { $0 &+ UInt64($1) } &+ UInt64(salt) &* 17)
        var out: [Float] = []
        var sumSquares: Float = 0
        for _ in 0..<8 {
            seed = seed &* 6364136223846793005 &+ 1
            let v = Float(seed % 1000) / 1000.0
            out.append(v)
            sumSquares += v * v
        }
        let norm = sqrt(sumSquares)
        if norm > 0 {
            for i in out.indices { out[i] /= norm }
        }
        return out
    }
}
