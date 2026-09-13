import Foundation
import GRPCCore
import InferstreamCore

/// `inferstream.v1.InferstreamService` — Tokenize / Detokenize / Embed /
/// EmbedStream / ListModels / Rerank. Same shape as the Rust extension
/// service. Embed is the TurboEmbed gRPC mapping (typed or packed bytes).
struct ExtensionService: Inferstream_V1_InferstreamService.SimpleServiceProtocol {
    let registry: Registry
    let tokenizers: TokenizerMap

    func tokenize(
        request: Inferstream_V1_TokenizeRequest,
        context: ServerContext
    ) async throws -> Inferstream_V1_TokenizeResponse {
        if request.texts.isEmpty {
            throw RPCError(code: .invalidArgument, message: "texts must not be empty")
        }
        let backend = try registry.require(request.modelName)
        let options = TokenizeOptions(
            addSpecialTokens: !request.noSpecialTokens,
            withOffsets: request.withOffsets,
            truncateTo: request.truncateTo > 0 ? Int(request.truncateTo) : nil,
            padToLongest: request.padToLongest
        )
        let encodings: [TokenEncoding]
        if let local = tokenizers.get(request.modelName) {
            encodings = try local.tokenize(request.texts, options: options)
        } else {
            encodings = try await backend.tokenize(request.texts, options: options)
        }
        var response = Inferstream_V1_TokenizeResponse()
        response.encodings = encodings.map { enc in
            var out = Inferstream_V1_Encoding()
            out.inputIds = enc.inputIds
            out.attentionMask = enc.attentionMask
            out.tokens = enc.tokens
            out.offsets = enc.offsets.map { pair in
                var off = Inferstream_V1_Offset()
                off.start = pair.0
                off.end = pair.1
                return off
            }
            return out
        }
        return response
    }

    func detokenize(
        request: Inferstream_V1_DetokenizeRequest,
        context: ServerContext
    ) async throws -> Inferstream_V1_DetokenizeResponse {
        if request.sequences.isEmpty {
            throw RPCError(code: .invalidArgument, message: "sequences must not be empty")
        }
        let backend = try registry.require(request.modelName)
        let sequences = request.sequences.map(\.ids)
        let texts: [String]
        if let local = tokenizers.get(request.modelName) {
            texts = try local.detokenize(sequences, skipSpecialTokens: request.skipSpecialTokens)
        } else {
            texts = try await backend.detokenize(
                sequences, skipSpecialTokens: request.skipSpecialTokens)
        }
        var response = Inferstream_V1_DetokenizeResponse()
        response.texts = texts
        return response
    }

    func embed(
        request: Inferstream_V1_EmbedRequest,
        context: ServerContext
    ) async throws -> Inferstream_V1_EmbedResponse {
        if request.texts.isEmpty {
            throw RPCError(code: .invalidArgument, message: "texts must not be empty")
        }
        let backend = try registry.require(request.modelName)
        var infer = Inference_ModelInferRequest()
        infer.modelName = request.modelName
        if !request.pooling.isEmpty {
            var p = Inference_InferParameter()
            p.stringParam = request.pooling
            infer.parameters["pooling"] = p
        }
        if request.hasNormalize {
            var p = Inference_InferParameter()
            p.boolParam = request.normalize
            infer.parameters["normalize"] = p
        }
        if request.truncateTo > 0 {
            var p = Inference_InferParameter()
            p.int64Param = Int64(request.truncateTo)
            infer.parameters["truncate"] = p
        }
        var input = Inference_ModelInferRequest.InferInputTensor()
        input.name = "text"
        input.datatype = OipDataType.bytes.rawValue
        input.shape = [Int64(request.texts.count)]
        infer.inputs = [input]
        infer.rawInputContents = [Tensor.packUTF8(request.texts)]
        let result: Inference_ModelInferResponse
        do {
            result = try await backend.infer(infer)
        } catch let error as ServeError {
            throw rpc(error)
        }
        guard let index = result.outputs.firstIndex(where: { $0.name == "embedding" }) else {
            throw RPCError(
                code: .internalError, message: "backend returned no output tensor named \"embedding\""
            )
        }
        let output = result.outputs[index]
        if output.datatype != OipDataType.fp32.rawValue {
            throw RPCError(
                code: .internalError,
                message: "output \"embedding\" must be FP32, backend returned \(output.datatype)")
        }
        guard result.rawOutputContents.indices.contains(index) else {
            throw RPCError(
                code: .internalError, message: "backend returned no raw content for \"embedding\"")
        }
        let values = try Tensor.unpackFP32(result.rawOutputContents[index])
        guard let dim64 = output.shape.last, dim64 > 0 else {
            throw RPCError(code: .internalError, message: "embedding output reported an empty shape")
        }
        let dim = Int(dim64)
        if values.count % dim != 0 {
            throw RPCError(
                code: .internalError,
                message: "embedding blob length \(values.count) is not a multiple of dim \(dim)")
        }
        var response = Inferstream_V1_EmbedResponse()
        response.dim = UInt32(dim)
        response.modelName = result.modelName
        response.modelVersion = result.modelVersion
        if request.outputFormat == .packedBytes {
            var blob = Data()
            blob.reserveCapacity(values.count * 4)
            for value in values {
                var le = value.bitPattern.littleEndian
                withUnsafeBytes(of: &le) { blob.append(contentsOf: $0) }
            }
            response.packedEmbeddings = blob
        } else {
            response.embeddings = values.chunks(of: dim).map { slice in
                var emb = Inferstream_V1_Embedding()
                emb.values = Array(slice)
                return emb
            }
        }
        return response
    }

    func embedStream(
        request: Inferstream_V1_EmbedRequest,
        response: RPCWriter<Inferstream_V1_EmbedChunk>,
        context: ServerContext
    ) async throws {
        let full = try await embed(request: request, context: context)
        if request.outputFormat == .packedBytes {
            let dim = Int(full.dim)
            let rowBytes = dim * 4
            let count = rowBytes == 0 ? 0 : full.packedEmbeddings.count / rowBytes
            for i in 0..<count {
                var chunk = Inferstream_V1_EmbedChunk()
                chunk.index = UInt32(i)
                let start = full.packedEmbeddings.startIndex + (i * rowBytes)
                let end = start + rowBytes
                chunk.packedRow = full.packedEmbeddings[start..<end]
                chunk.final = i + 1 == count
                try await response.write(chunk)
            }
            return
        }
        for (i, emb) in full.embeddings.enumerated() {
            var chunk = Inferstream_V1_EmbedChunk()
            chunk.index = UInt32(i)
            chunk.embedding = emb
            chunk.final = i + 1 == full.embeddings.count
            try await response.write(chunk)
        }
    }

    func listModels(
        request: Inferstream_V1_ListModelsRequest,
        context: ServerContext
    ) async throws -> Inferstream_V1_ListModelsResponse {
        var models: [Inferstream_V1_ModelInfo] = []
        for name in registry.names {
            guard let backend = registry.lookup(name) else { continue }
            let ready = await backend.modelReady()
            let meta = await backend.modelMetadata(name: name)
            var info = Inferstream_V1_ModelInfo()
            info.name = name
            info.backend = backend.id
            info.ready = ready
            info.platform = meta.platform
            info.versions = meta.versions
            info.embeddingDim = meta.embeddingDim
            info.hasTokenizer_p = tokenizers.contains(name) || backend.hasTokenizer
            models.append(info)
        }
        var response = Inferstream_V1_ListModelsResponse()
        response.models = models
        return response
    }

    func rerank(
        request: Inferstream_V1_RerankRequest,
        context: ServerContext
    ) async throws -> Inferstream_V1_RerankResponse {
        if request.documents.isEmpty {
            throw RPCError(code: .invalidArgument, message: "documents must not be empty")
        }
        let maxDocs = 32
        if request.documents.count > maxDocs {
            throw RPCError(
                code: .invalidArgument,
                message: "documents length \(request.documents.count) exceeds max_client_batch_size \(maxDocs)"
            )
        }
        let backend = try registry.require(request.modelName)
        let scores: [Float]
        do {
            scores = try await backend.rerank(query: request.query, documents: request.documents)
        } catch let error as ServeError {
            throw rpc(error)
        }
        var results = scores.enumerated().map { index, score in
            var row = Inferstream_V1_RerankResult()
            row.index = UInt32(index)
            row.score = score
            if request.returnDocuments {
                row.document = request.documents[index]
            }
            return row
        }
        results.sort { $0.score > $1.score }
        if request.topN > 0 {
            results = Array(results.prefix(Int(request.topN)))
        }
        var response = Inferstream_V1_RerankResponse()
        response.results = results
        return response
    }
}

extension Array {
    fileprivate func chunks(of size: Int) -> [ArraySlice<Element>] {
        stride(from: 0, to: count, by: size).map { start in
            self[start..<Swift.min(start + size, count)]
        }
    }
}
