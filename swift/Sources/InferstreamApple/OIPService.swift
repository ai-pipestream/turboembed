import Foundation
import GRPCCore
import InferstreamCore

/// KServe OIP V2 `GRPCInferenceService` plus Triton-shaped ModelStreamInfer.
struct OIPService: Inference_GRPCInferenceService.SimpleServiceProtocol {
    let registry: Registry

    func serverLive(
        request: Inference_ServerLiveRequest,
        context: ServerContext
    ) async throws -> Inference_ServerLiveResponse {
        var response = Inference_ServerLiveResponse()
        response.live = true
        return response
    }

    func serverReady(
        request: Inference_ServerReadyRequest,
        context: ServerContext
    ) async throws -> Inference_ServerReadyResponse {
        var response = Inference_ServerReadyResponse()
        response.ready = !registry.isEmpty
        return response
    }

    func modelReady(
        request: Inference_ModelReadyRequest,
        context: ServerContext
    ) async throws -> Inference_ModelReadyResponse {
        var response = Inference_ModelReadyResponse()
        if let backend = registry.lookup(request.name) {
            response.ready = await backend.modelReady()
        } else {
            response.ready = false
        }
        return response
    }

    func serverMetadata(
        request: Inference_ServerMetadataRequest,
        context: ServerContext
    ) async throws -> Inference_ServerMetadataResponse {
        var response = Inference_ServerMetadataResponse()
        response.name = "inferstream"
        response.version = "0.1.0"
        response.extensions = ["model_stream_infer", "inferstream.v1"]
        return response
    }

    func modelMetadata(
        request: Inference_ModelMetadataRequest,
        context: ServerContext
    ) async throws -> Inference_ModelMetadataResponse {
        let backend = try registry.require(request.name)
        let meta = await backend.modelMetadata(name: request.name)
        var response = Inference_ModelMetadataResponse()
        response.name = request.name
        response.versions = meta.versions
        response.platform = meta.platform
        var input = Inference_ModelMetadataResponse.TensorMetadata()
        input.name = "text"
        input.datatype = OipDataType.bytes.rawValue
        input.shape = [-1]
        var output = Inference_ModelMetadataResponse.TensorMetadata()
        output.name = "embedding"
        output.datatype = OipDataType.fp32.rawValue
        output.shape = [meta.embeddingDim > 0 ? meta.embeddingDim : -1]
        response.inputs = [input]
        response.outputs = [output]
        response.properties = meta.properties
        return response
    }

    func modelInfer(
        request: Inference_ModelInferRequest,
        context: ServerContext
    ) async throws -> Inference_ModelInferResponse {
        let backend = try registry.require(request.modelName)
        do {
            return try await backend.infer(request)
        } catch let error as ServeError {
            throw rpc(error)
        }
    }

    func modelStreamInfer(
        request: RPCAsyncSequence<Inference_ModelInferRequest, any Error>,
        response: RPCWriter<Inference_ModelStreamInferResponse>,
        context: ServerContext
    ) async throws {
        for try await inferRequest in request {
            do {
                let backend = try registry.require(inferRequest.modelName)
                try await backend.inferStream(inferRequest) { chunk in
                    var wrapped = Inference_ModelStreamInferResponse()
                    wrapped.inferResponse = chunk
                    try await response.write(wrapped)
                }
            } catch {
                var wrapped = Inference_ModelStreamInferResponse()
                wrapped.errorMessage = error.localizedDescription
                var skeleton = Inference_ModelInferResponse()
                skeleton.id = inferRequest.id
                skeleton.modelName = inferRequest.modelName
                wrapped.inferResponse = skeleton
                try await response.write(wrapped)
            }
        }
    }
}

func rpc(_ error: ServeError) -> RPCError {
    switch error {
    case .notFound(let m): RPCError(code: .notFound, message: m)
    case .invalid(let m): RPCError(code: .invalidArgument, message: m)
    case .unavailable(let m): RPCError(code: .unavailable, message: m)
    case .internalError(let m): RPCError(code: .internalError, message: m)
    }
}
