import Foundation
import MLX
import MLXEmbedders
import MLXLLM
import MLXLMCommon

/// In-process native MLX engine. Weights stay in unified memory; Metal is
/// the default device. Linked into inferstream-apple via the C ABI below.
final class Engine: @unchecked Sendable {
    private let lock = NSLock()
    private var lms: [String: ModelContainer] = [:]
    private var embeds: [String: EmbedderModelContainer] = [:]
    private let tokenizerLoader = HFTokenizerLoader()

    init() {
        _ = Device.gpu
    }

    func ping() throws -> [String: Any] {
        let a = MLXRandom.normal([64, 64])
        let b = MLXRandom.normal([64, 64])
        let c = (a.matmul(b)).sum()
        eval(c)
        let value = c.item(Float.self)
        let device = "\(Device.defaultDevice())"
        let metal = device.contains("gpu")
        var result: [String: Any] = [
            "mlx_version": "mlx-swift",
            "matmul_ok": value == value,
            "device": device,
            "metal_available": metal,
        ]
        if metal {
            result["active_memory"] = Memory.activeMemory
            result["peak_memory"] = Memory.peakMemory
        }
        return result
    }

    func embed(modelPath: String, texts: [String], normalize: Bool) throws -> (Int, [Float]) {
        try runBlocking {
            let container = try await self.loadEmbed(modelPath)
            return try await container.perform { context -> (Int, [Float]) in
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
                return (dim, rows.flatMap { $0 })
            }
        }
    }

    func generate(
        modelPath: String,
        prompt: String,
        maxTokens: Int,
        onToken: @escaping (String) -> Void
    ) throws -> (UInt32, Double) {
        let sink = TokenSink(onToken)
        return try runBlocking {
            let container = try await self.loadLM(modelPath)
            return try await container.perform { context -> (UInt32, Double) in
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
                        sink.send(chunk)
                    }
                    if let info = part.info {
                        tps = info.tokensPerSecond
                        count = UInt32(info.generationTokenCount)
                    }
                }
                return (count, tps)
            }
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

enum EngineError: Error, LocalizedError {
    case missingModel(String)
    var errorDescription: String? {
        switch self {
        case .missingModel(let path):
            "MLX model directory not found at \(path) — run `cargo xtask fetch --mlx` (or make fetch-mlx)"
        }
    }
}

final class TokenSink: @unchecked Sendable {
    private let impl: (String) -> Void
    init(_ impl: @escaping (String) -> Void) { self.impl = impl }
    func send(_ token: String) { impl(token) }
}

final class BlockingBox<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Result<T, any Error>?
    func set(_ result: Result<T, any Error>) {
        lock.lock()
        value = result
        lock.unlock()
    }
    func get() throws -> T {
        lock.lock()
        defer { lock.unlock() }
        switch value {
        case .success(let v): return v
        case .failure(let e): throw e
        case .none: throw EngineError.missingModel("blocking box empty")
        }
    }
}

func runBlocking<T: Sendable>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
    let box = BlockingBox<T>()
    let sem = DispatchSemaphore(value: 0)
    Task.detached {
        do {
            box.set(.success(try await body()))
        } catch {
            box.set(.failure(error))
        }
        sem.signal()
    }
    sem.wait()
    return try box.get()
}

func strdupError(_ message: String) -> UnsafeMutablePointer<CChar> {
    strdup(message) ?? strdup("mlx-engine error")!
}

func strdupJSON(_ obj: [String: Any]) throws -> UnsafeMutablePointer<CChar> {
    let data = try JSONSerialization.data(withJSONObject: obj, options: [])
    let text = String(data: data, encoding: .utf8) ?? "{}"
    return strdup(text)!
}

@_cdecl("mlx_engine_create")
public func mlx_engine_create(_ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?)
    -> OpaquePointer?
{
    do {
        _ = try runBlocking {
            let engine = Engine()
            _ = try engine.ping()
            return engine
        }
        let engine = Engine()
        return OpaquePointer(Unmanaged.passRetained(engine).toOpaque())
    } catch {
        err?.pointee = strdupError(String(describing: error))
        return nil
    }
}

@_cdecl("mlx_engine_destroy")
public func mlx_engine_destroy(_ engine: OpaquePointer?) {
    guard let engine else { return }
    Unmanaged<Engine>.fromOpaque(UnsafeRawPointer(engine)).release()
}

@_cdecl("mlx_engine_ping")
public func mlx_engine_ping(
    _ engine: OpaquePointer?,
    _ outJSON: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let engine else {
        err?.pointee = strdupError("null engine")
        return 1
    }
    let obj = Unmanaged<Engine>.fromOpaque(UnsafeRawPointer(engine)).takeUnretainedValue()
    do {
        let result = try obj.ping()
        outJSON.pointee = try strdupJSON(result)
        return 0
    } catch {
        err?.pointee = strdupError(String(describing: error))
        return 1
    }
}

@_cdecl("mlx_engine_embed")
public func mlx_engine_embed(
    _ engine: OpaquePointer?,
    _ modelPath: UnsafePointer<CChar>?,
    _ texts: UnsafePointer<UnsafePointer<CChar>?>?,
    _ nTexts: Int,
    _ normalize: Int32,
    _ outVectors: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>,
    _ outRows: UnsafeMutablePointer<Int>,
    _ outDim: UnsafeMutablePointer<Int>,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let engine, let modelPath, let texts else {
        err?.pointee = strdupError("null argument")
        return 1
    }
    let obj = Unmanaged<Engine>.fromOpaque(UnsafeRawPointer(engine)).takeUnretainedValue()
    let path = String(cString: modelPath)
    var batch = [String]()
    batch.reserveCapacity(nTexts)
    for i in 0..<nTexts {
        guard let p = texts[i] else {
            err?.pointee = strdupError("null text")
            return 1
        }
        batch.append(String(cString: p))
    }
    do {
        let (dim, values) = try obj.embed(
            modelPath: path, texts: batch, normalize: normalize != 0)
        let buf = UnsafeMutablePointer<Float>.allocate(capacity: values.count)
        buf.initialize(from: values, count: values.count)
        outVectors.pointee = buf
        outRows.pointee = batch.count
        outDim.pointee = dim
        return 0
    } catch {
        err?.pointee = strdupError(String(describing: error))
        return 1
    }
}

@_cdecl("mlx_engine_generate")
public func mlx_engine_generate(
    _ engine: OpaquePointer?,
    _ modelPath: UnsafePointer<CChar>?,
    _ prompt: UnsafePointer<CChar>?,
    _ maxTokens: UInt32,
    _ onToken: mlx_token_cb?,
    _ user: UnsafeMutableRawPointer?,
    _ outTokens: UnsafeMutablePointer<UInt32>,
    _ outTps: UnsafeMutablePointer<Double>,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let engine, let modelPath, let prompt else {
        err?.pointee = strdupError("null argument")
        return 1
    }
    let obj = Unmanaged<Engine>.fromOpaque(UnsafeRawPointer(engine)).takeUnretainedValue()
    let path = String(cString: modelPath)
    let text = String(cString: prompt)
    do {
        let (count, tps) = try obj.generate(
            modelPath: path, prompt: text, maxTokens: Int(maxTokens)
        ) { token in
            if let onToken {
                token.withCString { onToken($0, user) }
            }
        }
        outTokens.pointee = count
        outTps.pointee = tps
        return 0
    } catch {
        err?.pointee = strdupError(String(describing: error))
        return 1
    }
}

public typealias mlx_token_cb = @convention(c) (UnsafePointer<CChar>?, UnsafeMutableRawPointer?) ->
    Void

@_cdecl("mlx_engine_free")
public func mlx_engine_free(_ ptr: UnsafeMutableRawPointer?) {
    ptr?.deallocate()
}

@_cdecl("mlx_engine_free_str")
public func mlx_engine_free_str(_ ptr: UnsafeMutablePointer<CChar>?) {
    if let ptr {
        free(ptr)
    }
}
