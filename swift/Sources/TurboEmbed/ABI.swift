#if canImport(TurboEmbedC)
import TurboEmbedC
#endif

import Foundation
import InferstreamCore
import MlxEngine

/// `@_cdecl` export of the frozen C ABI (`include/turboembed.h`).
///
/// On a Mac this dylib is the Apple implementation of the same symbols
/// the C++ stub provides on Linux. `TURBOEMBED_DEVICE_METAL` talks to
/// `MlxEngine` (FP MiniLM, hidden-state mean+L2 on Metal). `mock-embed`
/// stays for ABI smoke only. Do not also link `native/turboembed/src/stub.cpp`.

private final class EngineBox: @unchecked Sendable {
    let device: turboembed_device
    var mockLoaded: Bool
    var mlx: MlxEngine.Engine?
    var mlxDevice: String = ""
    var metalAvailable: Bool = false
    var catalog: Catalog?
    var mlxAliases: [String: MlxAlias] = [:]
    var lastError: String = "" {
        didSet { refreshErrorPtr() }
    }
    private var lastErrorPtr: UnsafeMutablePointer<CChar>?

    init(device: turboembed_device) {
        self.device = device
        // Mock/CPU only when those devices were selected. AUTO/METAL/CUDA
        // never start with a CPU/mock fallback.
        self.mockLoaded = isExplicitCpu(device) || device == TURBOEMBED_DEVICE_MOCK
        refreshErrorPtr()
    }

    deinit {
        lastErrorPtr.map { free($0) }
    }

    var errorCString: UnsafePointer<CChar> {
        if let lastErrorPtr {
            return UnsafePointer(lastErrorPtr)
        }
        return StaticNames.empty.ptr
    }

    private func refreshErrorPtr() {
        lastErrorPtr.map { free($0) }
        lastErrorPtr = lastError.isEmpty ? nil : strdup(lastError)
    }
}

/// Thread-local-ish create error. The ABI is not Sync on one engine;
/// this slot is locked for the null-engine `turboembed_last_error` path.
private final class TLS: @unchecked Sendable {
    static let shared = TLS()
    private let lock = NSLock()
    private var message = ""
    private var ptr: UnsafeMutablePointer<CChar>?

    var createError: String {
        get {
            lock.lock()
            defer { lock.unlock() }
            return message
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            message = newValue
            ptr.map { free($0) }
            ptr = newValue.isEmpty ? nil : strdup(newValue)
        }
    }

    var createErrorCString: UnsafePointer<CChar> {
        lock.lock()
        defer { lock.unlock() }
        if let ptr {
            return UnsafePointer(ptr)
        }
        return StaticNames.empty.ptr
    }
}

private let mockAlias = "mock-embed"
private let mockDim: UInt32 = 8

/// AUTO / METAL = host GPU (Metal). Missing GPU → create fails, never CPU.
private func wantsHostGpu(_ device: turboembed_device) -> Bool {
    device == TURBOEMBED_DEVICE_METAL || device == TURBOEMBED_DEVICE_AUTO
}

private func isExplicitCpu(_ device: turboembed_device) -> Bool {
    device == TURBOEMBED_DEVICE_CPU || device == TURBOEMBED_DEVICE_OPENVINO_CPU
}

/// CUDA / TensorRT / OpenVINO GPU|NPU are not on this dylib. Fail loud.
private func refuseForeignGpu(_ device: turboembed_device) -> String? {
    switch device {
    case TURBOEMBED_DEVICE_CUDA, TURBOEMBED_DEVICE_TENSORRT,
        TURBOEMBED_DEVICE_OPENVINO_GPU, TURBOEMBED_DEVICE_OPENVINO_NPU:
        return
            "requested \(cDeviceName(device)); this dylib is Metal-only — refusing CPU fallback"
    default:
        return nil
    }
}

private func cDeviceName(_ device: turboembed_device) -> String {
    String(cString: turboembed_device_name(device))
}

private func isMockAlias(_ ptr: UnsafePointer<CChar>?, _ len: Int) -> Bool {
    guard let ptr else { return false }
    let n = len == 0 ? strlen(ptr) : len
    let bytes = UnsafeBufferPointer(start: ptr, count: n)
    let name = String(decoding: bytes.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    return name == "mock-embed" || name == "mock"
}

private func fnv1a(_ ptr: UnsafePointer<CChar>?, _ len: Int) -> UInt64 {
    var h: UInt64 = 14_695_981_039_346_656_037
    guard let ptr, len > 0 else { return h }
    for i in 0..<len {
        h ^= UInt64(UInt8(bitPattern: ptr[i]))
        h &*= 1_099_511_628_211
    }
    return h
}

private func mockRow(ptr: UnsafePointer<CChar>?, len: Int, into out: UnsafeMutablePointer<Float>, dim: UInt32) {
    var h = fnv1a(ptr, len)
    var sumSq: Double = 0
    for i in 0..<Int(dim) {
        h ^= h >> 30
        h &*= 0xbf58_476d_1ce4_e5b9
        h ^= h >> 27
        h &*= 0x94d0_49bb_1331_11eb
        h ^= h >> 31
        let raw = Int32(h & 0xffff) - 32768
        let v = Float(raw) / 32768.0
        out[i] = v
        sumSq += Double(v) * Double(v)
    }
    if sumSq > 0 {
        let inv = Float(1.0 / sumSq.squareRoot())
        for i in 0..<Int(dim) {
            out[i] *= inv
        }
    }
}

@_cdecl("turboembed_abi_version")
public func turboembed_abi_version() -> UInt32 { 1 }

@_cdecl("turboembed_status_name")
public func turboembed_status_name(_ status: turboembed_status) -> UnsafePointer<CChar> {
    switch status {
    case TURBOEMBED_OK: return stringPtr("OK")
    case TURBOEMBED_ERR_INVALID_ARGUMENT: return stringPtr("INVALID_ARGUMENT")
    case TURBOEMBED_ERR_NOT_FOUND: return stringPtr("NOT_FOUND")
    case TURBOEMBED_ERR_NOT_IMPLEMENTED: return stringPtr("NOT_IMPLEMENTED")
    case TURBOEMBED_ERR_UNAVAILABLE: return stringPtr("UNAVAILABLE")
    case TURBOEMBED_ERR_INTERNAL: return stringPtr("INTERNAL")
    case TURBOEMBED_ERR_OUT_OF_MEMORY: return stringPtr("OUT_OF_MEMORY")
    case TURBOEMBED_ERR_UNSUPPORTED_DEVICE: return stringPtr("UNSUPPORTED_DEVICE")
    default: return stringPtr("UNKNOWN")
    }
}

@_cdecl("turboembed_device_name")
public func turboembed_device_name(_ device: turboembed_device) -> UnsafePointer<CChar> {
    switch device {
    case TURBOEMBED_DEVICE_AUTO: return stringPtr("auto")
    case TURBOEMBED_DEVICE_CPU: return stringPtr("cpu")
    case TURBOEMBED_DEVICE_CUDA: return stringPtr("cuda")
    case TURBOEMBED_DEVICE_TENSORRT: return stringPtr("tensorrt")
    case TURBOEMBED_DEVICE_OPENVINO_CPU: return stringPtr("openvino-cpu")
    case TURBOEMBED_DEVICE_OPENVINO_GPU: return stringPtr("openvino-gpu")
    case TURBOEMBED_DEVICE_OPENVINO_NPU: return stringPtr("openvino-npu")
    case TURBOEMBED_DEVICE_METAL: return stringPtr("metal")
    case TURBOEMBED_DEVICE_MOCK: return stringPtr("mock")
    default: return stringPtr("unknown")
    }
}

@_cdecl("turboembed_last_error")
public func turboembed_last_error(_ engine: OpaquePointer?) -> UnsafePointer<CChar> {
    if let engine, let box = bridge(engine) {
        return box.errorCString
    }
    return TLS.shared.createErrorCString
}

@_cdecl("turboembed_engine_create")
public func turboembed_engine_create(
    _ device: turboembed_device,
    _ configPath: UnsafePointer<CChar>?,
    _ out: UnsafeMutablePointer<OpaquePointer?>?
) -> turboembed_status {
    guard let out else {
        TLS.shared.createError = "out pointer is null"
        return TURBOEMBED_ERR_INVALID_ARGUMENT
    }
    out.pointee = nil
    if let refusal = refuseForeignGpu(device) {
        TLS.shared.createError = refusal
        return TURBOEMBED_ERR_UNSUPPORTED_DEVICE
    }
    MlxProvider.ensureWorkspaceRoot()
    let box = EngineBox(device: device)
    if let configPath {
        let path = String(cString: configPath)
        if !path.isEmpty {
            box.catalog = try? Catalog.load(path: path)
        }
    }
    if box.catalog == nil {
        box.catalog = try? Catalog.builtin()
    }
    if wantsHostGpu(device) {
        do {
            let (engine, ping) = try MlxProvider.pingOrThrow()
            try MlxProvider.requireMetal(ping)
            box.mlx = engine
            box.mlxDevice = ping.device
            box.metalAvailable = true
            box.mockLoaded = false
            box.mlxAliases = MlxProvider.discover(catalog: box.catalog)
            fputs(
                "[turboembed] mlx ping device=\(ping.device) metal=true matmul_ok=\(ping.matmulOk) — FP MiniLM path is live\n",
                stderr)
        } catch {
            TLS.shared.createError =
                "requested \(cDeviceName(device)); \(error) — refusing CPU fallback"
            return TURBOEMBED_ERR_UNAVAILABLE
        }
    }
    out.pointee = OpaquePointer(Unmanaged.passRetained(box).toOpaque())
    TLS.shared.createError = ""
    return TURBOEMBED_OK
}

@_cdecl("turboembed_engine_destroy")
public func turboembed_engine_destroy(_ engine: OpaquePointer?) {
    guard let engine else { return }
    Unmanaged<EngineBox>.fromOpaque(UnsafeRawPointer(engine)).release()
}

@_cdecl("turboembed_list_models")
public func turboembed_list_models(
    _ engine: OpaquePointer?,
    _ outInfos: UnsafeMutablePointer<UnsafeMutablePointer<turboembed_model_info>?>?,
    _ outCount: UnsafeMutablePointer<Int>?
) -> turboembed_status {
    guard let box = bridge(engine), let outInfos, let outCount else {
        return TURBOEMBED_ERR_INVALID_ARGUMENT
    }
    var rows: [(String, UInt32, turboembed_device, Int32)] = [
        (
            mockAlias, mockDim, TURBOEMBED_DEVICE_MOCK,
            box.mockLoaded ? 1 : 0
        )
    ]
    let mlxRows = box.mlxAliases.values.sorted { $0.alias < $1.alias }
    for model in mlxRows {
        rows.append(
            (
                model.alias,
                model.dim,
                TURBOEMBED_DEVICE_METAL,
                model.ready ? 1 : 0
            ))
    }
    let infos = UnsafeMutablePointer<turboembed_model_info>.allocate(capacity: rows.count)
    for (i, row) in rows.enumerated() {
        infos[i] = turboembed_model_info(
            alias: allocCString(row.0),
            dim: row.1,
            device: row.2,
            ready: row.3
        )
    }
    outInfos.pointee = infos
    outCount.pointee = rows.count
    box.lastError = ""
    return TURBOEMBED_OK
}

@_cdecl("turboembed_model_list_free")
public func turboembed_model_list_free(
    _ infos: UnsafeMutablePointer<turboembed_model_info>?,
    _ count: Int
) {
    guard let infos else { return }
    for i in 0..<count {
        if let p = infos[i].alias.ptr {
            UnsafeMutablePointer(mutating: p).deallocate()
        }
    }
    infos.deallocate()
}

@_cdecl("turboembed_load_model")
public func turboembed_load_model(
    _ engine: OpaquePointer?,
    _ alias: UnsafePointer<CChar>?,
    _ aliasLen: Int
) -> turboembed_status {
    guard let box = bridge(engine), let alias else {
        return TURBOEMBED_ERR_INVALID_ARGUMENT
    }
    if isMockAlias(alias, aliasLen) {
        if box.device == TURBOEMBED_DEVICE_MOCK || isExplicitCpu(box.device) {
            box.mockLoaded = true
            box.lastError = ""
            return TURBOEMBED_OK
        }
        box.lastError =
            "mock-embed is ABI smoke only; GPU/AUTO paths never serve the 8-d FNV mock"
        return TURBOEMBED_ERR_NOT_IMPLEMENTED
    }
    if box.device == TURBOEMBED_DEVICE_MOCK {
        box.lastError =
            "catalog aliases are not served by mock; select METAL/AUTO for MiniLM — mock is smoke-only, never a silent substitute for missing Metal"
        return TURBOEMBED_ERR_NOT_IMPLEMENTED
    }
    let name = aliasName(alias, aliasLen)
    if let mlx = box.mlx {
        guard var model = box.mlxAliases[name] ?? MlxProvider.resolve(alias: name, catalog: box.catalog)
        else {
            box.lastError = MlxProviderError.missingWeights(name).localizedDescription
            return TURBOEMBED_ERR_NOT_FOUND
        }
        do {
            let warm = try MlxProvider.embed(
                engine: mlx,
                model: model,
                texts: ["hello world"],
                pooling: model.pooling,
                normalize: true,
                maxSeqLen: model.maxSeqLen
            )
            if name == "minilm" && warm.dim != 384 {
                throw MlxProviderError.fakeDim(alias: name, dim: warm.dim)
            }
            model.dim = UInt32(warm.dim)
            model.ready = true
            box.mlxAliases[name] = model
            box.lastError = ""
            fputs(
                "[turboembed] loaded \(name) from \(model.path) dim=\(warm.dim) pooling=\(model.pooling) device=\(box.mlxDevice)\n",
                stderr)
            return TURBOEMBED_OK
        } catch {
            box.lastError = String(describing: error)
            return TURBOEMBED_ERR_INTERNAL
        }
    }
    if isExplicitCpu(box.device) {
        box.lastError =
            "catalog alias \(name) is not served on explicit CPU; select METAL/AUTO for MiniLM — refusing silent GPU"
        return TURBOEMBED_ERR_NOT_IMPLEMENTED
    }
    box.lastError =
        "catalog alias \(name) needs METAL/AUTO (MLX). CUDA/ORT/GenAI are not on this dylib."
    return TURBOEMBED_ERR_NOT_IMPLEMENTED
}

@_cdecl("turboembed_embed_one")
public func turboembed_embed_one(
    _ engine: OpaquePointer?,
    _ alias: UnsafePointer<CChar>?,
    _ aliasLen: Int,
    _ text: UnsafePointer<CChar>?,
    _ textLen: Int,
    _ opts: UnsafePointer<turboembed_embed_options>?,
    _ out: UnsafeMutablePointer<UnsafeMutablePointer<turboembed_embed_result>?>?
) -> turboembed_status {
    var view = turboembed_str(ptr: text, len: textLen)
    return embedImpl(engine, alias, aliasLen, &view, 1, opts, out)
}

@_cdecl("turboembed_embed")
public func turboembed_embed(
    _ engine: OpaquePointer?,
    _ alias: UnsafePointer<CChar>?,
    _ aliasLen: Int,
    _ texts: UnsafePointer<turboembed_str>?,
    _ nTexts: Int,
    _ opts: UnsafePointer<turboembed_embed_options>?,
    _ out: UnsafeMutablePointer<UnsafeMutablePointer<turboembed_embed_result>?>?
) -> turboembed_status {
    embedImpl(engine, alias, aliasLen, texts, nTexts, opts, out)
}

@_cdecl("turboembed_embed_stream")
public func turboembed_embed_stream(
    _ engine: OpaquePointer?,
    _ alias: UnsafePointer<CChar>?,
    _ aliasLen: Int,
    _ texts: UnsafePointer<turboembed_str>?,
    _ nTexts: Int,
    _ opts: UnsafePointer<turboembed_embed_options>?,
    _ cb: turboembed_stream_cb?,
    _ userData: UnsafeMutableRawPointer?,
    _ out: UnsafeMutablePointer<UnsafeMutablePointer<turboembed_embed_result>?>?
) -> turboembed_status {
    var local: UnsafeMutablePointer<turboembed_embed_result>?
    let st = embedImpl(engine, alias, aliasLen, texts, nTexts, opts, &local)
    guard st == TURBOEMBED_OK, let result = local else { return st }
    if let cb {
        for i in 0..<Int(result.pointee.count) {
            let row = result.pointee.values.advanced(by: i * Int(result.pointee.dim))
            cb(userData, UInt32(i), row, result.pointee.dim, i + 1 == Int(result.pointee.count) ? 1 : 0)
        }
    }
    if let out {
        out.pointee = result
    } else {
        turboembed_embed_result_free(result)
    }
    return TURBOEMBED_OK
}

@_cdecl("turboembed_embed_result_free")
public func turboembed_embed_result_free(_ result: UnsafeMutablePointer<turboembed_embed_result>?) {
    guard let result else { return }
    if let values = UnsafeMutablePointer(mutating: result.pointee.values) {
        values.deallocate()
    }
    result.deallocate()
}

@_cdecl("turboembed_buffer_free")
public func turboembed_buffer_free(_ ptr: UnsafeMutableRawPointer?) {
    ptr?.deallocate()
}

@_cdecl("turboembed_register_provider")
public func turboembed_register_provider(
    _ vtbl: UnsafePointer<turboembed_provider_vtbl>?
) -> turboembed_status {
    _ = vtbl
    TLS.shared.createError = "turboembed_register_provider is reserved for MLX / model2vec plugins"
    return TURBOEMBED_ERR_NOT_IMPLEMENTED
}

private func embedImpl(
    _ engine: OpaquePointer?,
    _ alias: UnsafePointer<CChar>?,
    _ aliasLen: Int,
    _ texts: UnsafePointer<turboembed_str>?,
    _ nTexts: Int,
    _ opts: UnsafePointer<turboembed_embed_options>?,
    _ out: UnsafeMutablePointer<UnsafeMutablePointer<turboembed_embed_result>?>?
) -> turboembed_status {
    _ = opts
    guard let box = bridge(engine), let out else {
        return TURBOEMBED_ERR_INVALID_ARGUMENT
    }
    out.pointee = nil
    guard let alias, let texts, nTexts > 0 else {
        box.lastError = "null embed argument"
        return TURBOEMBED_ERR_INVALID_ARGUMENT
    }
    if !isMockAlias(alias, aliasLen) {
        return embedMlx(box, alias, aliasLen, texts, nTexts, opts, out)
    }
    if box.device != TURBOEMBED_DEVICE_MOCK && !isExplicitCpu(box.device) {
        box.lastError = "mock-embed is ABI smoke only; refusing 8-d FNV on GPU/AUTO"
        return TURBOEMBED_ERR_NOT_IMPLEMENTED
    }
    guard box.mockLoaded else {
        box.lastError = "mock-embed is not loaded"
        return TURBOEMBED_ERR_NOT_FOUND
    }
    let nFloats = nTexts * Int(mockDim)
    let values = UnsafeMutablePointer<Float>.allocate(capacity: nFloats)
    for i in 0..<nTexts {
        let view = texts[i]
        mockRow(ptr: view.ptr, len: view.len, into: values.advanced(by: i * Int(mockDim)), dim: mockDim)
    }
    let result = UnsafeMutablePointer<turboembed_embed_result>.allocate(capacity: 1)
    result.pointee = turboembed_embed_result(
        dim: mockDim,
        count: UInt32(nTexts),
        values: UnsafePointer(values),
        packed: UnsafeRawPointer(values).assumingMemoryBound(to: UInt8.self),
        packed_len: nFloats * MemoryLayout<Float>.size
    )
    out.pointee = result
    box.lastError = ""
    return TURBOEMBED_OK
}

private func embedMlx(
    _ box: EngineBox,
    _ alias: UnsafePointer<CChar>?,
    _ aliasLen: Int,
    _ texts: UnsafePointer<turboembed_str>?,
    _ nTexts: Int,
    _ opts: UnsafePointer<turboembed_embed_options>?,
    _ out: UnsafeMutablePointer<UnsafeMutablePointer<turboembed_embed_result>?>
) -> turboembed_status {
    guard let mlx = box.mlx else {
        box.lastError =
            "catalog embed requires a Metal MLX engine; create with TURBOEMBED_DEVICE_METAL"
        return TURBOEMBED_ERR_NOT_IMPLEMENTED
    }
    let name = aliasName(alias, aliasLen)
    guard var model = box.mlxAliases[name] ?? MlxProvider.resolve(alias: name, catalog: box.catalog)
    else {
        box.lastError = MlxProviderError.missingWeights(name).localizedDescription
        return TURBOEMBED_ERR_NOT_FOUND
    }
    switch MlxProvider.poolingName(opts, fallback: model.pooling) {
    case .failure(let err):
        box.lastError = err.localizedDescription
        return TURBOEMBED_ERR_INVALID_ARGUMENT
    case .success(let pooling):
        var batch = [String]()
        batch.reserveCapacity(nTexts)
        for i in 0..<nTexts {
            let view = texts![i]
            batch.append(stringView(view.ptr, view.len))
        }
        do {
            let result = try MlxProvider.embed(
                engine: mlx,
                model: model,
                texts: batch,
                pooling: pooling,
                normalize: MlxProvider.normalize(opts),
                maxSeqLen: MlxProvider.truncate(opts, fallback: model.maxSeqLen)
            )
            if name == "minilm" && result.dim != 384 {
                throw MlxProviderError.fakeDim(alias: name, dim: result.dim)
            }
            model.dim = UInt32(result.dim)
            model.ready = true
            box.mlxAliases[name] = model
            let values = UnsafeMutablePointer<Float>.allocate(capacity: result.values.count)
            values.initialize(from: result.values, count: result.values.count)
            let packed = UnsafeMutablePointer<turboembed_embed_result>.allocate(capacity: 1)
            packed.pointee = turboembed_embed_result(
                dim: UInt32(result.dim),
                count: UInt32(batch.count),
                values: UnsafePointer(values),
                packed: UnsafeRawPointer(values).assumingMemoryBound(to: UInt8.self),
                packed_len: result.values.count * MemoryLayout<Float>.size
            )
            out.pointee = packed
            box.lastError = ""
            return TURBOEMBED_OK
        } catch {
            box.lastError = String(describing: error)
            return TURBOEMBED_ERR_INTERNAL
        }
    }
}

private func aliasName(_ ptr: UnsafePointer<CChar>?, _ len: Int) -> String {
    stringView(ptr, len == 0 && ptr != nil ? strlen(ptr!) : len)
}

private func stringView(_ ptr: UnsafePointer<CChar>?, _ len: Int) -> String {
    guard let ptr, len > 0 else { return "" }
    let bytes = UnsafeBufferPointer(start: ptr, count: len)
    return String(decoding: bytes.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

private func allocCString(_ value: String) -> turboembed_str {
    let n = value.utf8.count + 1
    let alias = UnsafeMutablePointer<CChar>.allocate(capacity: n)
    value.utf8CString.withUnsafeBufferPointer { src in
        alias.update(from: src.baseAddress!, count: n)
    }
    return turboembed_str(ptr: alias, len: value.utf8.count)
}

private func bridge(_ engine: OpaquePointer?) -> EngineBox? {
    guard let engine else { return nil }
    return Unmanaged<EngineBox>.fromOpaque(UnsafeRawPointer(engine)).takeUnretainedValue()
}

private final class StickyCString: @unchecked Sendable {
    let ptr: UnsafePointer<CChar>
    init(_ value: String) {
        self.ptr = UnsafePointer(strdup(value)!)
    }
}

private enum StaticNames {
    static let ok = StickyCString("OK")
    static let invalid = StickyCString("INVALID_ARGUMENT")
    static let notFound = StickyCString("NOT_FOUND")
    static let notImpl = StickyCString("NOT_IMPLEMENTED")
    static let unavailable = StickyCString("UNAVAILABLE")
    static let internalErr = StickyCString("INTERNAL")
    static let oom = StickyCString("OUT_OF_MEMORY")
    static let badDevice = StickyCString("UNSUPPORTED_DEVICE")
    static let unknown = StickyCString("UNKNOWN")
    static let auto = StickyCString("auto")
    static let cpu = StickyCString("cpu")
    static let cuda = StickyCString("cuda")
    static let tensorrt = StickyCString("tensorrt")
    static let ovCpu = StickyCString("openvino-cpu")
    static let ovGpu = StickyCString("openvino-gpu")
    static let ovNpu = StickyCString("openvino-npu")
    static let metal = StickyCString("metal")
    static let mock = StickyCString("mock")
    static let unknownDev = StickyCString("unknown")
    static let empty = StickyCString("")
}

private func stringPtr(_ value: String) -> UnsafePointer<CChar> {
    switch value {
    case "OK": return StaticNames.ok.ptr
    case "INVALID_ARGUMENT": return StaticNames.invalid.ptr
    case "NOT_FOUND": return StaticNames.notFound.ptr
    case "NOT_IMPLEMENTED": return StaticNames.notImpl.ptr
    case "UNAVAILABLE": return StaticNames.unavailable.ptr
    case "INTERNAL": return StaticNames.internalErr.ptr
    case "OUT_OF_MEMORY": return StaticNames.oom.ptr
    case "UNSUPPORTED_DEVICE": return StaticNames.badDevice.ptr
    case "UNKNOWN": return StaticNames.unknown.ptr
    case "auto": return StaticNames.auto.ptr
    case "cpu": return StaticNames.cpu.ptr
    case "cuda": return StaticNames.cuda.ptr
    case "tensorrt": return StaticNames.tensorrt.ptr
    case "openvino-cpu": return StaticNames.ovCpu.ptr
    case "openvino-gpu": return StaticNames.ovGpu.ptr
    case "openvino-npu": return StaticNames.ovNpu.ptr
    case "metal": return StaticNames.metal.ptr
    case "mock": return StaticNames.mock.ptr
    case "unknown": return StaticNames.unknownDev.ptr
    default:
        return StickyCString(value).ptr
    }
}
