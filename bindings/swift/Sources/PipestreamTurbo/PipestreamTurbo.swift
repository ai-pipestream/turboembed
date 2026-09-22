// Pipestream Turbo for Swift.
//
// Handles are final classes that own one C handle each and release it in
// `deinit` (or `close()`); a child keeps its parent alive exactly as the C
// contract does, so releasing a parent first is safe. Every failing call
// throws `TurboError` with the graded status code, the 1-based field index
// of the offending descriptor field (0 when not about one field), and the
// library's message. Options mirror the C descriptors; `.model` and zero
// mean the bundle contract, and an option a device does not advertise is
// refused with `TURBO_E_UNSUPPORTED_OPTION` naming the field.

import CTurbo
import Foundation

/// A failed libturbo call.
public struct TurboError: Error, CustomStringConvertible {
    /// `TURBO_E_*` status code.
    public let code: Int32
    /// 1-based descriptor field index, or 0.
    public let field: UInt32
    /// The library's message.
    public let message: String

    /// Symbolic name of the code, for example `TURBO_E_BUSY`.
    public var statusName: String { String(cString: turbo_status_name(code)) }

    public var description: String {
        field == 0 ? "\(statusName): \(message)" : "\(statusName) (field \(field)): \(message)"
    }
}

/// Run a C call with a caller-owned error struct and throw on failure.
@discardableResult
func check(_ body: (UnsafeMutablePointer<turbo_error>) -> Int32) throws -> Int32 {
    var err = turbo_error()
    err.struct_size = UInt32(MemoryLayout<turbo_error>.size)
    let rc = withUnsafeMutablePointer(to: &err) { body($0) }
    if rc == TURBO_OK { return rc }
    let message = withUnsafePointer(to: &err.message) { p in
        p.withMemoryRebound(to: CChar.self, capacity: Int(TURBO_ERROR_MESSAGE_LEN)) { String(cString: $0) }
    }
    throw TurboError(code: rc, field: err.field, message: message)
}

/// Read a NUL-terminated fixed-size `char[]` tuple field.
func fixedString<T>(_ tuple: T) -> String {
    withUnsafePointer(to: tuple) { p in
        p.withMemoryRebound(to: CChar.self, capacity: MemoryLayout<T>.size) { String(cString: $0) }
    }
}

/// UTF-8 bytes of `strings`, kept alive while `body` runs, exposed as `turbo_text`.
func withTexts<R>(_ strings: [String], _ body: (UnsafePointer<turbo_text>, UInt32) throws -> R) rethrows -> R {
    var storage: [[UInt8]] = strings.map { Array($0.utf8) }
    var texts = [turbo_text](repeating: turbo_text(), count: strings.count)
    for i in 0..<strings.count {
        storage[i].withUnsafeMutableBufferPointer { buf in
            texts[i] = turbo_text(ptr: UnsafeRawPointer(buf.baseAddress ?? UnsafeMutablePointer<UInt8>.allocate(capacity: 1)).assumingMemoryBound(to: CChar.self), len: UInt64(buf.count))
        }
    }
    return try withExtendedLifetime(storage) {
        try texts.withUnsafeBufferPointer { try body($0.baseAddress!, UInt32(strings.count)) }
    }
}

func withText<R>(_ string: String, _ body: (turbo_text) throws -> R) rethrows -> R {
    var bytes = Array(string.utf8)
    if bytes.isEmpty { bytes = [0] }
    let len = UInt64(string.utf8.count)
    return try bytes.withUnsafeBufferPointer { buf in
        try body(turbo_text(ptr: UnsafeRawPointer(buf.baseAddress!).assumingMemoryBound(to: CChar.self), len: len))
    }
}

// MARK: - Enumerations

public enum DeviceKind: UInt32 { case cpu = 1, gpu = 2, igpu = 3, npu = 4, accel = 5 }
public enum SelectPolicy: UInt32 { case auto = 0, explicit = 1 }
public enum Task: UInt32 { case embed = 1, rerank = 2, classify = 3, tokenClassify = 4, generate = 5, tokenize = 6, run = 7, chunk = 8 }
public enum Modality: UInt32 { case text = 1, audio = 2, image = 3, video = 4 }
public enum CapStatus: UInt32 { case unsupported = 0, planned = 1, experimental = 2, supported = 3 }
public enum Truncate: UInt32 { case model = 0, none = 1, right = 2, left = 3 }
public enum PromptRole: UInt32 { case none = 0, query = 1, document = 2 }
public enum Normalize: UInt32 { case model = 0, none = 1, l2 = 2 }
public enum Pooling: UInt32 { case model = 0, mean = 1, cls = 2, last = 3 }
public enum OutputDType: UInt32 { case model = 0, f32 = 1, f16 = 2, i8 = 3 }
public enum Aggregation: UInt32 { case model = 0, none = 1, simple = 2, first = 3, max = 4 }
public enum Placement: UInt32 { case host = 1, pinned = 2, device = 3, shared = 4 }

// The Swift raw values above are checked against the C constants once, at
// first use, so a header change cannot silently skew them.
private let constantsVerified: Bool = {
    precondition(DeviceKind.cpu.rawValue == TURBO_DEVICE_CPU && DeviceKind.accel.rawValue == TURBO_DEVICE_ACCEL)
    precondition(SelectPolicy.auto.rawValue == TURBO_SELECT_AUTO && SelectPolicy.explicit.rawValue == TURBO_SELECT_EXPLICIT)
    precondition(Task.embed.rawValue == TURBO_TASK_EMBED && Task.chunk.rawValue == TURBO_TASK_CHUNK)
    precondition(Modality.text.rawValue == TURBO_MODALITY_TEXT && Modality.video.rawValue == TURBO_MODALITY_VIDEO)
    precondition(CapStatus.supported.rawValue == TURBO_CAP_SUPPORTED)
    precondition(Truncate.left.rawValue == TURBO_TRUNCATE_LEFT && PromptRole.document.rawValue == TURBO_PROMPT_DOCUMENT)
    precondition(Normalize.l2.rawValue == TURBO_NORMALIZE_L2 && Pooling.last.rawValue == TURBO_POOLING_LAST)
    precondition(OutputDType.i8.rawValue == TURBO_OUTPUT_I8 && Aggregation.max.rawValue == TURBO_AGGREGATE_MAX)
    precondition(Placement.shared.rawValue == TURBO_PLACE_SHARED)
    return true
}()

// MARK: - Runtime

/// Static description of one device.
public struct DeviceInfo {
    public let index: UInt32
    public let kind: DeviceKind
    public let ordinal: UInt32
    public let vendorId: UInt32
    /// `TURBO_CAP_*` bits.
    public let caps: UInt64
    public let memoryTotal: UInt64
    public let memoryFree: UInt64
    public let name: String
    public let vendor: String
    public let providerId: String
    public let providerVersion: String
    public let runtimeVersion: String
    public let driverVersion: String

    /// True when every bit of `bits` is advertised.
    public func has(_ bits: UInt64) -> Bool { caps & bits == bits }
}

/// One cell of the capability matrix.
public struct Capability {
    public let status: CapStatus
    public let cosineFloor: Float
    public let maxAbsError: Float
    public let deterministic: Bool
    public let notes: String
    public var offered: Bool { status != .unsupported }
}

/// The provider registry and device list.
public final class Runtime {
    let raw: OpaquePointer

    /// Create a runtime; `providerPaths` are provider libraries to load, and
    /// a failure to load any of them fails creation.
    public init(providerPaths: [String] = []) throws {
        _ = constantsVerified
        var out: OpaquePointer?
        _ = try withTexts(providerPaths) { texts, count in
            var desc = turbo_runtime_desc()
            desc.struct_size = UInt32(MemoryLayout<turbo_runtime_desc>.size)
            desc.n_provider_paths = count
            desc.provider_paths = count == 0 ? nil : texts
            try check { turbo_runtime_create(&desc, &out, $0) }
        }
        raw = out!
    }

    deinit { turbo_runtime_release(raw) }

    /// `TURBO_ABI_VERSION` of the library.
    public static var abiVersion: UInt32 { turbo_abi_version() }

    public func devices() throws -> [DeviceInfo] {
        var n: UInt32 = 0
        try check { turbo_runtime_device_count(raw, &n, $0) }
        return try (0..<n).map { try device($0) }
    }

    public func device(_ index: UInt32) throws -> DeviceInfo {
        var info = turbo_device_info()
        info.struct_size = UInt32(MemoryLayout<turbo_device_info>.size)
        try check { turbo_runtime_device_info(raw, index, &info, $0) }
        return DeviceInfo(
            index: index, kind: DeviceKind(rawValue: info.kind)!, ordinal: info.ordinal, vendorId: info.vendor_id,
            caps: info.caps, memoryTotal: info.memory_total, memoryFree: info.memory_free,
            name: fixedString(info.name), vendor: fixedString(info.vendor), providerId: fixedString(info.provider_id),
            providerVersion: fixedString(info.provider_version), runtimeVersion: fixedString(info.runtime_version),
            driverVersion: fixedString(info.driver_version))
    }

    /// Select a device. `.auto` never picks a CPU; an absent device throws
    /// `TURBO_E_DEVICE_NOT_FOUND`.
    public func selectDevice(policy: SelectPolicy = .auto, providerId: String = "", ordinal: UInt32 = 0) throws -> UInt32 {
        var out: UInt32 = 0
        try withText(providerId) { pid in
            try withText("") { vendor in
                var sel = turbo_device_selector()
                sel.struct_size = UInt32(MemoryLayout<turbo_device_selector>.size)
                sel.policy = policy.rawValue
                sel.ordinal = ordinal
                sel.provider_id = pid
                sel.vendor = vendor
                try check { turbo_runtime_select_device(raw, &sel, &out, $0) }
            }
        }
        return out
    }

    public func capability(device index: UInt32, task: Task, modality: Modality) throws -> Capability {
        var cap = turbo_capability()
        cap.struct_size = UInt32(MemoryLayout<turbo_capability>.size)
        try check { turbo_runtime_capability(raw, index, task.rawValue, modality.rawValue, &cap, $0) }
        return Capability(status: CapStatus(rawValue: cap.status)!, cosineFloor: cap.cosine_floor,
                          maxAbsError: cap.max_abs_error, deterministic: cap.deterministic != 0, notes: fixedString(cap.notes))
    }

    public func createContext(device index: UInt32) throws -> Context {
        try Context(runtime: self, device: index)
    }
}

// MARK: - Context and model

public final class Context {
    let raw: OpaquePointer
    public let deviceIndex: UInt32
    private let runtime: Runtime

    init(runtime: Runtime, device index: UInt32) throws {
        var out: OpaquePointer?
        var desc = turbo_context_desc()
        desc.struct_size = UInt32(MemoryLayout<turbo_context_desc>.size)
        try check { turbo_context_create(runtime.raw, index, &desc, &out, $0) }
        raw = out!
        deviceIndex = index
        self.runtime = runtime
    }

    deinit { turbo_context_release(raw) }

    public func loadModel(bundlePath: String) throws -> Model {
        try Model(context: self, bundlePath: bundlePath)
    }
}

/// What actually loaded.
public struct ModelInfo {
    public let task: Task
    public let dim: UInt32
    public let labels: [String]
    public let maxSeq: UInt32
    public let maxBatch: UInt32
    public let fullyAccelerated: Bool
    public let modelId: String
    public let providerId: String
}

public final class Model {
    let raw: OpaquePointer
    public let info: ModelInfo
    private let context: Context

    init(context: Context, bundlePath: String) throws {
        var out: OpaquePointer?
        try withText(bundlePath) { path in
            var desc = turbo_model_desc()
            desc.struct_size = UInt32(MemoryLayout<turbo_model_desc>.size)
            try check { turbo_model_load(context.raw, path, &desc, &out, $0) }
        }
        raw = out!
        self.context = context
        var mi = turbo_model_info()
        mi.struct_size = UInt32(MemoryLayout<turbo_model_info>.size)
        try check { turbo_model_get_info(out!, &mi, $0) }
        var labels: [String] = []
        for i in 0..<mi.n_labels {
            var t = turbo_text()
            try check { turbo_model_label(out!, i, &t, $0) }
            labels.append(t.len == 0 ? "" : String(decoding: UnsafeRawBufferPointer(start: t.ptr, count: Int(t.len)), as: UTF8.self))
        }
        info = ModelInfo(task: Task(rawValue: mi.task)!, dim: mi.dim, labels: labels, maxSeq: mi.max_seq, maxBatch: mi.max_batch,
                         fullyAccelerated: mi.fully_accelerated != 0, modelId: fixedString(mi.model_id), providerId: fixedString(mi.provider_id))
    }

    deinit { turbo_model_release(raw) }

    /// Create a session with fixed maxima; 0 means the model's default.
    public func createSession(maxBatch: UInt32 = 0, maxSeq: UInt32 = 0) throws -> Session {
        try Session(model: self, maxBatch: maxBatch, maxSeq: maxSeq)
    }
}

// MARK: - Options

public struct EmbedOptions {
    public var truncate: Truncate = .model
    public var maxTokens: UInt32 = 0
    public var promptRole: PromptRole = .none
    public var normalize: Normalize = .model
    public var pooling: Pooling = .model
    public var outputDim: UInt32 = 0
    public var outputDType: OutputDType = .model
    public init() {}

    var c: turbo_embed_options {
        var o = turbo_embed_options()
        o.struct_size = UInt32(MemoryLayout<turbo_embed_options>.size)
        o.truncate = truncate.rawValue; o.max_tokens = maxTokens; o.prompt_role = promptRole.rawValue
        o.normalize = normalize.rawValue; o.pooling = pooling.rawValue; o.output_dim = outputDim
        o.output_dtype = outputDType.rawValue
        return o
    }
}

public struct RerankOptions {
    public var truncate: Truncate = .model
    public var maxTokens: UInt32 = 0
    public var topN: UInt32 = 0
    public var returnSorted = false
    public var rawScores = false
    public init() {}

    var c: turbo_rerank_options {
        var o = turbo_rerank_options()
        o.struct_size = UInt32(MemoryLayout<turbo_rerank_options>.size)
        o.truncate = truncate.rawValue; o.max_tokens = maxTokens; o.top_n = topN
        o.return_sorted = returnSorted ? 1 : 0; o.raw_scores = rawScores ? 1 : 0
        return o
    }
}

public struct ClassifyOptions {
    public var truncate: Truncate = .model
    public var maxTokens: UInt32 = 0
    public var aggregation: Aggregation = .model
    public var rawScores = false
    public init() {}

    var c: turbo_classify_options {
        var o = turbo_classify_options()
        o.struct_size = UInt32(MemoryLayout<turbo_classify_options>.size)
        o.truncate = truncate.rawValue; o.max_tokens = maxTokens; o.aggregation = aggregation.rawValue
        o.raw_scores = rawScores ? 1 : 0
        return o
    }
}

// MARK: - Session and result

public final class Session {
    let raw: OpaquePointer
    public let model: Model

    init(model: Model, maxBatch: UInt32, maxSeq: UInt32) throws {
        var out: OpaquePointer?
        var desc = turbo_session_desc()
        desc.struct_size = UInt32(MemoryLayout<turbo_session_desc>.size)
        desc.max_batch = maxBatch
        desc.max_seq = maxSeq
        try check { turbo_session_create(model.raw, &desc, &out, $0) }
        raw = out!
        self.model = model
    }

    deinit { turbo_session_release(raw) }

    public func writeText(_ texts: [String], options: EmbedOptions = EmbedOptions()) throws {
        var o = options.c
        _ = try withTexts(texts) { t, n in try check { turbo_session_write_text(raw, t, n, &o, $0) } }
    }

    public func writePairs(query: String, documents: [String], options: RerankOptions = RerankOptions()) throws {
        var o = options.c
        try withText(query) { q in
            var qq = q
            _ = try withTexts(documents) { d, n in try check { turbo_session_write_pairs(raw, &qq, d, n, &o, $0) } }
        }
    }

    public func writeTextClassify(_ texts: [String], options: ClassifyOptions = ClassifyOptions()) throws {
        var o = options.c
        _ = try withTexts(texts) { t, n in try check { turbo_session_write_text_classify(raw, t, n, &o, $0) } }
    }

    /// Execute; the result leases this session until it is released.
    public func run() throws -> Result {
        var out: OpaquePointer?
        var o = turbo_run_options()
        o.struct_size = UInt32(MemoryLayout<turbo_run_options>.size)
        try check { turbo_session_run(raw, &o, &out, $0) }
        return Result(raw: out!, session: self)
    }

    public func stats() throws -> turbo_session_stats {
        var st = turbo_session_stats()
        st.struct_size = UInt32(MemoryLayout<turbo_session_stats>.size)
        try check { turbo_session_get_stats(raw, &st, $0) }
        return st
    }
}

/// One aggregated span from token classification.
public struct Span { public let row: UInt32; public let byteStart: UInt64; public let byteEnd: UInt64; public let label: UInt32; public let score: Float }

public final class Result {
    private var raw: OpaquePointer?
    private let session: Session

    init(raw: OpaquePointer, session: Session) { self.raw = raw; self.session = session }

    deinit { close() }

    /// Release the lease early (idempotent).
    public func close() {
        if let r = raw { turbo_result_release(r); raw = nil }
    }

    private func handle() -> OpaquePointer {
        guard let r = raw else { preconditionFailure("result is closed") }
        return r
    }

    public var outputCount: UInt32 {
        get throws {
            var info = turbo_result_info()
            info.struct_size = UInt32(MemoryLayout<turbo_result_info>.size)
            try check { turbo_result_get_info(handle(), &info, $0) }
            return info.n_outputs
        }
    }

    public var placement: Placement {
        get throws {
            var info = turbo_result_info()
            info.struct_size = UInt32(MemoryLayout<turbo_result_info>.size)
            try check { turbo_result_get_info(handle(), &info, $0) }
            return Placement(rawValue: info.placement)!
        }
    }

    /// Logical shape and name of output `index`.
    public func output(_ index: UInt32) throws -> (name: String, shape: [Int64]) {
        var t = turbo_tensor_info()
        t.struct_size = UInt32(MemoryLayout<turbo_tensor_info>.size)
        try check { turbo_result_output_info(handle(), index, &t, $0) }
        let shape = withUnsafePointer(to: t.shape) { p in
            p.withMemoryRebound(to: Int64.self, capacity: Int(TURBO_MAX_RANK)) { Array(UnsafeBufferPointer(start: $0, count: Int(t.ndim))) }
        }
        return (fixedString(t.name), shape)
    }

    /// Copy an `f32` output into Swift memory.
    public func readFloats(_ index: UInt32) throws -> [Float] {
        let (_, shape) = try output(index)
        let count = Int(shape.reduce(1, *))
        var out = [Float](repeating: 0, count: count)
        var written: UInt64 = 0
        _ = try out.withUnsafeMutableBytes { buf in
            try check { turbo_result_read(handle(), index, buf.baseAddress, UInt64(buf.count), &written, $0) }
        }
        return out
    }

    /// Copy an `i32` output into Swift memory.
    public func readInts(_ index: UInt32) throws -> [Int32] {
        let (_, shape) = try output(index)
        let count = Int(shape.reduce(1, *))
        var out = [Int32](repeating: 0, count: count)
        _ = try out.withUnsafeMutableBytes { buf in
            try check { turbo_result_read(handle(), index, buf.baseAddress, UInt64(buf.count), nil, $0) }
        }
        return out
    }

    public func spans() throws -> [Span] {
        var n: UInt32 = 0
        try check { turbo_result_spans(handle(), nil, 0, &n, $0) }
        if n == 0 { return [] }
        var raw = [turbo_span](repeating: turbo_span(), count: Int(n))
        _ = try raw.withUnsafeMutableBufferPointer { buf in
            try check { turbo_result_spans(handle(), buf.baseAddress, n, &n, $0) }
        }
        return raw.map { Span(row: $0.row, byteStart: $0.byte_start, byteEnd: $0.byte_end, label: $0.label, score: $0.score) }
    }
}
