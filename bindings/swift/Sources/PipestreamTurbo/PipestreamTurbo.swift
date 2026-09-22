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

/// Decode an ABI enumeration; an unknown value is `TURBO_E_INVALID_ENUM`,
/// never a trap.
func decodeEnum<T: RawRepresentable>(_ raw: UInt32, _ what: String) throws -> T where T.RawValue == UInt32 {
    guard let v = T(rawValue: raw) else {
        throw TurboError(code: TURBO_E_INVALID_ENUM, field: 0, message: "\(what) \(raw) is not a value this binding knows")
    }
    return v
}

/// Run a C call with a caller-owned error struct and throw on failure.
@discardableResult
func check(_ body: (UnsafeMutablePointer<turbo_error>) -> Int32) throws -> Int32 {
    var err = turbo_error()
    err.struct_size = UInt32(MemoryLayout<turbo_error>.size)
    let rc = withUnsafeMutablePointer(to: &err) { body($0) }
    if rc == TURBO_OK { return rc }
    throw makeError(rc, err)
}

/// The Swift error for a failed call's status and error record.
func makeError(_ rc: Int32, _ err: turbo_error) -> TurboError {
    var e = err
    let message = withUnsafePointer(to: &e.message) { p in
        p.withMemoryRebound(to: CChar.self, capacity: Int(TURBO_ERROR_MESSAGE_LEN)) { String(cString: $0) }
    }
    return TurboError(code: rc, field: err.field, message: message)
}

/// Read a NUL-terminated fixed-size `char[]` tuple field.
func fixedString<T>(_ tuple: T) -> String {
    withUnsafePointer(to: tuple) { p in
        p.withMemoryRebound(to: CChar.self, capacity: MemoryLayout<T>.size) { String(cString: $0) }
    }
}

/// UTF-8 bytes of `strings`, kept alive while `body` runs, exposed as `turbo_text`.
///
/// The bytes of every string are packed into one buffer and the views are
/// built inside that buffer's scope, so no pointer outlives the scope that
/// produced it. An empty string becomes `ptr == NULL, len == 0`, which
/// `turbo_types.h` permits, rather than a one-byte allocation nobody frees.
func withTexts<R>(_ strings: [String], _ body: (UnsafePointer<turbo_text>, UInt32) throws -> R) rethrows -> R {
    var flat: [UInt8] = []
    var extents: [(start: Int, count: Int)] = []
    extents.reserveCapacity(strings.count)
    for s in strings {
        let start = flat.count
        flat.append(contentsOf: s.utf8)
        extents.append((start, flat.count - start))
    }
    return try flat.withUnsafeBufferPointer { bytes in
        var texts = [turbo_text](repeating: turbo_text(), count: strings.count)
        for i in 0..<strings.count {
            let e = extents[i]
            let ptr: UnsafePointer<CChar>? = e.count == 0
                ? nil
                : UnsafeRawPointer(bytes.baseAddress! + e.start).assumingMemoryBound(to: CChar.self)
            texts[i] = turbo_text(ptr: ptr, len: UInt64(e.count))
        }
        return try texts.withUnsafeBufferPointer { try body($0.baseAddress!, UInt32(strings.count)) }
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
/// Element type of a tensor (`TURBO_DTYPE_*`), as reported by the library.
public enum DType: UInt32 {
    case bool = 1, u8 = 2, u16 = 3, u32 = 4, u64 = 5, i8 = 6, i16 = 7, i32 = 8, i64 = 9
    case f16 = 10, bf16 = 11, f32 = 12, f64 = 13, bytes = 14
}
public enum Aggregation: UInt32 { case model = 0, none = 1, simple = 2, first = 3, max = 4 }
public enum Placement: UInt32 { case host = 1, pinned = 2, device = 3, shared = 4 }
public enum FinishReason: UInt32 { case none = 0, eos = 1, stop = 2, length = 3, cancelled = 4 }

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
    precondition(DType.bool.rawValue == TURBO_DTYPE_BOOL && DType.f32.rawValue == TURBO_DTYPE_F32
        && DType.i32.rawValue == TURBO_DTYPE_I32 && DType.bytes.rawValue == TURBO_DTYPE_BYTES)
    precondition(Placement.shared.rawValue == TURBO_PLACE_SHARED)
    precondition(FinishReason.cancelled.rawValue == TURBO_FINISH_CANCELLED && FinishReason.length.rawValue == TURBO_FINISH_LENGTH)
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
    private var rawHandle: OpaquePointer?
    var raw: OpaquePointer {
        guard let r = rawHandle else { preconditionFailure("Runtime is closed") }
        return r
    }

    /// Release the C handle now (idempotent); children created from it stay
    /// valid because each keeps its own reference on the C side.
    public func close() {
        if let r = rawHandle { turbo_runtime_release(r); rawHandle = nil }
    }

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
        rawHandle = out!
    }

    deinit { close() }

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
            index: index, kind: try decodeEnum(info.kind, "device kind"), ordinal: info.ordinal, vendorId: info.vendor_id,
            caps: info.caps, memoryTotal: info.memory_total, memoryFree: info.memory_free,
            name: fixedString(info.name), vendor: fixedString(info.vendor), providerId: fixedString(info.provider_id),
            providerVersion: fixedString(info.provider_version), runtimeVersion: fixedString(info.runtime_version),
            driverVersion: fixedString(info.driver_version))
    }

    /// Select a device. `.auto` never picks a CPU; an absent device throws
    /// `TURBO_E_DEVICE_NOT_FOUND`.
    public func selectDevice(policy: SelectPolicy = .auto, providerId: String = "", ordinal: UInt32 = 0) throws -> UInt32 {
        var out: UInt32 = 0
        _ = try withText(providerId) { pid in
            _ = try withText("") { vendor in
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
        return Capability(status: try decodeEnum(cap.status, "capability status"), cosineFloor: cap.cosine_floor,
                          maxAbsError: cap.max_abs_error, deterministic: cap.deterministic != 0, notes: fixedString(cap.notes))
    }

    /// Load the tokenizer the bundle at `bundlePath` declares.
    public func createTokenizer(bundlePath: String) throws -> Tokenizer {
        try Tokenizer(runtime: self, bundlePath: bundlePath)
    }

    public func createContext(device index: UInt32) throws -> Context {
        try Context(runtime: self, device: index)
    }
}

// MARK: - Context and model

public final class Context {
    private var rawHandle: OpaquePointer?
    var raw: OpaquePointer {
        guard let r = rawHandle else { preconditionFailure("Context is closed") }
        return r
    }

    /// Release the C handle now (idempotent); children created from it stay
    /// valid because each keeps its own reference on the C side.
    public func close() {
        if let r = rawHandle { turbo_context_release(r); rawHandle = nil }
    }
    public let deviceIndex: UInt32
    private let runtime: Runtime

    init(runtime: Runtime, device index: UInt32) throws {
        var out: OpaquePointer?
        var desc = turbo_context_desc()
        desc.struct_size = UInt32(MemoryLayout<turbo_context_desc>.size)
        try check { turbo_context_create(runtime.raw, index, &desc, &out, $0) }
        rawHandle = out!
        deviceIndex = index
        self.runtime = runtime
    }

    deinit { close() }

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
    private var rawHandle: OpaquePointer?
    var raw: OpaquePointer {
        guard let r = rawHandle else { preconditionFailure("Model is closed") }
        return r
    }

    /// Release the C handle now (idempotent); children created from it stay
    /// valid because each keeps its own reference on the C side.
    public func close() {
        if let r = rawHandle { turbo_model_release(r); rawHandle = nil }
    }
    public let info: ModelInfo
    private let context: Context

    init(context: Context, bundlePath: String) throws {
        var out: OpaquePointer?
        _ = try withText(bundlePath) { path in
            var desc = turbo_model_desc()
            desc.struct_size = UInt32(MemoryLayout<turbo_model_desc>.size)
            try check { turbo_model_load(context.raw, path, &desc, &out, $0) }
        }
        // The instance is not fully initialized until `info` is assigned, so
        // `deinit` does not run if the work below throws. Release the handle
        // here rather than leaking the model, its weights, and the context
        // and runtime it retains.
        let handle = out!
        do {
            var mi = turbo_model_info()
            mi.struct_size = UInt32(MemoryLayout<turbo_model_info>.size)
            try check { turbo_model_get_info(handle, &mi, $0) }
            var labels: [String] = []
            for i in 0..<mi.n_labels {
                var t = turbo_text()
                try check { turbo_model_label(handle, i, &t, $0) }
                labels.append(t.len == 0 ? "" : String(decoding: UnsafeRawBufferPointer(start: t.ptr, count: Int(t.len)), as: UTF8.self))
            }
            info = ModelInfo(task: try decodeEnum(mi.task, "task"), dim: mi.dim, labels: labels, maxSeq: mi.max_seq, maxBatch: mi.max_batch,
                             fullyAccelerated: mi.fully_accelerated != 0, modelId: fixedString(mi.model_id), providerId: fixedString(mi.provider_id))
        } catch {
            turbo_model_release(handle)
            throw error
        }
        rawHandle = handle
        self.context = context
    }

    deinit { close() }

    /// Create a session with fixed maxima; 0 means the model's default.
    public func createSession(maxBatch: UInt32 = 0, maxSeq: UInt32 = 0) throws -> Session {
        try Session(model: self, maxBatch: maxBatch, maxSeq: maxSeq)
    }

    /// Create a generation on a generative model.
    public func createGeneration(_ desc: GenerateDesc = GenerateDesc()) throws -> Generation {
        try Generation(model: self, desc: desc)
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
    private var rawHandle: OpaquePointer?
    var raw: OpaquePointer {
        guard let r = rawHandle else { preconditionFailure("Session is closed") }
        return r
    }

    /// Release the C handle now (idempotent); children created from it stay
    /// valid because each keeps its own reference on the C side.
    public func close() {
        if let r = rawHandle { turbo_session_release(r); rawHandle = nil }
    }
    public let model: Model

    init(model: Model, maxBatch: UInt32, maxSeq: UInt32) throws {
        var out: OpaquePointer?
        var desc = turbo_session_desc()
        desc.struct_size = UInt32(MemoryLayout<turbo_session_desc>.size)
        desc.max_batch = maxBatch
        desc.max_seq = maxSeq
        try check { turbo_session_create(model.raw, &desc, &out, $0) }
        rawHandle = out!
        self.model = model
    }

    deinit { close() }

    public func writeText(_ texts: [String], options: EmbedOptions = EmbedOptions()) throws {
        var o = options.c
        _ = try withTexts(texts) { t, n in try check { turbo_session_write_text(raw, t, n, &o, $0) } }
    }

    public func writePairs(query: String, documents: [String], options: RerankOptions = RerankOptions()) throws {
        var o = options.c
        _ = try withText(query) { q in
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
            return try decodeEnum(info.placement, "placement")
        }
    }

    /// Logical shape, element type, and name of output `index`.
    ///
    /// A dynamic extent (`-1`, which `turbo_types.h` permits in a declared
    /// shape) has no element count, so it is refused here rather than
    /// turned into a negative buffer size by a caller that multiplies.
    public func output(_ index: UInt32) throws -> (name: String, dtype: DType, shape: [Int64]) {
        var t = turbo_tensor_info()
        t.struct_size = UInt32(MemoryLayout<turbo_tensor_info>.size)
        try check { turbo_result_output_info(handle(), index, &t, $0) }
        let shape = withUnsafePointer(to: t.shape) { p in
            p.withMemoryRebound(to: Int64.self, capacity: Int(TURBO_MAX_RANK)) { Array(UnsafeBufferPointer(start: $0, count: Int(t.ndim))) }
        }
        if let bad = shape.first(where: { $0 < 0 }) {
            throw TurboError(code: TURBO_E_INVALID_SHAPE, field: 0,
                             message: "output \(index) has the dynamic extent \(bad) in its shape \(shape); it has no element count to read")
        }
        return (fixedString(t.name), try decodeEnum(t.dtype, "dtype"), shape)
    }

    /// Copy an `f32` output into Swift memory. Another dtype is
    /// `TURBO_E_UNSUPPORTED_DTYPE`; use `readBytes` for the raw payload.
    public func readFloats(_ index: UInt32) throws -> [Float] {
        let (_, dtype, shape) = try output(index)
        guard dtype == .f32 else {
            throw TurboError(code: TURBO_E_UNSUPPORTED_DTYPE, field: 0, message: "output \(index) is \(dtype), not f32")
        }
        let count = Int(shape.reduce(1, *))
        var out = [Float](repeating: 0, count: count)
        var written: UInt64 = 0
        _ = try out.withUnsafeMutableBytes { buf in
            try check { turbo_result_read(handle(), index, buf.baseAddress, UInt64(buf.count), &written, $0) }
        }
        return Array(out[0..<(Int(written) / MemoryLayout<Float>.size)])
    }

    /// Copy an `i32` output into Swift memory. Another dtype is
    /// `TURBO_E_UNSUPPORTED_DTYPE`.
    public func readInts(_ index: UInt32) throws -> [Int32] {
        let (_, dtype, shape) = try output(index)
        guard dtype == .i32 else {
            throw TurboError(code: TURBO_E_UNSUPPORTED_DTYPE, field: 0, message: "output \(index) is \(dtype), not i32")
        }
        let count = Int(shape.reduce(1, *))
        var out = [Int32](repeating: 0, count: count)
        var written: UInt64 = 0
        _ = try out.withUnsafeMutableBytes { buf in
            try check { turbo_result_read(handle(), index, buf.baseAddress, UInt64(buf.count), &written, $0) }
        }
        return Array(out[0..<(Int(written) / MemoryLayout<Int32>.size)])
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

// MARK: - Generation

/// One chat message (`turbo_message`).
public struct Message {
    public var role: String
    public var content: String
    public init(role: String, content: String) { self.role = role; self.content = content }
    public static func user(_ content: String) -> Message { Message(role: "user", content: content) }
    public static func system(_ content: String) -> Message { Message(role: "system", content: content) }
    public static func assistant(_ content: String) -> Message { Message(role: "assistant", content: content) }
}

/// Generation parameters (`turbo_generate_desc`). Zero and empty values mean
/// the model's defaults; anything else is honored exactly or the generation
/// fails with `TURBO_E_UNSUPPORTED_OPTION` naming the field.
public struct GenerateDesc {
    public var maxNewTokens: UInt32 = 0
    public var minNewTokens: UInt32 = 0
    public var nSequences: UInt32 = 0
    public var temperature: Float = 0
    public var topK: UInt32 = 0
    public var topP: Float = 0
    public var minP: Float = 0
    public var repeatPenalty: Float = 0
    public var presencePenalty: Float = 0
    public var frequencyPenalty: Float = 0
    public var seed: UInt64? = nil
    public var stop: [String] = []
    public var stopTokens: [Int32] = []
    public var logitBias: [(token: Int32, bias: Float)] = []
    public var logprobs: UInt32 = 0
    public var echo: Bool = false
    public init() {}

    /// Run `body` with the C descriptor; every pointer it holds lives for the call.
    func withC<R>(_ body: (UnsafePointer<turbo_generate_desc>) throws -> R) throws -> R {
        var d = turbo_generate_desc()
        d.struct_size = UInt32(MemoryLayout<turbo_generate_desc>.size)
        d.max_new_tokens = maxNewTokens; d.min_new_tokens = minNewTokens; d.n_sequences = nSequences
        d.temperature = temperature; d.top_k = topK; d.top_p = topP; d.min_p = minP
        d.repeat_penalty = repeatPenalty; d.presence_penalty = presencePenalty; d.frequency_penalty = frequencyPenalty
        d.has_seed = seed == nil ? 0 : 1; d.seed = seed ?? 0
        d.logprobs = logprobs; d.structured_kind = UInt32(TURBO_STRUCTURED_NONE); d.echo = echo ? 1 : 0
        let biases = logitBias.map { turbo_logit_bias(token: $0.token, bias: $0.bias) }
        let stops = stopTokens
        return try withTexts(stop) { texts, n in
            d.n_stop = n; d.stop = n == 0 ? nil : texts
            return try stops.withUnsafeBufferPointer { st in
                d.n_stop_tokens = UInt32(st.count); d.stop_tokens = st.count == 0 ? nil : st.baseAddress
                return try biases.withUnsafeBufferPointer { lb in
                    d.n_logit_bias = UInt32(lb.count); d.logit_bias = lb.count == 0 ? nil : lb.baseAddress
                    return try withUnsafePointer(to: &d) { try body($0) }
                }
            }
        }
    }
}

/// One step of a generation, copied out of native memory.
public struct Chunk {
    public let sequence: UInt32
    public let tokens: [Int32]
    public let text: String
    public let logprobs: [Float]
    public let done: Bool
    public let finishReason: FinishReason
    public let promptTokens: UInt32
    public let generatedTokens: UInt32

    init(_ c: turbo_generation_chunk) throws {
        sequence = c.sequence
        tokens = c.n_tokens == 0 ? [] : Array(UnsafeBufferPointer(start: c.tokens, count: Int(c.n_tokens)))
        logprobs = c.n_logprobs == 0 ? [] : Array(UnsafeBufferPointer(start: c.logprobs, count: Int(c.n_logprobs)))
        text = c.text.len == 0 ? "" : String(decoding: UnsafeRawBufferPointer(start: c.text.ptr, count: Int(c.text.len)), as: UTF8.self)
        done = c.done != 0
        finishReason = try decodeEnum(c.finish_reason, "finish reason")
        promptTokens = c.prompt_tokens
        generatedTokens = c.generated_tokens
    }
}

/// A generation on a generative model: the pull iterator over
/// `turbo_generation_*`. Prompt once, then `step()` until a chunk is done.
/// `cancel()` may be called from another thread; the next step reports
/// `.cancelled`.
public final class Generation {
    private var rawHandle: OpaquePointer?
    var raw: OpaquePointer {
        guard let r = rawHandle else { preconditionFailure("Generation is closed") }
        return r
    }

    /// Release the C handle now (idempotent); children created from it stay
    /// valid because each keeps its own reference on the C side.
    public func close() {
        if let r = rawHandle { turbo_generation_release(r); rawHandle = nil }
    }
    public let model: Model

    init(model: Model, desc: GenerateDesc) throws {
        var out: OpaquePointer?
        _ = try desc.withC { d in try check { turbo_generation_create(model.raw, d, &out, $0) } }
        rawHandle = out!
        self.model = model
    }

    deinit { close() }

    /// Apply the chat template to `messages` and tokenize the prompt.
    public func prompt(_ messages: [Message]) throws {
        try withTexts(messages.map { $0.role }) { roles, n in
            try withTexts(messages.map { $0.content }) { contents, _ in
                var msgs = [turbo_message](repeating: turbo_message(), count: Int(n))
                for i in 0..<Int(n) { msgs[i] = turbo_message(role: roles[i], content: contents[i]) }
                _ = try msgs.withUnsafeBufferPointer { m in try check { turbo_generation_prompt(raw, m.baseAddress, n, $0) } }
            }
        }
    }

    /// Use caller-supplied prompt token ids.
    public func promptTokens(_ ids: [Int32]) throws {
        _ = try ids.withUnsafeBufferPointer { p in try check { turbo_generation_prompt_tokens(raw, p.baseAddress, UInt32(p.count), $0) } }
    }

    /// Produce the next chunk.
    public func step() throws -> Chunk {
        var c = turbo_generation_chunk()
        c.struct_size = UInt32(MemoryLayout<turbo_generation_chunk>.size)
        try check { turbo_generation_step(raw, &c, $0) }
        return try Chunk(c)
    }

    /// Step until done, handing every chunk to `sink`; a sink that returns
    /// false cancels, and the final chunk reports `.cancelled`.
    @discardableResult
    public func drain(_ sink: (Chunk) throws -> Bool) throws -> Chunk {
        while true {
            let c = try step()
            let go = try sink(c)
            if c.done { return c }
            if !go { try cancel() }
        }
    }

    /// Cancel; the next step reports `.cancelled`.
    public func cancel() throws {
        try check { turbo_generation_cancel(raw, $0) }
    }
}

// MARK: - Tokenizer

public struct TokenizerInfo {
    public let vocabSize: UInt32
    public let maxSeq: UInt32
    public let specialsPerSequence: UInt32
    public let padId: Int32
    public let bosId: Int32
    public let eosId: Int32
    public let unkId: Int32
    public let kind: String
    public let sha256: String
}

public struct EncodeOptions {
    public var addSpecialTokens: Bool = true
    public var truncate: Truncate = .model
    public var maxTokens: UInt32 = 0
    public var padTo: UInt32 = 0
    public var promptRole: PromptRole = .none
    public init() {}

    var c: turbo_encode_options {
        var o = turbo_encode_options()
        o.struct_size = UInt32(MemoryLayout<turbo_encode_options>.size)
        o.add_special_tokens = addSpecialTokens ? 1 : 0; o.truncate = truncate.rawValue
        o.max_tokens = maxTokens; o.pad_to = padTo; o.prompt_role = promptRole.rawValue
        return o
    }
}

/// Encoded rows: `ids` and `mask` are `rows x rowStride`, row-major;
/// `lengths` holds each row's live token count.
public struct Encoding {
    public let rows: Int
    public let rowStride: Int
    public let ids: [Int32]
    public let mask: [Int32]
    public let lengths: [UInt32]
    public func row(_ r: Int) -> [Int32] { Array(ids[r * rowStride ..< r * rowStride + Int(lengths[r])]) }
}

/// The tokenizer a bundle declares. Thread-safe and independent of any device.
public final class Tokenizer {
    private var rawHandle: OpaquePointer?
    var raw: OpaquePointer {
        guard let r = rawHandle else { preconditionFailure("Tokenizer is closed") }
        return r
    }

    /// Release the C handle now (idempotent); children created from it stay
    /// valid because each keeps its own reference on the C side.
    public func close() {
        if let r = rawHandle { turbo_tokenizer_release(r); rawHandle = nil }
    }
    public let info: TokenizerInfo
    private let runtime: Runtime

    init(runtime: Runtime, bundlePath: String) throws {
        var out: OpaquePointer?
        _ = try withText(bundlePath) { path in try check { turbo_tokenizer_create(runtime.raw, path, &out, $0) } }
        // As in `Model.init`: `info` is still unassigned, so a throw here
        // skips `deinit` and the tokenizer handle (and the runtime it
        // retains) would never be released.
        let handle = out!
        do {
            var ti = turbo_tokenizer_info()
            ti.struct_size = UInt32(MemoryLayout<turbo_tokenizer_info>.size)
            try check { turbo_tokenizer_get_info(handle, &ti, $0) }
            info = TokenizerInfo(vocabSize: ti.vocab_size, maxSeq: ti.max_seq, specialsPerSequence: ti.specials_per_sequence,
                                 padId: ti.pad_id, bosId: ti.bos_id, eosId: ti.eos_id, unkId: ti.unk_id,
                                 kind: fixedString(ti.kind), sha256: fixedString(ti.sha256))
        } catch {
            turbo_tokenizer_release(handle)
            throw error
        }
        rawHandle = handle
        self.runtime = runtime
    }

    deinit { close() }

    /// Encode `texts` into rows of `rowStride` ids, padded with the pad id and mask 0.
    public func encode(_ texts: [String], rowStride: Int, options: EncodeOptions = EncodeOptions()) throws -> Encoding {
        var ids = [Int32](repeating: 0, count: texts.count * rowStride)
        var mask = [Int32](repeating: 0, count: texts.count * rowStride)
        var lengths = [UInt32](repeating: 0, count: texts.count)
        var o = options.c
        _ = try withTexts(texts) { t, n in
            try ids.withUnsafeMutableBufferPointer { ip in
                try mask.withUnsafeMutableBufferPointer { mp in
                    try lengths.withUnsafeMutableBufferPointer { lp in
                        try check { turbo_tokenizer_encode(raw, t, n, &o, ip.baseAddress, mp.baseAddress, nil, UInt32(rowStride), lp.baseAddress, $0) }
                    }
                }
            }
        }
        return Encoding(rows: texts.count, rowStride: rowStride, ids: ids, mask: mask, lengths: lengths)
    }

    /// Decode ids to text.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) throws -> String {
        var capacity = max(64, ids.count * 8)
        while true {
            var buf = [UInt8](repeating: 0, count: capacity)
            var written: UInt64 = 0
            var e = turbo_error()
            e.struct_size = UInt32(MemoryLayout<turbo_error>.size)
            let rc = ids.withUnsafeBufferPointer { ip in
                buf.withUnsafeMutableBufferPointer { bp in
                    turbo_tokenizer_decode(raw, ip.baseAddress, UInt32(ip.count), skipSpecialTokens ? 1 : 0, bp.baseAddress, UInt64(bp.count), &written, &e)
                }
            }
            if rc == TURBO_E_CAPACITY {
                // The library reports the size it needs; a report that is
                // not larger than what was just offered is a contract
                // violation, not a reason to try the same size again.
                guard Int(written) > capacity else { throw makeError(rc, e) }
                capacity = Int(written)
                continue
            }
            if rc != TURBO_OK { throw makeError(rc, e) }
            return String(decoding: buf[0..<Int(written)], as: UTF8.self)
        }
    }

    /// Number of tokens `text` produces, without truncation or prefix.
    public func count(_ text: String, addSpecialTokens: Bool = true) throws -> UInt32 {
        var n: UInt32 = 0
        _ = try withText(text) { t in try check { turbo_tokenizer_count(raw, t, addSpecialTokens ? 1 : 0, &n, $0) } }
        return n
    }
}
