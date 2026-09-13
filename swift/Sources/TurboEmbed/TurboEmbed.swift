#if canImport(TurboEmbedC)
import TurboEmbedC
#endif

import Foundation

/// Safe Swift wrapper over the TurboEmbed C ABI.
///
/// Input strings are copied into a temporary UTF-8 buffer for the duration
/// of the call (`String` is not a stable contiguous view). `Embeddings`
/// owns the C result until it is released — `values` is only valid until then.
public final class Engine: @unchecked Sendable {
    private let raw: OpaquePointer

    public init(device: turboembed_device = TURBOEMBED_DEVICE_MOCK) throws {
        var out: OpaquePointer?
        let st = turboembed_engine_create(device, nil, &out)
        guard st == TURBOEMBED_OK, let engine = out else {
            throw TurboEmbedError.status(st, message: String(cString: turboembed_last_error(nil)))
        }
        self.raw = engine
    }

    deinit {
        turboembed_engine_destroy(raw)
    }

    public func load(alias: String) throws {
        let st = alias.withCString { turboembed_load_model(raw, $0, alias.utf8.count) }
        try throwIfNeeded(st)
    }

    public func embed(alias: String, texts: [String], options: EmbedOptions? = nil) throws -> Embeddings {
        if texts.isEmpty {
            throw TurboEmbedError.status(
                TURBOEMBED_ERR_INVALID_ARGUMENT, message: "texts must not be empty")
        }
        var cstrs: [UnsafeMutablePointer<CChar>] = []
        cstrs.reserveCapacity(texts.count)
        defer { cstrs.forEach { free($0) } }
        var views: [turboembed_str] = []
        views.reserveCapacity(texts.count)
        for text in texts {
            guard let dup = strdup(text) else {
                throw TurboEmbedError.status(
                    TURBOEMBED_ERR_OUT_OF_MEMORY, message: "strdup failed")
            }
            cstrs.append(dup)
            views.append(turboembed_str(ptr: UnsafePointer(dup), len: text.utf8.count))
        }
        var cOpts = options?.toC()
        var out: UnsafeMutablePointer<turboembed_embed_result>?
        let st = alias.withCString { cAlias in
            views.withUnsafeBufferPointer { buf in
                if var opts = cOpts {
                    return turboembed_embed(
                        raw, cAlias, alias.utf8.count, buf.baseAddress, buf.count, &opts, &out)
                }
                return turboembed_embed(
                    raw, cAlias, alias.utf8.count, buf.baseAddress, buf.count, nil, &out)
            }
        }
        try throwIfNeeded(st)
        guard let result = out else {
            throw TurboEmbedError.status(TURBOEMBED_ERR_INTERNAL, message: "null result")
        }
        return Embeddings(raw: result)
    }

    private func throwIfNeeded(_ st: turboembed_status) throws {
        if st != TURBOEMBED_OK {
            throw TurboEmbedError.status(st, message: String(cString: turboembed_last_error(raw)))
        }
    }
}

/// Engine-owned embed result. `values` is valid until this value is released.
public final class Embeddings: @unchecked Sendable {
    private let raw: UnsafeMutablePointer<turboembed_embed_result>

    fileprivate init(raw: UnsafeMutablePointer<turboembed_embed_result>) {
        self.raw = raw
    }

    deinit {
        turboembed_embed_result_free(raw)
    }

    public var dim: Int { Int(raw.pointee.dim) }
    public var count: Int { Int(raw.pointee.count) }

    public var values: UnsafeBufferPointer<Float> {
        UnsafeBufferPointer(start: raw.pointee.values, count: count * dim)
    }

    public var packed: UnsafeBufferPointer<UInt8> {
        UnsafeBufferPointer(start: raw.pointee.packed, count: raw.pointee.packed_len)
    }
}

/// Options forwarded to `turboembed_embed`. `nil` fields keep provider defaults.
public struct EmbedOptions: Sendable {
    public var pooling: turboembed_pooling
    /// `nil` = provider default.
    public var normalize: Bool?
    /// `nil` / 0 = provider default.
    public var truncateTo: UInt32?
    public var outputFormat: turboembed_output_format

    public init(
        pooling: turboembed_pooling = TURBOEMBED_POOLING_DEFAULT,
        normalize: Bool? = nil,
        truncateTo: UInt32? = nil,
        outputFormat: turboembed_output_format = TURBOEMBED_OUTPUT_TYPED
    ) {
        self.pooling = pooling
        self.normalize = normalize
        self.truncateTo = truncateTo
        self.outputFormat = outputFormat
    }

    fileprivate func toC() -> turboembed_embed_options {
        turboembed_embed_options(
            pooling: pooling,
            normalize: normalize.map { $0 ? 1 : 0 } ?? -1,
            truncate_to: truncateTo ?? 0,
            output_format: outputFormat
        )
    }
}

public enum TurboEmbedError: Error, CustomStringConvertible {
    case status(turboembed_status, message: String)

    public var description: String {
        switch self {
        case .status(let st, let message):
            "\(String(cString: turboembed_status_name(st))): \(message)"
        }
    }
}
