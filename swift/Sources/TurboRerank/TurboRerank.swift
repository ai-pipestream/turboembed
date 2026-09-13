#if canImport(TurboRerankC)
import TurboRerankC
#endif

import Foundation

/// Swift client over the frozen TurboRerank C ABI.
///
/// Metal compute lives in `native/turborerank/src/metal_api.mm`
/// (MTLResourceStorageModeShared + first-party kernels). This module
/// does **not** `@_cdecl` the ABI — that would collide with the C++
/// implementation the Rust crate links. See `docs/turborerank-swift.md`.

public enum TurboRerankError: Error, LocalizedError, Sendable {
    case status(turborerank_status, String)

    public var errorDescription: String? {
        switch self {
        case .status(let st, let detail):
            let name = String(cString: turborerank_status_name(st))
            return detail.isEmpty ? name : "\(name): \(detail)"
        }
    }
}

public final class TurboRerankEngine: @unchecked Sendable {
    private var raw: OpaquePointer?

    public init(device: turborerank_device, configPath: String? = nil) throws {
        var out: OpaquePointer?
        let st: turborerank_status
        if let configPath {
            st = configPath.withCString { turborerank_engine_create(device, $0, &out) }
        } else {
            st = turborerank_engine_create(device, nil, &out)
        }
        if st != TURBORERANK_OK || out == nil {
            throw TurboRerankError.status(st, String(cString: turborerank_last_error(nil)))
        }
        raw = out
    }

    deinit {
        if let raw {
            turborerank_engine_destroy(raw)
        }
    }

    public func load(alias: String) throws {
        let st = alias.withCString { turborerank_load_model(raw, $0, 0) }
        if st != TURBORERANK_OK {
            throw TurboRerankError.status(st, String(cString: turborerank_last_error(raw)))
        }
    }

    public func score(
        query: String,
        documents: [String],
        activation: turborerank_activation = TURBORERANK_ACT_IDENTITY,
        maxLength: UInt32 = 512
    ) throws -> [Float] {
        guard !documents.isEmpty else {
            throw TurboRerankError.status(
                TURBORERANK_ERR_INVALID_ARGUMENT, "documents must not be empty")
        }
        var scores = [Float](repeating: 0, count: documents.count)
        let qBytes = Array(query.utf8)
        let docBytes = documents.map { Array($0.utf8) }
        try qBytes.withUnsafeBufferPointer { qBuf in
            var views = [turborerank_str]()
            views.reserveCapacity(docBytes.count)
            func run(_ q: turborerank_str, _ dviews: [turborerank_str]) throws {
                var opts = turborerank_score_options(
                    truncation: TURBORERANK_TRUNC_LONGEST_FIRST,
                    activation: activation,
                    max_length: maxLength
                )
                var local = dviews
                let st = local.withUnsafeBufferPointer { dv in
                    scores.withUnsafeMutableBufferPointer { out in
                        turborerank_score(
                            raw,
                            nil,
                            0,
                            q,
                            dv.baseAddress,
                            dv.count,
                            &opts,
                            out.baseAddress
                        )
                    }
                }
                if st != TURBORERANK_OK {
                    throw TurboRerankError.status(st, String(cString: turborerank_last_error(raw)))
                }
            }
            let qView = turborerank_str(
                ptr: qBuf.baseAddress.map { UnsafeRawPointer($0).assumingMemoryBound(to: CChar.self) },
                len: qBuf.count
            )
            try withDocViews(docBytes) { dviews in
                try run(qView, dviews)
            }
        }
        return scores
    }
}

private func withDocViews(_ docs: [[UInt8]], _ body: ([turborerank_str]) throws -> Void) throws {
    if docs.isEmpty {
        try body([])
        return
    }
    try withDocViews(Array(docs.dropFirst())) { tail in
        try docs[0].withUnsafeBufferPointer { buf in
            var row = [turborerank_str]()
            row.append(
                turborerank_str(
                    ptr: buf.baseAddress.map {
                        UnsafeRawPointer($0).assumingMemoryBound(to: CChar.self)
                    },
                    len: buf.count
                ))
            row.append(contentsOf: tail)
            try body(row)
        }
    }
}
