#if canImport(TurboBufferC)
import TurboBufferC
#endif
#if canImport(TurboEmbedC)
import TurboEmbedC
#endif

import Foundation

/// Metal SHARED workspace rented from `include/turbo_buffer.h`.
///
/// Token / activation / result rows are `MTLResourceStorageModeShared`.
/// Lookup is `turbo_buffer_metal_lookup` only — no private MTL registry.
enum MetalArenaError: Error, LocalizedError, Sendable {
    case create(String)
    case rent(String)
    case wrap(String)
    case batch(String)

    var errorDescription: String? {
        switch self {
        case .create(let m), .rent(let m), .wrap(let m), .batch(let m):
            m
        }
    }
}

/// Per-engine Metal arena. Load rents token + activation slabs once.
/// Embed rents the result row (slab reuse after warmup → 0 allocs).
final class MetalArena: @unchecked Sendable {
    let raw: OpaquePointer
    var tokensIds = turbo_buffer_view()
    var tokensMask = turbo_buffer_view()
    var tokensTypes = turbo_buffer_view()
    var activations = turbo_buffer_view()
    let maxBatch: UInt32
    let maxSeq: UInt32
    let hidden: UInt32
    let dim: UInt32

    static let defaultMaxBatch: UInt32 = 32
    static let defaultMaxSeq: UInt32 = 256
    static let minilmHidden: UInt32 = 384

    init(
        maxBatch: UInt32 = MetalArena.defaultMaxBatch,
        maxSeq: UInt32 = MetalArena.defaultMaxSeq,
        hidden: UInt32 = MetalArena.minilmHidden,
        dim: UInt32 = MetalArena.minilmHidden
    ) throws {
        var arena: OpaquePointer?
        let st = turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_METAL, &arena)
        guard st == TURBO_BUFFER_OK, let arena else {
            throw MetalArenaError.create(
                "turbo_buffer_arena_create(METAL) failed: \(String(cString: turbo_buffer_last_error(nil))) — refusing CPU fallback"
            )
        }
        self.raw = arena
        self.maxBatch = maxBatch
        self.maxSeq = maxSeq
        self.hidden = hidden
        self.dim = dim
        do {
            try rentWorkspace()
        } catch {
            turbo_buffer_arena_destroy(arena)
            throw error
        }
    }

    deinit {
        returnWorkspace()
        turbo_buffer_arena_destroy(raw)
    }

    private func rentWorkspace() throws {
        try rentI32(&tokensIds, rows: maxBatch, cols: maxSeq, what: "input_ids")
        try rentI32(&tokensMask, rows: maxBatch, cols: maxSeq, what: "attention_mask")
        try rentI32(&tokensTypes, rows: maxBatch, cols: maxSeq, what: "token_type_ids")
        try rentF32(&activations, rows: maxBatch, cols: maxSeq * hidden, what: "activations")
        var warm = turbo_buffer_view()
        try rentF32(&warm, rows: maxBatch, cols: dim, what: "result_warmup")
        _ = turbo_buffer_arena_return(raw, &warm)
    }

    private func rentI32(
        _ view: inout turbo_buffer_view,
        rows: UInt32,
        cols: UInt32,
        what: String
    ) throws {
        try rent(&view, dtype: TURBO_BUFFER_DTYPE_I32, rows: rows, cols: cols, what: what)
    }

    private func rentF32(
        _ view: inout turbo_buffer_view,
        rows: UInt32,
        cols: UInt32,
        what: String
    ) throws {
        try rent(&view, dtype: TURBO_BUFFER_DTYPE_F32, rows: rows, cols: cols, what: what)
    }

    private func rent(
        _ view: inout turbo_buffer_view,
        dtype: turbo_buffer_dtype,
        rows: UInt32,
        cols: UInt32,
        what: String
    ) throws {
        let st = turbo_buffer_arena_rent(
            raw,
            dtype,
            TURBO_BUFFER_PLACE_SHARED,
            rows,
            cols,
            cols,
            &view
        )
        guard st == TURBO_BUFFER_OK, view.ptr != nil else {
            throw MetalArenaError.rent(
                "\(what): turbo_buffer_arena_rent SHARED failed: \(String(cString: turbo_buffer_last_error(raw)))"
            )
        }
        try requireMetalShared(view.ptr, what: what)
    }

    func rentResult(rows: UInt32, cols: UInt32) throws -> turbo_buffer_view {
        var view = turbo_buffer_view()
        try rentF32(&view, rows: rows, cols: cols, what: "result")
        return view
    }

    func returnView(_ view: inout turbo_buffer_view) {
        if view.ptr != nil {
            _ = turbo_buffer_arena_return(raw, &view)
        }
    }

    private func returnWorkspace() {
        returnView(&tokensIds)
        returnView(&tokensMask)
        returnView(&tokensTypes)
        returnView(&activations)
    }

    func requireMetalShared(_ ptr: UnsafeRawPointer?, what: String) throws {
        guard let ptr else {
            throw MetalArenaError.rent("\(what): null arena pointer")
        }
        if turbo_buffer_arena_owns(raw, ptr) != 1 {
            throw MetalArenaError.rent(
                "\(what): pointer is not turbo_buffer_arena_owns — refusing private malloc"
            )
        }
        if turbo_buffer_metal_owns(ptr) != 1 {
            throw MetalArenaError.rent(
                "\(what): pointer is not turbo_buffer_metal_owns — refusing private MTL"
            )
        }
        var native: UnsafeMutableRawPointer?
        var off = 0
        if turbo_buffer_metal_lookup(ptr, &native, &off) != 1 || native == nil {
            throw MetalArenaError.rent(
                "\(what): turbo_buffer_metal_lookup missed SHARED rent — no private MTL registry"
            )
        }
    }

    var inputIds: UnsafeMutablePointer<Int32> {
        tokensIds.ptr.assumingMemoryBound(to: Int32.self)
    }

    var attentionMask: UnsafeMutablePointer<Int32> {
        tokensMask.ptr.assumingMemoryBound(to: Int32.self)
    }

    var tokenTypes: UnsafeMutablePointer<Int32> {
        tokensTypes.ptr.assumingMemoryBound(to: Int32.self)
    }

    var hiddenStates: UnsafeMutablePointer<Float> {
        activations.ptr.assumingMemoryBound(to: Float.self)
    }
}

/// Result header + rented values view. `pub` is first so the ABI pointer
/// recovers this record on free (same layout as the C++ stub).
struct EmbedResultRec {
    var pub: turboembed_embed_result
    var view: turbo_buffer_view
    var arena: OpaquePointer?
}
