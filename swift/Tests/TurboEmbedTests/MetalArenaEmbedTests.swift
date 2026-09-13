#if canImport(TurboEmbedC)
import TurboEmbedC
#endif
#if canImport(TurboBufferC)
import TurboBufferC
#endif
import XCTest

final class MetalArenaEmbedTests: XCTestCase {
    func testMetalArenaCreateRentsSharedAndLookupHits() {
        var arena: OpaquePointer?
        let st = turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_METAL, &arena)
        XCTAssertEqual(st, TURBO_BUFFER_OK, String(cString: turbo_buffer_last_error(nil)))
        XCTAssertNotNil(arena)
        defer { turbo_buffer_arena_destroy(arena) }

        var ids = turbo_buffer_view()
        XCTAssertEqual(
            turbo_buffer_arena_rent(
                arena, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_SHARED, 32, 256, 256, &ids),
            TURBO_BUFFER_OK)
        XCTAssertEqual(turbo_buffer_arena_owns(arena, ids.ptr), 1)
        XCTAssertEqual(turbo_buffer_metal_owns(ids.ptr), 1)
        var native: UnsafeMutableRawPointer?
        var off = 0
        XCTAssertEqual(turbo_buffer_metal_lookup(ids.ptr, &native, &off), 1)
        XCTAssertNotNil(native)
        XCTAssertEqual(turbo_buffer_arena_return(arena, &ids), TURBO_BUFFER_OK)
    }

    func testMetalEmbedResultIsArenaOwnedAndZeroAllocsAfterWarmup() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: workspaceRoot() + "/models/mlx/minilm/model.safetensors"),
            "run make fetch-mlx ALIASES=minilm"
        )
        setenv("INFERSTREAM_ROOT", workspaceRoot(), 1)
        var engine: OpaquePointer?
        let create = turboembed_engine_create(TURBOEMBED_DEVICE_METAL, nil, &engine)
        XCTAssertEqual(create, TURBOEMBED_OK, String(cString: turboembed_last_error(nil)))
        XCTAssertNotNil(engine)
        defer { turboembed_engine_destroy(engine) }

        XCTAssertEqual(
            "minilm".withCString { turboembed_load_model(engine, $0, 0) },
            TURBOEMBED_OK,
            String(cString: turboembed_last_error(engine)))

        let hello = Array("hello world".utf8)
        try hello.withUnsafeBufferPointer { buf in
            var view = turboembed_str(
                ptr: buf.baseAddress.map {
                    UnsafeRawPointer($0).assumingMemoryBound(to: CChar.self)
                },
                len: buf.count)
            var opts = turboembed_embed_options(
                pooling: TURBOEMBED_POOLING_MEAN,
                normalize: 1,
                truncate_to: 256,
                output_format: TURBOEMBED_OUTPUT_TYPED
            )
            var first: UnsafeMutablePointer<turboembed_embed_result>?
            let st1 = "minilm".withCString { alias in
                turboembed_embed(engine, alias, 6, &view, 1, &opts, &first)
            }
            XCTAssertEqual(st1, TURBOEMBED_OK, String(cString: turboembed_last_error(engine)))
            XCTAssertEqual(first?.pointee.dim, 384)
            XCTAssertEqual(turbo_buffer_metal_owns(UnsafeRawPointer(first?.pointee.values)), 1)
            turboembed_embed_result_free(first)

            turbo_buffer_alloc_counter_reset()
            var second: UnsafeMutablePointer<turboembed_embed_result>?
            let st2 = "minilm".withCString { alias in
                turboembed_embed(engine, alias, 6, &view, 1, &opts, &second)
            }
            XCTAssertEqual(st2, TURBOEMBED_OK)
            XCTAssertEqual(turbo_buffer_alloc_counter(), 0)
            XCTAssertEqual(turbo_buffer_metal_owns(UnsafeRawPointer(second?.pointee.values)), 1)
            var native: UnsafeMutableRawPointer?
            var off = 0
            XCTAssertEqual(
                turbo_buffer_metal_lookup(
                    UnsafeRawPointer(second?.pointee.values), &native, &off), 1)
            XCTAssertNotNil(native)
            turboembed_embed_result_free(second)
        }
    }
}

private func workspaceRoot() -> String {
    if let env = ProcessInfo.processInfo.environment["INFERSTREAM_ROOT"], !env.isEmpty {
        return env
    }
    var dir = URL(filePath: FileManager.default.currentDirectoryPath)
    for _ in 0..<8 {
        if FileManager.default.fileExists(atPath: dir.appending(path: "include/turboembed.h").path)
        {
            return dir.path
        }
        let parent = dir.deletingLastPathComponent()
        if parent.path == dir.path { return FileManager.default.currentDirectoryPath }
        dir = parent
    }
    return FileManager.default.currentDirectoryPath
}
