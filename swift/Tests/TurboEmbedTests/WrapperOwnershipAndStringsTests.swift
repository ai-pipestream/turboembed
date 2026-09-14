#if canImport(TurboEmbed)
import TurboEmbed
#endif
#if canImport(TurboEmbedC)
import TurboEmbedC
#endif
import Dispatch
import XCTest

final class WrapperOwnershipAndStringsTests: XCTestCase {
    func testEmbedCopiesExactUTF8Spans() throws {
        let texts = ["", "a\0bc", "a", "Grüße 🌍"]
        let engine = try TurboEmbed.Engine(device: TURBOEMBED_DEVICE_MOCK)
        try engine.load(alias: "mock-embed")

        let wrapped = try engine.embed(alias: "mock-embed", texts: texts)
        let expected = try directMockEmbed(texts)

        XCTAssertEqual(wrapped.count, texts.count)
        XCTAssertEqual(wrapped.dim, expected.dim)
        XCTAssertEqual(Array(wrapped.values), expected.values)
        XCTAssertNotEqual(
            Array(wrapped.values[wrapped.dim..<(2 * wrapped.dim)]),
            Array(wrapped.values[(2 * wrapped.dim)..<(3 * wrapped.dim)]),
            "embedded NUL must remain part of the input span"
        )
    }

    func testEmbeddingsRetainsEngineUntilResultRelease() throws {
        var engine: TurboEmbed.Engine? = try TurboEmbed.Engine(device: TURBOEMBED_DEVICE_MOCK)
        weak var weakEngine = engine
        try engine?.load(alias: "mock-embed")
        var embeddings = try engine?.embed(alias: "mock-embed", texts: ["lifetime"])

        engine = nil
        XCTAssertNotNil(weakEngine)
        XCTAssertEqual(embeddings?.count, 1)
        XCTAssertEqual(embeddings?.values.count, 8)

        embeddings = nil
        XCTAssertNil(weakEngine)
    }

    func testConcurrentWrapperOperationsOnOneEngine() throws {
        let engine = try TurboEmbed.Engine(device: TURBOEMBED_DEVICE_MOCK)
        try engine.load(alias: "mock-embed")

        DispatchQueue.concurrentPerform(iterations: 64) { index in
            do {
                let embeddings = try engine.embed(
                    alias: "mock-embed", texts: ["concurrent \(index)"])
                XCTAssertEqual(embeddings.count, 1)
                XCTAssertEqual(embeddings.values.count, 8)
            } catch {
                XCTFail("concurrent embed \(index) failed: \(error)")
            }
        }
    }

    func testCreationErrorsAreIsolatedPerThread() {
        let firstSet = DispatchSemaphore(value: 0)
        let secondSet = DispatchSemaphore(value: 0)
        let finished = expectation(description: "both threads read their own creation error")
        finished.expectedFulfillmentCount = 2

        DispatchQueue.global().async {
            XCTAssertEqual(
                turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nil, nil),
                TURBOEMBED_ERR_INVALID_ARGUMENT
            )
            firstSet.signal()
            secondSet.wait()
            XCTAssertEqual(String(cString: turboembed_last_error(nil)), "out pointer is null")
            finished.fulfill()
        }
        DispatchQueue.global().async {
            firstSet.wait()
            XCTAssertEqual(turboembed_register_provider(nil), TURBOEMBED_ERR_NOT_IMPLEMENTED)
            secondSet.signal()
            XCTAssertEqual(
                String(cString: turboembed_last_error(nil)),
                "turboembed_register_provider is reserved for MLX / model2vec plugins"
            )
            finished.fulfill()
        }

        wait(for: [finished], timeout: 5)
    }

    func testABIRejectsUnrepresentableCountBeforeReadingViews() {
        var engine: OpaquePointer?
        XCTAssertEqual(
            turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nil, &engine),
            TURBOEMBED_OK
        )
        guard let engine else { return }
        defer { turboembed_engine_destroy(engine) }

        let text = Array("only one view exists".utf8)
        text.withUnsafeBufferPointer { bytes in
            var view = turboembed_str(
                ptr: bytes.baseAddress.map {
                    UnsafeRawPointer($0).assumingMemoryBound(to: CChar.self)
                },
                len: bytes.count
            )
            var result: UnsafeMutablePointer<turboembed_embed_result>?
            let oversizedCount = Int(UInt32.max) + 1
            let status = "mock-embed".withCString { alias in
                turboembed_embed(
                    engine, alias, 10, &view, oversizedCount, nil, &result)
            }

            XCTAssertEqual(status, TURBOEMBED_ERR_INVALID_ARGUMENT)
            XCTAssertNil(result)
            XCTAssertTrue(
                String(cString: turboembed_last_error(engine)).contains("count"))
        }
    }
}

private func directMockEmbed(_ texts: [String]) throws -> (dim: Int, values: [Float]) {
    var engine: OpaquePointer?
    let create = turboembed_engine_create(TURBOEMBED_DEVICE_MOCK, nil, &engine)
    guard create == TURBOEMBED_OK, let engine else {
        throw TurboEmbedError.status(create, message: String(cString: turboembed_last_error(nil)))
    }
    defer { turboembed_engine_destroy(engine) }

    let load = "mock-embed".withCString { turboembed_load_model(engine, $0, 10) }
    guard load == TURBOEMBED_OK else {
        throw TurboEmbedError.status(load, message: String(cString: turboembed_last_error(engine)))
    }

    var storage: [UnsafeMutablePointer<UInt8>] = []
    defer { storage.forEach { $0.deallocate() } }
    var views: [turboembed_str] = []
    for text in texts {
        let bytes = Array(text.utf8)
        guard !bytes.isEmpty else {
            views.append(turboembed_str(ptr: nil, len: 0))
            continue
        }
        let copy = UnsafeMutablePointer<UInt8>.allocate(capacity: bytes.count)
        bytes.withUnsafeBufferPointer { source in
            copy.update(from: source.baseAddress!, count: source.count)
        }
        storage.append(copy)
        views.append(
            turboembed_str(
                ptr: UnsafeRawPointer(copy).assumingMemoryBound(to: CChar.self),
                len: bytes.count
            )
        )
    }

    var result: UnsafeMutablePointer<turboembed_embed_result>?
    let status = "mock-embed".withCString { alias in
        views.withUnsafeBufferPointer { buffer in
            turboembed_embed(engine, alias, 10, buffer.baseAddress, buffer.count, nil, &result)
        }
    }
    guard status == TURBOEMBED_OK, let result else {
        throw TurboEmbedError.status(
            status, message: String(cString: turboembed_last_error(engine)))
    }
    defer { turboembed_embed_result_free(result) }
    let count = Int(result.pointee.count) * Int(result.pointee.dim)
    return (
        Int(result.pointee.dim),
        Array(UnsafeBufferPointer(start: result.pointee.values, count: count))
    )
}
