import Dispatch
import Foundation
import TurboEmbed
import TurboEmbedC
import XCTest

final class MetalWrapperOwnershipTests: XCTestCase {
    func testMetalResultRetainsItsEngineAndSurvivorRemainsUsable() throws {
        try requireMiniLM()
        var first: TurboEmbed.Engine? = try TurboEmbed.Engine(device: TURBOEMBED_DEVICE_METAL)
        let survivor = try TurboEmbed.Engine(device: TURBOEMBED_DEVICE_METAL)
        weak var owner = first
        try first!.load(alias: "minilm")
        try survivor.load(alias: "minilm")
        var retained: TurboEmbed.Embeddings? = try first!.embed(
            alias: "minilm", texts: ["hello world"])
        let reference = Array(retained!.values)
        XCTAssertEqual(reference.count, 384)

        first = nil
        XCTAssertNotNil(owner)
        for _ in 0..<4 {
            let result = try survivor.embed(alias: "minilm", texts: ["hello world"])
            assertNear(reference, Array(result.values))
            XCTAssertEqual(reference, Array(retained!.values), "retained result was overwritten")
        }
        retained = nil
        XCTAssertNil(owner)
        let result = try survivor.embed(alias: "minilm", texts: ["hello world"])
        assertNear(reference, Array(result.values))
    }

    func testIndependentMetalEnginesRunConcurrentWrapperCalls() throws {
        try requireMiniLM()
        let engines = try (0..<2).map { _ in
            let engine = try TurboEmbed.Engine(device: TURBOEMBED_DEVICE_METAL)
            try engine.load(alias: "minilm")
            return engine
        }
        let inputs = ["hello world", "Grüße 🌍"]
        let retained = try engines.enumerated().map { index, engine in
            try engine.embed(alias: "minilm", texts: [inputs[index]])
        }
        let references = retained.map { Array($0.values) }
        DispatchQueue.concurrentPerform(iterations: 16) { index in
            let engineIndex = index % engines.count
            do {
                let result = try engines[engineIndex].embed(
                    alias: "minilm", texts: [inputs[engineIndex]])
                assertNear(references[engineIndex], Array(result.values))
                XCTAssertEqual(references[engineIndex], Array(retained[engineIndex].values))
            } catch {
                XCTFail("Metal engine \(engineIndex) failed: \(error)")
            }
        }
    }
}

private func requireMiniLM() throws {
    let root: URL
    if let configured = ProcessInfo.processInfo.environment["INFERSTREAM_ROOT"] {
        root = URL(filePath: configured)
    } else {
        root = URL(filePath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    try XCTSkipUnless(
        FileManager.default.fileExists(atPath: root.appending(path: "models/mlx/minilm/model.safetensors").path),
        "requires the pinned MiniLM bundle; run make fetch-mlx ALIASES=minilm")
    setenv("INFERSTREAM_ROOT", root.path, 1)
}

private func assertNear(_ expected: [Float], _ actual: [Float], file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(expected.count, actual.count, file: file, line: line)
    for (a, b) in zip(expected, actual) {
        XCTAssertTrue(a.isFinite && b.isFinite, file: file, line: line)
        XCTAssertEqual(a, b, accuracy: 5e-5, file: file, line: line)
    }
}
