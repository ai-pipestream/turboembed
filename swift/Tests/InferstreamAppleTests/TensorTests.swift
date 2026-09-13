import Foundation
import InferstreamCore
import Testing

@Test func bytesRoundtrip() throws {
    let raw = Tensor.packUTF8(["hello", "", "inference"])
    let back = try Tensor.unpackUTF8(raw)
    #expect(back == ["hello", "", "inference"])
}

@Test func fp32Roundtrip() throws {
    let values: [Float] = [1.0, -2.5, 3.25e10, Float.leastNonzeroMagnitude]
    let raw = Tensor.packFP32(values)
    #expect(raw.count == 16)
    let back = try Tensor.unpackFP32(raw)
    #expect(back == values)
    var reused = Data()
    Tensor.packFP32(values, into: &reused)
    #expect(reused == raw)
}

@Test func outputScratchReusesAfterWarmup() {
    let scratch = OutputScratch()
    var first = scratch.rentBytes(minCap: 64)
    first.append(contentsOf: [1, 2, 3, 4])
    scratch.recycleBytes(first)
    scratch.resetCounters()
    for _ in 0..<8 {
        var slab = scratch.rentBytes(minCap: 64)
        #expect(slab.capacity >= 64)
        slab.append(contentsOf: [9, 8, 7, 6])
        scratch.recycleBytes(slab)
    }
    #expect(scratch.allocs == 0)
}

@Test func truncatedBytesRejected() {
    var raw = Tensor.packUTF8(["hello"])
    raw.removeLast()
    #expect(throws: TensorError.self) {
        _ = try Tensor.unpackBytes(raw)
    }
}

@Test func catalogExpandsAppleAliases() throws {
    let roots = Paths.searchRoots()
    let catalogURL = roots
        .map { $0.appending(path: "config/catalog.toml") }
        .first { FileManager.default.fileExists(atPath: $0.path) }
    try #require(catalogURL != nil)
    let catalog = try Catalog.load(path: catalogURL!.path)
    let minilm = try catalog.resolve(alias: "minilm", arch: .apple)
    #expect(minilm.backend == .mlx)
    #expect(minilm.path == "models/mlx/minilm")
    #expect(minilm.pooling == "mean")
    #expect(minilm.maxSeqLen == 256)
    let bge = try catalog.resolve(alias: "bge-small", arch: .apple)
    #expect(bge.backend == .mlx)
    #expect(bge.pooling == "cls")
    let qwen = try catalog.resolve(alias: "qwen-0.5b", arch: .apple)
    #expect(qwen.backend == .mlx)
    #expect(qwen.path == "models/mlx/qwen-0.5b")
    #expect(throws: CatalogError.self) {
        _ = try catalog.resolve(alias: "mpnet", arch: .apple)
    }
}

@Test func appleTomlServeList() throws {
    let roots = Paths.searchRoots()
    let configURL = roots
        .map { $0.appending(path: "config/apple.toml") }
        .first { FileManager.default.fileExists(atPath: $0.path) }
    try #require(configURL != nil)
    var config = try ServerConfig.load(path: configURL!.path)
    #expect(config.serve.contains("minilm"))
    #expect(config.serve.contains("bge-small"))
    #expect(config.serve.contains("default-llm"))
    #expect(config.auth.mode == .bearer)
    let catalog = try Catalog.load(
        path: configURL!.deletingLastPathComponent().appending(path: "catalog.toml").path)
    try config.expandServe(catalog: catalog)
    let names = Set(config.models.map(\.name))
    #expect(names.contains("minilm"))
    #expect(names.contains("bge-small"))
    #expect(names.contains("default-llm"))
    #expect(names.contains("qwen-0.5b"))
    #expect(names.contains("mock-embed"))
}
