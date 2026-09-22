// The conformance cases that define the binding, through the Swift API
// against the mock provider. Needs libturbo on the loader path and the mock
// bundle root in TURBO_BUNDLES (default: ../../testdata/bundles/mock).

import CTurbo
import Foundation
import PipestreamTurbo

// A self-contained runner rather than XCTest or Swift Testing: neither
// framework is available with the command line tools alone, and the
// conformance cases must run on any Mac with a Swift toolchain.

var failures: [String] = []

func expect(_ cond: Bool, _ what: String, file: String = #fileID, line: Int = #line) {
    if !cond { failures.append("\(file):\(line): \(what)") }
}

/// Thrown-error helper: the error a call throws, or nil.
func thrown(_ body: () throws -> Void) -> TurboError? {
    do { try body() } catch let e as TurboError { return e } catch { return nil }
    return nil
}

struct ConformanceTests {
    static let bundles: String = {
        if let env = ProcessInfo.processInfo.environment["TURBO_BUNDLES"] { return env }
        let here = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        return here.appendingPathComponent("../../../../testdata/bundles/mock").standardized.path
    }()

    func bundle(_ kind: String) -> String { Self.bundles + "/" + kind }

    /// The mock accelerator: AUTO never picks the mock CPU device.
    func mockDevice(_ rt: Runtime) throws -> UInt32 {
        let idx = try rt.selectDevice()
        let dev = try rt.device(idx)
        expect(dev.kind != .cpu, "AUTO must never select a CPU")
        expect(dev.providerId == "mock", "dev.providerId == \"mock\"")
        return idx
    }

    func abiVersionMatchesTheHeader() {
        expect(Runtime.abiVersion == TURBO_ABI_VERSION, "Runtime.abiVersion == TURBO_ABI_VERSION")
    }

    func devicesAreEnumeratedAndAutoNeverSelectsCpu() throws {
        let rt = try Runtime()
        let devices = try rt.devices()
        expect(!devices.isEmpty, "!devices.isEmpty")
        expect(devices.contains { $0.kind == .cpu }, "the mock offers a CPU device")
        expect(devices.contains { $0.kind != .cpu }, "and an accelerator")
        _ = try mockDevice(rt)
        let e = thrown { _ = try rt.selectDevice(policy: .explicit, providerId: "nonexistent") }
        expect(e?.code == TURBO_E_DEVICE_NOT_FOUND, "e?.code == TURBO_E_DEVICE_NOT_FOUND")
    }

    func capabilityCellsAreHonest() throws {
        let rt = try Runtime()
        let idx = try mockDevice(rt)
        let embed = try rt.capability(device: idx, task: .embed, modality: .text)
        expect(embed.status == .supported, "embed.status == .supported")
        expect(embed.offered, "embed.offered")
        let audio = try rt.capability(device: idx, task: .embed, modality: .audio)
        expect(audio.status == .unsupported, "audio.status == .unsupported")
    }

    func embeddingIsDeterministicAndUnitNorm() throws {
        let rt = try Runtime()
        let ctx = try rt.createContext(device: try mockDevice(rt))
        let model = try ctx.loadModel(bundlePath: bundle("embedding"))
        expect(model.info.task == .embed, "model.info.task == .embed")
        expect(model.info.providerId == "mock", "model.info.providerId == \"mock\"")
        let dim = Int(model.info.dim)
        expect(dim > 0, "dim > 0")
        let session = try model.createSession(maxBatch: 4)
        try session.writeText(["hello world", "hello world", "something else"])
        let result = try session.run()
        expect(try result.outputCount == 1, "try result.outputCount == 1")
        let (name, shape) = try result.output(0)
        expect(name == "embeddings", "name == \"embeddings\"")
        expect(shape == [3, Int64(dim)], "shape == [3, Int64(dim)]")
        expect(try result.placement == .host, "try result.placement == .host")
        let v = try result.readFloats(0)
        result.close()
        expect(v.count == 3 * dim, "v.count == 3 * dim")
        for row in 0..<3 {
            let norm = sqrt(v[row * dim..<(row + 1) * dim].reduce(0) { $0 + Double($1 * $1) })
            expect(abs(norm - 1.0) < 1e-4, "row \(row) is unit norm")
        }
        expect(Array(v[0..<dim]) == Array(v[dim..<2 * dim]), "identical texts embed identically")
        expect(try session.stats().runs == 1, "try session.stats().runs == 1")
    }

    func unsupportedOptionsNameTheirField() throws {
        let rt = try Runtime()
        let ctx = try rt.createContext(device: try mockDevice(rt))
        let dev = try rt.device(ctx.deviceIndex)
        expect(!dev.has(UInt64(TURBO_CAP_OPT_POOLING_OVERRIDE)), "the mock does not override pooling")
        let model = try ctx.loadModel(bundlePath: bundle("embedding"))
        let session = try model.createSession(maxBatch: 2, maxSeq: 16)
        var cls = EmbedOptions()
        cls.pooling = .cls
        let t = thrown { try session.writeText(["x"], options: cls) }
        expect(t?.code == TURBO_E_UNSUPPORTED_OPTION, "t?.code == TURBO_E_UNSUPPORTED_OPTION")
        expect(t?.field == 6, "pooling is field 6")
        expect(t?.statusName == "TURBO_E_UNSUPPORTED_OPTION", "t?.statusName == \"TURBO_E_UNSUPPORTED_OPTION\"")
        var big = EmbedOptions()
        big.maxTokens = 64
        let cap = thrown { try session.writeText(["x"], options: big) }
        expect(cap?.code == TURBO_E_CAPACITY, "a budget the session cannot hold is refused")
        expect(cap?.field == 3, "cap?.field == 3")
    }

    func rerankReturnsScoresAndSortedOrder() throws {
        let rt = try Runtime()
        let ctx = try rt.createContext(device: try mockDevice(rt))
        let model = try ctx.loadModel(bundlePath: bundle("reranker"))
        let session = try model.createSession(maxBatch: 4)
        var opts = RerankOptions()
        opts.returnSorted = true
        try session.writePairs(query: "what is turbo", documents: ["turbo is a library", "unrelated text", "what is turbo"], options: opts)
        let result = try session.run()
        expect(try result.outputCount == 2, "try result.outputCount == 2")
        let scores = try result.readFloats(0)
        let sorted = try result.readInts(1)
        expect(scores.count == 3, "scores.count == 3")
        expect(sorted.count == 3, "sorted.count == 3")
        for i in 1..<sorted.count {
            expect(scores[Int(sorted[i - 1])] >= scores[Int(sorted[i])], "sorted is descending")
        }
        expect(scores.allSatisfy { $0 >= 0 && $0 <= 1 }, "sigmoid scores")
    }

    func tokenClassificationYieldsSpansInsideTheText() throws {
        let rt = try Runtime()
        let ctx = try rt.createContext(device: try mockDevice(rt))
        let model = try ctx.loadModel(bundlePath: bundle("token-classifier"))
        expect(model.info.labels.first == "O", "model.info.labels.first == \"O\"")
        let session = try model.createSession(maxBatch: 2)
        let text = "Ada visited Berlin"
        try session.writeTextClassify([text])
        let result = try session.run()
        for span in try result.spans() {
            expect(span.row == 0, "span.row == 0")
            expect(span.byteStart < span.byteEnd, "span.byteStart < span.byteEnd")
            expect(span.byteEnd <= UInt64(text.utf8.count), "span.byteEnd <= UInt64(text.utf8.count)")
            expect(span.label > 0 && Int(span.label) < model.info.labels.count, "\(span)")
            expect(span.score > 0 && span.score <= 1, "span.score > 0 && span.score <= 1")
        }
    }

    func aHeldResultBlocksTheSessionAndParentsOutliveTheirChildren() throws {
        var rt: Runtime? = try Runtime()
        var ctx: Context? = try rt!.createContext(device: try mockDevice(rt!))
        var model: Model? = try ctx!.loadModel(bundlePath: bundle("embedding"))
        let session = try model!.createSession(maxBatch: 2)
        // Parents released first: the session keeps them alive.
        rt = nil; ctx = nil; model = nil
        try session.writeText(["still works"])
        let result = try session.run()
        let busy = thrown { try session.writeText(["again"]) }
        expect(busy?.code == TURBO_E_BUSY, "a live result leases the session")
        result.close()
        try session.writeText(["again"])
        let second = try session.run()
        expect(try second.outputCount == 1, "try second.outputCount == 1")
    }

    func concurrentUseOfOneSessionIsBusyNeverWrong() throws {
        let rt = try Runtime()
        let ctx = try rt.createContext(device: try mockDevice(rt))
        let model = try ctx.loadModel(bundlePath: bundle("embedding"))
        let session = try model.createSession(maxBatch: 8)
        let dim = Int(model.info.dim)
        let lock = NSLock()
        var ok = 0, busy = 0, other = 0
        let group = DispatchGroup()
        for _ in 0..<4 {
            group.enter()
            Thread {
                for _ in 0..<50 {
                    do {
                        try session.writeText(["thread text"])
                        let r = try session.run()
                        let v = try r.readFloats(0)
                        r.close()
                        lock.lock(); if v.count == dim { ok += 1 } else { other += 1 }; lock.unlock()
                    } catch let e as TurboError {
                        lock.lock(); if e.code == TURBO_E_BUSY { busy += 1 } else { other += 1 }; lock.unlock()
                    } catch {
                        lock.lock(); other += 1; lock.unlock()
                    }
                }
                group.leave()
            }.start()
        }
        group.wait()
        expect(other == 0, "only TURBO_E_BUSY is acceptable under contention")
        expect(ok > 0, "some runs complete")
    }
}


// MARK: - Runner

let suite = ConformanceTests()
let cases: [(String, () throws -> Void)] = [
    ("abiVersionMatchesTheHeader", suite.abiVersionMatchesTheHeader),
    ("devicesAreEnumeratedAndAutoNeverSelectsCpu", suite.devicesAreEnumeratedAndAutoNeverSelectsCpu),
    ("capabilityCellsAreHonest", suite.capabilityCellsAreHonest),
    ("embeddingIsDeterministicAndUnitNorm", suite.embeddingIsDeterministicAndUnitNorm),
    ("unsupportedOptionsNameTheirField", suite.unsupportedOptionsNameTheirField),
    ("rerankReturnsScoresAndSortedOrder", suite.rerankReturnsScoresAndSortedOrder),
    ("tokenClassificationYieldsSpansInsideTheText", suite.tokenClassificationYieldsSpansInsideTheText),
    ("aHeldResultBlocksTheSessionAndParentsOutliveTheirChildren", suite.aHeldResultBlocksTheSessionAndParentsOutliveTheirChildren),
    ("concurrentUseOfOneSessionIsBusyNeverWrong", suite.concurrentUseOfOneSessionIsBusyNeverWrong),
]
var passed = 0
for (name, body) in cases {
    let before = failures.count
    do { try body() } catch { failures.append("\(name): threw \(error)") }
    if failures.count == before { passed += 1; print("ok   \(name)") } else { print("FAIL \(name)") }
}
for f in failures { print("  \(f)") }
print("\(passed) passed, \(cases.count - passed) failed")
exit(failures.isEmpty ? 0 : 1)
