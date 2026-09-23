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
        let (name, dtype, shape) = try result.output(0)
        expect(name == "embeddings", "name == \"embeddings\"")
        expect(dtype == .f32, "dtype == .f32")
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

    /// An empty input row crosses as `ptr == NULL, len == 0`, which
    /// `turbo_types.h` permits; `withTexts` must not allocate a byte for it
    /// (nothing would free it) and must not let a buffer pointer escape the
    /// scope that produced it.
    func emptyTextsCrossAsNullViews() throws {
        let rt = try Runtime()
        let ctx = try rt.createContext(device: try mockDevice(rt))
        let model = try ctx.loadModel(bundlePath: bundle("embedding"))
        let dim = Int(model.info.dim)
        let session = try model.createSession(maxBatch: 3)
        for _ in 0..<64 {
            try session.writeText(["", "not empty", ""])
            let r = try session.run()
            let v = try r.readFloats(0)
            expect(v.count == 3 * dim, "v.count == 3 * dim")
            r.close()
        }
    }

    /// The readers report the output's dtype rather than reinterpreting it:
    /// an i32 output read as floats would be silently wrong numbers.
    func readersRefuseAMismatchedOutputDtype() throws {
        let rt = try Runtime()
        let ctx = try rt.createContext(device: try mockDevice(rt))
        let model = try ctx.loadModel(bundlePath: bundle("reranker"))
        let session = try model.createSession(maxBatch: 4)
        var opts = RerankOptions()
        opts.returnSorted = true
        try session.writePairs(query: "what is turbo", documents: ["turbo is a library", "unrelated text"], options: opts)
        let result = try session.run()
        expect(try result.output(0).dtype == .f32, "scores are f32")
        expect(try result.output(1).dtype == .i32, "sorted indices are i32")
        let asInts = thrown { _ = try result.readInts(0) }
        expect(asInts?.code == TURBO_E_UNSUPPORTED_DTYPE, "readInts on an f32 output is UNSUPPORTED_DTYPE")
        let asFloats = thrown { _ = try result.readFloats(1) }
        expect(asFloats?.code == TURBO_E_UNSUPPORTED_DTYPE, "readFloats on an i32 output is UNSUPPORTED_DTYPE")
        result.close()
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
        let rt = try Runtime()
        let ctx = try rt.createContext(device: try mockDevice(rt))
        let model = try ctx.loadModel(bundlePath: bundle("embedding"))
        let session = try model.createSession(maxBatch: 2)
        // Parents closed first, explicitly (not just dropped by ARC, which
        // the child's back-reference would prevent): the C side keeps them
        // alive for the session.
        model.close(); ctx.close(); rt.close()
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

    func generationStepsUntilLengthAndCancelIsReported() throws {
        let rt = try Runtime()
        let ctx = try rt.createContext(device: try mockDevice(rt))
        let model = try ctx.loadModel(bundlePath: bundle("generative"))
        expect(model.info.task == .generate, "a generative bundle")
        let prompt = [Message.user("say something")]
        var d = GenerateDesc(); d.maxNewTokens = 6
        let g = try model.createGeneration(d)
        try g.prompt(prompt)
        var tokens: [Int32] = []
        var text = ""
        let last = try g.drain { c in
            tokens += c.tokens; text += c.text
            if !c.done { expect(c.finishReason == .none, "an unfinished chunk names no reason") }
            return true
        }
        expect(last.done && last.finishReason == .length, "finishes with LENGTH")
        expect(!tokens.isEmpty && tokens.count <= 6, "max_new_tokens holds: \(tokens.count)")
        expect(!text.isEmpty, "the stream produced text")
        expect(tokens.allSatisfy { $0 >= 0 }, "token ids are never negative")
        var d2 = GenerateDesc(); d2.maxNewTokens = 50
        let g2 = try model.createGeneration(d2)
        try g2.prompt(prompt)
        let first = try g2.step()
        expect(first.promptTokens > 0 && first.sequence == 0 && !first.done, "the first chunk reports the prompt size")
        try g2.cancel()
        let end = try g2.step()
        expect(end.done && end.finishReason == .cancelled, "cancel is reported on the next step")
        let g3 = try model.createGeneration(d2)
        try g3.prompt(prompt)
        let stopped = try g3.drain { _ in false }
        expect(stopped.finishReason == .cancelled, "a stopping sink cancels")
    }

    func generationIsRepeatableWithASeedAndRefusesUnsupportedOptionsByField() throws {
        let rt = try Runtime()
        let ctx = try rt.createContext(device: try mockDevice(rt))
        let model = try ctx.loadModel(bundlePath: bundle("generative"))
        let prompt = [Message.user("say something")]
        var runs: [[Int32]] = []
        for _ in 0..<2 {
            var d = GenerateDesc(); d.maxNewTokens = 8; d.temperature = 0.9; d.seed = 42
            let g = try model.createGeneration(d)
            try g.prompt(prompt)
            var out: [Int32] = []
            try g.drain { c in out += c.tokens; return true }
            runs.append(out)
        }
        expect(runs[0] == runs[1], "one seed reproduces one token sequence")
        let dev = try rt.device(ctx.deviceIndex)
        if !dev.has(UInt64(TURBO_CAP_OPT_GEN_N)) {
            var d = GenerateDesc(); d.maxNewTokens = 2; d.nSequences = 3
            let e = thrown { _ = try model.createGeneration(d) }
            expect(e?.code == TURBO_E_UNSUPPORTED_OPTION, "n_sequences is refused: \(String(describing: e))")
            expect(e?.field == 4, "the rejection names n_sequences (field 4): \(String(describing: e))")
        }
    }

    func tokenizerEncodesDecodesAndCounts() throws {
        let rt = try Runtime()
        let tok = try rt.createTokenizer(bundlePath: Self.bundles + "/../minilm-tokenizer")
        expect(tok.info.kind == "wordpiece", "kind is wordpiece")
        expect(tok.info.vocabSize > 1000 && tok.info.specialsPerSequence == 2, "MiniLM tokenizer facts")
        var o = EncodeOptions(); o.maxTokens = 16
        let enc = try tok.encode(["hello world", "a longer sentence with several words"], rowStride: 16, options: o)
        expect(enc.lengths[0] == 4, "[CLS] hello world [SEP]")
        expect(enc.lengths[1] > enc.lengths[0], "the longer text has more tokens")
        for r in 0..<2 {
            for c in 0..<16 {
                expect(enc.mask[r * 16 + c] == (c < Int(enc.lengths[r]) ? 1 : 0), "mask row \(r) col \(c)")
                if c >= Int(enc.lengths[r]) { expect(enc.ids[r * 16 + c] == tok.info.padId, "padding carries the pad id") }
            }
        }
        expect(try tok.decode(enc.row(0)) == "hello world", "decode round-trips")
        expect(try tok.count("hello world") == 4 && tok.count("hello world", addSpecialTokens: false) == 2, "count")
        var none = EncodeOptions(); none.maxTokens = 6; none.truncate = .none
        let e = thrown { _ = try tok.encode(["one two three four five six seven eight nine ten"], rowStride: 6, options: none) }
        expect(e?.code == TURBO_E_CAPACITY, "NONE over budget is a capacity error: \(String(describing: e))")
    }

    func aHeldResultMakesEveryRunOnAnotherThreadBusy() throws {
        let rt = try Runtime()
        let ctx = try rt.createContext(device: try mockDevice(rt))
        let model = try ctx.loadModel(bundlePath: bundle("embedding"))
        let session = try model.createSession(maxBatch: 2)
        try session.writeText(["held"])
        let held = try session.run()
        let lock = NSLock()
        var busy = 0, other = 0
        let group = DispatchGroup()
        group.enter()
        Thread {
            for _ in 0..<20 {
                if let e = thrown({ try session.writeText(["contender"]) }) {
                    lock.lock(); if e.code == TURBO_E_BUSY { busy += 1 } else { other += 1 }; lock.unlock()
                } else {
                    lock.lock(); other += 1; lock.unlock()
                }
            }
            group.leave()
        }.start()
        group.wait()
        expect(busy == 20, "every attempt while the result is held is BUSY (\(busy))")
        expect(other == 0, "nothing else happened (\(other))")
        held.close()
        try session.writeText(["after"])
        _ = try session.run()
    }

    func cancelFromAnotherThreadIsNeverBusyAndEndsTheStream() throws {
        let rt = try Runtime()
        let ctx = try rt.createContext(device: try mockDevice(rt))
        let model = try ctx.loadModel(bundlePath: bundle("generative"))
        var d = GenerateDesc(); d.maxNewTokens = 64
        let g = try model.createGeneration(d)
        try g.prompt([Message.user("say something")])
        let first = try g.step()
        expect(!first.done, "first chunk is not the last")
        let group = DispatchGroup()
        var failed = false
        group.enter()
        Thread {
            if thrown({ try g.cancel() }) != nil { failed = true }
            group.leave()
        }.start()
        group.wait()
        expect(!failed, "cancel from another thread never fails")
        let next = try g.step()
        expect(next.done && next.finishReason == .cancelled, "the step after a cancel is the last one and says CANCELLED")
    }
}


// MARK: - Runner

let suite = ConformanceTests()
let cases: [(String, () throws -> Void)] = [
    ("abiVersionMatchesTheHeader", suite.abiVersionMatchesTheHeader),
    ("devicesAreEnumeratedAndAutoNeverSelectsCpu", suite.devicesAreEnumeratedAndAutoNeverSelectsCpu),
    ("capabilityCellsAreHonest", suite.capabilityCellsAreHonest),
    ("embeddingIsDeterministicAndUnitNorm", suite.embeddingIsDeterministicAndUnitNorm),
    ("emptyTextsCrossAsNullViews", suite.emptyTextsCrossAsNullViews),
    ("readersRefuseAMismatchedOutputDtype", suite.readersRefuseAMismatchedOutputDtype),
    ("unsupportedOptionsNameTheirField", suite.unsupportedOptionsNameTheirField),
    ("rerankReturnsScoresAndSortedOrder", suite.rerankReturnsScoresAndSortedOrder),
    ("tokenClassificationYieldsSpansInsideTheText", suite.tokenClassificationYieldsSpansInsideTheText),
    ("aHeldResultBlocksTheSessionAndParentsOutliveTheirChildren", suite.aHeldResultBlocksTheSessionAndParentsOutliveTheirChildren),
    ("concurrentUseOfOneSessionIsBusyNeverWrong", suite.concurrentUseOfOneSessionIsBusyNeverWrong),
    ("generationStepsUntilLengthAndCancelIsReported", suite.generationStepsUntilLengthAndCancelIsReported),
    ("generationIsRepeatableWithASeedAndRefusesUnsupportedOptionsByField", suite.generationIsRepeatableWithASeedAndRefusesUnsupportedOptionsByField),
    ("tokenizerEncodesDecodesAndCounts", suite.tokenizerEncodesDecodesAndCounts),
    ("aHeldResultMakesEveryRunOnAnotherThreadBusy", suite.aHeldResultMakesEveryRunOnAnotherThreadBusy),
    ("cancelFromAnotherThreadIsNeverBusyAndEndsTheStream", suite.cancelFromAnotherThreadIsNeverBusyAndEndsTheStream),
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
