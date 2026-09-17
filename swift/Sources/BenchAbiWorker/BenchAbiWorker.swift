// dlopen-only ABI leg of the Apple matched native-overhead pilot.
//
// This executable links **no** MLX and **no** swift-transformers: the only
// copy of those Objective-C classes in this process comes from
// `libTurboEmbed.dylib` itself, resolved with dlopen/dlsym. That is the
// point — running the ABI leg in the same process as the direct mlx-swift
// baseline duplicated the Tokenizers/MLX classes (objc collision warnings)
// and doubled the Metal resource footprint, which distorted ABI timings
// and eventually hit the Metal resource limit on the 18-case grid.
//
// The `bench-apple-overhead` orchestrator spawns one worker per case with
// the case parameters on argv. The worker dlopens the dylib, creates one
// METAL engine, loads `minilm`, warms up, prints `{"ready":true}` on
// stdout, and then serves newline-delimited commands on stdin, replying
// with one JSON object per line on stdout (diagnostics go to stderr):
//
//   collect                    -> {"values":[...]}   parity vectors (batch*dim)
//   steady                     -> {"allocs":N}       arena allocs on one request
//   time <maxSeconds> <maxReq> -> {"stats":{...}}    one timed repeat
//   concurrent <maxSeconds> <maxReq> -> {"engine_a":{...},"engine_b":{...}}
//   exit                       -> {"ok":true}        destroy engine and quit
//
// Exiting the process between cases releases the engine, the dylib's MLX
// graphs, and every Metal resource the case accumulated.

import ArgumentParser
import BenchOverheadCore
import Darwin
import Foundation

func emit(_ payload: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: payload)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

@main
struct BenchAbiWorker: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench-abi-worker",
        abstract:
            "dlopen-only libTurboEmbed.dylib leg of the Apple overhead pilot (no MLX / Transformers linked)"
    )

    @Option(help: "Path to libTurboEmbed.dylib.")
    var dylib: String

    @Option(help: "Rows per request.")
    var batch: Int

    @Option(help: "Target tokens per row.")
    var tokens: Int

    @Flag(help: "Mixed row lengths (see caseTexts).")
    var mixed = false

    @Option(help: "Untimed executions before reporting ready.")
    var warmup: Int = 20

    mutating func run() async throws {
        let warmupCount = warmup
        let abi = try AbiDylib(path: dylib)
        guard abi.abiVersion() == 1 else {
            throw fail("dylib reports ABI version \(abi.abiVersion()), expected 1")
        }
        let engine = try AbiEngine(dylib: abi)
        let benchCase = BenchCase(batch: batch, targetTokens: tokens, mixed: mixed)
        let texts = caseTexts(benchCase)
        let views = CTextViews(texts)
        for _ in 0..<warmupCount {
            try engine.embedOnce(views)
        }
        try emit(["ready": true, "case": benchCase.name])

        while let line = readLine(strippingNewline: true) {
            let parts = line.split(separator: " ").map(String.init)
            guard let cmd = parts.first else { continue }
            switch cmd {
            case "collect":
                let values = try engine.embedCollect(views)
                try emit(["values": values.map(Double.init)])
            case "steady":
                abi.allocCounterReset()
                try engine.embedOnce(views)
                try emit(["allocs": abi.allocCounter()])
            case "time":
                guard parts.count == 3, let maxSeconds = Double(parts[1]),
                    let maxRequests = Int(parts[2])
                else {
                    throw fail("bad time command: \(line)")
                }
                let stats = try await timedRepeat(
                    maxSeconds: maxSeconds, maxRequests: maxRequests
                ) {
                    try engine.embedOnce(views)
                }
                try emit(["stats": stats.json])
            case "concurrent":
                guard parts.count == 3, let maxSeconds = Double(parts[1]),
                    let maxRequests = Int(parts[2])
                else {
                    throw fail("bad concurrent command: \(line)")
                }
                let engineB = try AbiEngine(dylib: abi)
                let viewsB = CTextViews(texts)
                for _ in 0..<warmupCount {
                    try engineB.embedOnce(viewsB)
                }
                async let statsA = timedRepeat(
                    maxSeconds: maxSeconds, maxRequests: maxRequests
                ) {
                    try engine.embedOnce(views)
                }
                async let statsB = timedRepeat(
                    maxSeconds: maxSeconds, maxRequests: maxRequests
                ) {
                    try engineB.embedOnce(viewsB)
                }
                let (a, b) = try await (statsA, statsB)
                try emit(["engine_a": a.json, "engine_b": b.json])
            case "exit":
                try emit(["ok": true])
                return
            default:
                throw fail("unknown worker command: \(line)")
            }
        }
    }
}
