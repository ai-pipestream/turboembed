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
// All wire IO is raw POSIX read/write with stdio stdout forced unbuffered
// (see BenchOverheadCore/WorkerWire.swift for why — the 2026-09-17
// Machine C run deadlocked on the ready handshake through the previous
// FileHandle/stdio layers).
//
// Exiting the process between cases releases the engine, the dylib's MLX
// graphs, and every Metal resource the case accumulated.
//
// `--io-selftest` runs the same command loop with canned replies and no
// dylib, engine, model, or Metal — it exists only to prove the wire
// protocol under pipes and never produces bench numbers or receipts.
// `--io-selftest-parent` spawns this same binary in `--io-selftest` mode
// through the parent-side WorkerClient and exercises every command with
// tight deadlines; it exits nonzero on any protocol failure.

import ArgumentParser
import BenchOverheadCore
import Foundation

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

@main
struct BenchAbiWorker: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench-abi-worker",
        abstract:
            "dlopen-only libTurboEmbed.dylib leg of the Apple overhead pilot (no MLX / Transformers linked)"
    )

    @Option(help: "Path to libTurboEmbed.dylib (required unless --io-selftest).")
    var dylib: String?

    @Option(help: "Rows per request.")
    var batch: Int = 1

    @Option(help: "Target tokens per row.")
    var tokens: Int = 32

    @Flag(help: "Mixed row lengths (see caseTexts).")
    var mixed = false

    @Option(help: "Untimed executions before reporting ready.")
    var warmup: Int = 20

    @Flag(
        help:
            "Wire-protocol self-test worker: canned replies, no dylib/engine/Metal. Never a bench result."
    )
    var ioSelftest = false

    @Flag(
        help:
            "Wire-protocol self-test parent: spawns this binary with --io-selftest and exercises every command under deadlines."
    )
    var ioSelftestParent = false

    mutating func run() async throws {
        if ioSelftestParent {
            try runSelftestParent()
            return
        }
        installWorkerStdIO()
        if ioSelftest {
            try await runWorkerLoop(engine: nil, abi: nil)
            return
        }
        guard let dylib else {
            throw fail("--dylib is required (or pass --io-selftest)")
        }
        let abi = try AbiDylib(path: dylib)
        guard abi.abiVersion() == 1 else {
            throw fail("dylib reports ABI version \(abi.abiVersion()), expected 1")
        }
        let engine = try AbiEngine(dylib: abi)
        try await runWorkerLoop(engine: engine, abi: abi)
    }

    // MARK: - Worker command loop (real engine or selftest canned replies)

    private func runWorkerLoop(engine: AbiEngine?, abi: AbiDylib?) async throws {
        let warmupCount = warmup
        let benchCase = BenchCase(batch: batch, targetTokens: tokens, mixed: mixed)
        let texts = caseTexts(benchCase)
        let views = CTextViews(texts)
        if let engine {
            for _ in 0..<warmupCount {
                try engine.embedOnce(views)
            }
        }
        trace("warmup complete (\(engine == nil ? "selftest, skipped" : String(warmupCount))), sending ready")
        var ready: [String: Any] = ["ready": true, "case": benchCase.name]
        if engine == nil { ready["io_selftest"] = true }
        try emitReply(ready)

        let commands = LineChannel(fd: STDIN_FILENO)
        while let lineData = try commands.readLine(deadlineSeconds: nil, what: "worker stdin") {
            let line = String(decoding: lineData, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let parts = line.split(separator: " ").map(String.init)
            guard let cmd = parts.first else { continue }
            switch cmd {
            case "collect":
                let values: [Float]
                if let engine {
                    values = try engine.embedCollect(views)
                } else {
                    values = [Float](repeating: 0, count: benchCase.batch * kDim)
                }
                try emitReply(["values": values.map(Double.init)])
            case "steady":
                let allocs: UInt64
                if let engine, let abi {
                    abi.allocCounterReset()
                    try engine.embedOnce(views)
                    allocs = abi.allocCounter()
                } else {
                    allocs = 0
                }
                try emitReply(["allocs": allocs])
            case "time":
                guard parts.count == 3, let maxSeconds = Double(parts[1]),
                    let maxRequests = Int(parts[2])
                else {
                    throw fail("bad time command: \(line)")
                }
                let stats = try await timedRepeat(
                    maxSeconds: maxSeconds, maxRequests: maxRequests
                ) {
                    if let engine {
                        try engine.embedOnce(views)
                    } else {
                        usleep(200)
                    }
                }
                try emitReply(["stats": stats.json])
            case "concurrent":
                guard parts.count == 3, let maxSeconds = Double(parts[1]),
                    let maxRequests = Int(parts[2])
                else {
                    throw fail("bad concurrent command: \(line)")
                }
                if let engine, let abi {
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
                    try emitReply(["engine_a": a.json, "engine_b": b.json])
                } else {
                    async let statsA = timedRepeat(
                        maxSeconds: maxSeconds, maxRequests: maxRequests
                    ) { usleep(200) }
                    async let statsB = timedRepeat(
                        maxSeconds: maxSeconds, maxRequests: maxRequests
                    ) { usleep(200) }
                    let (a, b) = try await (statsA, statsB)
                    try emitReply(["engine_a": a.json, "engine_b": b.json])
                }
            case "exit":
                try emitReply(["ok": true])
                return
            default:
                throw fail("unknown worker command: \(line)")
            }
        }
        // Orchestrator closed our stdin without an exit command (e.g. it
        // died). Exit cleanly; the engine is destroyed with the process.
        trace("stdin closed without exit command, quitting")
    }

    private func trace(_ message: String) {
        FileHandle.standardError.write(
            Data("bench-abi-worker[\(ProcessInfo.processInfo.processIdentifier)]: \(message)\n".utf8))
    }

    // MARK: - Selftest parent (proves the wire protocol without Metal)

    private func runSelftestParent() throws {
        let selfPath = URL(fileURLWithPath: CommandLine.arguments[0]).path
        func check(_ condition: Bool, _ what: String) throws {
            guard condition else { throw fail("io-selftest: \(what)") }
        }

        // Handshake + every command, all under tight deadlines. A stuck
        // pipe fails here in seconds, not after wedging a bench host.
        let worker = try WorkerClient(
            executable: selfPath,
            arguments: ["--io-selftest", "--batch", "2", "--tokens", "16", "--warmup", "0"],
            label: "io-selftest worker", readySeconds: 30)

        let collect = try worker.request("collect", timeoutSeconds: 30)
        let values = collect["values"] as? [Any]
        try check(values?.count == 2 * kDim, "collect returned \(values?.count ?? -1) values, expected \(2 * kDim)")

        let steady = try worker.request("steady", timeoutSeconds: 30)
        try check((steady["allocs"] as? NSNumber) != nil, "steady reply missing allocs: \(steady)")

        let time = try worker.request("time 0.2 50", timeoutSeconds: 30)
        let stats = time["stats"] as? [String: Any]
        try check((stats?["p50_us"] as? NSNumber) != nil, "time reply missing stats.p50_us: \(time)")

        let concurrent = try worker.request("concurrent 0.2 50", timeoutSeconds: 30)
        try check(
            concurrent["engine_a"] is [String: Any] && concurrent["engine_b"] is [String: Any],
            "concurrent reply missing engine stats: \(concurrent)")

        worker.shutdown(timeoutSeconds: 30)

        // The fail-loud path: a worker that never becomes ready must
        // surface as a timeout error, not a hang. /bin/cat speaks no
        // protocol and never replies.
        do {
            _ = try WorkerClient(
                executable: "/bin/cat", arguments: [],
                label: "io-selftest silent worker", readySeconds: 2)
            throw fail("io-selftest: a silent worker did not trip the ready timeout")
        } catch let error as BenchError {
            let text = String(describing: error)
            guard text.contains("timed out") else { throw error }
        }

        FileHandle.standardError.write(
            Data("bench-abi-worker --io-selftest-parent: wire protocol OK (handshake, all commands, ready timeout)\n".utf8))
    }
}
