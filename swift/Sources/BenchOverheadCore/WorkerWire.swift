// Wire protocol between the `bench-apple-overhead` orchestrator and the
// isolated `bench-abi-worker` process: newline-delimited commands on the
// worker's stdin, one JSON object per line on its stdout.
//
// The 2026-09-17 Machine C run wedged on the very first case: the worker
// had finished warmup and was parked reading stdin while the orchestrator
// never observed the `{"ready":true}` line and blocked forever. Both ends
// of that handshake went through layers with buffering/queueing behavior
// we do not control (`FileHandle.write`/`FileHandle.read(upToCount:)`
// dispatch machinery in the parent, C stdio `readLine` in the worker),
// and the harness had no deadline, so the hang wedged the host instead of
// failing the make target.
//
// This file removes every such layer from the wire path:
//
// - All reads and writes are raw POSIX `read(2)`/`write(2)` on the pipe
//   file descriptors, with partial-write and EINTR handling. Nothing on
//   the wire path can sit in a userspace buffer.
// - The worker additionally forces C stdio stdout to unbuffered and
//   flushes it before every reply, so any stdio output from the dlopen'd
//   dylib cannot interleave into (or delay ahead of) a reply line.
// - Every parent-side read has a monotonic deadline enforced with
//   `poll(2)`. If the worker does not answer in time the orchestrator
//   kills it and fails loudly with the phase and pid instead of hanging.
// - The parent closes its copies of the worker's pipe ends right after
//   spawn, so a crashed worker produces EOF (a loud error) instead of a
//   silent forever-blocking read.
//
// The protocol itself is testable without Metal, models, or the dylib:
// `bench-abi-worker --io-selftest-parent` spawns itself in
// `--io-selftest` mode and exercises ready/collect/steady/time/
// concurrent/exit through this exact code path.

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif
import Foundation

// MARK: - Raw POSIX IO

public enum WireIO {
    /// Write all of `data` to `fd`, retrying on EINTR and partial writes.
    public static func writeAll(fd: Int32, _ data: Data) throws {
        try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let base = raw.baseAddress else { return }
            var offset = 0
            while offset < raw.count {
                let n = write(fd, base + offset, raw.count - offset)
                if n > 0 {
                    offset += n
                    continue
                }
                if n < 0 && errno == EINTR { continue }
                throw fail(
                    "write(fd \(fd)) failed after \(offset)/\(raw.count) bytes: "
                        + String(cString: strerror(errno)))
            }
        }
    }

    /// Write `data` followed by a newline.
    public static func writeLine(fd: Int32, _ data: Data) throws {
        var line = data
        line.append(UInt8(ascii: "\n"))
        try writeAll(fd: fd, line)
    }

    /// Monotonic now in nanoseconds (same clock as the timing helpers).
    public static func monotonicNs() -> UInt64 { nowNs() }
}

/// Reads newline-delimited lines from a raw file descriptor with an
/// optional monotonic deadline per line, using `poll(2)` + `read(2)`.
/// No stdio, no FileHandle — the buffer here is the only userspace
/// buffering on the read side.
public final class LineChannel {
    private let fd: Int32
    private var buffer = Data()
    private var sawEof = false

    public init(fd: Int32) {
        self.fd = fd
    }

    /// Next full line (without the trailing newline), or nil on clean EOF
    /// at a line boundary. Throws on timeout (naming `what`), on EOF in
    /// the middle of a line, and on read errors.
    public func readLine(deadlineSeconds: Double?, what: String) throws -> Data? {
        let deadlineNs: UInt64? = deadlineSeconds.map {
            WireIO.monotonicNs() + UInt64(max($0, 0) * 1e9)
        }
        while true {
            if let newline = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                let line = Data(buffer[buffer.startIndex..<newline])
                buffer.removeSubrange(buffer.startIndex...newline)
                return line
            }
            if sawEof {
                if buffer.isEmpty { return nil }
                throw fail("\(what): EOF in the middle of a line (\(buffer.count) bytes buffered)")
            }
            if let deadlineNs {
                let now = WireIO.monotonicNs()
                guard now < deadlineNs else {
                    throw fail("\(what): timed out waiting for a reply line")
                }
                let remainingMs = Int32(min((deadlineNs - now) / 1_000_000 + 1, 60_000))
                var pfd = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
                let rc = poll(&pfd, 1, remainingMs)
                if rc < 0 {
                    if errno == EINTR { continue }
                    throw fail("\(what): poll failed: " + String(cString: strerror(errno)))
                }
                if rc == 0 { continue }  // re-check deadline
            }
            var chunk = [UInt8](repeating: 0, count: 1 << 16)
            let n = read(fd, &chunk, chunk.count)
            if n > 0 {
                buffer.append(contentsOf: chunk[0..<n])
                continue
            }
            if n == 0 {
                sawEof = true
                continue
            }
            if errno == EINTR { continue }
            throw fail("\(what): read failed: " + String(cString: strerror(errno)))
        }
    }
}

// MARK: - Worker-side reply channel

/// Call once at worker startup, before any reply: ignore SIGPIPE (a
/// vanished parent must surface as a write error, not kill the worker
/// silently) and force C stdio stdout to unbuffered so nothing the
/// dlopen'd dylib prints can sit in front of a reply.
public func installWorkerStdIO() {
    signal(SIGPIPE, SIG_IGN)
    setvbuf(stdout, nil, _IONBF, 0)
}

/// Emit one JSON reply line on the worker's stdout via a raw write to
/// STDOUT_FILENO. Flushes stdio stdout first so any buffered stdio output
/// (from the dylib or diagnostics) cannot interleave into the line.
public func emitReply(_ payload: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: payload)
    fflush(stdout)
    try WireIO.writeLine(fd: STDOUT_FILENO, data)
}

// MARK: - Parent-side worker client

/// Spawns one worker process and speaks the line protocol to it with
/// hard deadlines. Every read has a timeout; a timeout or EOF kills the
/// worker and throws with the phase and pid, so a wedged worker fails
/// the run loudly instead of hanging it.
public final class WorkerClient {
    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let channel: LineChannel
    private let label: String

    public var pid: Int32 { process.processIdentifier }

    /// Spawn `executable` with `arguments` and block until it replies
    /// `{"ready":true}` (within `readySeconds`). The worker's stderr
    /// passes through to ours.
    public init(
        executable: String, arguments: [String], label: String, readySeconds: Double
    ) throws {
        self.label = label
        self.channel = LineChannel(fd: stdoutPipe.fileHandleForReading.fileDescriptor)
        signal(SIGPIPE, SIG_IGN)
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = FileHandle.standardError
        try process.run()
        // Close our copies of the worker's ends so a worker crash gives
        // this process EOF (a loud error below) instead of a read that
        // can never complete.
        try? stdinPipe.fileHandleForReading.close()
        try? stdoutPipe.fileHandleForWriting.close()
        let ready = try readReply(
            timeoutSeconds: readySeconds, what: "\(label): waiting for ready")
        guard ready["ready"] as? Bool == true else {
            killAndReap()
            throw fail("\(label): worker did not become ready: \(ready)")
        }
    }

    deinit {
        if process.isRunning {
            process.terminate()
        }
    }

    private func send(_ line: String) throws {
        guard process.isRunning else {
            throw fail("\(label): worker (pid \(pid)) exited before command: \(line)")
        }
        try WireIO.writeLine(
            fd: stdinPipe.fileHandleForWriting.fileDescriptor, Data(line.utf8))
    }

    private func readReply(timeoutSeconds: Double, what: String) throws -> [String: Any] {
        let lineData: Data?
        do {
            lineData = try channel.readLine(deadlineSeconds: timeoutSeconds, what: what)
        } catch {
            killAndReap()
            throw fail(
                "\(error) — killed worker pid \(pid); a healthy worker answers well inside "
                    + "\(Int(timeoutSeconds))s, so this is the fail-loud path for the "
                    + "hang previously observed on Machine C (sample the worker before "
                    + "retrying if it recurs)")
        }
        guard let lineData else {
            killAndReap()
            throw fail("\(what): worker (pid \(pid)) closed its output — see its stderr above")
        }
        guard let obj = try JSONSerialization.jsonObject(with: lineData) as? [String: Any]
        else {
            killAndReap()
            throw fail(
                "\(label): worker sent a non-JSON line: "
                    + String(decoding: lineData, as: UTF8.self))
        }
        return obj
    }

    /// Send one command line and read one JSON reply, both under the
    /// given deadline.
    public func request(_ line: String, timeoutSeconds: Double) throws -> [String: Any] {
        try send(line)
        return try readReply(timeoutSeconds: timeoutSeconds, what: "\(label): reply to '\(line)'")
    }

    /// Ask the worker to exit cleanly (bounded), then reap it. Escalates
    /// to SIGTERM if the worker ignores the request.
    public func shutdown(timeoutSeconds: Double = 30) {
        if (try? request("exit", timeoutSeconds: timeoutSeconds)) == nil, process.isRunning {
            process.terminate()
        }
        try? stdinPipe.fileHandleForWriting.close()
        process.waitUntilExit()
    }

    private func killAndReap() {
        if process.isRunning {
            process.terminate()
        }
        try? stdinPipe.fileHandleForWriting.close()
        process.waitUntilExit()
    }
}
