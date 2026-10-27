// The Metal backend's F16 linear-layer kernels timed alone, one shape at
// a time, for the GEMM receipt of docs/metal.md (The linear layers alone
// against MLX). Opt-in, outside the library and the build: it compiles
// core/metal/kernels.metal as the backend does, without fast math, and
// dispatches the kernels by name.
//
//   swiftc -O bench/metal-peers/gemm_time.swift -o gemm_time
//   ./gemm_time core/metal/kernels.metal [hidden...]
//
// Two modes, each shape in each tile the backend can run:
// - "bare": y = x w^T alone, as mlx_gemm.py times MLX's x @ w: F16
//   operands, F32 sums (the SIMD-group matrices are F32), F16 out, no
//   bias, no k split, one layer, so Q, K and V are one product of n = 3h.
// - "encoder": as the encoder runs the layer: Q, K and V as one dispatch
//   of three with their biases, the attention output split as auto
//   splits it, the feed-forward input with bias and GELU, its output.
// Each figure is one dispatch per command buffer, committed and waited
// for, from the host and from the buffer's GPU start and end. Only bare
// mode matches mlx_gemm.py's single product per mx.eval. Encoder mode
// is a separate diagnostic with different epilogues and output types.
// Each is TIMES timed after one untimed, with raw samples, least, mean,
// median and nearest-rank p99 in ms. No overlapping writes or assumed
// number of independent products enter this comparison.

import CryptoKit
import Foundation
import Metal

let times = 20

struct GemmParams {
    var m: UInt32, n: UInt32, k: UInt32, epilogue: UInt32, splits: UInt32, kchunk: UInt32
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write("usage: gemm_time <kernels.metal> [hidden...]\n".data(using: .utf8)!)
    exit(2)
}
// GEMM_ONLY=<hidden>,<m>,<shape>[,<mode>] times that one shape alone, so
// a driver can interleave it with mlx_gemm.py's run of the same shape.
let only = ProcessInfo.processInfo.environment["GEMM_ONLY"].map { $0.split(separator: ",").map(String.init) }
let hiddens = only.map { [Int($0[0])!] } ?? (args.count > 2 ? args[2...].map { Int($0)! } : [384, 768])
let sourceData = try Data(contentsOf: URL(fileURLWithPath: args[1]))
let source = String(decoding: sourceData, as: UTF8.self)
let digest = SHA256.hash(data: sourceData).map { String(format: "%02x", $0) }.joined()
let device = MTLCreateSystemDefaultDevice()!
let options = MTLCompileOptions()
if #available(macOS 15.0, *) { options.mathMode = .safe } else { options.fastMathEnabled = false }
let library = try device.makeLibrary(source: source, options: options)
var pipelines: [String: MTLComputePipelineState] = [:]
func pipeline(_ name: String) -> MTLComputePipelineState {
    if let p = pipelines[name] { return p }
    let p = try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
    pipelines[name] = p
    return p
}
let queue = device.makeCommandQueue()!
print("{\"side\":\"turbo\",\"kernels_sha256\":\"\(digest)\",\"device\":\"\(device.name)\","
    + "\"schema\":2,\"schedule\":\"single\",\"input_dtype\":\"float16\",\"accumulator_dtype\":\"float32\","
    + "\"weight_layout\":\"contiguous-n-k\",\"warmup\":1,\"times\":\(times)}")

func requireCompleted(_ cb: MTLCommandBuffer) {
    guard cb.status == .completed else {
        let message = cb.error?.localizedDescription ?? "command buffer did not complete"
        FileHandle.standardError.write("Metal failure: \(message)\n".data(using: .utf8)!)
        exit(1)
    }
}

// A private buffer filled from `fillHost` through a shared staging copy.
func buffer(_ bytes: Int, _ fillHost: ((UnsafeMutableRawPointer, Int) -> Void)? = nil) -> MTLBuffer {
    let b = device.makeBuffer(length: max(bytes, 16), options: .storageModePrivate)!
    guard let f = fillHost else { return b }
    let staging = device.makeBuffer(length: b.length, options: .storageModeShared)!
    f(staging.contents(), b.length)
    let cb = queue.makeCommandBuffer()!
    let blit = cb.makeBlitCommandEncoder()!
    blit.copy(from: staging, sourceOffset: 0, to: b, destinationOffset: 0, size: b.length)
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    requireCompleted(cb)
    return b
}
// F16 values of 1/16 with a pseudo-random sign: no product overflows, no
// denormal slows it.
func halves(_ p: UnsafeMutableRawPointer, _ bytes: Int) {
    let h = p.bindMemory(to: UInt16.self, capacity: bytes / 2)
    for i in 0..<(bytes / 2) { h[i] = 0x2c00 ^ UInt16(truncatingIfNeeded: (i &* 2654435761) >> 7) & 0x8000 }
}
func floats(_ p: UnsafeMutableRawPointer, _ bytes: Int) {
    let f = p.bindMemory(to: Float.self, capacity: bytes / 4)
    for i in 0..<(bytes / 4) { f[i] = 0.01 * Float(i % 7) - 0.03 }
}

struct Tile { let name: String; let bm: Int; let bn: Int; let threads: Int }
let tiles = [
    Tile(name: "h32", bm: 32, bn: 64, threads: 128),
    Tile(name: "h64", bm: 64, bn: 64, threads: 128),
    Tile(name: "h128", bm: 128, bn: 64, threads: 256),
    Tile(name: "h32n32", bm: 32, bn: 32, threads: 128),
]
// docs/metal.md: auto is h32n32 up to 1024 packed tokens, h64 past it.
let autoTokens = 1024
let spread = 128

func stats(_ v: [Double]) -> String {
    let s = v.sorted()
    let mean = s.reduce(0, +) / Double(s.count)
    let mid = s.count / 2
    let median = s.count % 2 == 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid]
    let p99 = s[Int(ceil(0.99 * Double(s.count))) - 1]
    return "{\"least\":\(s[0]),\"mean\":\(mean),\"median\":\(median),\"p99\":\(p99),\"samples\":\(v)}"
}

for mode in only.flatMap({ $0.count > 3 ? [$0[3]] : nil }) ?? ["bare", "encoder"] {
    for h in hiddens {
        // (name, n, k, layers in one dispatch, y half, epilogue)
        let shapes: [(String, Int, Int, Int, Bool, UInt32)] = mode == "bare"
            ? [("qkv", 3 * h, h, 1, true, 0), ("attn_out", h, h, 1, true, 0), ("ffn_in", 4 * h, h, 1, true, 0),
               ("ffn_out", h, 4 * h, 1, true, 0)]
            : [("qkv", h, h, 3, true, 1), ("attn_out", h, h, 1, false, 0), ("ffn_in", 4 * h, h, 1, true, 2),
               ("ffn_out", h, 4 * h, 1, false, 0)]
        for m in only.map({ [Int($0[1])!] }) ?? [128, 512, 1024, 4096, 16384] {
            for (shape, n, k, z, yh, epilogue) in shapes where only == nil || only![2] == shape {
                let x = buffer(m * k * 2, halves)
                let w = (0..<3).map { _ in buffer(n * k * 2, halves) }
                let bias = buffer(n * 4, floats)
                for tile in tiles + [Tile(name: "auto", bm: 0, bn: 0, threads: 0)] {
                    let t = tile.name == "auto" ? (m <= autoTokens ? tiles[3] : tiles[1]) : tile
                    let count = (n + t.bn - 1) / t.bn * ((m + t.bm - 1) / t.bm)
                    // In encoder mode, k is split as the encoder splits a
                    // single layer with no epilogue whose tiles leave most
                    // of the GPU idle, bounded as it bounds it by the
                    // session's capacity over the run's tokens; its output
                    // is then F32. The session here holds exactly the run,
                    // cap the run's tokens rounded up to 32, so where the
                    // run fills it (every M here) that bound is 1.
                    var splits = 1
                    let cap = (m + 31) / 32 * 32
                    if mode == "encoder" && tile.name == "auto" && z == 1 && epilogue == 0 && count < spread {
                        splits = max(1, min((spread + count - 1) / count, k / 64, cap / m, 8))
                    }
                    let kchunk = ((k + splits - 1) / splits + 31) / 32 * 32
                    splits = (k + kchunk - 1) / kchunk
                    let yHalf = yh && splits == 1
                    let y = (0..<3).map { _ in buffer(splits * m * n * (yHalf ? 2 : 4)) }
                    let pso = pipeline("gemm_\(t.name)_h\(yHalf ? "h" : "f")")
                    if pso.maxTotalThreadsPerThreadgroup < t.threads { continue }
                    var p = GemmParams(m: UInt32(m), n: UInt32(n), k: UInt32(k), epilogue: epilogue,
                                       splits: UInt32(splits), kchunk: UInt32(kchunk))
                    func run() -> (Double, Double) {
                        let cb = queue.makeCommandBuffer()!
                        let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
                        enc.setComputePipelineState(pso)
                        enc.setBuffer(x, offset: 0, index: 0)
                        for j in 0..<3 {
                            enc.setBuffer(w[j], offset: 0, index: 1 + j)
                            enc.setBuffer(bias, offset: 0, index: 4 + j)
                            enc.setBuffer(y[j], offset: 0, index: 7 + j)
                        }
                        enc.setBytes(&p, length: MemoryLayout<GemmParams>.size, index: 10)
                        enc.dispatchThreadgroups(
                            MTLSize(width: (n + t.bn - 1) / t.bn, height: (m + t.bm - 1) / t.bm, depth: z * splits),
                            threadsPerThreadgroup: MTLSize(width: t.threads, height: 1, depth: 1))
                        enc.endEncoding()
                        let t0 = DispatchTime.now().uptimeNanoseconds
                        cb.commit()
                        cb.waitUntilCompleted()
                        let t1 = DispatchTime.now().uptimeNanoseconds
                        requireCompleted(cb)
                        let gpu = (cb.gpuEndTime - cb.gpuStartTime) * 1000
                        let host = Double(t1 - t0) / 1e6
                        guard gpu.isFinite && host.isFinite && gpu > 0 && host > 0 else {
                            FileHandle.standardError.write("invalid timing sample\n".data(using: .utf8)!)
                            exit(1)
                        }
                        return (gpu, host)
                    }
                    var single: [(Double, Double)] = []
                    _ = run()
                    for _ in 0..<times { single.append(run()) }
                    let flops = 2.0 * Double(m) * Double(n) * Double(k) * Double(z)
                    print("{\"side\":\"turbo\",\"mode\":\"\(mode)\",\"hidden\":\(h),\"shape\":\"\(shape)\",\"m\":\(m),"
                        + "\"n\":\(n),\"k\":\(k),\"layers\":\(z),\"tile\":\"\(tile.name)\",\"splits\":\(splits),"
                        + "\"flops\":\(Int(flops)),\"epilogue\":\(epilogue),\"output_dtype\":\"\(yHalf ? "float16" : "float32")\","
                        + "\"single_gpu\":\(stats(single.map { $0.0 })),\"single_host\":\(stats(single.map { $0.1 }))}")
                }
            }
        }
    }
}
