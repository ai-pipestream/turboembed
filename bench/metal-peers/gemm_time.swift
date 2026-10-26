// The Metal backend's F16 linear-layer kernels timed alone, one shape at
// a time, for the GEMM receipt of docs/metal.md (The F16 tile): each
// shape of an encoder layer, (n, k) = (3h, h) for Q, K and V in one
// dispatch of three, (h, h), (4h, h) and (h, 4h), at each token count, in
// each tile the backend can run, with the operand types, epilogue and
// k split the encoder gives that layer. Opt-in, outside the library and
// the build: it compiles core/metal/kernels.metal as the backend does,
// without fast math, and dispatches the kernels by name.
//
//   swift bench/metal-peers/gemm_time.swift core/metal/kernels.metal [hidden...]
//
// For each shape and tile it encodes REPS dispatches in one command
// buffer, TIMES times after one untimed buffer, and prints one JSON line:
// the least and the median time per dispatch in ms, from the buffer's GPU
// start and end and, as mlx_gemm.py times MLX, from the host (commit to
// completion), and the TFLOPS at the least GPU time.

import Foundation
import Metal

let reps = 20
let times = 7

struct GemmParams {
    var m: UInt32, n: UInt32, k: UInt32, epilogue: UInt32, splits: UInt32, kchunk: UInt32
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write("usage: gemm_time.swift <kernels.metal> [hidden...]\n".data(using: .utf8)!)
    exit(2)
}
let hiddens = args.count > 2 ? args[2...].map { Int($0)! } : [384, 768]
let source = try String(contentsOfFile: args[1], encoding: .utf8)
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

func buffer(_ bytes: Int) -> MTLBuffer {
    let b = device.makeBuffer(length: max(bytes, 16), options: .storageModePrivate)!
    return b
}

// Fill a private buffer with small F16 values (0x2c00 is 1/16), so no
// product overflows and no denormals slow it.
func fill(_ b: MTLBuffer) {
    let staging = device.makeBuffer(length: b.length, options: .storageModeShared)!
    let p = staging.contents().bindMemory(to: UInt16.self, capacity: b.length / 2)
    for i in 0..<(b.length / 2) { p[i] = 0x2c00 ^ UInt16((i * 2654435761 >> 7) & 0x8000) }
    let cb = queue.makeCommandBuffer()!
    let blit = cb.makeBlitCommandEncoder()!
    blit.copy(from: staging, sourceOffset: 0, to: b, destinationOffset: 0, size: b.length)
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
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

for h in hiddens {
    // (name, n, k, layers in one dispatch, x half, y half, epilogue)
    let shapes: [(String, Int, Int, Int, Bool, Bool, UInt32)] = [
        ("qkv", h, h, 3, true, true, 1),
        ("attn_out", h, h, 1, true, false, 0),
        ("ffn_in", 4 * h, h, 1, true, true, 2),
        ("ffn_out", h, 4 * h, 1, true, false, 0),
    ]
    for m in [128, 512, 1024, 4096, 16384] {
        for (shape, n, k, z, xh, yh, epilogue) in shapes {
            let x = buffer(m * k * 2)
            fill(x)
            let w = (0..<3).map { _ -> MTLBuffer in let b = buffer(n * k * 2); fill(b); return b }
            let bias = buffer(n * 4)
            for tile in tiles + [Tile(name: "auto", bm: 0, bn: 0, threads: 0)] {
                let t = tile.name == "auto" ? (m <= autoTokens ? tiles[3] : tiles[1]) : tile
                let count = (n + t.bn - 1) / t.bn * ((m + t.bm - 1) / t.bm)
                // The encoder splits k for a single layer with no epilogue
                // whose tiles leave most of the GPU idle; its output is F32.
                var splits = 1
                if tile.name == "auto" && z == 1 && epilogue == 0 && count < spread {
                    splits = max(1, min((spread + count - 1) / count, k / 64, 8))
                }
                let kchunk = ((k + splits - 1) / splits + 31) / 32 * 32
                splits = (k + kchunk - 1) / kchunk
                let yHalf = yh && splits == 1
                let y = (0..<3).map { _ in buffer(splits * m * n * (yHalf ? 2 : 4)) }
                let name = "gemm_\(t.name)_\(xh ? "h" : "f")\(yHalf ? "h" : "f")"
                var p = GemmParams(m: UInt32(m), n: UInt32(n), k: UInt32(k), epilogue: epilogue,
                                   splits: UInt32(splits), kchunk: UInt32(kchunk))
                let pso = pipeline(name)
                if pso.maxTotalThreadsPerThreadgroup < t.threads { continue }
                var ms: [Double] = []
                var host: [Double] = []
                for i in 0...times {
                    let cb = queue.makeCommandBuffer()!
                    let enc = cb.makeComputeCommandEncoder()!
                    enc.setComputePipelineState(pso)
                    enc.setBuffer(x, offset: 0, index: 0)
                    for j in 0..<3 {
                        enc.setBuffer(w[j], offset: 0, index: 1 + j)
                        enc.setBuffer(bias, offset: 0, index: 4 + j)
                        enc.setBuffer(y[j], offset: 0, index: 7 + j)
                    }
                    enc.setBytes(&p, length: MemoryLayout<GemmParams>.size, index: 10)
                    for _ in 0..<reps {
                        enc.dispatchThreadgroups(
                            MTLSize(width: (n + t.bn - 1) / t.bn, height: (m + t.bm - 1) / t.bm, depth: z * splits),
                            threadsPerThreadgroup: MTLSize(width: t.threads, height: 1, depth: 1))
                    }
                    enc.endEncoding()
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    cb.commit()
                    cb.waitUntilCompleted()
                    let t1 = DispatchTime.now().uptimeNanoseconds
                    if i > 0 {
                        ms.append((cb.gpuEndTime - cb.gpuStartTime) * 1000 / Double(reps))
                        host.append(Double(t1 - t0) / 1e6 / Double(reps))
                    }
                }
                ms.sort()
                host.sort()
                let flops = 2.0 * Double(m) * Double(n) * Double(k) * Double(z)
                let line = String(
                    format: "{\"side\":\"turbo\",\"hidden\":%d,\"shape\":\"%@\",\"m\":%d,\"n\":%d,\"k\":%d,\"layers\":%d,"
                        + "\"tile\":\"%@\",\"splits\":%d,\"least_ms\":%.4f,\"median_ms\":%.4f,\"host_least_ms\":%.4f,"
                        + "\"host_median_ms\":%.4f,\"tflops\":%.3f}",
                    h, shape, m, n, k, z, tile.name, splits, ms[0], ms[ms.count / 2], host[0], host[host.count / 2],
                    flops / ms[0] / 1e9)
                print(line)
            }
        }
    }
}
