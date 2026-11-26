// Scratch sweep (#812): bare y = x w^T, F16 in/out, GPU time per dispatch,
// tiles interleaved per sample; checks every tile's output bits against the first.
import Foundation
import Metal
struct GemmParams { var m: UInt32, n: UInt32, k: UInt32, epilogue: UInt32, splits: UInt32, kchunk: UInt32 }
let env = ProcessInfo.processInfo.environment
let src = try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
let dev = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions(); opts.mathMode = .safe
let lib = try dev.makeLibrary(source: src, options: opts)
let q = dev.makeCommandQueue()!
// tiles: name:bm:bn:threads
let tiles = (env["TILES"] ?? "h32n32a:32:32:128,h64:64:64:128").split(separator: ",").map { s -> (String, Int, Int, Int) in
  let f = s.split(separator: ":").map(String.init); return (f[0], Int(f[1])!, Int(f[2])!, Int(f[3])!) }
struct MP { var M: Int32, N: Int32, K: Int32, lda: Int32, ldb: Int32, ldd: Int32, tn: Int32, tm: Int32
  var sa: Int64, sb: Int64, sd: Int64, swz: Int32, kit: Int32, bnd: Int32 }
let mlib = try dev.makeLibrary(URL: URL(fileURLWithPath: env["MLXLIB"] ?? "/nonexistent"))
func mpso(_ name: String) -> MTLComputePipelineState {
  let fc = MTLFunctionConstantValues(); var f = false, t = true
  fc.setConstantValue(&f, type: .bool, index: 10); fc.setConstantValue(&f, type: .bool, index: 100)
  fc.setConstantValue(&f, type: .bool, index: 110)
  for i in 200...202 { fc.setConstantValue(&t, type: .bool, index: i) }
  return try! dev.makeComputePipelineState(function: try! mlib.makeFunction(name: name, constantValues: fc)) }
// mlxnt: MLX's nt kernel on our [n, k] weights, the M2's choice for x w^T.
let psos = tiles.map { $0.0 == "mlxnt" ? mpso("steel_gemm_fused_nt_float16_float16_bm64_bn32_bk32_wm2_wn2")
  : try! dev.makeComputePipelineState(function: lib.makeFunction(name: "gemm_\($0.0)_hh")!) }
let times = Int(env["TIMES"] ?? "20")!
func buf(_ bytes: Int, _ fill: Bool) -> MTLBuffer {
  let b = dev.makeBuffer(length: bytes, options: .storageModeShared)!
  if fill { let h = b.contents().bindMemory(to: UInt16.self, capacity: bytes/2)
    for i in 0..<(bytes/2) { h[i] = (0x2c00 + UInt16(truncatingIfNeeded: (i &* 2654435761) >> 13) & 0x1ff) ^ UInt16(truncatingIfNeeded: (i &* 2654435761) >> 7) & 0x8000 } }
  return b }
let shapesEnv = env["SHAPES"] ?? "384,128;768,4096;768,16384"
for hs in shapesEnv.split(separator: ";") {
  let hm = hs.split(separator: ",").map { Int($0)! }; let h = hm[0], m = hm[1]
  for (shape, n, k) in [("qkv", 3*h, h), ("attn_out", h, h), ("ffn_in", 4*h, h), ("ffn_out", h, 4*h)] {
    if let only = env["SHAPE"], only != shape { continue }
    let x = buf(m*k*2, true), w = buf(n*k*2, true)
    let ys = tiles.map { _ in buf(m*n*2, false) }
    var p = GemmParams(m: UInt32(m), n: UInt32(n), k: UInt32(k), epilogue: 0, splits: 1, kchunk: UInt32(k))
    func run(_ t: Int) -> Double {
      if tiles[t].0 == "mlxnt" {
        let tn = n / 32, tm = m / 64
        var mp = MP(M: Int32(m), N: Int32(n), K: Int32(k), lda: Int32(k), ldb: Int32(k), ldd: Int32(n), tn: Int32(tn), tm: Int32(tm), sa: 0, sb: 0, sd: 0, swz: 0, kit: Int32(k / 32), bnd: 1)
        let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(psos[t]); e.setBuffer(x, offset: 0, index: 0); e.setBuffer(w, offset: 0, index: 1); e.setBuffer(ys[t], offset: 0, index: 3)
        e.setBytes(&mp, length: MemoryLayout<MP>.stride, index: 4)
        e.dispatchThreadgroups(MTLSize(width: tn, height: tm, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed else { print("FAIL mlx"); exit(1) }
        return (cb.gpuEndTime - cb.gpuStartTime) * 1000
      }
      let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
      e.setComputePipelineState(psos[t]); e.setBuffer(x, offset: 0, index: 0)
      for j in 0..<3 { e.setBuffer(w, offset: 0, index: 1+j); e.setBuffer(x, offset: 0, index: 4+j); e.setBuffer(ys[t], offset: 0, index: 7+j) }
      e.setBytes(&p, length: MemoryLayout<GemmParams>.size, index: 10)
      e.setBuffer(x, offset: 0, index: 11); e.setBuffer(x, offset: 0, index: 12)
      let (_, bm, bn, th) = tiles[t]
      e.dispatchThreadgroups(MTLSize(width: (n+bn-1)/bn, height: (m+bm-1)/bm, depth: 1), threadsPerThreadgroup: MTLSize(width: th, height: 1, depth: 1))
      e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
      guard cb.status == .completed else { print("FAIL \(tiles[t].0) \(cb.error!)"); exit(1) }
      return (cb.gpuEndTime - cb.gpuStartTime) * 1000 }
    var s = tiles.map { _ in [Double]() }
    for t in tiles.indices { _ = run(t) }
    for i in 0..<times { for j in tiles.indices { let t = (i + j) % tiles.count; s[t].append(run(t)) } }
    let ref = Data(bytes: ys[0].contents(), count: m*n*2)
    var line = "h=\(h) m=\(m) \(shape)"
    for t in tiles.indices {
      let v = s[t].sorted(); let med = v[v.count/2]
      let same = Data(bytes: ys[t].contents(), count: m*n*2) == ref
      line += String(format: " | %@ %.3f%@", tiles[t].0, med, same ? "" : " BITDIFF") }
    print(line); fflush(stdout)
  }
}
