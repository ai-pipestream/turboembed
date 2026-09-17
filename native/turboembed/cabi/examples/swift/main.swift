// External Swift consumer example for the packaged turboembed.h C ABI.
//
// Build against an extracted SDK prefix (no SwiftPM, no source checkout):
//
//   swiftc main.swift \
//     -import-objc-header "$SDK/include/turboembed.h" \
//     -L "$SDK/lib" -lTurboEmbed \
//     -Xlinker -rpath -Xlinker "$SDK/lib" \
//     -o turboembed_embed_swift
//
// Usage: turboembed_embed_swift <catalog.toml> <alias> [auto|metal|cuda|cpu|mock]
//
// Creates an engine on the requested device (default metal), loads the
// catalog alias, embeds one short and one Unicode text as a batch, and
// verifies the reported dimension and L2 norms. Device policy is the
// library's: a missing accelerator is a loud error, never a CPU fallback.

import Foundation

func fail(_ what: String, _ engine: OpaquePointer?) -> Int32 {
    FileHandle.standardError.write(
        Data("\(what) failed: \(String(cString: turboembed_last_error(engine)))\n".utf8))
    return 1
}

let args = CommandLine.arguments
guard args.count >= 3 && args.count <= 4 else {
    FileHandle.standardError.write(
        Data("usage: \(args[0]) <catalog.toml> <alias> [auto|metal|cuda|cpu|mock]\n".utf8))
    exit(2)
}
let catalog = args[1]
let alias = args[2]
let deviceName = args.count == 4 ? args[3] : "metal"

let device: turboembed_device
switch deviceName {
case "auto": device = TURBOEMBED_DEVICE_AUTO
case "metal": device = TURBOEMBED_DEVICE_METAL
case "cuda": device = TURBOEMBED_DEVICE_CUDA
case "cpu": device = TURBOEMBED_DEVICE_CPU
case "mock": device = TURBOEMBED_DEVICE_MOCK
default:
    FileHandle.standardError.write(Data("unknown device \(deviceName)\n".utf8))
    exit(2)
}

print("abi_version=\(turboembed_abi_version())")

var engine: OpaquePointer?
guard turboembed_engine_create(device, catalog, &engine) == TURBOEMBED_OK, let engine else {
    exit(fail("engine create", nil))
}
defer { turboembed_engine_destroy(engine) }

guard alias.withCString({ turboembed_load_model(engine, $0, alias.utf8.count) }) == TURBOEMBED_OK
else {
    exit(fail("load_model", engine))
}

var infos: UnsafeMutablePointer<turboembed_model_info>?
var nInfos = 0
guard turboembed_list_models(engine, &infos, &nInfos) == TURBOEMBED_OK, let infos else {
    exit(fail("list_models", engine))
}
for i in 0..<nInfos {
    let info = infos[i]
    let name = String(
        decoding: UnsafeBufferPointer(
            start: UnsafeRawPointer(info.alias.ptr)?.assumingMemoryBound(to: UInt8.self),
            count: info.alias.len),
        as: UTF8.self)
    print("model=\(name) device=\(String(cString: turboembed_device_name(info.device))) dim=\(info.dim)")
}
turboembed_model_list_free(infos, nInfos)

let inputs = ["hello world", "das Straßenpflaster glänzt — 東京"]
var buffers: [UnsafeMutablePointer<UInt8>] = []
defer { buffers.forEach { $0.deallocate() } }
var views: [turboembed_str] = inputs.map { text in
    let bytes = Array(text.utf8)
    let copy = UnsafeMutablePointer<UInt8>.allocate(capacity: bytes.count)
    bytes.withUnsafeBufferPointer { copy.update(from: $0.baseAddress!, count: $0.count) }
    buffers.append(copy)
    return turboembed_str(
        ptr: UnsafeRawPointer(copy).assumingMemoryBound(to: CChar.self), len: bytes.count)
}

var opts = turboembed_embed_options(
    pooling: TURBOEMBED_POOLING_DEFAULT,
    normalize: -1,  // catalog default
    truncate_to: 0,
    output_format: TURBOEMBED_OUTPUT_TYPED
)

var result: UnsafeMutablePointer<turboembed_embed_result>?
let st = alias.withCString { cAlias in
    turboembed_embed(engine, cAlias, alias.utf8.count, &views, views.count, &opts, &result)
}
guard st == TURBOEMBED_OK, let result else {
    exit(fail("embed", engine))
}
let dim = Int(result.pointee.dim)
let count = Int(result.pointee.count)
guard dim > 0, count == inputs.count, result.pointee.values != nil else {
    FileHandle.standardError.write(Data("embed returned an empty result\n".utf8))
    turboembed_embed_result_free(result)
    exit(1)
}
var normsOk = true
for row in 0..<count {
    var sumSq = 0.0
    for d in 0..<dim {
        let v = Double(result.pointee.values[row * dim + d])
        sumSq += v * v
    }
    let norm = sumSq.squareRoot()
    print(String(format: "row=%d dim=%d norm=%.6f", row, dim, norm))
    if abs(norm - 1.0) > 1e-3 { normsOk = false }
}
turboembed_embed_result_free(result)

guard normsOk else {
    FileHandle.standardError.write(Data("L2 norms are off; catalog normalize=true expected\n".utf8))
    exit(1)
}
print("embed=PASS device=\(deviceName)")
